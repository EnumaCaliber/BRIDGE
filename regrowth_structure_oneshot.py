"""
Structured RL-based One-Shot Regrowth
======================================
Channel restoration via DependencyGraph (using index_map / pruned_dense_map)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import copy
import os
import time
import wandb
import random
from torch.distributions import Categorical
from collections import deque
import torch_pruning as tp

from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.analysis_tools import (
    BlockwiseFeatureExtractor, compute_block_ssim, load_model_name,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# Structured Pruning Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_original_channels(dense_model):
    return {n: m.out_channels for n, m in dense_model.named_modules()
            if isinstance(m, nn.Conv2d)}


def compute_channel_sparsity(model, original_channels):
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total     += original_channels[name]
            remaining += m.out_channels
    return (1 - remaining / total) if total > 0 else 0.0


def get_target_layers(pruned_dense_map):
    """Layers with non-empty pruned_dense_map entries are still restorable."""
    return [name for name, pruned_list in pruned_dense_map.items()
            if len(pruned_list) > 0]


def get_layer_capacities(pruned_dense_map, target_layers):
    """Restorable channel count per layer = length of pruned_dense_map entry."""
    return [len(pruned_dense_map[name]) for name in target_layers]


# ═══════════════════════════════════════════════════════════════════════════════
# DependencyGraph-based Channel Restoration
# ═══════════════════════════════════════════════════════════════════════════════

def _get_dense_weight(layer_name, dense_idx, dim, dense_named):
    """
    Extract specific channel slices from the dense model weight tensor.

    Args:
        layer_name  : dot-separated module name (e.g. 'layer1.0.conv1')
        dense_idx   : list of channel indices in the dense model
        dim         : 0 → out_channels axis,  1 → in_channels axis
        dense_named : pre-built dict(dense_model.named_modules())

    Returns:
        Tensor of shape [len(dense_idx), ...] or None if layer not found.
    """
    m = dense_named.get(layer_name)
    if m is None:
        return None
    w     = m.weight.data
    idx_t = torch.tensor(dense_idx, device=w.device)
    return w.index_select(dim, idx_t)


def regrow_channels_dg(model, dense_model, example_inputs,
                       allocation_dense_idx, index_map,
                       pruned_dense_map, device):
    """
    Restore pruned channels back into the model using DependencyGraph.
    Weights are copied from the corresponding dense model channels
    (falls back to zeros / identity values when unavailable).
    index_map and pruned_dense_map are updated in-place.

    Supported layer types
    ─────────────────────
    Conv2d  depthwise   groups == in_channels == out_channels
    Conv2d  out         expanding out_channels
    Conv2d  in          expanding in_channels
    BatchNorm2d         num_features expansion
    Linear  in          expanding in_features  (VGG/EfficientNet head)
    Linear  out         expanding out_features (SE excitation)

    Args:
        model                : sparse model (modified in-place)
        dense_model          : original dense model (read-only reference)
        example_inputs       : single-sample tensor for DG tracing
        allocation_dense_idx : {layer_name: [dense_idx, ...]}
        index_map            : {layer_name: [survived_dense_idx]}  ← updated in-place
        pruned_dense_map     : {layer_name: [pruned_dense_idx]}    ← updated in-place
        device               : torch device string
    """
    named_modules = dict(model.named_modules())
    dense_named   = dict(dense_model.named_modules())

    for target_name, dense_indices in allocation_dense_idx.items():
        if not dense_indices:
            continue

        # Skip indices already present in index_map
        # (a previous layer's DG pass may have pulled them in transitively)
        already_survived = set(index_map.get(target_name, []))
        dense_indices = [idx for idx in dense_indices if idx not in already_survived]
        if not dense_indices:
            print(f"  [DG] {target_name}: all indices already survived, skipping")
            continue

        target_conv = named_modules.get(target_name)
        if target_conv is None:
            print(f"  [DG] {target_name} not found, skipping")
            continue

        num_grow = len(dense_indices)

        # Rebuild DG every time the model structure changes
        DG = tp.DependencyGraph().build_dependency(model, example_inputs)
        group = DG.get_pruning_group(
            target_conv, tp.prune_conv_out_channels, idxs=[0]
        )

        already_processed = set()  # guard against double-expansion of depthwise layers

        for dep, _ in group:
            layer   = dep.target.module
            handler = dep.handler
            name    = dep.target.name

            # ── Skip parameter-free nodes (ReLU, Add, etc. → _ElementWiseOp)
            if not list(layer.parameters(recurse=False)):
                print(f"    · skip [{name}]: {type(layer).__name__} (no params)")
                continue

            # ── Case 1: Conv2d — depthwise (groups == in_channels == out_channels)
            #    Must be checked before the regular out/in cases.
            #    DG emits both prune_conv_out_channels and prune_conv_in_channels
            #    for the same depthwise layer, so we guard with already_processed.
            if (isinstance(layer, nn.Conv2d)
                    and layer.groups > 1
                    and layer.groups == layer.in_channels):

                if name in already_processed:
                    continue
                already_processed.add(name)

                old_w   = layer.weight.data              # [C, 1, kH, kW]
                dense_w = _get_dense_weight(name, dense_indices, dim=0, dense_named=dense_named)
                new_w   = dense_w if dense_w is not None else \
                          torch.zeros(num_grow, 1, *old_w.shape[2:], device=device)

                layer.weight       = nn.Parameter(torch.cat([old_w, new_w], dim=0))
                layer.out_channels += num_grow
                layer.in_channels  += num_grow
                layer.groups       += num_grow

                if layer.bias is not None:
                    dense_layer = dense_named.get(name)
                    new_b = (dense_layer.bias.data[dense_indices]
                             if dense_layer is not None and dense_layer.bias is not None
                             else torch.zeros(num_grow, device=device))
                    layer.bias = nn.Parameter(torch.cat([layer.bias.data, new_b]))

                print(f"    ✓ conv dw  [{name}]: out/in/groups → {layer.out_channels}")

            # ── Case 2: Conv2d — regular, expanding out_channels
            elif (isinstance(layer, nn.Conv2d)
                  and handler == tp.prune_conv_out_channels):

                old_w   = layer.weight.data              # [out, in, kH, kW]
                dense_w = _get_dense_weight(name, dense_indices, dim=0, dense_named=dense_named)

                if dense_w is not None:
                    # Align in_channels: sparse in_ch may be < dense in_ch
                    min_in = min(old_w.shape[1], dense_w.shape[1])
                    new_w  = torch.zeros(num_grow, old_w.shape[1],
                                         *old_w.shape[2:], device=device)
                    new_w[:, :min_in] = dense_w[:, :min_in]
                else:
                    new_w = torch.zeros(num_grow, old_w.shape[1],
                                        *old_w.shape[2:], device=device)

                layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=0))
                layer.out_channels += num_grow

                if layer.bias is not None:
                    dense_layer = dense_named.get(name)
                    new_b = (dense_layer.bias.data[dense_indices]
                             if dense_layer is not None and dense_layer.bias is not None
                             else torch.zeros(num_grow, device=device))
                    layer.bias = nn.Parameter(torch.cat([layer.bias.data, new_b]))

                print(f"    ✓ conv out [{name}]: "
                      f"{old_w.shape[0]} → {layer.out_channels}")

            # ── Case 3: Conv2d — regular, expanding in_channels
            elif (isinstance(layer, nn.Conv2d)
                  and handler == tp.prune_conv_in_channels):

                old_w   = layer.weight.data              # [out, in, kH, kW]
                dense_w = _get_dense_weight(name, dense_indices, dim=1, dense_named=dense_named)

                if dense_w is not None:
                    # Align out_channels: sparse out_ch may be < dense out_ch
                    min_out = min(old_w.shape[0], dense_w.shape[0])
                    new_w   = torch.zeros(old_w.shape[0], num_grow,
                                          *old_w.shape[2:], device=device)
                    new_w[:min_out] = dense_w[:min_out]
                else:
                    new_w = torch.zeros(old_w.shape[0], num_grow,
                                        *old_w.shape[2:], device=device)

                layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=1))
                layer.in_channels += num_grow
                print(f"    ✓ conv in  [{name}]: "
                      f"{old_w.shape[1]} → {layer.in_channels}")

            # ── Case 4: BatchNorm2d
            elif isinstance(layer, nn.BatchNorm2d):
                old_ch   = layer.num_features
                dense_bn = dense_named.get(name)

                # Learnable affine params
                for attr, default_fill in [('weight', 1.0), ('bias', 0.0)]:
                    old = getattr(layer, attr).data
                    new = (getattr(dense_bn, attr).data[dense_indices]
                           if dense_bn is not None
                           else torch.full((num_grow,), default_fill, device=device))
                    setattr(layer, attr, nn.Parameter(torch.cat([old, new])))

                # Running statistics
                for attr, default_fill in [('running_mean', 0.0), ('running_var', 1.0)]:
                    old = getattr(layer, attr)
                    new = (getattr(dense_bn, attr)[dense_indices]
                           if dense_bn is not None
                           else torch.full((num_grow,), default_fill, device=device))
                    setattr(layer, attr, torch.cat([old, new]))

                layer.num_features += num_grow
                print(f"    ✓ bn       [{name}]: {old_ch} → {layer.num_features}")

            # ── Case 5: Linear — expanding in_features
            #    Triggered when the last conv before a flatten feeds into a Linear
            #    (VGG16 classifier, EfficientNet/ShuffleNet head, etc.)
            elif (isinstance(layer, nn.Linear)
                  and handler == tp.prune_linear_in_channels):

                old_w        = layer.weight.data          # [out_features, in_features]
                dense_linear = dense_named.get(name)

                if dense_linear is not None:
                    new_w   = dense_linear.weight.data[:, dense_indices]
                    min_out = min(old_w.shape[0], new_w.shape[0])
                    pad_w   = torch.zeros(old_w.shape[0], num_grow, device=device)
                    pad_w[:min_out] = new_w[:min_out]
                    new_w = pad_w
                else:
                    new_w = torch.zeros(old_w.shape[0], num_grow, device=device)

                layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=1))
                layer.in_features += num_grow
                print(f"    ✓ linear in [{name}]: "
                      f"{old_w.shape[1]} → {layer.in_features}")

            # ── Case 6: Linear — expanding out_features
            #    Triggered inside SE blocks (EfficientNet) where the squeeze Linear
            #    output feeds into the excitation Linear whose width == channel count.
            elif (isinstance(layer, nn.Linear)
                  and handler == tp.prune_linear_out_channels):

                old_w        = layer.weight.data          # [out_features, in_features]
                dense_linear = dense_named.get(name)

                if dense_linear is not None:
                    new_w   = dense_linear.weight.data[dense_indices]
                    min_in  = min(old_w.shape[1], new_w.shape[1])
                    pad_w   = torch.zeros(num_grow, old_w.shape[1], device=device)
                    pad_w[:, :min_in] = new_w[:, :min_in]
                    new_w = pad_w
                else:
                    new_w = torch.zeros(num_grow, old_w.shape[1], device=device)

                layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=0))
                layer.out_features += num_grow

                if layer.bias is not None:
                    new_b = (dense_linear.bias.data[dense_indices]
                             if dense_linear is not None and dense_linear.bias is not None
                             else torch.zeros(num_grow, device=device))
                    layer.bias = nn.Parameter(torch.cat([layer.bias.data, new_b]))

                print(f"    ✓ linear out[{name}]: "
                      f"{old_w.shape[0]} → {layer.out_features}")

            # ── Fallback: unhandled combination → hard fail so nothing is silently skipped
            else:
                raise NotImplementedError(
                    f"Unhandled combination — "
                    f"layer type: '{type(layer).__name__}', "
                    f"handler: '{handler}', "
                    f"name: [{name}]. "
                    f"Please add a case for this."
                )

        # ── Sync index_map and pruned_dense_map (in-place)
        index_map[target_name] = sorted(
            index_map[target_name] + list(dense_indices))
        pruned_dense_map[target_name] = sorted(
            set(pruned_dense_map[target_name]) - set(dense_indices))

        # Refresh named_modules after structural change before processing next layer
        named_modules = dict(model.named_modules())

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SSIM Layer Selector
# ═══════════════════════════════════════════════════════════════════════════════

class SSIMLayerSelector:
    @staticmethod
    def update_search_space(sparse_model, pretrained_model,
                            data_loader_ref, target_layers,
                            threshold=0.0, num_batches=64):
        block_dict = {'all_layers': target_layers}
        ext_pre  = BlockwiseFeatureExtractor(pretrained_model, block_dict)
        ext_spar = BlockwiseFeatureExtractor(sparse_model,     block_dict)
        with torch.no_grad():
            feats_pre  = ext_pre.extract_block_features(data_loader_ref,  num_batches=num_batches)
            feats_spar = ext_spar.extract_block_features(data_loader_ref, num_batches=num_batches)
        block_ssim = compute_block_ssim(feats_pre, feats_spar).get('all_layers', {})

        ssim_dict, selected = {}, []
        for lname in target_layers:
            score = float(block_ssim.get(lname, 0.5))
            ssim_dict[lname] = score
            if score < threshold:
                selected.append(lname)

        print(f"\n  ── SSIM Layer Selection (threshold < {threshold:+.3f}) ──")
        for n, s in ssim_dict.items():
            flag = "  ← SEARCH" if n in selected else ""
            print(f"    {n}: {s:+.4f}{flag}")

        if not selected:
            worst = min(ssim_dict, key=ssim_dict.get)
            selected = [worst]
            print(f"  Fallback → {worst} ({ssim_dict[worst]:+.4f})")
        print(f"  Search space: {len(selected)}/{len(target_layers)} layers\n")
        return selected, ssim_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Taylor Channel Scorer
# ═══════════════════════════════════════════════════════════════════════════════

class TaylorChannelScorer:
    def __init__(self, dense_model, device='cuda'):
        self.dense_model = dense_model
        self.device      = device

    def compute(self, target_layers, data_loader, n_batches=10):
        self.dense_model.train()
        self.dense_model.zero_grad()
        crit = nn.CrossEntropyLoss()
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= n_batches: break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            crit(self.dense_model(inputs), targets).backward()
        self.dense_model.eval()

        d_mods, channel_scores = dict(self.dense_model.named_modules()), {}
        for lname in target_layers:
            d_m = d_mods.get(lname)
            if d_m is None or d_m.weight.grad is None:
                channel_scores[lname] = {}
                continue
            w    = d_m.weight.detach()        # [out_ch, in_ch, kH, kW]
            grad = d_m.weight.grad.detach()   # [out_ch, in_ch, kH, kW]
            scores_t = (grad * w).abs().sum(dim=tuple(range(1, w.dim())))  # [out_ch]
            channel_scores[lname] = {ch: scores_t[ch].item() for ch in range(d_m.out_channels)}
            print(f"  {lname}: {d_m.out_channels} ch  max={scores_t.max():.3e}")
        return channel_scores


def select_channels_by_taylor(channel_scores, pruned_dense_map, layer_name, n_restore):
    """
    pruned_dense_map[layer_name]: dense indices not yet restored for this layer.
    Returns the top-n_restore indices ranked by Taylor score.
    """
    scores     = channel_scores.get(layer_name, {})
    pruned_set = set(pruned_dense_map.get(layer_name, []))
    if not pruned_set or n_restore == 0:
        return []
    ranked = sorted(pruned_set, key=lambda c: scores.get(c, 0.0), reverse=True)
    return ranked[:min(n_restore, len(ranked))]


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Controller
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthAgent(nn.Module):
    def __init__(self, alloc_space_size, hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.DEVICE  = device
        self.nhid    = hidden_size

        self.lstm          = nn.LSTMCell(alloc_space_size + context_dim, hidden_size)
        self.alloc_decoder = nn.Linear(hidden_size, alloc_space_size)
        self.hidden        = self.init_hidden()

    def forward(self, prev_logits, context_vec):
        if prev_logits.dim() == 1: prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1: context_vec = context_vec.unsqueeze(0)
        h, c = self.lstm(torch.cat([prev_logits, context_vec], dim=-1), self.hidden)
        self.hidden = (h, c)
        return self.alloc_decoder(h)

    def init_hidden(self):
        return (torch.zeros(1, self.nhid, device=self.DEVICE),
                torch.zeros(1, self.nhid, device=self.DEVICE))


# ═══════════════════════════════════════════════════════════════════════════════
# RL Policy Gradient
# ═══════════════════════════════════════════════════════════════════════════════

class OneshotStructuredRegrowthPG:

    def __init__(self, config, model_sparse, dense_model,
                 channel_scores, index_map, pruned_dense_map,
                 original_channels, example_inputs,
                 target_layers, train_loader, test_loader,
                 device, wandb_run=None):

        self.NUM_EPOCHS         = config['num_epochs']
        self.ALPHA              = config['learning_rate']
        self.HIDDEN_SIZE        = config['hidden_size']
        self.BETA               = config['entropy_coef']
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.005)
        self.DEVICE             = device
        self.ALLOC_SPACE        = config['alloc_space_size']
        self.NUM_STEPS          = len(target_layers)
        self.CONTEXT_DIM        = config.get('context_dim', 3)
        self.BASELINE_DECAY     = config.get('baseline_decay', 0.9)

        self.acc_threshold    = config['acc_threshold']
        self.target_restore_ch = config['target_restore_ch']

        self.model_sparse      = model_sparse
        self.dense_model       = dense_model
        self.target_layers     = target_layers
        self.train_loader      = train_loader
        self.test_loader       = test_loader
        self.channel_scores    = channel_scores
        self.index_map         = index_map
        # pruned_dense_map is deepcopied per episode; the master copy is never mutated
        self.pruned_dense_map  = pruned_dense_map
        self.original_channels = original_channels
        self.example_inputs    = example_inputs

        self.layer_capacities = config['layer_capacities']
        self.total_capacity   = max(sum(self.layer_capacities), 1)

        self.early_stop_patience  = config.get('early_stop_patience', 40)
        self.min_epochs           = config.get('min_epochs', 50)
        self.reward_std_threshold = config.get('reward_std_threshold', 0.002)
        self.reward_window_size   = config.get('reward_window_size', 20)
        self.finetune_epochs      = config.get('finetune_epochs', 5)

        self._best_model_state = None
        self._best_reward_seen = float('-inf')
        self._best_index_map   = None
        self._best_pruned_map  = None  # preserve pruned_dense_map of the best episode

        self.run            = wandb_run
        self.model_name     = config.get('model_name')
        self.method         = config.get('method', 'structured_oneshot')
        self.checkpoint_dir = config.get('checkpoint_dir', './structured_oneshot_rl_ckpts')
        self.model_sparsity = config.get('model_sparsity', '0.5')

        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta           = config.get('start_beta', 0.4)
        self.end_beta             = config.get('end_beta', 0.04)
        self.decay_fraction       = config.get('decay_fraction', 0.4)

        self.agent = RegrowthAgent(
            alloc_space_size=self.ALLOC_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE,
        ).to(self.DEVICE)

        self.adam            = optim.Adam(self.agent.parameters(), lr=self.ALPHA)
        self.reward_baseline = None
        self.layer_priority  = [(n, i) for i, n in enumerate(target_layers)]

        print(f"  Acc threshold  : {self.acc_threshold:.2f}%")
        print(f"  Target restore : {self.target_restore_ch} channels")
        print(f"  Layers ({len(target_layers)}):")
        for n, i in self.layer_priority:
            print(f"    {n}: cap={self.layer_capacities[i]} "
                  f"pruned={len(self.pruned_dense_map.get(n, []))}")

    def get_entropy_coef(self, epoch):
        if not self.use_entropy_schedule:
            return self.BETA
        de = self.NUM_EPOCHS * self.decay_fraction
        if epoch < de:
            return self.start_beta - (self.start_beta - self.end_beta) * (epoch / de)
        return self.end_beta

    def _save_dir(self):
        d = os.path.join(self.checkpoint_dir,
                         f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        return d

    def evaluate_model(self, model, full_eval=False):
        max_b = None if full_eval else 20
        return quick_eval(model, self.test_loader, self.DEVICE, max_batches=max_b)

    def mini_finetune(self, model, epochs=5, lr=3e-4):
        model.train()
        opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        crit = nn.CrossEntropyLoss()
        best_acc, best_state = 0.0, None
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
            acc = self.evaluate_model(model, full_eval=True)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
            model.train()
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

    def calculate_loss(self, alloc_logits, wlp, beta):
        loss  = -torch.mean(wlp)
        p_a   = F.softmax(alloc_logits, dim=1)
        ent   = -torch.mean(torch.sum(p_a * F.log_softmax(alloc_logits, dim=1), dim=1))
        return loss - beta * ent, ent

    def solve_environment(self):
        best_reward, best_reward_ep = float('-inf'), 0
        reward_window = deque(maxlen=self.reward_window_size)
        stop_reason   = ""

        for epoch in range(self.NUM_EPOCHS):
            ep_wlp, ep_alloc_logits, reward, sparsity = self.play_episode(epoch)

            reward_window.append(reward)
            if reward > best_reward:
                best_reward, best_reward_ep = reward, epoch

            beta      = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(ep_alloc_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.adam.step()

            no_imp  = epoch - best_reward_ep
            rwd_std = float(np.std(list(reward_window))) if len(reward_window) > 1 else float('inf')

            if self.run:
                self.run.log({
                    'epoch': epoch + 1, 'reward': reward,
                    'best_reward': best_reward,
                    'loss': loss.item(), 'entropy': ent.item(),
                    'beta': beta, 'sparsity': sparsity,
                    'no_improve': no_imp, 'rwd_std': rwd_std,
                })

            print(f"Ep {epoch + 1:3d}/{self.NUM_EPOCHS} | "
                  f"Rwd={reward:+.4f} Best={best_reward:+.4f} | "
                  f"Loss={loss.item():.4f} | "
                  f"Ent={ent.item():.4f} | NoImp={no_imp} | Std={rwd_std:.5f}")

            if no_imp >= self.early_stop_patience:
                stop_reason = f"NoImp {self.early_stop_patience}"
                break
            if (epoch >= self.min_epochs
                    and len(reward_window) >= self.reward_window_size
                    and rwd_std < self.reward_std_threshold):
                stop_reason = f"Std {rwd_std:.5f} < threshold"
                break

        print(f"\nBest reward: {best_reward:+.4f}"
              + (f"  [{stop_reason}]" if stop_reason else ""))

    def play_episode(self, epoch):
        t0 = time.time()
        self.agent.hidden = self.agent.init_hidden()
        prev_logits       = torch.zeros(1, self.ALLOC_SPACE, device=self.DEVICE)
        all_log_probs, alloc_masked_logits = [], []

        target_ch = self.target_restore_ch

        # ── Allocation decision (channels per layer)
        ratio_opts = (torch.arange(self.ALLOC_SPACE, device=self.DEVICE, dtype=torch.float)
                      / (self.ALLOC_SPACE - 1))
        remaining  = target_ch
        sel_counts, pnames = [], []

        for p_idx, (lname, orig_idx) in enumerate(self.layer_priority):
            cap = int(self.layer_capacities[orig_idx])
            ctx = torch.tensor([
                (p_idx + 1) / (self.NUM_STEPS + 1),
                cap / self.total_capacity,
                remaining / target_ch if target_ch > 0 else 0.0,
            ], dtype=torch.float, device=self.DEVICE).unsqueeze(0)

            logits  = self.agent(prev_logits, ctx).squeeze(0)
            eff_max = min(cap, remaining)
            c_opts  = torch.round(ratio_opts * eff_max).to(torch.long)

            # Keep only the first occurrence of each unique count;
            # duplicates get masked so the agent distributes over distinct choices.
            seen_vals, dedup = set(), torch.zeros(self.ALLOC_SPACE, dtype=torch.bool,
                                                   device=self.DEVICE)
            for j, v in enumerate(c_opts.tolist()):
                if v not in seen_vals:
                    seen_vals.add(v)
                    dedup[j] = True

            feasible = (c_opts <= remaining) & dedup
            if not feasible.any(): feasible[0] = True

            masked = torch.where(feasible, logits, torch.full_like(logits, -1e9))
            dist   = Categorical(probs=F.softmax(masked, dim=0))
            action = dist.sample()
            chosen = min(int(c_opts[action].item()), cap, remaining)
            remaining = max(remaining - chosen, 0)

            sel_counts.append(chosen)
            pnames.append(lname)
            all_log_probs.append(dist.log_prob(action))
            alloc_masked_logits.append(masked)
            prev_logits = logits.unsqueeze(0)

        ep_log_probs    = torch.stack(all_log_probs)
        ep_alloc_logits = torch.stack(alloc_masked_logits)

        # ── Greedy second pass: fill any leftover budget by Taylor score
        if remaining > 0:
            ranked = sorted(
                range(len(self.layer_priority)),
                key=lambda i: max(
                    (self.channel_scores.get(self.layer_priority[i][0], {}).get(ch, 0.0)
                     for ch in self.pruned_dense_map.get(self.layer_priority[i][0], [])),
                    default=0.0,
                ),
                reverse=True,
            )
            for i in ranked:
                if remaining <= 0:
                    break
                _, orig_idx_g = self.layer_priority[i]
                slack = int(self.layer_capacities[orig_idx_g]) - sel_counts[i]
                if slack <= 0:
                    continue
                add = min(slack, remaining)
                sel_counts[i] += add
                remaining -= add
            print(f"  [Greedy] remaining after fill={remaining}")

        # ── Episode-local pruned_dense_map
        # (used for Taylor selection and DG update; master copy is never mutated)
        ep_pruned_map = copy.deepcopy(self.pruned_dense_map)

        # ── Select concrete dense indices via Taylor scores
        allocation_dense_idx = {}
        actual_restored = 0
        for lname, n_add in zip(pnames, sel_counts):
            if n_add == 0:
                continue
            chs = select_channels_by_taylor(
                self.channel_scores, ep_pruned_map, lname, n_add)
            if chs:
                allocation_dense_idx[lname] = chs
                actual_restored += len(chs)
                print(f"    {lname}: +{len(chs)} ch (top3: {chs[:3]})")

        print(f"  [Alloc] {actual_restored}/{target_ch}  | {list(allocation_dense_idx.keys())}")

        # ── Restore channels via DependencyGraph
        ep_index_map = copy.deepcopy(self.index_map)
        new_model = regrow_channels_dg(
            model=copy.deepcopy(self.model_sparse),
            dense_model=self.dense_model,
            example_inputs=self.example_inputs,
            allocation_dense_idx=allocation_dense_idx,
            index_map=ep_index_map,
            pruned_dense_map=ep_pruned_map,
            device=self.DEVICE,
        )

        pre_ft_model = copy.deepcopy(new_model)
        self.mini_finetune(new_model, epochs=self.finetune_epochs)

        accuracy = self.evaluate_model(new_model, full_eval=True)
        sparsity = compute_channel_sparsity(new_model, self.original_channels)

        reward = (accuracy - self.acc_threshold) / 100.0

        print(f"  [Reward] acc={accuracy:.2f}%  threshold={self.acc_threshold:.2f}%  "
              f"Δ={accuracy - self.acc_threshold:+.2f}pp  "
              f"reward={reward:+.4f}  [{time.time() - t0:.1f}s]")

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model_state = copy.deepcopy(pre_ft_model.state_dict())
            self._best_index_map   = copy.deepcopy(ep_index_map)
            self._best_pruned_map  = copy.deepcopy(ep_pruned_map)
            self._save_best(epoch, reward, accuracy, pre_ft_model,
                            ep_index_map, ep_pruned_map)

        if self.reward_baseline is None:
            self.reward_baseline = reward
        adv = float(np.clip(
            (reward - self.reward_baseline) / max(self.REWARD_TEMPERATURE, 1e-6),
            -10.0, 10.0))
        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline
                                + (1 - self.BASELINE_DECAY) * reward)

        adv_t  = torch.tensor(adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp = torch.sum(ep_log_probs * adv_t).unsqueeze(0)

        return ep_wlp, ep_alloc_logits, reward, sparsity

    def _save_best(self, epoch, reward, accuracy, model,
                   index_map, pruned_dense_map):
        p = os.path.join(self._save_dir(), f'best_ep{epoch + 1}_rwd{reward:+.4f}.pth')
        model_to_save = copy.deepcopy(model)
        for m in model_to_save.modules():
            m._forward_hooks.clear()
            m._backward_hooks.clear()
            m._forward_pre_hooks.clear()
        torch.save({
            'model'           : model_to_save,
            'index_map'       : index_map,
            'pruned_dense_map': pruned_dense_map,
        }, p)
        print(f"  ✓ Best (pre-finetune): reward={reward:+.4f}  "
              f"mini_ft_acc={accuracy:.2f}% → {p}")
        if self.run:
            self.run.log({"best_reward": reward, "best_mini_ft_acc": accuracy,
                          "best_epoch": epoch + 1})


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def quick_eval(model, test_loader, device, max_batches=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name',         type=str,   default='resnet20')
    parser.add_argument('--data_dir',       type=str,   default='./data')
    parser.add_argument('--method',         type=str,   default='structured_oneshot')
    parser.add_argument('--pruned_ckpt',    type=str,
                        default="resnet20/ckpt_structured_iterative/step10_sp0.973.pth")
    parser.add_argument('--acc_threshold',  type=float, default=50.0)
    parser.add_argument('--sparsity_delta', type=float, default=0.04)
    parser.add_argument('--num_epochs',     type=int,   default=300)
    parser.add_argument('--learning_rate',  type=float, default=3e-4)
    parser.add_argument('--hidden_size',    type=int,   default=64)
    parser.add_argument('--entropy_coef',   type=float, default=0.5)
    parser.add_argument('--reward_temperature', type=float, default=0.005)
    parser.add_argument('--start_beta',     type=float, default=0.40)
    parser.add_argument('--end_beta',       type=float, default=0.04)
    parser.add_argument('--decay_fraction', type=float, default=0.4)
    parser.add_argument('--alloc_space_size',  type=int, default=11)
    parser.add_argument('--ssim_threshold',    type=float, default=0.0)
    parser.add_argument('--ssim_num_batches',  type=int,   default=64)
    parser.add_argument('--taylor_batches',    type=int,   default=10)
    parser.add_argument('--finetune_epochs',   type=int,   default=50)
    parser.add_argument('--early_stop_patience',  type=int,   default=40)
    parser.add_argument('--min_epochs',           type=int,   default=50)
    parser.add_argument('--reward_std_threshold', type=float, default=0.002)
    parser.add_argument('--reward_window_size',   type=int,   default=20)
    parser.add_argument('--save_dir',   type=str,   default='./structured_rl_ckpts')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--no_wandb',   action='store_true', default=False)
    parser.add_argument('--dataset',    type=str,   default='CIFAR10')
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--val_split',  type=float, default=0.1)
    parser.add_argument('--num_workers',type=int,   default=15)
    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_loader, _, test_loader = data_loader(
        data_dir=args.data_dir, val_split=args.val_split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        dataset=args.dataset,
    )
    example_inputs = next(iter(train_loader))[0][:1].to(device)

    # ── Load dense model
    dense_model = model_loader(args.m_name, device)
    load_model_name(dense_model, f'./{args.m_name}/checkpoint', args.m_name)
    dense_model.eval()
    original_channels = get_original_channels(dense_model)

    # ── Load pruned checkpoint (saved by prune_structure_ratio.py)
    ckpt = torch.load(args.pruned_ckpt, map_location=device, weights_only=False)
    pruned_model     = ckpt['model'].to(device)
    index_map        = ckpt['out_index_map']
    pruned_dense_map = ckpt['pruned_out_dense_map']
    pruned_model.eval()

    sp0  = compute_channel_sparsity(pruned_model, original_channels)
    acc0 = quick_eval(pruned_model, test_loader, device)
    print(f"\nStarting → Acc={acc0:.2f}%  Sparsity={sp0:.4f}")

    # ── Restorable layers (non-empty entries in pruned_dense_map)
    target_layers    = get_target_layers(pruned_dense_map)
    layer_capacities = get_layer_capacities(pruned_dense_map, target_layers)

    if sum(layer_capacities) == 0:
        print("No channels to restore. Exiting.")
        return

    total_original_ch = sum(original_channels.values())
    want_ch           = int(total_original_ch * args.sparsity_delta)
    print(f"\n{len(target_layers)} pruned layers  "
          f"want_restore={want_ch} ch  (delta={args.sparsity_delta:.4f})")

    selected_layers, ssim_scores = SSIMLayerSelector.update_search_space(
        sparse_model=pruned_model, pretrained_model=dense_model,
        data_loader_ref=test_loader, target_layers=target_layers,
        threshold=args.ssim_threshold, num_batches=args.ssim_num_batches,
    )
    layer_capacities = get_layer_capacities(pruned_dense_map, selected_layers)

    # If selected layers don't have enough capacity, add more by ascending SSIM score
    if sum(layer_capacities) < want_ch:
        print(f"  [Warning] selected capacity {sum(layer_capacities)} < target {want_ch}, "
              f"adding layers by SSIM order...")
        remaining_layers = sorted(
            [ln for ln in target_layers if ln not in selected_layers],
            key=lambda ln: ssim_scores.get(ln, 1.0)
        )
        for lname in remaining_layers:
            selected_layers.append(lname)
            layer_capacities = get_layer_capacities(pruned_dense_map, selected_layers)
            print(f"    added {lname} (SSIM={ssim_scores.get(lname, 1.0):.4f}), "
                  f"capacity={sum(layer_capacities)}")
            if sum(layer_capacities) >= want_ch:
                break

    target_restore_ch = min(want_ch, sum(layer_capacities))
    print(f"  Final selected_layers={len(selected_layers)}  "
          f"target_restore_ch={target_restore_ch}")

    print("Computing Taylor scores...")
    channel_scores = TaylorChannelScorer(dense_model=dense_model, device=device).compute(
        target_layers=selected_layers,
        data_loader=train_loader, n_batches=args.taylor_batches,
    )

    layer_capacities = get_layer_capacities(pruned_dense_map, selected_layers)

    if not args.no_wandb:
        run = wandb.init(
            project="structured_oneshot_rl_regrowth",
            name=f"{args.m_name}_sp{sp0:.3f}_delta{args.sparsity_delta}",
            config=vars(args) | {"start_acc": acc0, "start_sp": sp0,
                                 "target_restore_ch": target_restore_ch},
        )
        for lname, sc in ssim_scores.items():
            run.log({f"ssim/{lname}": sc})
    else:
        run = None

    config = {
        'num_epochs': args.num_epochs, 'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size, 'entropy_coef': args.entropy_coef,
        'alloc_space_size': args.alloc_space_size,
        'layer_capacities': layer_capacities, 'model_name': args.m_name,
        'reward_temperature': args.reward_temperature,
        'checkpoint_dir': args.save_dir,
        'start_beta': args.start_beta, 'end_beta': args.end_beta,
        'decay_fraction': args.decay_fraction,
        'early_stop_patience': args.early_stop_patience,
        'min_epochs': args.min_epochs,
        'reward_std_threshold': args.reward_std_threshold,
        'reward_window_size': args.reward_window_size,
        'model_sparsity': f"sp{sp0:.3f}", 'method': args.method,
        'finetune_epochs': args.finetune_epochs,
        'acc_threshold': args.acc_threshold,
        'target_restore_ch': target_restore_ch,
        'sparsity_delta': args.sparsity_delta,
    }

    pg = OneshotStructuredRegrowthPG(
        config=config, model_sparse=pruned_model, dense_model=dense_model,
        channel_scores=channel_scores,
        index_map=index_map,
        pruned_dense_map=pruned_dense_map,
        original_channels=original_channels,
        example_inputs=example_inputs,
        target_layers=selected_layers,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device, wandb_run=run,
    )

    pg.solve_environment()

    if pg._best_model_state is not None:
        best_model = copy.deepcopy(pruned_model)
        best_model.load_state_dict(pg._best_model_state)
    else:
        best_model = pruned_model

    final_acc = quick_eval(best_model, test_loader, device)
    final_sp  = compute_channel_sparsity(best_model, original_channels)

    print(f"\n{'=' * 70}")
    print(f"DONE | Acc={final_acc:.2f}%  Sp={final_sp:.4f}  "
          f"Δacc={final_acc - acc0:+.2f}pp  Δsp={final_sp - sp0:+.4f}")
    print(f"{'=' * 70}")

    if run:
        run.log({"final/accuracy": final_acc, "final/sparsity": final_sp,
                 "final/delta_acc": final_acc - acc0})
        run.finish()


if __name__ == '__main__':
    main()