"""
Structured RL-based Iterative Regrowth
=======================================
Multiple rounds of channel restoration via DependencyGraph.
Each round grows sparsity_delta fraction of original channels,
using the best model from the previous round as the starting point.

Usage:
    python regrowth_structure_iterative.py \
        --pruned_ckpt resnet20/ckpt_structured_iterative/step10_sp0.973.pth \
        --m_name resnet20 \
        --sparsity_delta 0.04 \
        --num_iters 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import copy
import os
import re
import time
import wandb
import random
from pathlib import Path
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
# Baseline (channel-sparsity → accuracy lookup table)
# ═══════════════════════════════════════════════════════════════════════════════

def load_baseline_from_folder(baseline_dir, device, test_loader):
    """Scan a folder of structured pruning checkpoints and build a sparsity→acc table.

    Checkpoints must contain 'model' (full model object).
    The model with the most total Conv2d channels is treated as the dense reference.
    Returns (baseline_dict, dense_model, original_channels).
    """
    pth_files = sorted(Path(baseline_dir).glob('*.pth'))
    if not pth_files:
        raise ValueError(f"No .pth files found in {baseline_dir}")

    # ── Pass 1: load all models, find the densest one as reference
    loaded = []
    for pth_file in pth_files:
        try:
            ckpt  = torch.load(pth_file, map_location=device, weights_only=False)
            model = ckpt['model'].to(device) if isinstance(ckpt, dict) else ckpt.to(device)
            model.eval()
            total_ch = sum(m.out_channels for m in model.modules()
                           if isinstance(m, nn.Conv2d))
            loaded.append((pth_file, model, total_ch))
        except Exception as e:
            print(f"  Error loading {pth_file.name}: {e}")

    if not loaded:
        raise ValueError(f"No valid checkpoints in {baseline_dir}")

    dense_model = max(loaded, key=lambda t: t[2])[1]
    original_channels = {n: m.out_channels for n, m in dense_model.named_modules()
                         if isinstance(m, nn.Conv2d)}

    # ── Pass 2: compute sparsity and accuracy for each checkpoint
    print('=' * 60)
    baseline_dict = {}
    for pth_file, model, _ in loaded:
        try:
            sp = compute_channel_sparsity(model, original_channels)
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    _, pred = model(x).max(1)
                    total   += y.size(0)
                    correct += pred.eq(y).sum().item()
            acc = 100.0 * correct / total
            baseline_dict[sp] = acc
            print(f"  {pth_file.name}: sp={sp:.4f}  acc={acc:.2f}%")
        except Exception as e:
            print(f"  Error evaluating {pth_file.name}: {e}")

    baseline_dict = dict(sorted(baseline_dict.items()))
    if not baseline_dict:
        raise ValueError(f"No valid checkpoints in {baseline_dir}")
    print(f"\nBaseline: {len(baseline_dict)} pts  "
          f"sp=[{min(baseline_dict):.4f}…{max(baseline_dict):.4f}]  "
          f"acc=[{min(baseline_dict.values()):.2f}%…{max(baseline_dict.values()):.2f}%]")
    print('=' * 60 + '\n')
    return baseline_dict, dense_model, original_channels


class BaselineInterpolator:
    def __init__(self, table: dict):
        self.points = {float(k): float(v) for k, v in table.items()}
        pts = sorted(self.points.items())
        print(f"  [Baseline] {len(pts)} pts: sp {pts[0][0]:.4f}→{pts[-1][0]:.4f}")

    def get_baseline_acc(self, sparsity: float) -> float:
        pts = sorted(self.points.items())
        if sparsity <= pts[0][0]:  return pts[0][1]
        if sparsity >= pts[-1][0]: return pts[-1][1]
        for i in range(len(pts) - 1):
            s1, a1 = pts[i];  s2, a2 = pts[i + 1]
            if s1 <= sparsity <= s2:
                t = (sparsity - s1) / (s2 - s1 + 1e-12)
                return a1 + t * (a2 - a1)
        return pts[-1][1]


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
    return [name for name, lst in pruned_dense_map.items() if len(lst) > 0]


def get_layer_capacities(pruned_dense_map, target_layers):
    return [len(pruned_dense_map[name]) for name in target_layers]


# ═══════════════════════════════════════════════════════════════════════════════
# DependencyGraph-based Channel Restoration
# ═══════════════════════════════════════════════════════════════════════════════

def _get_dense_weight(layer_name, dense_idx, dim, dense_named):
    m = dense_named.get(layer_name)
    if m is None:
        return None
    w     = m.weight.data
    idx_t = torch.tensor(dense_idx, device=w.device)
    return w.index_select(dim, idx_t)


def _find_survived_in_ch(lname, in_ch_sparse, index_map, exclude=None):
    """Return the dense in_channel indices currently present in the sparse layer.

    Searches index_map for an entry whose length matches in_ch_sparse.
    Falls back to range(in_ch_sparse) if none found (in_ch not pruned).
    """
    for name, survived in index_map.items():
        if name == exclude:
            continue
        if len(survived) == in_ch_sparse:
            return survived
    return list(range(in_ch_sparse))


def regrow_channels_dg(model, dense_model, example_inputs,
                       allocation_dense_idx, index_map,
                       pruned_dense_map, device):
    """
    Restore pruned channels back into the model using DependencyGraph.
    Weights copied from dense model. index_map / pruned_dense_map updated in-place.

    Args:
        allocation_dense_idx : {layer_name: [dense_channel_indices_to_restore]}
        index_map            : {layer_name: [survived_dense_idx]}  ← updated in-place
        pruned_dense_map     : {layer_name: [pruned_dense_idx]}    ← updated in-place
    """
    named_modules = dict(model.named_modules())
    dense_named   = dict(dense_model.named_modules())

    for target_name, dense_indices in allocation_dense_idx.items():
        if not dense_indices:
            continue

        already_survived = set(index_map.get(target_name, []))
        dense_indices = [idx for idx in dense_indices if idx not in already_survived]
        if not dense_indices:
            continue

        target_conv = named_modules.get(target_name)
        if target_conv is None:
            continue

        num_grow   = len(dense_indices)
        old_out_ch = target_conv.out_channels

        # ── Rebuild DG from current model structure each restoration
        DG = tp.DependencyGraph().build_dependency(model, example_inputs)
        group = DG.get_pruning_group(target_conv, tp.prune_conv_out_channels,
                                     idxs=list(range(num_grow)))

        for dep, _ in group:
            m     = dep.target.module
            fn    = dep.handler
            lname = next((n for n, mod in named_modules.items() if mod is m), None)

            # ── out_channels expansion (target conv + BN)
            if fn == tp.prune_conv_out_channels or (
                    fn == tp.prune_batchnorm_out_channels and isinstance(m, nn.Conv2d)):
                # Conv2d: expand filter weight rows
                new_w = _get_dense_weight(lname, dense_indices, 0, dense_named)
                if new_w is None:
                    new_w = torch.zeros(num_grow, *m.weight.shape[1:], device=device)
                elif new_w.shape[1] != m.in_channels:
                    # Feeder layer is also pruned; select its survived in_channel columns.
                    survived_in = _find_survived_in_ch(lname, m.in_channels, index_map,
                                                       exclude=lname)
                    if len(survived_in) == m.in_channels:
                        sel = torch.tensor(survived_in, device=device)
                        new_w = new_w[:, sel, :, :]
                    else:
                        new_w = torch.zeros(num_grow, m.in_channels, *m.weight.shape[2:],
                                            device=device)
                m.weight = nn.Parameter(torch.cat([m.weight.data, new_w], dim=0))
                if hasattr(m, 'bias') and m.bias is not None:
                    d_m = dense_named.get(lname)
                    if d_m is not None and d_m.bias is not None:
                        nb = d_m.bias.data[torch.tensor(dense_indices, device=device)]
                    else:
                        nb = torch.zeros(num_grow, device=device)
                    m.bias = nn.Parameter(torch.cat([m.bias.data, nb], dim=0))
                m.out_channels += num_grow

            elif fn == tp.prune_batchnorm_out_channels and isinstance(m, nn.BatchNorm2d):
                # BN: expand γ, β, running stats — exactly once
                d_m  = dense_named.get(lname)
                idx_t = torch.tensor(dense_indices, device=device)
                m.num_features += num_grow
                for attr in ('running_mean', 'running_var'):
                    old = getattr(m, attr)
                    new_val = (getattr(d_m, attr)[idx_t]
                               if d_m is not None
                               else torch.ones(num_grow, device=device))
                    setattr(m, attr, torch.cat([old, new_val]))
                if m.weight is not None:
                    nw = (d_m.weight.data[idx_t]
                          if d_m is not None else torch.ones(num_grow, device=device))
                    m.weight = nn.Parameter(torch.cat([m.weight.data, nw]))
                if m.bias is not None:
                    nb2 = (d_m.bias.data[idx_t]
                           if (d_m is not None and d_m.bias is not None)
                           else torch.zeros(num_grow, device=device))
                    m.bias = nn.Parameter(torch.cat([m.bias.data, nb2]))

            # ── in_channels expansion (next conv / linear)
            elif fn in (tp.prune_conv_in_channels, tp.prune_linear_in_channels):
                new_w = _get_dense_weight(lname, dense_indices, 1, dense_named)
                if new_w is None:
                    in_shape = list(m.weight.shape)
                    in_shape[1] = num_grow
                    new_w = torch.zeros(in_shape, device=device)
                elif new_w.shape[0] != m.weight.shape[0]:
                    # This layer's own out_channels are also pruned; select survived rows.
                    survived_out = index_map.get(lname)
                    if survived_out is not None and len(survived_out) == m.weight.shape[0]:
                        sel = torch.tensor(survived_out, device=device)
                        new_w = new_w[sel, :, :, :] if new_w.dim() == 4 else new_w[sel, :]
                    else:
                        out_ch = m.weight.shape[0]
                        in_shape = list(m.weight.shape)
                        in_shape[1] = num_grow
                        new_w = torch.zeros(in_shape, device=device)
                m.weight = nn.Parameter(torch.cat([m.weight.data, new_w], dim=1))
                if isinstance(m, nn.Conv2d):
                    m.in_channels += num_grow
                elif isinstance(m, nn.Linear):
                    m.in_features += num_grow

        # ── Sync index_map and pruned_dense_map in-place
        index_map[target_name] = sorted(index_map[target_name] + list(dense_indices))
        pruned_dense_map[target_name] = sorted(
            set(pruned_dense_map[target_name]) - set(dense_indices))

        print(f"  [DG] {target_name}: +{num_grow} ch  "
              f"(out: {old_out_ch}→{target_conv.out_channels}  "
              f"still_pruned={len(pruned_dense_map[target_name])})")


# ═══════════════════════════════════════════════════════════════════════════════
# SSIM Layer Selector (structured: operates on Conv2d out-channel degradation)
# ═══════════════════════════════════════════════════════════════════════════════

class SSIMLayerSelector:
    @staticmethod
    def update_search_space(sparse_model, pretrained_model, data_loader_ref,
                            target_layers, threshold=0.0, num_batches=64):
        block_dict = {'all_layers': target_layers}
        ext_pre  = BlockwiseFeatureExtractor(pretrained_model, block_dict)
        ext_spar = BlockwiseFeatureExtractor(sparse_model,     block_dict)

        with torch.no_grad():
            feats_pre  = ext_pre.extract_block_features(data_loader_ref,  num_batches=num_batches)
            feats_spar = ext_spar.extract_block_features(data_loader_ref, num_batches=num_batches)

        block_ssim = compute_block_ssim(feats_pre, feats_spar).get('all_layers', {})

        ssim_dict, selected = {}, []
        for lname in target_layers:
            score = float(block_ssim.get(lname, 1.0))
            ssim_dict[lname] = score
            if score < threshold:
                selected.append(lname)

        print(f"\n  ── SSIM Layer Selection  (threshold < {threshold:+.3f}) ──")
        for n, s in ssim_dict.items():
            flag = "  ← SEARCH" if n in selected else ""
            print(f"    {n}: {s:+.4f}{flag}")

        if not selected:
            worst = min(ssim_dict, key=ssim_dict.get)
            selected = [worst]
            print(f"  No layer below threshold → fallback: {worst} ({ssim_dict[worst]:+.4f})")

        print(f"  Search space: {len(selected)}/{len(target_layers)} layers\n")
        return selected, ssim_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Taylor Channel Scorer
# ═══════════════════════════════════════════════════════════════════════════════

class TaylorChannelScorer:
    def __init__(self, dense_model, device='cuda'):
        self.model  = dense_model
        self.device = device

    def compute(self, target_layers, data_loader, n_batches=10):
        self.model.train()
        crit = nn.CrossEntropyLoss()
        channel_scores = {n: {} for n in target_layers}

        for i, (x, y) in enumerate(data_loader):
            if i >= n_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            crit(self.model(x), y).backward()
            for lname in target_layers:
                d_m = dict(self.model.named_modules()).get(lname)
                if d_m is None or d_m.weight.grad is None:
                    continue
                w, g = d_m.weight.data, d_m.weight.grad
                scores_t = (g * w).abs().sum(dim=tuple(range(1, w.dim())))
                for ch, sc in enumerate(scores_t.tolist()):
                    channel_scores[lname][ch] = channel_scores[lname].get(ch, 0.0) + sc

        self.model.eval()
        return channel_scores


def select_channels_by_taylor(channel_scores, pruned_dense_map, layer_name, n_restore):
    pruned_set = set(pruned_dense_map.get(layer_name, []))
    scores = {ch: sc for ch, sc in channel_scores.get(layer_name, {}).items()
              if ch in pruned_set}
    if not scores:
        return []
    top = sorted(scores, key=scores.__getitem__, reverse=True)
    return top[:n_restore]


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Controller
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthAgent(nn.Module):
    def __init__(self, budget_space_size, alloc_space_size, hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.DEVICE   = device
        self.input_dim = max(budget_space_size, alloc_space_size)
        self.lstm     = nn.LSTMCell(self.input_dim + context_dim, hidden_size)
        self.budget_decoder = nn.Linear(hidden_size, budget_space_size)
        self.alloc_decoder  = nn.Linear(hidden_size, alloc_space_size)
        self.hidden   = self.init_hidden()

    def forward(self, prev_logits, context_vec, step='alloc'):
        if prev_logits.dim() == 1:  prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1:  context_vec = context_vec.unsqueeze(0)
        pad = self.input_dim - prev_logits.shape[-1]
        if pad > 0:
            prev_logits = F.pad(prev_logits, (0, pad))
        h, c = self.lstm(torch.cat([prev_logits, context_vec], dim=-1), self.hidden)
        self.hidden = (h, c)
        return self.budget_decoder(h) if step == 'budget' else self.alloc_decoder(h)

    def init_hidden(self):
        d = next(self.parameters()).device if len(list(self.parameters())) > 0 else self.DEVICE
        return (torch.zeros(1, self.lstm.hidden_size, device=d),
                torch.zeros(1, self.lstm.hidden_size, device=d))


# ═══════════════════════════════════════════════════════════════════════════════
# RL Policy Gradient (one search round)
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredRegrowthPG:
    def __init__(self, config, model_sparse, dense_model,
                 channel_scores, index_map, pruned_dense_map,
                 original_channels, example_inputs,
                 target_layers, train_loader, test_loader,
                 device, baseline_interp=None, wandb_run=None, iter_idx=0):

        self.NUM_EPOCHS    = config['num_epochs']
        self.BUDGET_SPACE  = config['budget_space_size']
        self.ALLOC_SPACE   = config['alloc_space_size']
        self.DEVICE        = device
        self.BASELINE_DECAY = config.get('baseline_decay', 0.9)
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.005)

        self.model_sparse      = model_sparse.to(device)
        self.dense_model       = dense_model
        self.channel_scores    = channel_scores
        self.index_map         = index_map
        self.pruned_dense_map  = pruned_dense_map
        self.original_channels = original_channels
        self.example_inputs    = example_inputs
        self.target_layers     = target_layers
        self.train_loader      = train_loader
        self.test_loader       = test_loader

        self.layer_capacities = config['layer_capacities']
        self.total_capacity   = max(sum(self.layer_capacities), 1)
        self.acc_threshold    = config['acc_threshold']  # fallback when no interpolator
        self.baseline_interp  = baseline_interp
        self.finetune_epochs  = config.get('finetune_epochs', 50)

        # Budget options: evenly spaced between min_restore_frac and max_restore_frac
        # fractions of total DENSE channels (not restorable capacity)
        total_dense_ch = max(config.get('total_original_ch', self.total_capacity), 1)
        min_ch = max(1, int(total_dense_ch * config.get('min_restore_frac', 0.005)))
        max_ch = max(min_ch, int(total_dense_ch * config.get('max_restore_frac', 0.010)))
        self.budget_counts = [
            max(1, int(min_ch + (max_ch - min_ch) * i / max(self.BUDGET_SPACE - 1, 1)))
            for i in range(self.BUDGET_SPACE)
        ]
        print(f"  Budget options: {self.budget_counts[0]}…{self.budget_counts[-1]} ch  "
              f"({self.BUDGET_SPACE} choices)")

        self.early_stop_patience   = config.get('early_stop_patience', 40)
        self.min_epochs            = config.get('min_epochs', 50)
        self.reward_std_threshold  = config.get('reward_std_threshold', 0.002)
        self.reward_window_size    = config.get('reward_window_size', 20)

        self.use_entropy_schedule = True
        self.start_beta    = config.get('start_beta', 0.4)
        self.end_beta      = config.get('end_beta', 0.04)
        self.decay_fraction = config.get('decay_fraction', 0.4)

        self._best_reward_seen = float('-inf')
        self._best_model       = None
        self._best_index_map   = None
        self._best_pruned_map  = None

        self.run         = wandb_run
        self.iter_idx    = iter_idx
        self.model_name  = config.get('model_name', 'model')
        self.method      = config.get('method', 'structured_iterative')
        self.checkpoint_dir = config.get('checkpoint_dir', './structured_rl_ckpts')
        self.model_sparsity = config.get('model_sparsity', 'sp0.000')

        self.NUM_STEPS = len(target_layers)
        self.reward_baseline = None

        self.layer_priority = [(n, i) for i, n in enumerate(target_layers)]

        self.agent = RegrowthAgent(
            budget_space_size=self.BUDGET_SPACE,
            alloc_space_size=self.ALLOC_SPACE,
            hidden_size=config.get('hidden_size', 64),
            context_dim=3,
            device=device,
        ).to(device)
        self.adam = optim.Adam(self.agent.parameters(), lr=config.get('learning_rate', 3e-4))

        print(f"  [Iter {iter_idx+1}] capacity={self.total_capacity} ch  "
              f"acc_threshold={self.acc_threshold:.2f}%  "
              f"layers={len(target_layers)}")

    def get_entropy_coef(self, epoch):
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
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                if not full_eval and i >= 20:
                    break
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                _, pred = model(x).max(1)
                total   += y.size(0)
                correct += pred.eq(y).sum().item()
        return 100.0 * correct / total

    def mini_finetune(self, model, epochs=50, lr=3e-4):
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
            model.eval()
            acc = self.evaluate_model(model, full_eval=False)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
            model.train()
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

    def calculate_loss(self, budget_logits, alloc_logits, wlp, beta):
        loss  = -torch.mean(wlp)
        p_b   = F.softmax(budget_logits, dim=1)
        ent_b = -torch.mean(torch.sum(p_b * F.log_softmax(budget_logits, dim=1), dim=1))
        p_a   = F.softmax(alloc_logits, dim=1)
        ent_a = -torch.mean(torch.sum(p_a * F.log_softmax(alloc_logits, dim=1), dim=1))
        ent   = (ent_b + ent_a) / 2.0
        return loss - beta * ent, ent

    def _save_best(self, epoch, reward, accuracy, model, index_map, pruned_dense_map):
        p = os.path.join(self._save_dir(),
                         f'iter{self.iter_idx+1}_ep{epoch+1}_rwd{reward:+.4f}.pth')
        m_copy = copy.deepcopy(model)
        for mod in m_copy.modules():
            mod._forward_hooks.clear()
            mod._backward_hooks.clear()
            mod._forward_pre_hooks.clear()
        torch.save({
            'model'           : m_copy,
            'index_map'       : index_map,
            'pruned_dense_map': pruned_dense_map,
        }, p)
        print(f"  ✓ Best (post-ft): reward={reward:+.4f}  acc={accuracy:.2f}% → {p}")
        if self.run:
            pfx = f"iter{self.iter_idx+1}"
            self.run.log({f"{pfx}/best_reward": reward,
                          f"{pfx}/best_mini_ft_acc": accuracy,
                          f"{pfx}/best_epoch": epoch + 1})

    def solve_environment(self):
        """Run RL search for one regrowth round.

        Returns:
            (best_model, best_index_map, best_pruned_map, best_reward)
        """
        best_reward, best_reward_ep = float('-inf'), 0
        reward_window = deque(maxlen=self.reward_window_size)
        stop_reason   = ""

        for epoch in range(self.NUM_EPOCHS):
            ep_wlp, ep_budget_logits, ep_alloc_logits, reward, sparsity = self.play_episode(epoch)

            reward_window.append(reward)
            if reward > best_reward:
                best_reward, best_reward_ep = reward, epoch

            beta      = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(ep_budget_logits, ep_alloc_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.adam.step()

            no_imp  = epoch - best_reward_ep
            rwd_std = float(np.std(list(reward_window))) if len(reward_window) > 1 else float('inf')

            pfx = f"iter{self.iter_idx+1}"
            if self.run:
                self.run.log({
                    f"{pfx}/epoch":      epoch + 1,
                    f"{pfx}/reward":     reward,
                    f"{pfx}/best_reward": best_reward,
                    f"{pfx}/loss":       loss.item(),
                    f"{pfx}/entropy":    ent.item(),
                    f"{pfx}/beta":       beta,
                    f"{pfx}/sparsity":   sparsity,
                    f"{pfx}/no_improve": no_imp,
                    f"{pfx}/rwd_std":    rwd_std,
                })

            print(f"[Iter{self.iter_idx+1}] Ep {epoch+1:3d}/{self.NUM_EPOCHS} | "
                  f"Rwd={reward:+.4f} Best={best_reward:+.4f} | "
                  f"Loss={loss.item():.4f} Ent={ent.item():.4f} | "
                  f"NoImp={no_imp} Std={rwd_std:.5f}")

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

        return (self._best_model, self._best_index_map,
                self._best_pruned_map, best_reward)

    def play_episode(self, epoch):
        t0 = time.time()
        self.agent.hidden = self.agent.init_hidden()
        prev_logits       = torch.zeros(1, self.BUDGET_SPACE, device=self.DEVICE)
        all_log_probs, budget_masked_logits, alloc_masked_logits = [], [], []

        # ── Budget decision: agent picks total channels to restore this episode
        b_ctx    = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float,
                                 device=self.DEVICE).unsqueeze(0)
        b_logits = self.agent(prev_logits, b_ctx, step='budget').squeeze(0)
        b_dist   = Categorical(probs=F.softmax(b_logits, dim=0))
        b_action = b_dist.sample()
        target_ch = min(int(self.budget_counts[b_action.item()]), self.total_capacity)

        all_log_probs.append(b_dist.log_prob(b_action))
        budget_masked_logits.append(b_logits)
        prev_logits = b_logits.unsqueeze(0)
        print(f"  [Budget] action={b_action.item()}  target={target_ch} ch")

        ratio_opts = torch.arange(1, self.ALLOC_SPACE + 1,
                                   device=self.DEVICE, dtype=torch.float) / self.ALLOC_SPACE
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

            seen_vals, dedup = set(), torch.zeros(self.ALLOC_SPACE, dtype=torch.bool,
                                                   device=self.DEVICE)
            for j, v in enumerate(c_opts.tolist()):
                if v not in seen_vals:
                    seen_vals.add(v)
                    dedup[j] = True

            feasible = (c_opts <= remaining) & dedup
            if not feasible.any():
                feasible[0] = True

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

        ep_log_probs        = torch.stack(all_log_probs)
        ep_budget_logits    = torch.stack(budget_masked_logits)
        ep_alloc_logits     = torch.stack(alloc_masked_logits)

        # ── Greedy second pass: fill leftover budget by Taylor score
        if remaining > 0:
            scores_remaining = {
                pnames[i]: sum(
                    self.channel_scores.get(pnames[i], {}).get(ch, 0.0)
                    for ch in self.pruned_dense_map.get(pnames[i], [])
                ) / max(self.layer_capacities[self.layer_priority[i][1]], 1)
                for i in range(len(pnames))
                if self.layer_capacities[self.layer_priority[i][1]] > sel_counts[i]
            }
            while remaining > 0 and scores_remaining:
                best_name = max(scores_remaining, key=scores_remaining.__getitem__)
                i = pnames.index(best_name)
                cap = self.layer_capacities[self.layer_priority[i][1]]
                give = min(remaining, cap - sel_counts[i])
                if give <= 0:
                    scores_remaining.pop(best_name)
                    continue
                sel_counts[i] += give
                remaining -= give
                scores_remaining.pop(best_name)

        allocation = {n: c for n, c in zip(pnames, sel_counts) if c > 0}
        print(f"  [Alloc] total={sum(sel_counts)}/{target_ch} | {allocation}")

        # ── Build episode copies of model and maps
        ep_index_map  = copy.deepcopy(self.index_map)
        ep_pruned_map = copy.deepcopy(self.pruned_dense_map)

        # Structured pruning model has no reparameterization → deepcopy is safe
        new_model = copy.deepcopy(self.model_sparse).to(self.DEVICE)

        # ── Convert allocation counts → dense channel indices via Taylor
        allocation_dense_idx = {}
        for lname, n_restore in allocation.items():
            dense_idx = select_channels_by_taylor(
                self.channel_scores, ep_pruned_map, lname, n_restore)
            if dense_idx:
                allocation_dense_idx[lname] = dense_idx

        regrow_channels_dg(
            model=new_model, dense_model=self.dense_model,
            example_inputs=self.example_inputs,
            allocation_dense_idx=allocation_dense_idx,
            index_map=ep_index_map, pruned_dense_map=ep_pruned_map,
            device=self.DEVICE,
        )

        sparsity = compute_channel_sparsity(new_model, self.original_channels)

        self.mini_finetune(new_model, epochs=self.finetune_epochs)
        accuracy = self.evaluate_model(new_model, full_eval=True)

        if self.baseline_interp is not None:
            baseline_acc = self.baseline_interp.get_baseline_acc(sparsity)
        else:
            baseline_acc = self.acc_threshold
        reward = (accuracy - baseline_acc) / 100.0

        print(f"  [Reward] acc={accuracy:.2f}%  baseline={baseline_acc:.2f}%  "
              f"Δ={accuracy - baseline_acc:+.2f}pp  reward={reward:+.4f}  "
              f"[{time.time() - t0:.1f}s]")

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model       = copy.deepcopy(new_model)   # post-finetune
            self._best_index_map   = copy.deepcopy(ep_index_map)
            self._best_pruned_map  = copy.deepcopy(ep_pruned_map)
            self._save_best(epoch, reward, accuracy,
                            new_model, ep_index_map, ep_pruned_map)

        if self.reward_baseline is None:
            self.reward_baseline = reward
        adv = float(np.clip(
            (reward - self.reward_baseline) / max(self.REWARD_TEMPERATURE, 1e-6),
            -10.0, 10.0))
        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline
                                + (1 - self.BASELINE_DECAY) * reward)

        adv_t  = torch.tensor(adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp = torch.sum(ep_log_probs * adv_t).unsqueeze(0)

        return ep_wlp, ep_budget_logits, ep_alloc_logits, reward, sparsity


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def quick_eval(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
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
    parser.add_argument('--method',         type=str,   default='structured_iterative')
    parser.add_argument('--pruned_ckpt',    type=str,
                        default='resnet20/ckpt_structured_iterative/step05_sp0.875.pth')
    parser.add_argument('--baseline_dir',       type=str,   default= "resnet20/ckpt_structured_iterative",
                        help='Folder of pruned checkpoints to build sparsity→acc baseline')
    parser.add_argument('--num_iters',          type=int,   default=5)
    parser.add_argument('--target_sparsity',    type=float, default=0.0,
                        help='Stop early when channel sparsity falls below this (0=disabled)')
    parser.add_argument('--num_epochs',         type=int,   default=300)
    parser.add_argument('--learning_rate',      type=float, default=3e-4)
    parser.add_argument('--hidden_size',        type=int,   default=64)
    parser.add_argument('--entropy_coef',       type=float, default=0.5)
    parser.add_argument('--reward_temperature', type=float, default=0.005)
    parser.add_argument('--start_beta',         type=float, default=0.40)
    parser.add_argument('--end_beta',           type=float, default=0.04)
    parser.add_argument('--decay_fraction',     type=float, default=0.4)
    parser.add_argument('--budget_space_size',  type=int,   default=6)
    parser.add_argument('--min_restore_frac',   type=float, default=0.005,
                        help='Min channels to restore as fraction of total dense channels')
    parser.add_argument('--max_restore_frac',   type=float, default=0.01,
                        help='Max channels to restore as fraction of total dense channels')
    parser.add_argument('--alloc_space_size',   type=int,   default=10)
    parser.add_argument('--ssim_threshold',     type=float, default=0.0)
    parser.add_argument('--ssim_num_batches',   type=int,   default=64)
    parser.add_argument('--taylor_batches',     type=int,   default=10)
    parser.add_argument('--finetune_epochs',    type=int,   default=40)
    parser.add_argument('--early_stop_patience',   type=int,   default=40)
    parser.add_argument('--min_epochs',            type=int,   default=50)
    parser.add_argument('--reward_std_threshold',  type=float, default=0.002)
    parser.add_argument('--reward_window_size',    type=int,   default=20)
    parser.add_argument('--save_dir',   type=str,   default='./structured_rl_ckpts')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--no_wandb',   action='store_true', default=True)
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

    # ── Dense model + baseline
    if args.baseline_dir:
        print("\nBuilding baseline from folder (dense model extracted automatically)...")
        baseline_dict, dense_model, original_channels = load_baseline_from_folder(
            args.baseline_dir, device, test_loader)
        baseline_interp = BaselineInterpolator(baseline_dict)
    else:
        dense_model = model_loader(args.m_name, device)
        load_model_name(dense_model, f'./{args.m_name}/checkpoint', args.m_name)
        original_channels = get_original_channels(dense_model)
        baseline_interp = None
        print("  No baseline_dir provided; reward = acc - cur_acc\n")

    dense_model.eval()
    total_original_ch = sum(original_channels.values())

    # ── Load starting checkpoint
    ckpt             = torch.load(args.pruned_ckpt, map_location=device, weights_only=False)
    current_model    = ckpt['model'].to(device)
    index_map        = ckpt.get('index_map',        ckpt.get('out_index_map'))
    pruned_dense_map = ckpt.get('pruned_dense_map', ckpt.get('pruned_out_dense_map'))
    current_model.eval()

    acc0 = quick_eval(current_model, test_loader, device)
    sp0  = compute_channel_sparsity(current_model, original_channels)
    print(f"\nStarting → Acc={acc0:.2f}%  Sparsity={sp0:.4f}")
    if baseline_interp is not None:
        print(f"  Baseline at sp={sp0:.4f}: {baseline_interp.get_baseline_acc(sp0):.2f}%\n")

    if not args.no_wandb:
        run = wandb.init(
            project='structured_iterative_rl_regrowth',
            name=f"{args.m_name}_sp{sp0:.3f}_bfrac{args.min_restore_frac:.2f}-{args.max_restore_frac:.2f}",
            config=vars(args) | {'start_acc': acc0, 'start_sp': sp0},
        )
        run.log({'iter_summary/iteration': 0,
                 'iter_summary/accuracy': acc0,
                 'iter_summary/sparsity': sp0})
    else:
        run = None

    # ── Iterative regrowth loop
    for iter_idx in range(args.num_iters):
        cur_sp  = compute_channel_sparsity(current_model, original_channels)
        cur_acc = quick_eval(current_model, test_loader, device)

        print(f"\n{'#' * 70}")
        print(f"  ITER {iter_idx+1}/{args.num_iters}  |  "
              f"sp={cur_sp:.4f}  acc={cur_acc:.2f}%")
        print(f"{'#' * 70}\n")

        # ── Restorable layers
        all_target_layers = get_target_layers(pruned_dense_map)
        if not all_target_layers:
            print("No restorable channels left. Done.")
            break

        # ── SSIM layer selection
        selected_layers, ssim_scores = SSIMLayerSelector.update_search_space(
            sparse_model=current_model, pretrained_model=dense_model,
            data_loader_ref=test_loader, target_layers=all_target_layers,
            threshold=args.ssim_threshold, num_batches=args.ssim_num_batches,
        )
        if run:
            for lname, sc in ssim_scores.items():
                run.log({f'ssim/{lname}': sc, 'ssim_iter': iter_idx + 1})

        layer_capacities = get_layer_capacities(pruned_dense_map, selected_layers)
        total_capacity   = sum(layer_capacities)

        if total_capacity == 0:
            print("  Nothing to restore this iteration. Done.")
            break

        print(f"  selected_layers={len(selected_layers)}  total_capacity={total_capacity} ch")

        # ── Taylor scores (from dense model)
        print("Computing Taylor scores...")
        channel_scores = TaylorChannelScorer(dense_model=dense_model, device=device).compute(
            target_layers=selected_layers,
            data_loader=train_loader, n_batches=args.taylor_batches,
        )

        sp_label = f"iter{iter_idx}_sp{cur_sp:.4f}"
        config = {
            'num_epochs':            args.num_epochs,
            'learning_rate':         args.learning_rate,
            'hidden_size':           args.hidden_size,
            'entropy_coef':          args.entropy_coef,
            'budget_space_size':     args.budget_space_size,
            'min_restore_frac':      args.min_restore_frac,
            'max_restore_frac':      args.max_restore_frac,
            'alloc_space_size':      args.alloc_space_size,
            'layer_capacities':      layer_capacities,
            'total_original_ch':     total_original_ch,
            'model_name':            args.m_name,
            'reward_temperature':    args.reward_temperature,
            'checkpoint_dir':        args.save_dir,
            'start_beta':            args.start_beta,
            'end_beta':              args.end_beta,
            'decay_fraction':        args.decay_fraction,
            'finetune_epochs':       args.finetune_epochs,
            'early_stop_patience':   args.early_stop_patience,
            'min_epochs':            args.min_epochs,
            'reward_std_threshold':  args.reward_std_threshold,
            'reward_window_size':    args.reward_window_size,
            'model_sparsity':        sp_label,
            'method':                args.method,
            'acc_threshold':         cur_acc,   # fallback when baseline_interp is None
        }

        pg = StructuredRegrowthPG(
            config=config,
            model_sparse=current_model,
            dense_model=dense_model,
            channel_scores=channel_scores,
            index_map=index_map,
            pruned_dense_map=pruned_dense_map,
            original_channels=original_channels,
            example_inputs=example_inputs,
            target_layers=selected_layers,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            baseline_interp=baseline_interp,
            wandb_run=run,
            iter_idx=iter_idx,
        )

        t_start = time.time()
        best_model, best_index_map, best_pruned_map, best_reward = pg.solve_environment()
        elapsed = time.time() - t_start

        if best_model is None:
            print("  No improving episode found; keeping current model.")
            best_model       = current_model
            best_index_map   = index_map
            best_pruned_map  = pruned_dense_map

        iter_acc = quick_eval(best_model, test_loader, device)
        iter_sp  = compute_channel_sparsity(best_model, original_channels)

        print(f"\n  [Iter {iter_idx+1}] acc={iter_acc:.2f}%  sp={iter_sp:.4f}  "
              f"Δacc={iter_acc - cur_acc:+.2f}pp  "
              f"time={elapsed:.1f}s ({elapsed/3600:.2f}h)")

        if run:
            run.log({
                'iter_summary/iteration':   iter_idx + 1,
                'iter_summary/best_reward': best_reward,
                'iter_summary/accuracy':    iter_acc,
                'iter_summary/sparsity':    iter_sp,
                'iter_summary/delta_acc_pp': iter_acc - cur_acc,
                'iter_summary/time_s':      elapsed,
            })

        # ── Advance state for next iteration
        current_model    = best_model
        index_map        = best_index_map
        pruned_dense_map = best_pruned_map

        if args.target_sparsity > 0 and iter_sp <= args.target_sparsity:
            print(f"  Target sparsity {args.target_sparsity:.4f} reached. Stopping.")
            break

    # ── Final summary
    final_acc = quick_eval(current_model, test_loader, device)
    final_sp  = compute_channel_sparsity(current_model, original_channels)
    print(f"\n{'=' * 70}")
    print(f"DONE | Acc={final_acc:.2f}%  Sp={final_sp:.4f}  "
          f"Δacc={final_acc - acc0:+.2f}pp  Δsp={final_sp - sp0:+.4f}")
    print(f"{'=' * 70}")

    if run:
        run.log({'final/accuracy': final_acc,
                 'final/sparsity': final_sp,
                 'final/delta_acc_pp': final_acc - acc0})
        run.finish()


if __name__ == '__main__':
    main()
