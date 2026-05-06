"""
Structured RL-based One-Shot Regrowth
======================================
Weight transfer fix: match_to_dense 找通道对应关系，按对应关系复制权重。
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
# Taylor Importance
# ═══════════════════════════════════════════════════════════════════════════════

class PrecomputedTaylorImportance(tp.importance.Importance):
    def __init__(self, channel_scores, id_to_name):
        self.channel_scores = channel_scores
        self.id_to_name     = id_to_name

    def __call__(self, group, **kwargs):
        for dep, _ in group:
            m    = dep.target.module
            name = self.id_to_name.get(id(m))
            if isinstance(m, nn.Conv2d) and name and name in self.channel_scores:
                scores = self.channel_scores[name]
                return torch.tensor(
                    [scores.get(i, 0.0) for i in range(m.out_channels)],
                    dtype=torch.float)
        for dep, _ in group:
            m = dep.target.module
            if isinstance(m, nn.Conv2d):
                return m.weight.detach().abs().sum(dim=[1, 2, 3])
        return torch.ones(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Structured Pruning Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_original_channels(dense_model):
    return {n: m.out_channels for n, m in dense_model.named_modules()
            if isinstance(m, nn.Conv2d)}


def get_current_config(model):
    return {n: m.out_channels for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d)}


def compute_channel_sparsity(model, original_channels):
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total     += original_channels[name]
            remaining += m.out_channels
    return (1 - remaining / total) if total > 0 else 0.0


def get_target_layers(dense_model, pruned_model):
    d_mods = dict(dense_model.named_modules())
    return [name for name, p_m in pruned_model.named_modules()
            if isinstance(p_m, nn.Conv2d)
            and name in d_mods
            and p_m.out_channels < d_mods[name].out_channels]


def get_layer_capacities(dense_model, pruned_model, target_layers):
    d_mods = dict(dense_model.named_modules())
    p_mods = dict(pruned_model.named_modules())
    caps = []
    for name in target_layers:
        d_m, p_m = d_mods.get(name), p_mods.get(name)
        caps.append(max(d_m.out_channels - p_m.out_channels, 0)
                    if d_m and p_m else 0)
    return caps


# ═══════════════════════════════════════════════════════════════════════════════
# Channel Matching
# ═══════════════════════════════════════════════════════════════════════════════

def match_to_dense(model, dense_model, layer_name):
    """返回 {model_ch_idx: dense_ch_idx}，通过权重相似度匹配。"""
    d_m = dict(dense_model.named_modules())[layer_name]
    m_m = dict(model.named_modules()).get(layer_name)
    if m_m is None:
        return {}
    d_w    = d_m.weight.data.cpu()
    m_w    = m_m.weight.data.cpu()
    min_in = min(d_w.shape[1], m_w.shape[1])
    mapping = {}
    for m_idx in range(m_w.shape[0]):
        diffs = (d_w[:, :min_in] - m_w[m_idx, :min_in].unsqueeze(0)).abs().sum(dim=[1, 2, 3])
        mapping[m_idx] = int(diffs.argmin().item())
    return mapping


def build_pruned_to_dense(pruned_model, dense_model, target_layers):
    """预计算 pruned model 每层的通道对应关系，只需算一次。"""
    print("Building pruned↔dense channel mapping…")
    result = {}
    for lname in target_layers:
        result[lname] = match_to_dense(pruned_model, dense_model, lname)
        kept = set(result[lname].values())
        pruned_set = set(range(dict(dense_model.named_modules())[lname].out_channels)) - kept
        print(f"  {lname}: {len(pruned_set)} pruned channels")
    return result


def apply_config_with_matched_transfer(pruned_model, dense_model,
                                       new_config, original_channels,
                                       example_inputs, channel_scores,
                                       pruned_to_dense, target_layers):
    """
    重建模型并做正确的权重转移：
    dense → tp 剪到 new_config → 找 new↔dense 对应 → 按对应关系复制 pruned 权重
    """
    # 重建结构
    model = copy.deepcopy(dense_model)
    pruning_ratio_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in new_config:
            sp = 1 - new_config[name] / original_channels[name]
            if sp > 0:
                pruning_ratio_dict[module] = sp

    if pruning_ratio_dict:
        id_to_name = {id(m): n for n, m in model.named_modules()}
        pruner = tp.pruner.MagnitudePruner(
            model, example_inputs,
            importance=PrecomputedTaylorImportance(channel_scores, id_to_name),
            iterative_steps=1, pruning_ratio=0,
            pruning_ratio_dict=pruning_ratio_dict,
        )
        pruner.step()

    # 找 new_model 每个通道对应 dense 哪个 index
    new_to_dense = {lname: match_to_dense(model, dense_model, lname)
                    for lname in target_layers}

    # dense_idx → pruned_idx 反向映射
    dense_to_pruned = {
        lname: {d: p for p, d in pruned_to_dense[lname].items()}
        for lname in target_layers
    }

    p_mods = dict(pruned_model.named_modules())
    with torch.no_grad():
        # target layers 的 Conv：按通道对应关系复制
        for lname in target_layers:
            n_m = dict(model.named_modules()).get(lname)
            p_m = p_mods.get(lname)
            if n_m is None or p_m is None:
                continue
            d2p    = dense_to_pruned[lname]
            n2d    = new_to_dense[lname]
            min_in = min(n_m.weight.shape[1], p_m.weight.shape[1])
            for n_idx, d_idx in n2d.items():
                if d_idx in d2p:
                    p_idx = d2p[d_idx]
                    n_m.weight.data[n_idx, :min_in] = p_m.weight.data[p_idx, :min_in]
                    if n_m.bias is not None and p_m.bias is not None:
                        n_m.bias.data[n_idx] = p_m.bias.data[p_idx]
                # else: 新通道，保留 dense 权重

        # 非 target layer 的 Conv：按 index 复制
        for name, n_m in model.named_modules():
            if not isinstance(n_m, nn.Conv2d) or name in target_layers:
                continue
            p_m = p_mods.get(name)
            if p_m is None:
                continue
            n_out = min(n_m.weight.shape[0], p_m.weight.shape[0])
            n_in  = min(n_m.weight.shape[1], p_m.weight.shape[1])
            n_m.weight.data[:n_out, :n_in] = p_m.weight.data[:n_out, :n_in]
            if n_m.bias is not None and p_m.bias is not None:
                n_m.bias.data[:n_out] = p_m.bias.data[:n_out]

        # BN
        for lname, n_m in model.named_modules():
            if not isinstance(n_m, nn.BatchNorm2d):
                continue
            p_m = p_mods.get(lname)
            if p_m is None:
                continue
            if lname in new_to_dense:
                n2d = new_to_dense[lname]
                d2p = dense_to_pruned.get(lname, {})
                for n_idx, d_idx in n2d.items():
                    if n_idx >= n_m.num_features:
                        continue
                    if d_idx in d2p:
                        p_idx = d2p[d_idx]
                        if p_idx >= p_m.num_features:
                            continue
                        n_m.weight.data[n_idx]       = p_m.weight.data[p_idx]
                        n_m.bias.data[n_idx]         = p_m.bias.data[p_idx]
                        n_m.running_mean.data[n_idx] = p_m.running_mean.data[p_idx]
                        n_m.running_var.data[n_idx]  = p_m.running_var.data[p_idx]
            else:
                n = min(n_m.num_features, p_m.num_features)
                n_m.weight.data[:n]       = p_m.weight.data[:n]
                n_m.bias.data[:n]         = p_m.bias.data[:n]
                n_m.running_mean.data[:n] = p_m.running_mean.data[:n]
                n_m.running_var.data[:n]  = p_m.running_var.data[:n]

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

    def compute(self, sparse_model, target_layers, data_loader, n_batches=10):
        sparse_model.train()
        sparse_model.zero_grad()
        crit = nn.CrossEntropyLoss()
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= n_batches: break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            crit(sparse_model(inputs), targets).backward()
        sparse_model.eval()

        d_mods, channel_scores = dict(self.dense_model.named_modules()), {}
        for lname in target_layers:
            d_m = d_mods.get(lname)
            p_m = dict(sparse_model.named_modules()).get(lname)
            if d_m is None or p_m is None or p_m.weight.grad is None:
                channel_scores[lname] = {}
                continue
            p_grad    = p_m.weight.grad.detach()
            d_weight  = d_m.weight.detach()
            mean_grad = p_grad.mean(dim=0)
            scores = {}
            for ch in range(d_m.out_channels):
                d_w    = d_weight[ch]
                min_in = min(mean_grad.shape[0], d_w.shape[0])
                scores[ch] = (mean_grad[:min_in] * d_w[:min_in]).abs().sum().item()
            channel_scores[lname] = scores
            print(f"  {lname}: {d_m.out_channels} ch  max={max(scores.values()):.3e}")
        return channel_scores


def select_channels_by_taylor(channel_scores, pruned_to_dense, layer_name, n_restore):
    """从被剪掉的通道里，按 Taylor 分数选 top-n。"""
    scores     = channel_scores.get(layer_name, {})
    kept_dense = set(pruned_to_dense.get(layer_name, {}).values())
    pruned_set = set(scores.keys()) - kept_dense
    if not pruned_set or n_restore == 0:
        return []
    ranked = sorted(pruned_set, key=lambda c: scores.get(c, 0.0), reverse=True)
    return ranked[:min(n_restore, len(ranked))]


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Controller
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthAgent(nn.Module):
    def __init__(self, budget_space_size, alloc_space_size,
                 hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.DEVICE    = device
        self.nhid      = hidden_size
        self.input_dim = max(budget_space_size, alloc_space_size)

        self.lstm           = nn.LSTMCell(self.input_dim + context_dim, hidden_size)
        self.budget_decoder = nn.Linear(hidden_size, budget_space_size)
        self.alloc_decoder  = nn.Linear(hidden_size, alloc_space_size)
        self.hidden         = self.init_hidden()

    def forward(self, prev_logits, context_vec, step='alloc'):
        if prev_logits.dim() == 1: prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1: context_vec = context_vec.unsqueeze(0)
        pad = self.input_dim - prev_logits.shape[-1]
        if pad > 0:
            prev_logits = F.pad(prev_logits, (0, pad))
        h, c = self.lstm(torch.cat([prev_logits, context_vec], dim=-1), self.hidden)
        self.hidden = (h, c)
        return self.budget_decoder(h) if step == 'budget' else self.alloc_decoder(h)

    def init_hidden(self):
        return (torch.zeros(1, self.nhid, device=self.DEVICE),
                torch.zeros(1, self.nhid, device=self.DEVICE))


# ═══════════════════════════════════════════════════════════════════════════════
# RL Policy Gradient
# ═══════════════════════════════════════════════════════════════════════════════

class OneshotStructuredRegrowthPG:

    def __init__(self, config, model_sparse, dense_model,
                 channel_scores, pruned_to_dense,
                 original_channels, example_inputs,
                 target_layers, train_loader, test_loader,
                 device, wandb_run=None):

        self.NUM_EPOCHS         = config['num_epochs']
        self.ALPHA              = config['learning_rate']
        self.HIDDEN_SIZE        = config['hidden_size']
        self.BETA               = config['entropy_coef']
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.005)
        self.DEVICE             = device
        self.BUDGET_SPACE       = config['budget_space_size']
        self.ALLOC_SPACE        = config['alloc_space_size']
        self.NUM_STEPS          = len(target_layers)
        self.CONTEXT_DIM        = config.get('context_dim', 3)
        self.BASELINE_DECAY     = config.get('baseline_decay', 0.9)

        self.acc_threshold = config['acc_threshold']
        self.budget_bonus  = config.get('budget_bonus', 0.02)

        self.model_sparse      = model_sparse
        self.dense_model       = dense_model
        self.target_layers     = target_layers
        self.train_loader      = train_loader
        self.test_loader       = test_loader
        self.channel_scores    = channel_scores
        self.pruned_to_dense   = pruned_to_dense
        self.original_channels = original_channels
        self.example_inputs    = example_inputs
        self.pruned_config     = get_current_config(model_sparse)

        self.layer_capacities = config['layer_capacities']
        self.total_capacity   = max(sum(self.layer_capacities), 1)

        self.early_stop_patience  = config.get('early_stop_patience', 40)
        self.min_epochs           = config.get('min_epochs', 50)
        self.reward_std_threshold = config.get('reward_std_threshold', 0.002)
        self.reward_window_size   = config.get('reward_window_size', 20)
        self.finetune_epochs      = config.get('finetune_epochs', 5)

        self._best_model_state = None
        self._best_reward_seen = float('-inf')

        self.run            = wandb_run
        self.model_name     = config.get('model_name')
        self.method         = config.get('method', 'structured_oneshot')
        self.checkpoint_dir = config.get('checkpoint_dir', './structured_oneshot_rl_ckpts')
        self.model_sparsity = config.get('model_sparsity', '0.5')

        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta           = config.get('start_beta', 0.4)
        self.end_beta             = config.get('end_beta', 0.04)
        self.decay_fraction       = config.get('decay_fraction', 0.4)

        min_ch = max(1, config['target_restore_ch'] // 2)
        max_ch = min(config['target_restore_ch'], self.total_capacity)
        self.budget_options = [int(x) for x in
                               np.linspace(min_ch, max_ch, self.BUDGET_SPACE).tolist()]
        print(f"  Budget options: {self.budget_options}")

        self.agent = RegrowthAgent(
            budget_space_size=self.BUDGET_SPACE,
            alloc_space_size=self.ALLOC_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE,
        ).to(self.DEVICE)

        self.adam            = optim.Adam(self.agent.parameters(), lr=self.ALPHA)
        self.reward_baseline = None
        self.layer_priority  = [(n, i) for i, n in enumerate(target_layers)]

        print(f"  Acc threshold : {self.acc_threshold:.2f}%")
        print(f"  Budget bonus  : {self.budget_bonus}")
        print(f"  Layers ({len(target_layers)}):")
        for n, i in self.layer_priority:
            print(f"    {n}: cap={self.layer_capacities[i]}")

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
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                if not full_eval and i >= 20: break
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                _, pred = model(x).max(1)
                total   += y.size(0)
                correct += pred.eq(y).sum().item()
        return 100.0 * correct / total

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

    def calculate_loss(self, budget_logits, alloc_logits, wlp, beta):
        loss  = -torch.mean(wlp)
        p_b   = F.softmax(budget_logits, dim=1)
        ent_b = -torch.mean(torch.sum(p_b * F.log_softmax(budget_logits, dim=1), dim=1))
        p_a   = F.softmax(alloc_logits,  dim=1)
        ent_a = -torch.mean(torch.sum(p_a * F.log_softmax(alloc_logits,  dim=1), dim=1))
        ent   = (ent_b + ent_a) / 2.0
        return loss - beta * ent, ent

    def solve_environment(self):
        best_reward, best_reward_ep = float('-inf'), 0
        reward_window = deque(maxlen=self.reward_window_size)
        stop_reason   = ""

        for epoch in range(self.NUM_EPOCHS):
            ep_wlp, ep_budget_logits, ep_alloc_logits, reward, sparsity, budget_ch = \
                self.play_episode(epoch)

            reward_window.append(reward)
            if reward > best_reward:
                best_reward, best_reward_ep = reward, epoch

            beta      = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(ep_budget_logits, ep_alloc_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            no_imp  = epoch - best_reward_ep
            rwd_std = float(np.std(list(reward_window))) if len(reward_window) > 1 else float('inf')

            if self.run:
                self.run.log({
                    'epoch': epoch + 1, 'reward': reward,
                    'best_reward': best_reward,
                    'loss': loss.item(), 'entropy': ent.item(),
                    'beta': beta, 'budget_ch': budget_ch,
                    'sparsity': sparsity, 'no_improve': no_imp, 'rwd_std': rwd_std,
                })

            print(f"Ep {epoch + 1:3d}/{self.NUM_EPOCHS} | "
                  f"Rwd={reward:+.4f} Best={best_reward:+.4f} | "
                  f"BudgetCh={budget_ch} | Loss={loss.item():.4f} | "
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
        prev_logits       = torch.zeros(1, self.BUDGET_SPACE, device=self.DEVICE)
        all_log_probs, budget_masked_logits, alloc_masked_logits = [], [], []

        # Budget
        b_ctx    = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float,
                                device=self.DEVICE).unsqueeze(0)
        b_logits = self.agent(prev_logits, b_ctx, step='budget').squeeze(0)
        b_dist   = Categorical(probs=F.softmax(b_logits, dim=0))
        b_action = b_dist.sample()
        target_ch = self.budget_options[b_action.item()]

        all_log_probs.append(b_dist.log_prob(b_action))
        budget_masked_logits.append(b_logits)
        prev_logits = b_logits.unsqueeze(0)
        print(f"  [Budget] {target_ch} channels to restore")

        # Allocation
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

            logits   = self.agent(prev_logits, ctx, step='alloc').squeeze(0)
            eff_max  = min(cap, remaining)
            c_opts   = torch.round(ratio_opts * eff_max).to(torch.long)
            feasible = c_opts <= remaining
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

        ep_log_probs     = torch.stack(all_log_probs)
        ep_budget_logits = torch.stack(budget_masked_logits)
        ep_alloc_logits  = torch.stack(alloc_masked_logits)

        allocation      = {n: int(c) for n, c in zip(pnames, sel_counts) if c > 0}
        actual_restored = sum(sel_counts)
        usage_ratio     = actual_restored / max(target_ch, 1)
        print(f"  [Alloc] {actual_restored}/{target_ch} | {allocation}")

        # 用 Taylor 分数选具体通道，更新 new_config
        new_config = dict(self.pruned_config)
        for lname, n_add in allocation.items():
            chs = select_channels_by_taylor(
                self.channel_scores, self.pruned_to_dense, lname, n_add)
            if chs:
                new_config[lname] = min(
                    self.pruned_config[lname] + len(chs),
                    self.original_channels.get(lname, self.pruned_config[lname])
                )
                print(f"    {lname}: +{len(chs)} ch (top3: {chs[:3]})")

        # 重建 + 正确权重转移
        new_model = apply_config_with_matched_transfer(
            pruned_model=self.model_sparse,
            dense_model=self.dense_model,
            new_config=new_config,
            original_channels=self.original_channels,
            example_inputs=self.example_inputs,
            channel_scores=self.channel_scores,
            pruned_to_dense=self.pruned_to_dense,
            target_layers=self.target_layers,
        ).to(self.DEVICE)

        pre_ft_state = copy.deepcopy(new_model.state_dict())  # 涨了但未 finetune
        self.mini_finetune(new_model, epochs=self.finetune_epochs)

        accuracy = self.evaluate_model(new_model, full_eval=True)
        sparsity = compute_channel_sparsity(new_model, self.original_channels)

        acc_term    = (accuracy - self.acc_threshold) / 100.0
        budget_term = self.budget_bonus * usage_ratio
        reward      = acc_term + budget_term

        print(f"  [Reward] acc={accuracy:.2f}%  threshold={self.acc_threshold:.2f}%  "
              f"Δ={accuracy - self.acc_threshold:+.2f}pp  "
              f"usage={usage_ratio:.2f}  reward={reward:+.4f}  [{time.time() - t0:.1f}s]")

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model_state = pre_ft_state  # 存涨了但未 finetune 的
            self._save_best(epoch, reward, accuracy, pre_ft_state)

        if self.reward_baseline is None:
            self.reward_baseline = reward
        adv = float(np.clip(
            (reward - self.reward_baseline) / max(self.REWARD_TEMPERATURE, 1e-6),
            -10.0, 10.0))
        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline
                                + (1 - self.BASELINE_DECAY) * reward)

        adv_t  = torch.tensor(adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp = torch.sum(ep_log_probs * adv_t).unsqueeze(0)

        return ep_wlp, ep_budget_logits, ep_alloc_logits, reward, sparsity, target_ch

    def _save_best(self, epoch, reward, accuracy, state_dict):
        p = os.path.join(self._save_dir(), f'best_ep{epoch + 1}_rwd{reward:+.4f}.pth')
        torch.save({
            'epoch': epoch, 'reward': reward, 'accuracy_mini_ft': accuracy,
            'model_state_dict': state_dict,  # 涨了但未 finetune 的权重
        }, p)
        print(f"  ✓ Best (pre-finetune): reward={reward:+.4f}  mini_ft_acc={accuracy:.2f}% → {p}")
        if self.run:
            self.run.log({"best_reward": reward, "best_mini_ft_acc": accuracy,
                          "best_epoch": epoch + 1})


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
    parser.add_argument('--m_name',         type=str,   default='effnet')
    parser.add_argument('--data_dir',       type=str,   default='./data')
    parser.add_argument('--method',         type=str,   default='structured_oneshot')
    parser.add_argument('--pruned_ckpt',    type=str,   default="effnet/ckpt_after_prune_structured_oneshot/pruned_structured_l1_sp0.9_it1.pth")
    parser.add_argument('--acc_threshold',  type=float, default=50.0)
    parser.add_argument('--sparsity_delta', type=float, default=0.04)
    parser.add_argument('--budget_bonus',   type=float, default=0.02)
    parser.add_argument('--num_epochs',     type=int,   default=300)
    parser.add_argument('--learning_rate',  type=float, default=3e-4)
    parser.add_argument('--hidden_size',    type=int,   default=64)
    parser.add_argument('--entropy_coef',   type=float, default=0.5)
    parser.add_argument('--reward_temperature', type=float, default=0.005)
    parser.add_argument('--start_beta',     type=float, default=0.40)
    parser.add_argument('--end_beta',       type=float, default=0.04)
    parser.add_argument('--decay_fraction', type=float, default=0.4)
    parser.add_argument('--budget_space_size', type=int, default=5)
    parser.add_argument('--alloc_space_size',  type=int, default=11)
    parser.add_argument('--ssim_threshold',    type=float, default=0.0)
    parser.add_argument('--ssim_num_batches',  type=int,   default=64)
    parser.add_argument('--taylor_batches',    type=int,   default=10)
    parser.add_argument('--finetune_epochs',   type=int,   default=5)
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

    dense_model = model_loader(args.m_name, device)
    load_model_name(dense_model, f'./{args.m_name}/checkpoint', args.m_name)
    dense_model.eval()
    original_channels = get_original_channels(dense_model)

    pruned_model = torch.load(args.pruned_ckpt, map_location=device, weights_only=False)
    pruned_model.eval()

    sp0  = compute_channel_sparsity(pruned_model, original_channels)
    acc0 = quick_eval(pruned_model, test_loader, device)
    print(f"\nStarting → Acc={acc0:.2f}%  Sparsity={sp0:.4f}")

    target_layers    = get_target_layers(dense_model, pruned_model)
    layer_capacities = get_layer_capacities(dense_model, pruned_model, target_layers)
    total_original_ch = sum(original_channels.values())
    target_restore_ch = min(int(total_original_ch * args.sparsity_delta),
                            sum(layer_capacities))
    print(f"\n{len(target_layers)} pruned layers  "
          f"target_restore={target_restore_ch} ch  (delta={args.sparsity_delta:.4f})")

    if sum(layer_capacities) == 0:
        print("No channels to restore. Exiting.")
        return

    selected_layers, ssim_scores = SSIMLayerSelector.update_search_space(
        sparse_model=pruned_model, pretrained_model=dense_model,
        data_loader_ref=test_loader, target_layers=target_layers,
        threshold=args.ssim_threshold, num_batches=args.ssim_num_batches,
    )

    print("Computing Taylor scores…")
    channel_scores = TaylorChannelScorer(dense_model=dense_model, device=device).compute(
        sparse_model=pruned_model, target_layers=selected_layers,
        data_loader=train_loader, n_batches=args.taylor_batches,
    )

    # 预计算 pruned↔dense 通道对应（只需一次）
    pruned_to_dense  = build_pruned_to_dense(pruned_model, dense_model, selected_layers)
    layer_capacities = get_layer_capacities(dense_model, pruned_model, selected_layers)

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
        'budget_space_size': args.budget_space_size,
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
        'budget_bonus': args.budget_bonus,
    }

    pg = OneshotStructuredRegrowthPG(
        config=config, model_sparse=pruned_model, dense_model=dense_model,
        channel_scores=channel_scores, pruned_to_dense=pruned_to_dense,
        original_channels=original_channels, example_inputs=example_inputs,
        target_layers=selected_layers, train_loader=train_loader,
        test_loader=test_loader, device=device, wandb_run=run,
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