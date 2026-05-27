"""
Structured RL-based One-Shot Regrowth
======================================
DependencyGraph 기반 채널 복원 (index_map / pruned_dense_map 활용)
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
    """pruned_dense_map에 비어있지 않은 층 = 아직 복원 가능한 층"""
    return [name for name, pruned_list in pruned_dense_map.items()
            if len(pruned_list) > 0]


def get_layer_capacities(pruned_dense_map, target_layers):
    """각 층에서 복원 가능한 채널 수 = pruned_dense_map 길이"""
    return [len(pruned_dense_map[name]) for name in target_layers]


# ═══════════════════════════════════════════════════════════════════════════════
# DependencyGraph 기반 채널 복원
# ═══════════════════════════════════════════════════════════════════════════════

def regrow_channels_dg(model, example_inputs, allocation_dense_idx, index_map, device):
    """
    allocation_dense_idx : {layer_name: [dense_idx, ...]}
        - 이번 episode에서 복원할 층별 dense index 목록
    index_map : {layer_name: [survived_dense_idx, ...]}
        - 이 에피소드 전용 복사본을 받아 in-place 업데이트

    Returns: 채널이 복원된 새 모델 (in-place 수정 후 반환)
    """
    named_modules = dict(model.named_modules())

    for target_name, dense_indices in allocation_dense_idx.items():
        if not dense_indices:
            continue

        # 过滤掉已经在 index_map 里的（前一层 DG 连带涨进来的）
        already_survived = set(index_map.get(target_name, []))
        dense_indices = [idx for idx in dense_indices if idx not in already_survived]
        if not dense_indices:
            print(f"  [DG] {target_name} 全部已存在，跳过")
            continue

        target_conv = named_modules.get(target_name)
        if target_conv is None:
            print(f"  [DG] {target_name} not found, skip")
            continue

        num_grow = len(dense_indices)

        # DG는 모델 구조가 바뀔 때마다 재빌드
        DG = tp.DependencyGraph().build_dependency(model, example_inputs)
        group = DG.get_pruning_group(
            target_conv, tp.prune_conv_out_channels, idxs=[0]
        )

        for dep, _ in group:
            layer   = dep.target.module
            handler = dep.handler
            name    = dep.target.name

            if isinstance(layer, nn.Conv2d) and handler == tp.prune_conv_out_channels:
                old_w = layer.weight.data
                new_w = torch.zeros(num_grow, old_w.shape[1],
                                    *old_w.shape[2:], device=device)
                layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=0))
                if layer.bias is not None:
                    layer.bias = nn.Parameter(
                        torch.cat([layer.bias.data,
                                   torch.zeros(num_grow, device=device)]))
                layer.out_channels += num_grow
                print(f"    ✓ conv out [{name}]: "
                      f"{old_w.shape[0]} → {layer.out_channels}")

            elif isinstance(layer, nn.Conv2d) and handler == tp.prune_conv_in_channels:
                old_w = layer.weight.data
                new_w = torch.zeros(old_w.shape[0], num_grow,
                                    *old_w.shape[2:], device=device)
                layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=1))
                layer.in_channels += num_grow
                print(f"    ✓ conv in  [{name}]: "
                      f"{old_w.shape[1]} → {layer.in_channels}")

            elif isinstance(layer, nn.BatchNorm2d):
                old_ch = layer.num_features
                for attr, fill in [('weight', 1.0), ('bias', 0.0)]:
                    old = getattr(layer, attr).data
                    setattr(layer, attr, nn.Parameter(
                        torch.cat([old, torch.full((num_grow,), fill, device=device)])))
                for attr, fill in [('running_mean', 0.0), ('running_var', 1.0)]:
                    old = getattr(layer, attr)
                    setattr(layer, attr,
                            torch.cat([old, torch.full((num_grow,), fill, device=device)]))
                layer.num_features += num_grow
                print(f"    ✓ bn       [{name}]: {old_ch} → {layer.num_features}")

        # index_map 업데이트: 복원된 dense index를 survived 목록에 추가
        index_map[target_name] = sorted(
            index_map[target_name] + list(dense_indices))

        # 다음 층 처리를 위해 named_modules 갱신
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


def select_channels_by_taylor(channel_scores, pruned_dense_map, layer_name, n_restore):
    """
    pruned_dense_map[layer_name] : 이 층에서 아직 복원 안 된 dense index 목록
    → Taylor 점수 기준 상위 n_restore개 반환
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
                 channel_scores, index_map, pruned_dense_map,  # ← 변경
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
        self.index_map         = index_map          # ← 추가
        self.pruned_dense_map  = pruned_dense_map   # ← 추가 (dense idx 기준 pruned 목록)
        self.original_channels = original_channels
        self.example_inputs    = example_inputs

        self.layer_capacities = config['layer_capacities']
        self.total_capacity   = max(sum(self.layer_capacities), 1)

        self.early_stop_patience  = config.get('early_stop_patience', 40)
        self.min_epochs           = config.get('min_epochs', 50)
        self.reward_std_threshold = config.get('reward_std_threshold', 0.002)
        self.reward_window_size   = config.get('reward_window_size', 20)
        self.finetune_epochs      = config.get('finetune_epochs', 5)

        self._best_model_state  = None
        self._best_reward_seen  = float('-inf')
        self._best_index_map    = None  # ← 최적 index_map 저장

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

        # ── Budget 결정
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

        # ── Allocation 결정 (층별 채널 수)
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

        # ── Taylor 점수로 구체적 dense index 선택
        allocation_dense_idx = {}   # {layer_name: [dense_idx, ...]}
        actual_restored = 0
        for lname, n_add in zip(pnames, sel_counts):
            if n_add == 0:
                continue
            chs = select_channels_by_taylor(
                self.channel_scores, self.pruned_dense_map, lname, n_add)
            if chs:
                allocation_dense_idx[lname] = chs
                actual_restored += len(chs)
                print(f"    {lname}: +{len(chs)} ch (top3: {chs[:3]})")

        usage_ratio = actual_restored / max(target_ch, 1)
        print(f"  [Alloc] {actual_restored}/{target_ch}  | {list(allocation_dense_idx.keys())}")

        # ── DependencyGraph로 채널 복원
        ep_index_map = copy.deepcopy(self.index_map)  # episode 전용 index_map
        new_model = regrow_channels_dg(
            model=copy.deepcopy(self.model_sparse),
            example_inputs=self.example_inputs,
            allocation_dense_idx=allocation_dense_idx,
            index_map=ep_index_map,
            device=self.DEVICE,
        )

        pre_ft_model  = copy.deepcopy(new_model)
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
            self._best_model_state = copy.deepcopy(pre_ft_model.state_dict())
            self._best_index_map   = copy.deepcopy(ep_index_map)  # ← 최적 index_map 보존
            self._save_best(epoch, reward, accuracy, pre_ft_model, ep_index_map)

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

    def _save_best(self, epoch, reward, accuracy, model, index_map):
        p = os.path.join(self._save_dir(), f'best_ep{epoch + 1}_rwd{reward:+.4f}.pth')
        model_to_save = copy.deepcopy(model)
        for m in model_to_save.modules():
            m._forward_hooks.clear()
            m._backward_hooks.clear()
            m._forward_pre_hooks.clear()
        # index_map도 함께 저장 (나중에 추가 regrow 또는 분석에 사용)
        torch.save({
            'model': model_to_save,
            'index_map': index_map,
        }, p)
        print(f"  ✓ Best (pre-finetune): reward={reward:+.4f}  "
              f"mini_ft_acc={accuracy:.2f}% → {p}")
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
    parser.add_argument('--pruned_ckpt',    type=str,
                        default="effnet/ckpt_after_prune_structured_oneshot/pruned_structured_l1_sp0.9_it1.pth")
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

    # ── Dense 모델 로드
    dense_model = model_loader(args.m_name, device)
    load_model_name(dense_model, f'./{args.m_name}/checkpoint', args.m_name)
    dense_model.eval()
    original_channels = get_original_channels(dense_model)

    # ── 剪枝 체크포인트 로드 (index_map / pruned_dense_map 포함)
    ckpt = torch.load(args.pruned_ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        pruned_model    = ckpt['model'].to(device)
        index_map       = ckpt['index_map']        # {layer: [survived_dense_idx]}
        pruned_dense_map = ckpt['pruned_dense_map'] # {layer: [pruned_dense_idx]}
    else:
        # 구형 포맷 호환 (모델만 저장된 경우)
        pruned_model     = ckpt.to(device) if not isinstance(ckpt, dict) else ckpt
        index_map        = {}
        pruned_dense_map = {}
        print("[Warning] index_map / pruned_dense_map not found in checkpoint!")
    pruned_model.eval()

    sp0  = compute_channel_sparsity(pruned_model, original_channels)
    acc0 = quick_eval(pruned_model, test_loader, device)
    print(f"\nStarting → Acc={acc0:.2f}%  Sparsity={sp0:.4f}")

    # ── 复原可能的层（pruned_dense_map 基准）
    target_layers    = get_target_layers(pruned_dense_map)
    layer_capacities = get_layer_capacities(pruned_dense_map, target_layers)

    if sum(layer_capacities) == 0:
        print("No channels to restore. Exiting.")
        return

    total_original_ch = sum(original_channels.values())
    want_ch           = int(total_original_ch * args.sparsity_delta)
    print(f"\n{len(target_layers)} pruned layers  "
          f"want_restore={want_ch} ch  (delta={args.sparsity_delta:.4f})")

    # ── SSIM으로 손상 큰 층만 검색 공간으로 좁힘
    selected_layers, ssim_scores = SSIMLayerSelector.update_search_space(
        sparse_model=pruned_model, pretrained_model=dense_model,
        data_loader_ref=test_loader, target_layers=target_layers,
        threshold=args.ssim_threshold, num_batches=args.ssim_num_batches,
    )
    layer_capacities = get_layer_capacities(pruned_dense_map, selected_layers)

    # ── selected_layers 容量不足时，按 SSIM 分数从低到高逐层补入
    if sum(layer_capacities) < want_ch:
        print(f"  [Warning] selected_layers 容量 {sum(layer_capacities)} < 目标 {want_ch}，按 SSIM 补层...")
        remaining_layers = sorted(
            [l for l in target_layers if l not in selected_layers],
            key=lambda l: ssim_scores.get(l, 1.0)
        )
        for l in remaining_layers:
            selected_layers.append(l)
            layer_capacities = get_layer_capacities(pruned_dense_map, selected_layers)
            print(f"    补入 {l} (SSIM={ssim_scores.get(l, 1.0):.4f})，当前容量={sum(layer_capacities)}")
            if sum(layer_capacities) >= want_ch:
                break

    target_restore_ch = min(want_ch, sum(layer_capacities))
    print(f"  最终 selected_layers={len(selected_layers)} 层  "
          f"target_restore_ch={target_restore_ch}")

    # ── Taylor 중요도 계산
    print("Computing Taylor scores…")
    channel_scores = TaylorChannelScorer(dense_model=dense_model, device=device).compute(
        sparse_model=pruned_model, target_layers=selected_layers,
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
        channel_scores=channel_scores,
        index_map=index_map,               # ← 변경
        pruned_dense_map=pruned_dense_map,  # ← 변경
        original_channels=original_channels,
        example_inputs=example_inputs,
        target_layers=selected_layers,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device, wandb_run=run,
    )

    pg.solve_environment()

    # ── 최종 모델 복원
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