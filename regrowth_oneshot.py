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
from tqdm import tqdm

from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.analysis_tools import (
    load_model_name, prune_weights_reparam, count_pruned_params,
    BlockwiseFeatureExtractor, compute_block_ssim,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# Saliency
# ═══════════════════════════════════════════════════════════════════════════════

class SaliencyComputer:
    def __init__(self, model, criterion, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.accumulated_grads = {}
        self.grad_count = 0

    def reset(self):
        self.accumulated_grads = {}
        self.grad_count = 0

    def compute_saliency_scores(self, data_loader, target_layers):
        self.model.eval()
        self.reset()
        print("\nComputing saliency scores…")

        module_dict = dict(self.model.named_modules())
        for lname in target_layers:
            m = module_dict.get(lname)
            if m is not None and hasattr(m, 'weight'):
                self.accumulated_grads[lname] = torch.zeros(m.weight.shape, device=self.device)

        for inputs, labels in tqdm(data_loader, desc="  grad accum"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            loss = self.criterion(self.model(inputs), labels)
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            for param, grad in zip(self.model.parameters(), grads):
                if grad is None:
                    continue
                for name, p in self.model.named_parameters():
                    if p is param:
                        for lname in target_layers:
                            if name == f"{lname}.weight":
                                self.accumulated_grads[lname] += (
                                    grad.pow(2).detach() * param.data.pow(2).detach())
                                break
                        break
            self.grad_count += 1

        saliency_dict = {}
        for lname in target_layers:
            if lname in self.accumulated_grads:
                sal = self.accumulated_grads[lname] / max(self.grad_count, 1)
                saliency_dict[lname] = sal.cpu()
                print(f"  {lname}: mean={sal.mean():.3e}  max={sal.max():.3e}")
        print("Saliency done.\n")
        return saliency_dict


# ═══════════════════════════════════════════════════════════════════════════════
# SSIM Layer Selector
# ═══════════════════════════════════════════════════════════════════════════════

class SSIMLayerSelector:
    @staticmethod
    def update_search_space(sparse_model, pretrained_model,
                            data_loader_ref, threshold=0.0, num_batches=64):
        all_masked = [name for name, m in sparse_model.named_modules()
                      if hasattr(m, 'weight_mask') and len(name) > 0]

        block_dict = {'all_layers': all_masked}
        ext_pre = BlockwiseFeatureExtractor(pretrained_model, block_dict)
        ext_spar = BlockwiseFeatureExtractor(sparse_model, block_dict)

        with torch.no_grad():
            feats_pre = ext_pre.extract_block_features(data_loader_ref, num_batches=num_batches)
            feats_spar = ext_spar.extract_block_features(data_loader_ref, num_batches=num_batches)

        block_ssim = compute_block_ssim(feats_pre, feats_spar).get('all_layers', {})

        ssim_dict, selected = {}, []
        for lname in all_masked:
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
            print(f"  No layer below {threshold:+.3f} → fallback: {worst} ({ssim_dict[worst]:+.4f})")

        print(f"  Search space: {len(selected)}/{len(all_masked)} layers\n")
        return selected, ssim_dict


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Controller
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthAgent(nn.Module):
    def __init__(self, action_dim, hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.DEVICE = device
        self.nhid = hidden_size
        self.action_dim = action_dim

        self.lstm = nn.LSTMCell(action_dim + context_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, action_dim)
        self.hidden = self.init_hidden()

    def forward(self, prev_logits, context_vec):
        if prev_logits.dim() == 1: prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1: context_vec = context_vec.unsqueeze(0)
        h, c = self.lstm(torch.cat([prev_logits, context_vec], dim=-1), self.hidden)
        self.hidden = (h, c)
        return self.decoder(h)

    def init_hidden(self):
        return (torch.zeros(1, self.nhid, device=self.DEVICE),
                torch.zeros(1, self.nhid, device=self.DEVICE))


# ═══════════════════════════════════════════════════════════════════════════════
# Saliency-based Regrowth
# ═══════════════════════════════════════════════════════════════════════════════

class SaliencyBasedRegrowth:
    @staticmethod
    @torch.no_grad()
    def apply_regrowth(model, layer_name, saliency_tensor, num_weights,
                       init_strategy='zero', device='cuda'):
        module = dict(model.named_modules()).get(layer_name)
        if module is None or not hasattr(module, 'weight_mask'):
            return 0, []

        mask = module.weight_mask
        sal = saliency_tensor.to(device)
        pruned = (mask == 0)
        if not pruned.any():
            return 0, []

        sal_m = sal.clone()
        sal_m[~pruned] = -float('inf')
        flat = sal_m.flatten()
        k = min(num_weights, (flat > -float('inf')).sum().item())
        if k == 0:
            return 0, []

        _, top_k = torch.topk(flat, k=k)
        regrown = []
        for fi in top_k:
            idx = np.unravel_index(fi.cpu().item(), sal.shape)
            regrown.append(idx)
            mask[idx] = 1.0
            wp = getattr(module, 'weight_orig', module.weight)
            if init_strategy == 'zero':
                wp.data[idx] = 0.0
            elif init_strategy == 'kaiming':
                fi_n, _ = nn.init._calculate_fan_in_and_fan_out(wp)
                b = np.sqrt(6.0 / fi_n)
                wp.data[idx] = torch.empty(1).uniform_(-b, b).item()
            elif init_strategy == 'xavier':
                fi_n, fo_n = nn.init._calculate_fan_in_and_fan_out(wp)
                b = np.sqrt(6.0 / (fi_n + fo_n))
                wp.data[idx] = torch.empty(1).uniform_(-b, b).item()
        return len(regrown), regrown


# ═══════════════════════════════════════════════════════════════════════════════
# RL Policy Gradient
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthPolicyGradient:
    def __init__(self, config, model_pretrained, model_99,
                 target_layers, train_loader, test_loader, device, wandb_run=None):

        self.NUM_EPOCHS = config['num_epochs']
        self.ALPHA = config['learning_rate']
        self.HIDDEN_SIZE = config['hidden_size']
        self.BETA = config['entropy_coef']
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.01)
        self.DEVICE = device
        self.ACTION_SPACE = config['action_space_size']
        self.NUM_STEPS = len(target_layers)
        self.CONTEXT_DIM = config.get('context_dim', 3)
        self.BASELINE_DECAY = config.get('baseline_decay', 0.9)

        self.model_pretrained = model_pretrained.to(device)
        self.model_99 = model_99.to(device)
        self.target_layers = target_layers
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.target_regrow = config['target_regrow']
        self.layer_capacities = config['layer_capacities']
        self.total_capacity = max(sum(self.layer_capacities), 1)
        self.init_strategy = config.get('init_strategy', 'zero')
        self.budget_bonus = config.get('budget_bonus', 0.02)  # weight for budget utilization term

        self.early_stop_patience = config.get('early_stop_patience', 40)
        self.min_epochs = config.get('min_epochs', 50)
        self.reward_std_threshold = config.get('reward_std_threshold', 0.002)
        self.reward_window_size = config.get('reward_window_size', 20)
        self.acc_baseline = config.get('acc_baseline', None)

        self._best_model_state = None
        self._best_reward_seen = float('-inf')

        self.run = wandb_run
        self.model_name = config.get('model_name')
        self.method = config.get('method', 'oneshot')
        self.checkpoint_dir = config.get('checkpoint_dir', './rl_saliency_checkpoints')
        self.model_sparsity = config.get('model_sparsity', '0.98')

        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta = config.get('start_beta', 0.4)
        self.end_beta = config.get('end_beta', 0.004)
        self.decay_fraction = config.get('decay_fraction', 0.4)

        print("\nComputing saliency scores…")
        self.saliency_dict = SaliencyComputer(
            model=self.model_pretrained,
            criterion=nn.CrossEntropyLoss(),
            device=self.DEVICE,
        ).compute_saliency_scores(
            data_loader=self.train_loader,
            target_layers=self.target_layers,
        )

        self.agent = RegrowthAgent(
            action_dim=self.ACTION_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE,
        ).to(self.DEVICE)

        self.adam = optim.Adam(self.agent.parameters(), lr=self.ALPHA)
        self.reward_baseline = None
        self.layer_priority = [(n, i) for i, n in enumerate(target_layers)]

        # Evaluate sparse model once — used as reward zero-point
        self.before_accuracy = self.evaluate_model(self.model_99, full_eval=True)
        print(f"  Sparse model accuracy (reward zero-point): {self.before_accuracy:.2f}%")

        print(f"  Search-space layers ({len(target_layers)}):")
        for n, i in self.layer_priority:
            print(f"    {i + 1}. {n}  cap={self.layer_capacities[i]}")

    def get_entropy_coef(self, epoch):
        if not self.use_entropy_schedule:
            return self.BETA
        de = self.NUM_EPOCHS * self.decay_fraction
        if epoch < de:
            return self.start_beta - (self.start_beta - self.end_beta) * (epoch / de)
        return self.end_beta

    def _create_model_copy(self, src):
        m = model_loader(self.model_name, self.DEVICE)
        prune_weights_reparam(m)
        m.load_state_dict(src.state_dict())
        return m

    def calculate_sparsity(self, model):
        total, pruned = 0, 0
        for _, m in model.named_modules():
            if hasattr(m, 'weight_mask'):
                total += m.weight_mask.numel()
                pruned += (m.weight_mask == 0).sum().item()
        return (100.0 * pruned / total if total > 0 else 0.0), total, pruned

    def evaluate_model(self, model, full_eval=False):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                if not full_eval and i >= 20: break
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                _, pred = model(x).max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        return 100.0 * correct / total

    def mini_finetune(self, model, epochs=50, lr=3e-4):
        model.train()
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        crit = nn.CrossEntropyLoss()
        best_acc, best_state = 0.0, None
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                    _, pred = model(x).max(1)
                    total += y.size(0)
                    correct += pred.eq(y).sum().item()
            acc = 100.0 * correct / total
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
            model.train()
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

    def calculate_loss(self, epoch_logits, weighted_log_probs, beta):
        loss = -torch.mean(weighted_log_probs)
        p = F.softmax(epoch_logits, dim=1)
        ent = -torch.mean(torch.sum(p * F.log_softmax(epoch_logits, dim=1), dim=1))
        return loss - beta * ent, ent

    def solve_environment(self):
        best_reward, best_alloc, best_regrow = float('-inf'), None, None
        best_reward_ep = 0
        reward_window = deque(maxlen=self.reward_window_size)
        stop_reason = ""

        for epoch in range(self.NUM_EPOCHS):
            ep_wlp, ep_logits, reward, accuracy, improvement, usage_ratio, alloc, sparsity, regrow = self.play_episode(epoch)

            reward_window.append(reward)
            if reward > best_reward:
                best_reward, best_reward_ep = reward, epoch
                best_alloc, best_regrow = alloc, copy.deepcopy(regrow)

            beta = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(ep_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.adam.step()

            no_imp = epoch - best_reward_ep
            rwd_std = float(np.std(list(reward_window))) if len(reward_window) > 1 else float('inf')

            if self.run:
                self.run.log({
                    "epoch": epoch + 1,
                    "acc": accuracy,
                    "improvement_pp": improvement,
                    "budget_usage": usage_ratio,
                    "reward": reward,
                    "loss": loss.item(),
                    "entropy": ent.item(),
                    "beta": beta,
                    "sparsity": sparsity,
                    "no_improve": no_imp,
                    "rwd_std": rwd_std,
                })

            print(f"Ep {epoch + 1:3d}/{self.NUM_EPOCHS} | "
                  f"Acc={accuracy:.2f}% Δ={improvement:+.2f}pp | "
                  f"Usage={usage_ratio:.2f} | Rwd={reward:+.4f} Best={best_reward:+.4f} | "
                  f"Loss={loss.item():.4f} | Ent={ent.item():.4f} | "
                  f"NoImp={no_imp} | Std={rwd_std:.5f}")

            if no_imp >= self.early_stop_patience:
                stop_reason = f"NoImp {self.early_stop_patience}"
                print(f"\n>>> Early stop: {stop_reason}")
                break
            if (epoch >= self.min_epochs
                    and len(reward_window) >= self.reward_window_size
                    and rwd_std < self.reward_std_threshold):
                stop_reason = f"Std {rwd_std:.5f} < {self.reward_std_threshold}"
                print(f"\n>>> Early stop: {stop_reason}")
                break

        print(f"\nBest improvement: {best_reward * 100:+.2f}pp  "
              f"(sparse baseline: {self.before_accuracy:.2f}%)"
              + (f"  [{stop_reason}]" if stop_reason else ""))

        best_model = self._create_model_copy(self.model_99)
        if self._best_model_state is not None:
            best_model.load_state_dict(self._best_model_state)
        return best_alloc, best_reward, best_regrow, best_model

    def play_episode(self, epoch):
        self.agent.hidden = self.agent.init_hidden()
        prev_logits = torch.zeros(1, self.ACTION_SPACE, device=self.DEVICE)

        ratio_opts = (torch.arange(1, self.ACTION_SPACE + 1, device=self.DEVICE, dtype=torch.float)
                      / self.ACTION_SPACE)
        remaining = int(self.target_regrow)
        total_budget = remaining
        log_probs, masked_logits_list = [], []
        sel_counts, pnames = [], []

        for p_idx, (lname, orig_idx) in enumerate(self.layer_priority):
            cap = int(self.layer_capacities[orig_idx])
            ctx = torch.tensor([
                p_idx / max(self.NUM_STEPS - 1, 1),
                cap / self.total_capacity,
                remaining / total_budget if total_budget > 0 else 0.0,
            ], dtype=torch.float, device=self.DEVICE).unsqueeze(0)

            logits = self.agent(prev_logits, ctx).squeeze(0)
            eff_max = min(cap, remaining)
            c_opts = torch.round(ratio_opts * eff_max).to(torch.long)
            seen_vals, dedup = set(), torch.zeros(self.ACTION_SPACE, dtype=torch.bool, device=self.DEVICE)
            for j, v in enumerate(c_opts.tolist()):
                if v not in seen_vals:
                    seen_vals.add(v)
                    dedup[j] = True
            feasible = (c_opts <= remaining) & dedup
            if not feasible.any(): feasible[0] = True

            masked = torch.where(feasible, logits, torch.full_like(logits, -1e9))
            dist = Categorical(probs=F.softmax(masked, dim=0))
            action = dist.sample()
            chosen = min(int(c_opts[action].item()), cap, remaining)
            remaining = max(remaining - chosen, 0)

            sel_counts.append(chosen)
            pnames.append(lname)
            log_probs.append(dist.log_prob(action))
            masked_logits_list.append(masked)
            prev_logits = logits.unsqueeze(0)

        ep_log_probs = torch.stack(log_probs)
        ep_logits = torch.stack(masked_logits_list)
        allocation = {n: int(c) for n, c in zip(pnames, sel_counts) if c > 0}
        print(f"  [Alloc] {sum(sel_counts)}/{total_budget} | {allocation}")

        model_copy = self._create_model_copy(self.model_99)
        regrow_indices = {}
        for lname, num_w in allocation.items():
            sal = self.saliency_dict.get(lname)
            if sal is not None:
                _, idxs = SaliencyBasedRegrowth.apply_regrowth(
                    model=model_copy, layer_name=lname, saliency_tensor=sal,
                    num_weights=num_w, init_strategy=self.init_strategy,
                    device=self.DEVICE)
                regrow_indices[lname] = idxs

        pre_ft_model = self._create_model_copy(self.model_99)
        pre_ft_model.load_state_dict(model_copy.state_dict())
        self.mini_finetune(model_copy, epochs=50)

        accuracy = self.evaluate_model(model_copy, full_eval=True)
        sparsity, _, _ = self.calculate_sparsity(model_copy)

        actual_regrown = sum(len(v) for v in regrow_indices.values())
        usage_ratio = actual_regrown / max(self.target_regrow, 1)   # 0 → 1

        improvement = accuracy - self.before_accuracy               # pp gain over sparse model
        acc_term = improvement / 100.0                              # negative if accuracy drops
        budget_term = self.budget_bonus * usage_ratio               # always ≥ 0, rewards using budget
        reward = acc_term + budget_term

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model_state = copy.deepcopy(pre_ft_model.state_dict())
            self._save_best_model(epoch, reward, accuracy, pre_ft_model)

        if self.acc_baseline is not None and accuracy / 100.0 > self.acc_baseline:
            self._save_baseline_model(epoch, accuracy, pre_ft_model)

        print(f"  [Reward] acc={accuracy:.2f}%  Δ={improvement:+.2f}pp  "
              f"usage={usage_ratio:.2f}({actual_regrown}/{self.target_regrow})  "
              f"reward={reward:+.4f}(acc={acc_term:+.4f} + budget={budget_term:.4f})")

        if self.reward_baseline is None:
            self.reward_baseline = reward
        adv = float(np.clip(
            (reward - self.reward_baseline) / max(self.REWARD_TEMPERATURE, 1e-6),
            -100.0, 100.0))
        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline
                                + (1 - self.BASELINE_DECAY) * reward)

        adv_t = torch.tensor(adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp = torch.sum(ep_log_probs * adv_t).unsqueeze(0)

        return ep_wlp, ep_logits, reward, accuracy, improvement, usage_ratio, allocation, sparsity, regrow_indices

    def _save_dir(self):
        d = os.path.join(self.checkpoint_dir,
                         f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        return d

    def _save_best_model(self, epoch, reward, accuracy, pre_ft_model):
        p = os.path.join(self._save_dir(), f'best_ep{epoch + 1}_rwd{reward:+.4f}.pth')
        torch.save(pre_ft_model, p)
        print(f"  ✓ Best saved (pre-ft): reward={reward:+.4f}  mini_ft_acc={accuracy:.2f}% → {p}")
        if self.run:
            self.run.log({"best_reward": reward, "best_mini_ft_acc": accuracy, "best_epoch": epoch + 1})

    def _save_baseline_model(self, epoch, accuracy, pre_ft_model):
        p = os.path.join(self._save_dir(), f'baseline_exceeded_ep{epoch + 1}_acc{accuracy:.2f}.pth')
        torch.save(pre_ft_model, p)
        print(f"  ✓ Baseline exceeded (pre-ft): mini_ft_acc={accuracy:.2f}% → {p}")
        if self.run:
            self.run.log({"baseline_exceeded_acc": accuracy, "baseline_exceeded_epoch": epoch + 1})





# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name', type=str, default='resnet20')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_sparsity', type=str, default='0.98')
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--entropy_coef', type=float, default=0.5)
    parser.add_argument('--reward_temperature', type=float, default=0.005)
    parser.add_argument('--start_beta', type=float, default=0.40)
    parser.add_argument('--end_beta', type=float, default=0.04)
    parser.add_argument('--decay_fraction', type=float, default=0.4)
    parser.add_argument('--action_space_size', type=int, default=11)
    parser.add_argument('--regrow_step', type=float, default=0.01)
    parser.add_argument('--budget_bonus', type=float, default=0.005,
                        help='Weight for budget utilization term in reward (encourages using full regrowth budget)')
    parser.add_argument('--init_strategy', type=str, default='zero',
                        choices=['zero', 'kaiming', 'xavier', 'magnitude'])
    parser.add_argument('--ssim_threshold', type=float, default=0.0)
    parser.add_argument('--ssim_num_batches', type=int, default=64)
    parser.add_argument('--initial_ckpt', type=str, default="resnet20/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.98.pth",
                        help='Path to the pruned sparse model checkpoint')
    parser.add_argument('--early_stop_patience', type=int, default=50)
    parser.add_argument('--min_epochs', type=int, default=50)
    parser.add_argument('--reward_std_threshold', type=float, default=0.002)
    parser.add_argument('--reward_window_size', type=int, default=20)
    parser.add_argument('--acc_baseline', type=float, default=0.8117)
    parser.add_argument('--save_dir', type=str, default='./rl_saliency_oneshot')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    # data_loader
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='dataset CIFAR10, tiny_imagenet')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=15)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = data_loader(
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=args.dataset,
    )

    print("Loading models…")
    model_pretrained = model_loader(args.m_name, device)
    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)
    model_pretrained.eval()

    model_99 = model_loader(args.m_name, device)
    prune_weights_reparam(model_99)
    model_99.load_state_dict(torch.load(args.initial_ckpt))

    # Auto-select target layers via feature-map SSIM
    target_layers, ssim_scores = SSIMLayerSelector.update_search_space(
        sparse_model=model_99,
        pretrained_model=model_pretrained,
        data_loader_ref=test_loader,
        threshold=args.ssim_threshold,
        num_batches=args.ssim_num_batches,
    )

    layer_capacities = []
    for lname in target_layers:
        m = dict(model_99.named_modules())[lname]
        layer_capacities.append(int((m.weight_mask == 0).sum().item())
                                if hasattr(m, 'weight_mask') else 0)

    total_weights, _, _ = count_pruned_params(model_99)
    target_regrow = min(int(total_weights * args.regrow_step), sum(layer_capacities))

    print(f"Total weights: {total_weights}  Target regrow: {target_regrow}  "
          f"Total capacity: {sum(layer_capacities)}")

    if not args.no_wandb:
        run = wandb.init(
            project="ICCAD_saliency_final",
            name=f"regrowth_{args.m_name}_{args.model_sparsity}",
            config=vars(args),
        )
        for lname, sc in ssim_scores.items():
            run.log({f"ssim/{lname}": sc})
    else:
        run = None

    config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'entropy_coef': args.entropy_coef,
        'action_space_size': args.action_space_size,
        'target_regrow': target_regrow,
        'layer_capacities': layer_capacities,
        'model_name': args.m_name,
        'method': 'oneshot',
        'reward_temperature': args.reward_temperature,
        'checkpoint_dir': args.save_dir,
        'start_beta': args.start_beta,
        'end_beta': args.end_beta,
        'decay_fraction': args.decay_fraction,
        'init_strategy': args.init_strategy,
        'budget_bonus': args.budget_bonus,
        'early_stop_patience': args.early_stop_patience,
        'min_epochs': args.min_epochs,
        'reward_std_threshold': args.reward_std_threshold,
        'reward_window_size': args.reward_window_size,
        'acc_baseline': args.acc_baseline,
        'model_sparsity': args.model_sparsity,
    }

    pg = RegrowthPolicyGradient(
        config=config,
        model_pretrained=model_pretrained,
        model_99=model_99,
        target_layers=target_layers,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        wandb_run=run,
    )

    before_acc = pg.evaluate_model(model_99, full_eval=True)
    before_sp, _, _ = pg.calculate_sparsity(model_99)
    print(f"\nBefore: acc={before_acc:.2f}%  sparsity={before_sp:.2f}%")

    t_start = time.time()
    best_alloc, best_reward, _, _ = pg.solve_environment()
    elapsed = time.time() - t_start

    summary = (
        f"model        : {args.m_name}\n"
        f"sparsity     : {args.model_sparsity}\n"
        f"sparse_acc   : {before_acc:.2f}%\n"
        f"best_mini_ft : {before_acc + best_reward * 100:.2f}%  (Δ={best_reward * 100:+.2f}pp)\n"
        f"search_time  : {elapsed:.1f}s  ({elapsed/3600:.2f}h)\n"
        f"best_alloc   : {best_alloc}\n"
    )
    print(f"\n{'=' * 60}")
    print(summary, end='')
    print(f"{'=' * 60}")

    log_path = os.path.join(pg._save_dir(), 'search_summary.txt')
    with open(log_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved → {log_path}")

    if run:
        run.log({"search_time_s": elapsed})
        run.finish()


if __name__ == '__main__':
    main()