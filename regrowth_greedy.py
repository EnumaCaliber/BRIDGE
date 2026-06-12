"""
Greedy SSIM-based Iterative Regrowth — Baseline for RL comparison.

Two allocation modes
--------------------
fill  : fill the lowest-SSIM layer completely, then move to the next.
prop  : distribute budget proportional to (1 - SSIM); leftover filled greedily.

Within each layer, weights are selected by saliency score (grad² × weight²)
computed once on the pretrained dense model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import copy
import os
import re
import random
import time
from pathlib import Path
from tqdm import tqdm

from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.analysis_tools import (
    load_model_name, prune_weights_reparam, count_pruned_params,
    BlockwiseFeatureExtractor, compute_block_ssim,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Seed / Misc
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sparsity(model):
    total, pruned = 0, 0
    for _, m in model.named_modules():
        if hasattr(m, 'weight_mask'):
            total  += m.weight_mask.numel()
            pruned += (m.weight_mask == 0).sum().item()
    return 100.0 * pruned / total if total > 0 else 0.0


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


def get_layer_capacities(model, target_layers):
    caps = {}
    mods = dict(model.named_modules())
    for name in target_layers:
        m = mods.get(name)
        caps[name] = int((m.weight_mask == 0).sum().item()) \
            if (m is not None and hasattr(m, 'weight_mask')) else 0
    return caps


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline table
# ═══════════════════════════════════════════════════════════════════════════════

def load_baseline_from_folder(model_name, model_dir, device, test_loader):
    model_path = Path(model_dir)
    pth_files  = sorted(model_path.glob("*.pth"))
    if not pth_files:
        raise ValueError(f"No .pth files found in {model_dir}")

    print("=" * 60)
    baseline_dict = {}
    for pth_file in pth_files:
        match = re.search(r'(\d+\.\d+)', pth_file.name)
        if not match:
            print(f"  Warning: cannot parse sparsity from {pth_file.name}")
            continue
        sparsity = float(match.group(1))
        try:
            sd     = torch.load(pth_file, map_location=device, weights_only=True)
            merged = {}
            for k, v in sd.items():
                if k.endswith('_orig'):
                    merged[k[:-5]] = v * sd[k[:-5] + '_mask']
                elif not k.endswith('_mask'):
                    merged[k] = v
            m = model_loader(model_name, device)
            m.load_state_dict(merged)
            m.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    _, pred = m(x).max(1)
                    total   += y.size(0)
                    correct += pred.eq(y).sum().item()
            acc = 100.0 * correct / total
            baseline_dict[sparsity] = acc
            print(f"  {pth_file.name}: sp={sparsity:.4f}  acc={acc:.2f}%")
        except Exception as e:
            print(f"  Error {pth_file.name}: {e}")

    baseline_dict = dict(sorted(baseline_dict.items()))
    if not baseline_dict:
        raise ValueError(
            f"No valid baseline checkpoints in {model_dir}.\n"
            f"Files: {[f.name for f in pth_files]}\n"
            f"Expected filenames with a float, e.g. 'model_0.9903.pth'."
        )
    print(f"\nBaseline: {len(baseline_dict)} pts  "
          f"sp {min(baseline_dict):.4f}~{max(baseline_dict):.4f}  "
          f"acc {min(baseline_dict.values()):.2f}~{max(baseline_dict.values()):.2f}%")
    print("=" * 60 + "\n")
    return baseline_dict


class BaselineInterpolator:
    def __init__(self, table):
        self.pts = sorted((float(k), float(v)) for k, v in table.items())

    def get(self, sparsity):
        pts = self.pts
        if sparsity <= pts[0][0]:  return pts[0][1]
        if sparsity >= pts[-1][0]: return pts[-1][1]
        for i in range(len(pts) - 1):
            s1, a1 = pts[i];  s2, a2 = pts[i + 1]
            if s1 <= sparsity <= s2:
                t = (sparsity - s1) / (s2 - s1 + 1e-12)
                return a1 + t * (a2 - a1)
        return pts[-1][1]


# ═══════════════════════════════════════════════════════════════════════════════
# SSIM layer selector
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ssim_scores(sparse_model, dense_model, data_loader_ref, num_batches=64):
    all_masked = [n for n, m in sparse_model.named_modules()
                  if hasattr(m, 'weight_mask') and n]

    block_dict = {'all': all_masked}
    ext_d = BlockwiseFeatureExtractor(dense_model,  block_dict)
    ext_s = BlockwiseFeatureExtractor(sparse_model, block_dict)
    with torch.no_grad():
        fd = ext_d.extract_block_features(data_loader_ref, num_batches=num_batches)
        fs = ext_s.extract_block_features(data_loader_ref, num_batches=num_batches)

    block_ssim = compute_block_ssim(fd, fs).get('all', {})
    ssim_dict  = {n: float(block_ssim.get(n, 0.5)) for n in all_masked}

    print(f"\n  ── SSIM scores (ascending = higher priority) ──")
    for n, s in sorted(ssim_dict.items(), key=lambda x: x[1]):
        print(f"    {n}: {s:+.4f}")
    print()
    return all_masked, ssim_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Saliency
# ═══════════════════════════════════════════════════════════════════════════════

def compute_saliency(model, data_loader, target_layers, device):
    """Fisher-information saliency: grad² × weight² accumulated over dataset."""
    model.eval()
    crit     = nn.CrossEntropyLoss()
    mods     = dict(model.named_modules())
    accum    = {n: torch.zeros(mods[n].weight.shape, device=device)
                for n in target_layers if n in mods and hasattr(mods[n], 'weight')}
    name_map = {id(p): n for n, p in model.named_parameters()}
    count    = 0

    print("Computing saliency…")
    for x, y in tqdm(data_loader, desc="  grad accum"):
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss  = crit(model(x), y)
        grads = torch.autograd.grad(loss, model.parameters(),
                                    create_graph=False, allow_unused=True)
        for param, grad in zip(model.parameters(), grads):
            if grad is None:
                continue
            pname = name_map.get(id(param), '')
            for lname in target_layers:
                if pname == f'{lname}.weight':
                    accum[lname] += grad.pow(2).detach() * param.data.pow(2).detach()
                    break
        count += 1

    sal = {n: (v / max(count, 1)).cpu() for n, v in accum.items()}
    for n, s in sal.items():
        print(f"  {n}: mean={s.mean():.3e}  max={s.max():.3e}")
    print("Saliency done.\n")
    return sal


# ═══════════════════════════════════════════════════════════════════════════════
# Greedy allocation
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_allocate(ssim_dict, capacities, budget, mode):
    """
    Returns {layer_name: n_weights} with sum ≤ budget.

    mode='fill' : fill lowest-SSIM layers one by one until budget exhausted.
    mode='prop' : allocate proportional to (1-SSIM), then fill leftover greedily.
    """
    sorted_layers = sorted(ssim_dict.items(), key=lambda x: x[1])  # ascending SSIM

    if mode == 'fill':
        allocation = {}
        remaining  = budget
        for lname, _ in sorted_layers:
            cap = capacities.get(lname, 0)
            if cap == 0 or remaining == 0:
                continue
            give = min(cap, remaining)
            allocation[lname] = give
            remaining -= give
        return allocation

    # mode == 'prop'
    weights   = {n: max(0.0, 1.0 - s) for n, s in ssim_dict.items()}
    total_w   = sum(weights.values())
    remaining = budget
    allocation = {}

    if total_w > 0:
        for lname, _ in sorted_layers:
            cap = capacities.get(lname, 0)
            if cap == 0:
                continue
            proportion = weights[lname] / total_w
            give = min(cap, int(round(proportion * budget)))
            allocation[lname] = allocation.get(lname, 0) + give
            remaining -= give

    # Fill leftover greedily (lowest SSIM first)
    for lname, _ in sorted_layers:
        if remaining <= 0:
            break
        cap   = capacities.get(lname, 0)
        slack = cap - allocation.get(lname, 0)
        if slack > 0:
            add = min(slack, remaining)
            allocation[lname] = allocation.get(lname, 0) + add
            remaining -= add

    return {k: v for k, v in allocation.items() if v > 0}


# ═══════════════════════════════════════════════════════════════════════════════
# Saliency-based weight regrowth
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def apply_regrowth(model, layer_name, saliency, num_weights, init_strategy, device):
    m = dict(model.named_modules()).get(layer_name)
    if m is None or not hasattr(m, 'weight_mask'):
        return 0
    mask   = m.weight_mask
    sal    = saliency.to(device)
    pruned = (mask == 0)
    if not pruned.any():
        return 0

    sal_m         = sal.clone()
    sal_m[~pruned] = -float('inf')
    flat          = sal_m.flatten()
    k             = min(num_weights, (flat > -float('inf')).sum().item())
    if k == 0:
        return 0

    _, top_k = torch.topk(flat, k=k)
    wp       = getattr(m, 'weight_orig', m.weight)
    for fi in top_k:
        idx     = np.unravel_index(fi.cpu().item(), sal.shape)
        mask[idx] = 1.0
        if init_strategy == 'zero':
            wp.data[idx] = 0.0
        elif init_strategy == 'kaiming':
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(wp)
            b = np.sqrt(6.0 / fan_in)
            wp.data[idx] = torch.empty(1).uniform_(-b, b).item()
        elif init_strategy == 'xavier':
            fi_n, fo_n = nn.init._calculate_fan_in_and_fan_out(wp)
            b = np.sqrt(6.0 / (fi_n + fo_n))
            wp.data[idx] = torch.empty(1).uniform_(-b, b).item()
    return k


# ═══════════════════════════════════════════════════════════════════════════════
# Finetune
# ═══════════════════════════════════════════════════════════════════════════════

def finetune(model, train_loader, test_loader, epochs, lr, device, patience=10):
    model.train()
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, no_improve = 0.0, None, 0
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        acc = quick_eval(model, test_loader, device)
        if acc > best_acc:
            best_acc, best_state, no_improve = acc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        model.train()
        if no_improve >= patience:
            print(f"    finetune early stop at ep {ep + 1}/{epochs} (no improve {patience})")
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return best_acc


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name',         type=str,   default='effnet')
    parser.add_argument('--data_dir',       type=str,   default='./data')
    parser.add_argument('--baseline_dir',   type=str,
                        default='./effnet/ckpt_after_prune_0.3_epoch_finetune_40/')
    parser.add_argument('--initial_ckpt',   type=str,
                        default='./effnet/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth')
    parser.add_argument('--greedy_mode',    type=str,   default='prop',
                        choices=['fill', 'prop'],
                        help='fill=lowest-SSIM-first; prop=proportional to (1-SSIM)')
    parser.add_argument('--budget_frac',    type=float, default=0.001,
                        help='Fraction of total weights to restore per iteration '
                             '(set --num_iters 1 for one-shot)')
    parser.add_argument('--num_iters',      type=int,   default=5,
                        help='Number of regrowth iterations; set to 1 for one-shot')
    parser.add_argument('--finetune_epochs', type=int,   default=40)
    parser.add_argument('--finetune_lr',     type=float, default=3e-4)
    parser.add_argument('--finetune_patience', type=int, default=40)
    parser.add_argument('--ssim_num_batches', type=int, default=64)
    parser.add_argument('--init_strategy',  type=str,   default='zero',
                        choices=['zero', 'kaiming', 'xavier'])
    parser.add_argument('--save_dir',       type=str,   default='./greedy_ckpts')
    parser.add_argument('--resume_iter',    type=int,   default=0)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--dataset',        type=str,   default='CIFAR10')
    parser.add_argument('--batch_size',     type=int,   default=128)
    parser.add_argument('--val_split',      type=float, default=0.1)
    parser.add_argument('--num_workers',    type=int,   default=15)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode_str = "ONE-SHOT" if args.num_iters == 1 else f"ITERATIVE x{args.num_iters}"
    print(f"Device: {device}  greedy-{args.greedy_mode}  [{mode_str}]  "
          f"budget_frac={args.budget_frac}")

    train_loader, _, test_loader = data_loader(
        data_dir=args.data_dir, val_split=args.val_split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        dataset=args.dataset,
    )

    # ── Baseline interpolation table
    baseline_dict  = load_baseline_from_folder(
        args.m_name, args.baseline_dir, device, test_loader)
    baseline_interp = BaselineInterpolator(baseline_dict)

    # ── Total prunable weights (for budget scaling)
    _ref = model_loader(args.m_name, device)
    prune_weights_reparam(_ref)
    _ref.load_state_dict(torch.load(args.initial_ckpt, map_location=device,
                                    weights_only=True))
    total_weights, _, _ = count_pruned_params(_ref)
    del _ref
    print(f"Total prunable weights: {total_weights:,}\n")

    # ── Dense model (for SSIM reference and saliency)
    dense_model = model_loader(args.m_name, device)
    load_model_name(dense_model, f'./{args.m_name}/checkpoint', args.m_name)
    dense_model.eval()

    # ── Masked layers for saliency
    _init = model_loader(args.m_name, device)
    prune_weights_reparam(_init)
    _init.load_state_dict(torch.load(args.initial_ckpt, map_location=device,
                                     weights_only=True))
    all_masked = [n for n, m in _init.named_modules()
                  if hasattr(m, 'weight_mask') and n]
    del _init

    saliency_dict = compute_saliency(dense_model, train_loader, all_masked, device)

    # ── Current sparse model
    current_model = model_loader(args.m_name, device)
    prune_weights_reparam(current_model)
    if args.resume_iter > 0:
        ckp = os.path.join(args.save_dir,
                           f'{args.m_name}/greedy_{args.greedy_mode}/'
                           f'iter_{args.resume_iter - 1}/best_grown_model.pth')
        assert os.path.exists(ckp), f"Resume ckpt not found: {ckp}"
        current_model.load_state_dict(torch.load(ckp, map_location=device,
                                                  weights_only=True))
        print(f"Resumed from iter {args.resume_iter}: {ckp}")
    else:
        current_model.load_state_dict(torch.load(args.initial_ckpt, map_location=device,
                                                  weights_only=True))

    acc0 = quick_eval(current_model, test_loader, device)
    sp0  = get_sparsity(current_model)
    print(f"Start → Acc={acc0:.2f}%  Sp={sp0:.2f}%  "
          f"baseline@sp={baseline_interp.get(sp0/100):.2f}%\n")

    budget_per_iter = max(1, int(total_weights * args.budget_frac))
    print(f"Budget per iter: {budget_per_iter} weights "
          f"({args.budget_frac:.4f} × {total_weights:,})\n")

    results = []

    for iter_idx in range(args.resume_iter, args.num_iters):
        t0     = time.time()
        cur_sp = get_sparsity(current_model)
        cur_acc = quick_eval(current_model, test_loader, device)
        bline  = baseline_interp.get(cur_sp / 100.0)

        print(f"\n{'#' * 70}")
        print(f"  ITER {iter_idx + 1}/{args.num_iters}  |  "
              f"sp={cur_sp:.2f}%  acc={cur_acc:.2f}%  "
              f"baseline={bline:.2f}%  gap={cur_acc - bline:+.2f}pp")
        print(f"{'#' * 70}\n")

        # ── SSIM scores for all masked layers
        target_layers, ssim_dict = compute_ssim_scores(
            sparse_model=current_model, dense_model=dense_model,
            data_loader_ref=test_loader,
            num_batches=args.ssim_num_batches,
        )

        capacities = get_layer_capacities(current_model, target_layers)
        if sum(capacities.values()) == 0:
            print("  All pruned weights restored. Stopping.")
            break

        # ── Greedy allocation
        allocation = greedy_allocate(
            ssim_dict={n: ssim_dict[n] for n in target_layers},
            capacities=capacities,
            budget=min(budget_per_iter, sum(capacities.values())),
            mode=args.greedy_mode,
        )
        total_alloc = sum(allocation.values())
        print(f"  [Greedy-{args.greedy_mode}] allocated {total_alloc} weights:")
        for lname, n in sorted(allocation.items(), key=lambda x: ssim_dict[x[0]]):
            print(f"    {lname}: {n}  (SSIM={ssim_dict[lname]:+.4f})")

        # ── Apply regrowth
        grown_model = model_loader(args.m_name, device)
        prune_weights_reparam(grown_model)
        grown_model.load_state_dict(copy.deepcopy(current_model.state_dict()))

        actual_grown = 0
        for lname, num_w in allocation.items():
            sal = saliency_dict.get(lname)
            if sal is None:
                continue
            k = apply_regrowth(grown_model, lname, sal, num_w,
                               args.init_strategy, device)
            actual_grown += k

        new_sp = get_sparsity(grown_model)
        print(f"\n  Regrown {actual_grown} weights  "
              f"sp: {cur_sp:.2f}% → {new_sp:.2f}%")

        # ── Finetune
        best_acc = finetune(grown_model, train_loader, test_loader,
                            args.finetune_epochs, args.finetune_lr, device,
                            patience=args.finetune_patience)
        final_sp  = get_sparsity(grown_model)
        base_now  = baseline_interp.get(final_sp / 100.0)
        delta_pp  = best_acc - base_now

        elapsed = time.time() - t0
        print(f"\n  [Iter {iter_idx + 1}] acc={best_acc:.2f}%  sp={final_sp:.2f}%  "
              f"baseline={base_now:.2f}%  Δbaseline={delta_pp:+.2f}pp  "
              f"[{elapsed:.0f}s]")

        results.append({
            'iter': iter_idx + 1, 'acc': best_acc, 'sparsity': final_sp,
            'baseline': base_now, 'delta_pp': delta_pp,
        })

        # ── Save
        save_dir = os.path.join(
            args.save_dir,
            f'{args.m_name}/greedy_{args.greedy_mode}/iter_{iter_idx}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'best_grown_model.pth')
        torch.save(grown_model.state_dict(), save_path)
        print(f"  ✓ Saved → {save_path}")

        current_model = grown_model

    # ── Summary
    print(f"\n{'=' * 70}")
    print(f"  greedy-{args.greedy_mode}  budget_frac={args.budget_frac}  "
          f"init={args.init_strategy}")
    print(f"  {'Iter':>4}  {'Acc':>7}  {'Sp':>7}  {'Baseline':>9}  {'Δpp':>6}")
    for r in results:
        print(f"  {r['iter']:>4}  {r['acc']:>7.2f}  {r['sparsity']:>7.2f}  "
              f"{r['baseline']:>9.2f}  {r['delta_pp']:>+6.2f}")

    final_acc = quick_eval(current_model, test_loader, device)
    final_sp  = get_sparsity(current_model)
    print(f"\nFINAL  Acc={final_acc:.2f}%  Sp={final_sp:.2f}%  "
          f"Δbaseline={final_acc - baseline_interp.get(final_sp/100):+.2f}pp")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
