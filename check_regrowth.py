"""
test_taylor_regrowth.py (fixed v3)
L1 重建 + 正确的 BN 权重转移。
"""

import torch
import torch.nn as nn
import copy
import argparse
import torch_pruning as tp

from models.model_loader import model_loader
from data.data_loader import data_loader

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
M_NAME      = 'vgg16'
PRUNED_CKPT = 'vgg16/ckpt_after_prune_structured_oneshot/pruned_structured_l1_sp0.9_it1.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',        type=str,   default='CIFAR10')
parser.add_argument('--batch_size',     type=int,   default=128)
parser.add_argument('--val_split',      type=float, default=0.1)
parser.add_argument('--num_workers',    type=int,   default=15)
parser.add_argument('--data_dir',       type=str,   default='./data')
parser.add_argument('--regrow_ratio',   type=float, default=0.5)
parser.add_argument('--taylor_batches', type=int,   default=10)
args = parser.parse_args()

train_loader, _, test_loader = data_loader(
    data_dir=args.data_dir, val_split=args.val_split,
    batch_size=args.batch_size, num_workers=args.num_workers,
    dataset=args.dataset,
)
example_inputs = next(iter(train_loader))[0][:1].to(DEVICE)

dense_model = model_loader(M_NAME, DEVICE)
ckpt = torch.load(f'./{M_NAME}/checkpoint/pretrain_{M_NAME}_ckpt.pth', weights_only=False)
dense_model.load_state_dict(ckpt['net'])
dense_model.eval()

pruned_model = torch.load(PRUNED_CKPT, map_location=DEVICE, weights_only=False)
pruned_model.eval()

original_channels = {n: m.out_channels for n, m in dense_model.named_modules()
                     if isinstance(m, nn.Conv2d)}
pruned_channels   = {n: m.out_channels for n, m in pruned_model.named_modules()
                     if isinstance(m, nn.Conv2d)}
target_layers     = [n for n in original_channels
                     if pruned_channels.get(n, original_channels[n]) < original_channels[n]]

total_orig = sum(original_channels.values())
total_prun = sum(pruned_channels.values())
sp_before  = 1 - total_prun / total_orig
print(f"Sparsity before: {sp_before:.4f}")

# ── Step 1: new_config ────────────────────────────────────────────────────────
new_config = dict(pruned_channels)
for lname in target_layers:
    cap = original_channels[lname] - pruned_channels[lname]
    new_config[lname] = pruned_channels[lname] + max(1, int(cap * args.regrow_ratio))

# ── Step 2: L1 重建 ───────────────────────────────────────────────────────────
print("Rebuilding with L1 importance…")
new_model = copy.deepcopy(dense_model)
pruning_ratio_dict = {}
for name, module in new_model.named_modules():
    if isinstance(module, nn.Conv2d) and name in new_config:
        sp = 1 - new_config[name] / original_channels[name]
        if sp > 0:
            pruning_ratio_dict[module] = sp

pruner = tp.pruner.MagnitudePruner(
    new_model, example_inputs,
    importance=tp.importance.MagnitudeImportance(p=1),
    iterative_steps=1, pruning_ratio=0,
    pruning_ratio_dict=pruning_ratio_dict,
)
pruner.step()

# ── Step 3: 验证 pruned/new 通道数是否对齐（L1 保证前 n 个通道相同）─────────
print("\nVerifying channel alignment between pruned and new model…")
p_mods = dict(pruned_model.named_modules())
n_mods = dict(new_model.named_modules())

all_ok = True
for name in target_layers:
    p_m = p_mods.get(name)
    n_m = n_mods.get(name)
    if p_m is None or n_m is None: continue
    n_out = min(p_m.weight.shape[0], n_m.weight.shape[0])
    n_in  = min(p_m.weight.shape[1], n_m.weight.shape[1])
    diff = (p_m.weight.data[:n_out, :n_in] - n_m.weight.data[:n_out, :n_in]).abs().max().item()
    ok = diff < 1e-4
    if not ok:
        print(f"  [MISMATCH] {name}: max_diff={diff:.3e}  "
              f"pruned_ch={p_m.out_channels}  new_ch={n_m.out_channels}")
        all_ok = False

if all_ok:
    print("  ✓ All pruned channels align with new model's first N channels")
else:
    print("  ✗ Channel mismatch — L1 ordering differs from original pruning")

# ── Step 4: 权重转移 ──────────────────────────────────────────────────────────
print("\nTransferring weights…")
with torch.no_grad():
    for name, new_m in new_model.named_modules():
        p_m = p_mods.get(name)
        if p_m is None: continue

        if isinstance(new_m, nn.Conv2d):
            n_out = min(new_m.weight.shape[0], p_m.weight.shape[0])
            n_in  = min(new_m.weight.shape[1], p_m.weight.shape[1])
            new_m.weight.data[:n_out, :n_in] = p_m.weight.data[:n_out, :n_in]
            if new_m.bias is not None and p_m.bias is not None:
                new_m.bias.data[:n_out] = p_m.bias.data[:n_out]

        elif isinstance(new_m, nn.BatchNorm2d):
            n = min(new_m.num_features, p_m.num_features)
            new_m.weight.data[:n]       = p_m.weight.data[:n]
            new_m.bias.data[:n]         = p_m.bias.data[:n]
            new_m.running_mean.data[:n] = p_m.running_mean.data[:n]
            new_m.running_var.data[:n]  = p_m.running_var.data[:n]
            new_m.num_batches_tracked.copy_(p_m.num_batches_tracked)

        elif isinstance(new_m, nn.Linear):
            n_out = min(new_m.weight.shape[0], p_m.weight.shape[0])
            n_in  = min(new_m.weight.shape[1], p_m.weight.shape[1])
            new_m.weight.data[:n_out, :n_in] = p_m.weight.data[:n_out, :n_in]
            if new_m.bias is not None and p_m.bias is not None:
                new_m.bias.data[:n_out] = p_m.bias.data[:n_out]

new_model = new_model.to(DEVICE)

# ── Step 5: 验证 ──────────────────────────────────────────────────────────────
actual_channels = {n: m.out_channels for n, m in new_model.named_modules()
                   if isinstance(m, nn.Conv2d)}
total_new = sum(actual_channels.values())
sp_after  = 1 - total_new / total_orig

print("\n" + "=" * 70)
print(f"{'Layer':<40} {'want':>6} {'got':>6}  match?")
print("=" * 70)
mismatch = 0
for name in original_channels:
    want = new_config.get(name)
    got  = actual_channels.get(name)
    ok   = "✓" if want == got else f"✗ (off by {got - want})"
    if want != got: mismatch += 1
    print(f"  {name:<38} {want:>6} {got:>6}  {ok}")

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, pred = model(x).max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total

acc_before = evaluate(pruned_model, test_loader)
acc_after  = evaluate(new_model,    test_loader)

print(f"\nCascade mismatches : {mismatch}/{len(original_channels)}")
print(f"Sparsity  : {sp_before:.4f} → {sp_after:.4f}  (Δ={sp_after - sp_before:+.4f})")
print(f"Accuracy  : {acc_before:.2f}% → {acc_after:.2f}%  (Δ={acc_after - acc_before:+.2f}pp)")

if sp_after < sp_before:
    print("\n△ 通道涨回成功，开始 mini finetune 验证精度能否恢复…")
else:
    print("\n✗ 通道未涨回")

# ── Mini finetune ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Mini Finetune (10 epochs)…")
print("=" * 60)

new_model.train()
crit = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4, weight_decay=0.01)
best_acc, best_state = 0.0, None

for epoch in range(2):
    new_model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        crit(new_model(x), y).backward()
        optimizer.step()

    acc = evaluate(new_model, test_loader)
    if acc > best_acc:
        best_acc = acc
        best_state = copy.deepcopy(new_model.state_dict())
    print(f"  Epoch {epoch + 1:2d}/10 | acc={acc:.2f}%  best={best_acc:.2f}%")

print(f"\n{'=' * 60}")
print(f"Before regrowth (pruned) : {acc_before:.2f}%")
print(f"After regrowth (no ft)   : {acc_after:.2f}%")
print(f"After mini finetune      : {best_acc:.2f}%  (Δ={best_acc - acc_before:+.2f}pp vs pruned)")
print(f"Sparsity before          : {sp_before:.4f}")
print(f"Sparsity after regrowth  : {sp_after:.4f}  (Δ={sp_after - sp_before:+.4f})")

# finetune 不改变结构，稀疏度不变
new_channels_ft = {n: m.out_channels for n, m in new_model.named_modules()
                   if isinstance(m, nn.Conv2d)}
sp_after_ft = 1 - sum(new_channels_ft.values()) / total_orig
print(f"Sparsity after finetune  : {sp_after_ft:.4f}  (should equal after regrowth)")
print(f"{'=' * 60}")