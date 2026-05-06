"""
eval_regrown.py
加载涨回去的结构化稀疏模型，测稀疏度和精度。
"""

import torch
import torch.nn as nn
import argparse

from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.analysis_tools import load_model_name

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--m_name',      type=str, default="effnet")
parser.add_argument('--regrown_ckpt',type=str, default ="structured_rl_ckpts/effnet/structured_oneshot/sp0.899/best_ep1_rwd+0.1976.pth",
                    help='涨回去的模型路径（best_ep*.pth）')
parser.add_argument('--pruned_ckpt', type=str, default= None,
                    help='对比用的 pruned model 路径（可选）')
parser.add_argument('--data_dir',    type=str, default='./data')
parser.add_argument('--dataset',     type=str, default='CIFAR10')
parser.add_argument('--batch_size',  type=int, default=128)
parser.add_argument('--val_split',   type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=15)
args = parser.parse_args()


# ── 加载数据 ──────────────────────────────────────────────────────────────────
_, _, test_loader = data_loader(
    data_dir=args.data_dir, val_split=args.val_split,
    batch_size=args.batch_size, num_workers=args.num_workers,
    dataset=args.dataset,
)


# ── 加载 dense model（用于计算稀疏度基准）────────────────────────────────────
dense_model = model_loader(args.m_name, DEVICE)
load_model_name(dense_model, f'./{args.m_name}/checkpoint', args.m_name)
dense_model.eval()

original_channels = {n: m.out_channels for n, m in dense_model.named_modules()
                     if isinstance(m, nn.Conv2d)}
total_orig = sum(original_channels.values())


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def compute_channel_sparsity(model):
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total     += original_channels[name]
            remaining += m.out_channels
    return (1 - remaining / total) if total > 0 else 0.0


def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, pred = model(x).max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def print_channel_table(model, label):
    channels = {n: m.out_channels for n, m in model.named_modules()
                if isinstance(m, nn.Conv2d)}
    sp = compute_channel_sparsity(model)
    total_ch = sum(channels.values())
    print(f"\n{'=' * 65}")
    print(f"  {label}  (sparsity={sp:.4f}  total_ch={total_ch}/{total_orig})")
    print(f"{'=' * 65}")
    print(f"  {'Layer':<40} {'dense':>6} {'model':>7} {'pruned%':>8}")
    print(f"  {'-' * 63}")
    for name in original_channels:
        d_ch  = original_channels[name]
        m_ch  = channels.get(name, '?')
        ratio = f"{100*(d_ch - m_ch)/d_ch:.1f}%" if isinstance(m_ch, int) else '?'
        diff  = f"({d_ch - m_ch:+d})" if isinstance(m_ch, int) else ''
        print(f"  {name:<40} {d_ch:>6} {m_ch:>6}{diff:<5} {ratio:>8}")


# ── 加载 regrown model ────────────────────────────────────────────────────────
print(f"\nLoading regrown model: {args.regrown_ckpt}")
ckpt = torch.load(args.regrown_ckpt, map_location=DEVICE, weights_only=False)

# 支持两种存储格式：直接 state_dict 或带 key 的 dict
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    print(f"  epoch={ckpt.get('epoch', '?')}  "
          f"reward={ckpt.get('reward', '?')}  "
          f"mini_ft_acc={ckpt.get('accuracy_mini_ft', '?')}")
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt

# 用 state_dict 推断模型结构（结构化剪枝后的模型需要先 load 整个模型）
# 尝试直接 torch.load（如果存的是整个模型）
try:
    regrown_model = torch.load(args.regrown_ckpt, map_location=DEVICE, weights_only=False)
    if not isinstance(regrown_model, nn.Module):
        raise ValueError("Not a nn.Module")
except Exception:
    # 存的是 state_dict，需要先重建结构
    # 结构化剪枝模型需要知道剪后的结构，这里暂时报错提示
    print("  ✗ 存储格式是 state_dict，需要先重建模型结构才能加载。")
    print("  请确认保存时用的是 torch.save(model, path) 而非 torch.save(model.state_dict(), path)")
    exit(1)

if isinstance(regrown_model, dict):
    print("  ✗ 文件是 dict，不是 nn.Module。请用 torch.save(model, path) 保存整个模型。")
    exit(1)

regrown_model = regrown_model.to(DEVICE)
regrown_model.eval()

# ── 打印表格 ──────────────────────────────────────────────────────────────────
print_channel_table(regrown_model, "Regrown Model")

sp_regrown  = compute_channel_sparsity(regrown_model)
acc_regrown = evaluate(regrown_model)

# ── 对比 pruned model（可选）────────────────────────────────────────────────
if args.pruned_ckpt:
    print(f"\nLoading pruned model: {args.pruned_ckpt}")
    pruned_model = torch.load(args.pruned_ckpt, map_location=DEVICE, weights_only=False)
    if isinstance(pruned_model, dict) and 'model_state_dict' in pruned_model:
        # state_dict 格式，跳过通道表格
        sp_pruned  = None
        acc_pruned = None
        print("  (pruned model 是 state_dict 格式，跳过通道对比)")
    else:
        pruned_model = pruned_model.to(DEVICE)
        pruned_model.eval()
        print_channel_table(pruned_model, "Pruned Model (before regrowth)")
        sp_pruned  = compute_channel_sparsity(pruned_model)
        acc_pruned = evaluate(pruned_model)
else:
    sp_pruned  = None
    acc_pruned = None

# ── 汇总 ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  Summary")
print(f"{'=' * 65}")
if sp_pruned is not None:
    print(f"  Pruned   → Sparsity: {sp_pruned:.4f}  Acc: {acc_pruned:.2f}%")
print(f"  Regrown  → Sparsity: {sp_regrown:.4f}  Acc: {acc_regrown:.2f}%  (no finetune)")
if sp_pruned is not None:
    print(f"  Δ        → Sparsity: {sp_regrown - sp_pruned:+.4f}  "
          f"Acc: {acc_regrown - acc_pruned:+.2f}pp")
print(f"{'=' * 65}\n")