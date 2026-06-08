"""
Test sparsity and accuracy of a saved structured regrowth checkpoint.

Usage:
    python test_structure_regrowth_model.py --ckpt <path_to_ckpt.pth> \
        --m_name effnet --dense_ckpt_dir ./effnet/checkpoint
"""

import torch
import torch.nn as nn
import argparse

from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.analysis_tools import load_model_name


def get_original_channels(model):
    return {n: m.out_channels for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d)}


def compute_channel_sparsity(model, original_channels):
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total     += original_channels[name]
            remaining += m.out_channels
    return (1 - remaining / total) if total > 0 else 0.0


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def print_layer_channels(model, original_channels):
    print(f"\n{'Layer':<45} {'Dense':>6} {'Now':>6} {'Pruned':>7}")
    print("-" * 68)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            orig = original_channels[name]
            curr = m.out_channels
            diff = orig - curr
            flag = "  ←" if diff > 0 else ""
            print(f"  {name:<43} {orig:>6} {curr:>6} {diff:>7}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',           type=str, default="resnet20/ckpt_structured_iterative/step10_sp0.973.pth")
    parser.add_argument('--m_name',         type=str, default='resnet20')
    parser.add_argument('--dense_ckpt_dir', type=str, default=None)
    parser.add_argument('--data_dir',       type=str, default='./data')
    parser.add_argument('--dataset',        type=str, default='CIFAR10')
    parser.add_argument('--batch_size',     type=int, default=128)
    parser.add_argument('--num_workers',    type=int, default=4)
    parser.add_argument('--no_eval',        action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load dense model for reference channel counts
    dense_model = model_loader(args.m_name, device)
    ckpt_dir = args.dense_ckpt_dir or f'./{args.m_name}/checkpoint'
    load_model_name(dense_model, ckpt_dir, args.m_name)
    original_channels = get_original_channels(dense_model)
    total_dense_ch = sum(original_channels.values())

    # ── Load regrowth checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model      = ckpt['model'].to(device)
    pruned_map = ckpt.get('pruned_out_dense_map', {})

    sparsity   = compute_channel_sparsity(model, original_channels)
    total_curr = sum(m.out_channels for n, m in model.named_modules()
                     if isinstance(m, nn.Conv2d) and n in original_channels)

    print(f"\nCheckpoint : {args.ckpt}")
    print(f"Dense  total channels : {total_dense_ch}")
    print(f"Current total channels: {total_curr}")
    print(f"Sparsity              : {sparsity:.4f}  ({sparsity*100:.2f}%)")
    print(f"Restorable remaining  : {sum(len(v) for v in pruned_map.values())} ch")

    print_layer_channels(model, original_channels)

    if not args.no_eval:
        _, _, test_loader = data_loader(
            data_dir=args.data_dir, batch_size=args.batch_size,
            num_workers=args.num_workers, dataset=args.dataset,
        )
        acc = evaluate(model, test_loader, device)
        print(f"\nTest accuracy: {acc:.2f}%")


if __name__ == '__main__':
    main()
