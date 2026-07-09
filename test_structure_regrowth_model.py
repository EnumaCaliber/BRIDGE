"""
Test sparsity and accuracy of saved structured regrowth checkpoints.

Single file:
    python test_structure_regrowth_model.py --ckpt <path.pth> --m_name resnet20

Whole folder (exports Excel):
    python test_structure_regrowth_model.py --ckpt_dir ./structured_rl_ckpts/resnet20 \
        --m_name resnet20 --out results.xlsx
"""

import os
import glob
import torch
import torch.nn as nn
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    print(f"\n{'Layer':<45} {'Dense':>6} {'Now':>6} {'Pruned':>7} {'Sp%':>7}")
    print("-" * 76)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            orig = original_channels[name]
            curr = m.out_channels
            diff = orig - curr
            sp   = 100.0 * diff / orig if orig > 0 else 0.0
            flag = "  ←" if diff > 0 else ""
            print(f"  {name:<43} {orig:>6} {curr:>6} {diff:>7} {sp:>6.1f}%{flag}")


def eval_one(ckpt_path, original_channels, test_loader, device, no_eval=False):
    """Load one checkpoint and return a result dict."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ckpt['model'].to(device)

    pruned_map  = ckpt.get('pruned_dense_map',
                  ckpt.get('pruned_out_dense_map', {}))
    sparsity    = compute_channel_sparsity(model, original_channels)
    total_dense = sum(original_channels.values())
    total_curr  = sum(m.out_channels for n, m in model.named_modules()
                      if isinstance(m, nn.Conv2d) and n in original_channels)
    restorable  = sum(len(v) for v in pruned_map.values())

    acc = evaluate(model, test_loader, device) if not no_eval else None

    return {
        'file':       os.path.basename(ckpt_path),
        'sparsity':   round(sparsity, 4),
        'sparsity_%': round(sparsity * 100, 2),
        'total_ch':   total_curr,
        'dense_ch':   total_dense,
        'restorable': restorable,
        'acc_%':      round(acc, 2) if acc is not None else None,
        '_model':     model,
    }


def main():
    parser = argparse.ArgumentParser()
    # single-file mode
    parser.add_argument('--ckpt',           type=str,  default=None)
    # folder mode
    parser.add_argument('--ckpt_dir',       type=str,  default="./vgg16/ckpt_structured_iterative_hessian")
    parser.add_argument('--out',            type=str,  default='results.csv',
                        help='Excel output path (folder mode)')

    parser.add_argument('--m_name',         type=str,  default='vgg16')
    parser.add_argument('--dense_ckpt_dir', type=str,  default=None)
    parser.add_argument('--data_dir',       type=str,  default='./data')
    parser.add_argument('--dataset',        type=str,  default='CIFAR10')
    parser.add_argument('--batch_size',     type=int,  default=128)
    parser.add_argument('--val_split',      type=float,default=0.1)
    parser.add_argument('--num_workers',    type=int,  default=4)
    parser.add_argument('--no_eval',        action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Dense model
    dense_model = model_loader(args.m_name, device)
    ckpt_dir = args.dense_ckpt_dir or f'./{args.m_name}/checkpoint'
    load_model_name(dense_model, ckpt_dir, args.m_name)
    original_channels = get_original_channels(dense_model)

    # ── Data
    _, _, test_loader = data_loader(
        data_dir=args.data_dir, val_split=args.val_split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        dataset=args.dataset,
    )

    # ── Collect checkpoints
    if args.ckpt_dir:
        paths = sorted(glob.glob(os.path.join(args.ckpt_dir, '**', '*.pth'), recursive=True))
        if not paths:
            print(f"No .pth files found in {args.ckpt_dir}")
            return
    elif args.ckpt:
        paths = [args.ckpt]
    else:
        parser.error("Provide --ckpt or --ckpt_dir")

    # ── Evaluate
    rows = []
    for i, p in enumerate(paths):
        print(f"\n[{i+1}/{len(paths)}] {p}")
        try:
            r = eval_one(p, original_channels, test_loader, device, no_eval=args.no_eval)
            print(f"  Sparsity={r['sparsity_%']:.2f}%  "
                  + (f"Acc={r['acc_%']:.2f}%" if r['acc_%'] is not None else "no_eval"))

            if len(paths) == 1:
                print_layer_channels(r['_model'], original_channels)

            r.pop('_model')
            rows.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({'file': os.path.basename(p), 'error': str(e)})

    # ── Print summary table
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(df.to_string(index=False))

    # ── Save results
    if args.ckpt_dir or (args.ckpt and not args.no_eval):
        out_path = args.out
        if out_path.endswith('.csv'):
            df.to_csv(out_path, index=False)
        else:
            df.to_excel(out_path, index=False)
        print(f"\nSaved → {out_path}")

    # ── Plot sparsity vs accuracy
    plot_df = df.dropna(subset=['sparsity_%', 'acc_%']).copy()
    if len(plot_df) >= 2:
        plot_df = plot_df.sort_values('sparsity_%')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(plot_df['sparsity_%'], plot_df['acc_%'],
                marker='o', linewidth=1.5, markersize=5, color='steelblue')
        for _, row in plot_df.iterrows():
            ax.annotate(f"acc={row['acc_%']:.1f}%\nsp={row['sparsity_%']:.1f}%",
                        (row['sparsity_%'], row['acc_%']),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=7, color='dimgray')
        ax.set_xlabel('Channel Sparsity (%)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{args.m_name}  —  Sparsity vs Accuracy')
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        stem = os.path.splitext(args.out)[0]
        plot_path = stem + '_plot.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot   → {plot_path}")


if __name__ == '__main__':
    main()
