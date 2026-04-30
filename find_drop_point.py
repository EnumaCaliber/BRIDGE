import torch
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.model_loader import model_loader
from data.data_loader import data_loader
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--m_name', type=str, default='resnet20')
parser.add_argument('--model_dir', type=str, default="./resnet20/iterative", help='Directory containing pth files')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=15)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")


def get_sparsity_from_state_dict(sd):
    total = 0
    zeros = 0
    for k, v in sd.items():
        if k.endswith('_mask'):
            total += v.numel()
            zeros += (v == 0).sum().item()
    return zeros / total * 100 if total > 0 else 0


def load_model(ckpt_path, model_name, device):
    sd = torch.load(ckpt_path, map_location=device)
    sparsity = get_sparsity_from_state_dict(sd)
    merged = {}
    for k, v in sd.items():
        if k.endswith('_orig'):
            base = k[:-5]
            merged[base] = v * sd[base + '_mask']
        elif not k.endswith('_mask'):
            merged[k] = v
    model = model_loader(model_name, device)
    model.load_state_dict(merged)
    return model, sparsity


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def find_drop_point_by_residual(df):
    df = df.sort_values('sparsity').reset_index(drop=True)

    print("\n" + "=" * 70)
    print("Finding Critical Drop Point via Linear Regression")
    print("=" * 70)

    baseline_acc = df['accuracy'].iloc[0]
    print(f"Baseline accuracy: {baseline_acc:.2f}%\n")

    critical_idx = None
    critical_residual = 0

    for i in range(3, len(df)):
        X = df['sparsity'].iloc[:i].values.reshape(-1, 1)
        y = df['accuracy'].iloc[:i].values
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        y_pred = model_lr.predict(X)
        residuals = np.abs(y - y_pred)
        current_residual = residuals[-1]
        avg_residual = np.mean(residuals[:-1]) if i > 1 else 0
        if avg_residual > 0 and current_residual > 3 * avg_residual:
            critical_idx = i - 1
            critical_residual = current_residual
            break

    if critical_idx is not None:
        critical_point = df.iloc[critical_idx]
        prev_point = df.iloc[critical_idx - 1] if critical_idx > 0 else None

        print(f"⚠️  Critical point detected at index {critical_idx}:")
        print(f"   Sparsity: {critical_point['sparsity']:.2f}%")
        print(f"   Accuracy: {critical_point['accuracy']:.2f}%")
        print(f"   Residual: {critical_residual:.4f} (vs avg {avg_residual:.4f})")

        if prev_point is not None:
            drop = prev_point['accuracy'] - critical_point['accuracy']
            print(f"   Accuracy drop from previous: {drop:.2f}%")

        return critical_point
    else:
        print("✓ No critical drop point detected (accuracy degrades gradually)")
        return None


def find_drop_point_by_segmented_fit(df):
    df = df.sort_values('sparsity').reset_index(drop=True)
    best_split_idx = None
    min_total_error = float('inf')
    for split_idx in range(2, len(df) - 2):
        X_left = df['sparsity'].iloc[:split_idx + 1].values.reshape(-1, 1)
        y_left = df['accuracy'].iloc[:split_idx + 1].values
        model_left = LinearRegression()
        model_left.fit(X_left, y_left)
        error_left = np.mean((y_left - model_left.predict(X_left)) ** 2)

        X_right = df['sparsity'].iloc[split_idx:].values.reshape(-1, 1)
        y_right = df['accuracy'].iloc[split_idx:].values
        model_right = LinearRegression()
        model_right.fit(X_right, y_right)
        error_right = np.mean((y_right - model_right.predict(X_right)) ** 2)

        total_error = error_left + error_right

        if total_error < min_total_error:
            min_total_error = total_error
            best_split_idx = split_idx

    if best_split_idx is not None:
        split_point = df.iloc[best_split_idx]

        print(f"\n📊 Segmented regression analysis:")
        print(f"   Best split point at sparsity: {split_point['sparsity']:.2f}%")
        print(f"   Accuracy at split: {split_point['accuracy']:.2f}%")

        X_left = df['sparsity'].iloc[:best_split_idx + 1].values.reshape(-1, 1)
        y_left = df['accuracy'].iloc[:best_split_idx + 1].values
        model_left = LinearRegression()
        model_left.fit(X_left, y_left)

        X_right = df['sparsity'].iloc[best_split_idx:].values.reshape(-1, 1)
        y_right = df['accuracy'].iloc[best_split_idx:].values
        model_right = LinearRegression()
        model_right.fit(X_right, y_right)

        print(f"   Left slope: {model_left.coef_[0]:.4f} (gentle degradation)")
        print(f"   Right slope: {model_right.coef_[0]:.4f} (steep degradation)")
        print(f"   Slope change: {abs(model_right.coef_[0] - model_left.coef_[0]):.4f}")

        return split_point

    return None


def plot_accuracy_vs_sparsity(df, critical_point, split_point, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(df['sparsity'], df['accuracy'], 'o-', color='steelblue',
             linewidth=2, markersize=8, label='Accuracy')


    if critical_point is not None:
        plt.scatter(critical_point['sparsity'], critical_point['accuracy'],
                    color='red', s=200, marker='X', zorder=5,
                    edgecolors='black', linewidths=2, label='Critical Point')


    if split_point is not None:
        plt.scatter(split_point['sparsity'], split_point['accuracy'],
                    color='orange', s=150, marker='s', zorder=5,
                    edgecolors='black', linewidths=2, label='Split Point')

    plt.xlabel('Sparsity (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Sparsity ({args.m_name})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    train_loader, val_loader, test_loader = data_loader(
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=args.dataset,
    )

    model_dir = Path(args.model_dir)
    pth_files = sorted(model_dir.glob("*.pth"))

    if not pth_files:
        print(f"No .pth files found in {args.model_dir}")
        return

    print(f"Found {len(pth_files)} models\n")

    results = []
    for pth_file in pth_files:
        print(f"Evaluating: {pth_file.name}")

        try:
            model, sparsity = load_model(pth_file, args.m_name, device)
            accuracy = evaluate(model, test_loader, device)

            results.append({
                'filename': pth_file.name,
                'sparsity': sparsity,
                'accuracy': accuracy
            })

            print(f"  Sparsity: {sparsity:.2f}%, Accuracy: {accuracy:.2f}%")

        except Exception as e:
            print(f"  Error: {e}")

    df = pd.DataFrame(results)
    df = df.sort_values('sparsity').reset_index(drop=True)

    df.to_csv(model_dir / 'evaluation_results.csv', index=False)


    critical_point = find_drop_point_by_residual(df)
    split_point = find_drop_point_by_segmented_fit(df)

    print("\n" + "=" * 70)
    print("All Results (sorted by sparsity):")
    print("=" * 70)
    for i, row in df.iterrows():
        marker = ""
        if critical_point is not None and row['sparsity'] == critical_point['sparsity']:
            marker = "  ← CRITICAL POINT (residual method)"
        elif split_point is not None and row['sparsity'] == split_point['sparsity']:
            marker = "  ← SPLIT POINT (segmented regression)"
        print(f"Sparsity: {row['sparsity']:6.2f}%  |  Accuracy: {row['accuracy']:.2f}%{marker}")


    if critical_point is not None:
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATION:")
        print(f"   Maximum safe sparsity: {critical_point['sparsity']:.2f}%")
        print(f"   Expected accuracy: {critical_point['accuracy']:.2f}%")
        print(f"   (Accuracy starts to degrade rapidly beyond this point)")
        print("=" * 70)
    elif split_point is not None:
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATION:")
        print(f"   Maximum safe sparsity: {split_point['sparsity']:.2f}%")
        print(f"   Expected accuracy: {split_point['accuracy']:.2f}%")
        print("=" * 70)

    # 画图
    plot_path = model_dir / 'accuracy_vs_sparsity.png'
    plot_accuracy_vs_sparsity(df, critical_point, split_point, plot_path)


if __name__ == '__main__':
    main()