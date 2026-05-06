import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np
from transformers import get_cosine_schedule_with_warmup

from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.analysis_tools import prune_weights_reparam
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def full_finetune(model, train_loader, test_loader, device,
                  epochs, lr, save_path, patience, save_interval, interval_dir):
    print(f"\n{'=' * 70}")
    print(f"Full Finetune  epochs={epochs}  lr={lr}  patience={patience}")
    print(f"{'=' * 70}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_acc, best_epoch, no_improve = 0.0, 0, 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
            scheduler.step()

        acc = evaluate(model, test_loader, device)

        if acc > best_acc:
            best_acc, best_epoch, no_improve = acc, epoch + 1, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:4d}/{epochs} | acc={acc:.2f}%  best={best_acc:.2f}% (ep {best_epoch})")

        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            import os
            os.makedirs(interval_dir, exist_ok=True)
            p = f"{interval_dir}/epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), p)
            print(f"  ✓ Interval ckpt → {p}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improve for {patience} epochs)")
            break

    print(f"\nBest accuracy: {best_acc:.2f}%  (epoch {best_epoch})")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name', type=str, default='vgg16')
    # TODO
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the regrown sparse model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'tiny_imagenet'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=15)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save_path', type=str, default=None,
                        help='Where to save the best model. Defaults to <model_path>_finetuned.pth')
    parser.add_argument('--save_interval', type=int, default=0,
                        help='Save a checkpoint every N epochs (0 = disabled)')
    parser.add_argument('--interval_dir', type=str, default=None,
                        help='Directory for interval checkpoints. Defaults to same dir as save_path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")


    if args.save_path is None:
        base = os.path.splitext(args.model_path)[0]
        args.save_path = f"{base}_finetuned.pth"
    if args.interval_dir is None:
        args.interval_dir = os.path.join(os.path.dirname(args.save_path), 'interval_ckpts')

    print(f"Model     : {args.m_name}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Save path : {args.save_path}")

    train_loader, val_loader, test_loader = data_loader(
        data_dir=args.data_dir, val_split=args.val_split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        dataset=args.dataset,
    )

    model = model_loader(args.m_name, device)
    prune_weights_reparam(model)
    model_state = torch.load(args.model_path,weights_only=False)['model_state_dict']
    model.load_state_dict(model_state)

    before_acc = evaluate(model, test_loader, device)
    print(f"\nAccuracy before finetune: {before_acc:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    best_acc = full_finetune(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        patience=args.patience,
        save_interval=args.save_interval,
        interval_dir=args.interval_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"Before finetune : {before_acc:.2f}%")
    print(f"After finetune  : {best_acc:.2f}%  (Δ={best_acc - before_acc:+.2f}pp)")
    print(f"Saved → {args.save_path}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()