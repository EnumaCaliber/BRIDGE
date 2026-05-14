import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import numpy as np
import random
from models.model_loader import model_loader
from data.data_loader import data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--m_name', type=str, default='effnet')
parser.add_argument('--model_path', type=str, default='./rl_saliency_checkpoints/effnet/iterative/iter_0/best_grown_model.pth')
# parser.add_argument('--model_path', type=str, default='./rl_saliency_checkpoints/effnet/oneshot/0.94fullfinetune/final_model_0.94.pth')
# parser.add_argument('--model_path', type=str, default='./vgg16/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.99.pth')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--data_dir', type=str, default='./data')
# data_loader
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--val_split',  type=float, default=0.1)
parser.add_argument('--num_workers',type=int, default=15)
parser.add_argument('--data_dir',   type=str, default='./data')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn

def get_sparsity(sd):
    total = sum(v.numel() for k, v in sd.items() if k.endswith('_mask'))
    zeros = sum((v == 0).sum().item() for k, v in sd.items() if k.endswith('_mask'))
    return zeros / total * 100 if total > 0 else 0.0


def load_model(ckpt_path, model_name, device):
    sd = torch.load(ckpt_path, map_location=device)

    merged, done = {}, set()
    for k in sd:
        if k in done:
            continue
        if k.endswith('_orig'):
            base = k[:-5]
            merged[base] = sd[k] * sd[base + '_mask']
            done.update([k, base + '_mask'])
        elif not k.endswith('_mask'):
            merged[k] = sd[k]

    model = model_loader(model_name, device)
    model.load_state_dict(merged)
    return model, sd





criterion = nn.CrossEntropyLoss()
# def evaluate(model, loader, device):
#     model.eval()
#     correct, total_loss, total = 0, 0, 0
#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             out = model(x)
#             total_loss += F.cross_entropy(out, y).item()
#             correct += out.argmax(1).eq(y).sum().item()
#             total += y.size(0)
#     return 100.0 * correct / total, total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return acc, test_loss


# ── Main ──────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, val_loader, test_loader = data_loader(
    data_dir=args.data_dir,
    val_split=args.val_split,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    dataset= args.dataset,
)

ckpt_path = Path(args.model_path)
model, sd = load_model(ckpt_path, args.m_name, device)
# model, sd = model.to(device), sd.to(device)
sparsity = get_sparsity(sd)
acc, loss = evaluate(model, test_loader, device)

print(f"\n{'=' * 50}")
print(f"File      : {ckpt_path.name}")
print(f"Sparsity  : {sparsity:.2f}%")
print(f"Accuracy  : {acc:.2f}%")
print(f"Loss      : {loss:.4f}")
print(f"{'=' * 50}\n")
