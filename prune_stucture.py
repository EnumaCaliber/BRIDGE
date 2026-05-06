import torch.optim as optim
import argparse
import wandb
import copy
from functools import partial
import torch_pruning as tp

from utils.train import trainer_loader
from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.tools import *
from utils.pruners import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Structured Iterative Prune')
parser.add_argument('--m_name', type=str, default="resnet20")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pruner', type=str, default='l1',
                    choices=['l1', 'lamp', 'taylor'])
parser.add_argument('--target_sp', type=float, default=0.99,
                    help='channel sparsity')
parser.add_argument('--iterative_steps', type=int, default=25,
                    help='')
parser.add_argument('--finetune_steps', type=int, default=40 * 313,
                    help='')
args = parser.parse_args()

import random
import numpy as np

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, val_loader, test_loader = data_loader(data_dir='./data')

# -------------------------------------------------------
#
# -------------------------------------------------------
net = model_loader(args.m_name, device)
cktp = f'pretrain_{args.m_name}_ckpt.pth'
checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}')
net.load_state_dict(checkpoint['net'])
net.eval()

dense_model = copy.deepcopy(net)  # 备份 dense

# -------------------------------------------------------
#
# -------------------------------------------------------
example_inputs, _ = next(iter(train_loader))
example_inputs = example_inputs[:1].to(device)

# -------------------------------------------------------
#
# -------------------------------------------------------
original_channels = {}
for name, m in net.named_modules():
    if isinstance(m, nn.Conv2d):
        original_channels[name] = m.out_channels


# -------------------------------------------------------
# function
# -------------------------------------------------------
def get_importance(pruner_name, model, train_loader, device):
    if pruner_name == 'l1':
        return tp.importance.MagnitudeImportance(p=1)
    elif pruner_name == 'lamp':
        return tp.importance.LAMPImportance()
    elif pruner_name == 'taylor':
        importance = tp.importance.TaylorImportance()
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = nn.CrossEntropyLoss()(model(inputs), targets)
            loss.backward()
            break
        model.eval()
        return importance


def compute_channel_sparsity(model):
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total += original_channels[name]
            remaining += m.out_channels
    return 1 - remaining / total


def get_flops_params(model):
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, params


# -------------------------------------------------------
#  pruner
# -------------------------------------------------------
ignored_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]

importance = get_importance(args.pruner, net, train_loader, device)

pruner = tp.pruner.MagnitudePruner(
    net,
    example_inputs,
    importance=importance,
    iterative_steps=args.iterative_steps,  # GMP 核心：分多步剪
    ch_sparsity=args.target_sp,  # 最终目标稀疏率
    ignored_layers=ignored_layers,
)

macs_before, params_before = get_flops_params(net)
print(f"before prune | MACs: {macs_before / 1e6:.2f}M  Params: {params_before / 1e6:.2f}M")

# -------------------------------------------------------
# save path
# -------------------------------------------------------
target_folder = f'./{args.m_name}/ckpt_structured_iterative'
os.makedirs(target_folder, exist_ok=True)

trainer = trainer_loader()
opt_post = {
    "optimizer": partial(optim.AdamW, lr=0.0003),
    "steps": 40 * 313,
    "scheduler": None
}

# -------------------------------------------------------
# GMP 迭代主循环
# -------------------------------------------------------
for step in range(args.iterative_steps):
    print(f"\n{'=' * 50}")
    print(f"Step {step + 1}/{args.iterative_steps}")


    pruner.step()
    sparsity = compute_channel_sparsity(net)
    macs, params = get_flops_params(net)
    print(f"  channel sparsity : {sparsity:.2%}")
    print(f"  MACs  : {macs / 1e6:.2f}M  ({macs_before / macs:.2f}x)")
    print(f"  Params: {params / 1e6:.2f}M  ({params_before / params:.2f}x)")

    # ③ finetune
    result_log = trainer(net, opt_post, train_loader, test_loader,
                         patience=20)


    formatted_sp = f"{sparsity:.3f}"
    save_path = os.path.join(
        target_folder,
        f'step{step + 1:02d}_sp{formatted_sp}.pth'
    )
    torch.save(net, save_path)



torch.save(dense_model, os.path.join(target_folder, f'dense_{args.m_name}.pth'))

print(f"\n{'=' * 50}")
print(f"final channel sparsity: {compute_channel_sparsity(net):.2%}")
wandb.finish()
