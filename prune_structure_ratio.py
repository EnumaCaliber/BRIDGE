import torch.optim as optim
import argparse
import copy
from functools import partial
import torch_pruning as tp

from utils.train import trainer_loader
from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.tools import *
from utils.pruners import *

import random
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Structured Iterative Prune')
parser.add_argument('--m_name', type=str, default="resnet20",
                    choices=['resnet20', 'vgg16', 'efficientnet', 'shufflenet'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pruner', type=str, default='l1', choices=['l1', 'lamp', 'taylor'])
parser.add_argument('--ratio_per_step', type=float, default=0.3)
parser.add_argument('--iterative_steps', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='CIFAR10')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, val_loader, test_loader = data_loader(
    data_dir=args.data_dir,
    val_split=args.val_split,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    dataset=args.dataset,
)

net = model_loader(args.m_name, device)
cktp = f'pretrain_{args.m_name}_ckpt.pth'
checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}')
net.load_state_dict(checkpoint['net'])
net.eval()

dense_model = copy.deepcopy(net)

example_inputs, _ = next(iter(train_loader))
example_inputs = example_inputs[:1].to(device)


original_out_channels = {}
for name, m in net.named_modules():
    if isinstance(m, nn.Conv2d):
        original_out_channels[name] = m.out_channels


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
        if isinstance(m, nn.Conv2d) and name in original_out_channels:
            total     += original_out_channels[name]
            remaining += m.out_channels
    return 1 - remaining / total


def get_flops_params(model):
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, params


def prune_one_step(model, ratio, example_inputs, pruner_name,
                   train_loader, device, out_index_map):

    ignored = [m for m in model.modules() if isinstance(m, nn.Linear)]
    importance = get_importance(pruner_name, model, train_loader, device)
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=ratio,
        ignored_layers=ignored,
    )

    module_to_name = {m: n for n, m in model.named_modules()}

    for group in pruner.step(interactive=True):
        for dep, idxs in group:
            module  = dep.target.module
            handler = dep.handler


            if not isinstance(module, nn.Conv2d):
                continue
            if handler != tp.prune_conv_out_channels:
                continue

            name = module_to_name.get(module)
            if name is None or name not in out_index_map:
                continue

            pruned_local = set(idxs)
            out_index_map[name] = [
                out_index_map[name][i]
                for i in range(len(out_index_map[name]))
                if i not in pruned_local
            ]


        group.prune()


    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in out_index_map:
            pruned_out = sorted(
                set(range(original_out_channels[name])) - set(out_index_map[name])
            )
            if pruned_out:
                print(f"  [{name}] survived: {out_index_map[name]}")
                print(f"  [{name}] pruned  : {pruned_out}")



macs_before, params_before = get_flops_params(net)
print(f"before prune | MACs: {macs_before / 1e6:.2f}M  Params: {params_before / 1e6:.2f}M")

target_folder = f'./{args.m_name}/ckpt_structured_iterative'
os.makedirs(target_folder, exist_ok=True)

trainer = trainer_loader()



if args.dataset == "CIFAR10":
    opt_post = {
        "optimizer": partial(optim.AdamW, lr=0.0003),
        "steps": 40 * 313,
        "scheduler": None
    }
elif args.dataset == "tiny_imagenet":
    opt_post = {
        "optimizer": partial(optim.AdamW, lr=0.0003),
        "steps": 40 * 702,
        "scheduler": None
    }



# main 
out_index_map = {}
for name, m in net.named_modules():
    if isinstance(m, nn.Conv2d):
        out_index_map[name] = list(range(m.out_channels))

for step in range(args.iterative_steps):
    print(f"\n{'=' * 50}")
    print(f"Step {step + 1}/{args.iterative_steps}")

    prune_one_step(
        net, args.ratio_per_step, example_inputs,
        args.pruner, train_loader, device,
        out_index_map,
    )

    sparsity = compute_channel_sparsity(net)
    macs, params = get_flops_params(net)
    print(f"  ratio_per_step   : {args.ratio_per_step:.2%}")
    print(f"  channel sparsity : {sparsity:.2%}")
    print(f"  MACs  : {macs / 1e6:.2f}M  ({macs_before / macs:.2f}x)")
    print(f"  Params: {params / 1e6:.2f}M  ({params_before / params:.2f}x)")

    result_log = trainer(net, opt_post, train_loader, test_loader, patience=40)
    print(f"  Test acc={result_log[0]:.2f}%  Train acc={result_log[2]:.2f}%")

    pruned_out_dense_map = {
        name: sorted(set(range(original_out_channels[name])) - set(out_index_map[name]))
        for name in out_index_map
    }

    formatted_sp = f"{sparsity:.3f}"
    save_path = os.path.join(target_folder, f'step{step + 1:02d}_sp{formatted_sp}.pth')
    torch.save({
        'model':                net,
        'out_index_map':        out_index_map,        # live out_channel -> dense index
        'pruned_out_dense_map': pruned_out_dense_map, # prune out_channel dense index
    }, save_path)

torch.save(dense_model, os.path.join(target_folder, f'dense_{args.m_name}.pth'))
print(f"\n{'=' * 50}")
print(f"final channel sparsity: {compute_channel_sparsity(net):.2%}")