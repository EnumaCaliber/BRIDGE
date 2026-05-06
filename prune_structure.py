'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import os
import argparse
from functools import partial

from utils.train import trainer_loader
from models.model_loader import model_loader
from data.data_loader import data_loader
from utils.tools import  *
from utils.pruners import *
import random
import numpy as np


parser = argparse.ArgumentParser(description='Prune')

parser.add_argument('--m_name', type=str, default="resnet20",
                    help='Model name (e.g., resnet18, vgg16, etc.)')
parser.add_argument('--pruner', type=str, default='lamp', help='pruning method')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--m_prune', type=str, default='iterate', help="oneshot and iterate")
parser.add_argument('--iter_start', type=int, default=1, help='start iteration for pruning')
parser.add_argument('--iter_end', type=int, default=15, help='end iteration for pruning')
parser.add_argument('--oneshot', type=int, default=0.95, help='end iteration for pruning')
parser.add_argument('--sparsity', type=float, default=0.3, help='end iteration for pruning')
parser.add_argument('--dataset', type= str, default= "CIFAR10", help='dataset CIFAR10, tiny_imagenet')
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
print(f"Random seed set to: {args.seed}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, val_loader, test_loader = data_loader(
    data_dir=args.data_dir,
    val_split=args.val_split,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    dataset= args.dataset,
)

net = model_loader(args.m_name, device)
cktp = f'pretrain_{args.m_name}_ckpt.pth'
checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}')
net.load_state_dict(checkpoint['net'])
net.eval()

true_mask = 1 - get_model_sparsity(net)
print(f'true mask {true_mask}')

pruner = weight_pruner_loader(args.pruner)
trainer = trainer_loader()
prune_weights_reparam(net)

""" PRUNE AND RETRAIN """

target_folder = f'./{args.m_name}/{args.m_prune}'
os.makedirs(target_folder, exist_ok=True)

if args.m_prune == 'iterate':
    sparsity = args.sparsity
    opt_post = {
        "optimizer": partial(optim.AdamW, lr=0.0003),
        "steps": 40*313,  # 40000 for iterative, 400000 for one-shot
        "scheduler": None
    }
else:
    sparsity = args.oneshot
    opt_post = {
        "optimizer": partial(optim.AdamW, lr=0.0003),
        "steps": 400*313,  # 40000 for iterative, 400000 for one-shot
        "scheduler": None
    }

for it in range(args.iter_start, args.iter_end + 1):
    print(f"Pruning for iteration {it}: METHOD: {args.pruner}")
    pruner(net, sparsity)
    result_log = trainer(net, opt_post, train_loader, test_loader,  patience=50)

    print(f"  Iteration {it} results: Test acc={result_log[0]:.2f}%, Train acc={result_log[2]:.2f}%")
    true_mask = 1 - get_model_sparsity(net)
    formatted_mask = round(true_mask, 4)
    target_path_mask = os.path.join(target_folder, f'{args.m_prune}_{formatted_mask}.pth')
    torch.save(net.state_dict(), target_path_mask)

