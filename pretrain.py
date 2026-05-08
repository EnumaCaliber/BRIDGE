import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
import numpy as np


from models.model_loader import model_loader
from data.data_loader import data_loader

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--m_name',type=str, default= "vgg16",
                    help='Model name (e.g., resnet18, vgg16, etc.)')
parser.add_argument('--pruner', type=str, help='pruning method')
# parser.add_argument('--iter_start', type=int, default=1, help='start iteration for pruning')
# parser.add_argument('--iter_end', type=int, default=1, help='end iteration for pruning')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--max_epochs', type=int, default=1000, help='maximum pretraining epochs')
parser.add_argument('--patience', type=int, default=30, help='early stopping patience (epochs without improvement)')
parser.add_argument('--data_dir',   type=str, default='./data')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--val_split',  type=float, default=0.1)
parser.add_argument('--num_workers',type=int, default=15)
parser.add_argument('--dataset', type= str, default= "CIFAR10", help='dataset CIFAR10, tiny_imagenet')



args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  
epochs_without_improvement = 0  


train_loader, val_loader, test_loader = data_loader(
    data_dir=args.data_dir,
    val_split=args.val_split,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    dataset= args.dataset,
)

net = model_loader(args.m_name, device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def get_ckpt_path() -> str:
    """Return the path where the best checkpoint is saved."""
    folder = os.path.join(args.m_name, 'checkpoint')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"pretrain_{args.m_name}_ckpt.pth")

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%', end=' | ')


def evaluate(epoch):
    global best_acc
    global epochs_without_improvement
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):  # val_loader
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_val_loss = test_loss / len(val_loader)
    acc = 100.*correct/total
    print(f'Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%', end='')
    


    if acc > best_acc:
        print(f' NEW BEST (prev: {best_acc:.2f}%)')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        target_folder = os.path.join(args.m_name, 'checkpoint')
        os.makedirs(target_folder, exist_ok=True)
        ckpt_filename = f"pretrain_{args.m_name}_ckpt.pth"
        target_path = os.path.join(target_folder, ckpt_filename)
        torch.save(state, target_path)
        best_acc = acc
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement <= 5:
            print(f' (no improvement: {epochs_without_improvement})')
        elif epochs_without_improvement % 10 == 0:
            print(f' (no improvement: {epochs_without_improvement} epochs)')
        else:
            print()
    
    return epochs_without_improvement >= args.patience

def test() -> float:
    """Load the best saved checkpoint, then evaluate on the test set.
 
    Returns:
        Test accuracy of the best model (%).
    """
    ckpt_path = get_ckpt_path()
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Best checkpoint not found at '{ckpt_path}'. "
            "Make sure training has run at least one epoch."
        )
 
    # ↓ Load best weights (not the last-epoch weights still in `net`)
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    print(f'\n==> Loaded best checkpoint '
          f'(epoch {checkpoint["epoch"]}, val acc {checkpoint["acc"]:.2f}%)')
 
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
 
    return 100. * correct / total


if args.resume != True:
    print(f"\nStarting pretraining for up to {args.max_epochs} epochs (patience: {args.patience})")
    print(f"Early stopping if no improvement for {args.patience} consecutive epochs\n")

    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        should_stop = evaluate(epoch)
        scheduler.step()

        if should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"No improvement for {args.patience} epochs")
            print(f"Best validation accuracy: {best_acc:.2f}%\n")
            break
    else:
        print(f"\nReached maximum epochs ({args.max_epochs})")
        print(f"Best validation accuracy: {best_acc:.2f}%\n")


acc = test()
print(f'best acc for {args.m_name}', acc)