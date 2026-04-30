import torch
import torch.nn as nn
from copy import deepcopy
import os
import torch.nn.utils.prune as prune


# Preliminaries. Not to be exported.

def _is_prunable_module(m):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d))


def _get_sparsity(tsr):
    total = tsr.numel()
    nnz = tsr.nonzero().size(0)
    return nnz / total


def _get_nnz(tsr):
    return tsr.nonzero().size(0)


# Modules

def get_weights(model):
    weights = []
    for m in model.modules():
        if _is_prunable_module(m):
            weights.append(m.weight)
    return weights


def get_convweights(model):
    weights = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weights.append(m.weight)
    return weights


def get_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(m)
    return modules


def get_convmodules(model):
    modules = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            modules.append(m)
    return modules


def get_copied_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(deepcopy(m).cpu())
    return modules


def get_model_sparsity(model):
    prunables = 0
    nnzs = 0
    for m in model.modules():
        if _is_prunable_module(m):
            prunables += m.weight.data.numel()
            nnzs += m.weight.data.nonzero().size(0)
    return nnzs / prunables


def get_sparsities(model):
    return [_get_sparsity(m.weight.data) for m in model.modules() if _is_prunable_module(m)]


def get_nnzs(model):
    return [_get_nnz(m.weight.data) for m in model.modules() if _is_prunable_module(m)]


def load_model(model, ckpt_path):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
    target_path = os.path.join(ckpt_path, "ckpt.pth")
    checkpoint = torch.load(target_path)

    # Strip 'module.' prefix from keys
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in checkpoint['net'].items():
        name = k.replace('module.', '')  # remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return model, best_acc, start_epoch


def remove_pruning_reparam(model):
    for m in model.modules():
        if hasattr(m, 'weight_mask'):
            try:
                prune.remove(m, 'weight')
            except ValueError:
                continue  # skip if already removed or not pruned


def prune_weights_reparam(model):
    module_list = get_modules(model)  # your utility to get all prunable layers
    for m in module_list:
        prune.identity(m, name="weight")