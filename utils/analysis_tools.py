import os
import math
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def load_model(net, ckpt_path):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
    target_path = os.path.join(ckpt_path, "pretrain_densenet_ckpt.pth")
    checkpoint = torch.load(target_path)

    # # Strip 'module.' prefix from keys
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()

    # for k, v in checkpoint['net'].items():
    #     name = k.replace('module.', '')  # remove 'module.' prefix
    #     new_state_dict[name] = v

    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net, best_acc, start_epoch


def load_model_name(net, ckpt_path, name):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
    target_path = os.path.join(ckpt_path, f"pretrain_{name}_ckpt.pth")
    checkpoint = torch.load(target_path)

    # # Strip 'module.' prefix from keys
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()

    # for k, v in checkpoint['net'].items():
    #     name = k.replace('module.', '')  # remove 'module.' prefix
    #     new_state_dict[name] = v

    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net, best_acc, start_epoch


def _is_prunable_module(m):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d))


def get_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(m)
    return modules


def get_layer_by_name(model, name):
    modules = dict(model.named_modules())
    return modules[name]


def prune_weights_reparam(model):
    module_list = get_modules(model)  # your utility to get all prunable layers
    for m in module_list:
        prune.identity(m, name="weight")


def get_pruned_weights(model):
    weights = []
    for m in model.modules():
        if hasattr(m, 'weight_orig') and hasattr(m, 'weight_mask'):
            w_eff = m.weight_orig.data * m.weight_mask.data  # Elementwise multiply
            weights.append(w_eff.view(-1))  # flatten
    if len(weights) == 0:
        raise ValueError("No pruned weights found (missing weight_orig and weight_mask?)")
    all_weights = torch.cat(weights)
    return all_weights


def pruned_weight_distribution(model, model_name, sparsity, save_path="./weight_distributions",
                               title="Weight Distributions"):
    # name_sparsity = f'{model_name}_{sparsity}'
    target_folder = os.path.join(save_path, model_name)
    os.makedirs(target_folder, exist_ok=True)

    all_weights = get_pruned_weights(model)
    nonzero_weights = all_weights[all_weights != 0].cpu().numpy()

    plt.hist(nonzero_weights, bins=200)
    # plt.xlim(-0.2, 0.2)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Pruned {model_name}, Sparsity 0.{sparsity}')

    plt.grid(True)

    filename = f"{model_name}_{sparsity}_hist.png"
    plt.tight_layout()
    target_path = os.path.join(target_folder, filename)
    plt.savefig(target_path, dpi=300)
    plt.close()

    print(f"Saved weight distribution plots to: {save_path}")


def plot_weight_distributions(model, model_name, save_path="./weight_distributions", title="Weight Distributions"):
    target_folder = os.path.join(save_path, model_name)
    os.makedirs(target_folder, exist_ok=True)

    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.extend(param.data.cpu().numpy().flatten())

    plt.hist(all_weights, bins=200)
    plt.xlim(-0.2, 0.2)
    plt.title("All Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    title = f"{model_name} Weight Distribution"
    plt.title(title)

    filename = f"{model_name}_hist.png"
    plt.tight_layout()
    target_path = os.path.join(target_folder, filename)
    plt.savefig(target_path, dpi=300)
    plt.close()

    print(f"Saved weight distribution plots to: {save_path}")


def analyze_features(features_dict, title="Features"):
    """Analyze and print statistics about extracted features"""
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")

    for layer_name, features in features_dict.items():
        shape = features.shape
        mean_val = features.mean().item()
        std_val = features.std().item()
        min_val = features.min().item()
        max_val = features.max().item()

        print(f"Layer: {layer_name}")
        print(f"  Shape: {shape}")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std:  {std_val:.6f}")
        print(f"  Min:  {min_val:.6f}")
        print(f"  Max:  {max_val:.6f}")
        print(f"  Non-zero ratio: {(features != 0).float().mean().item():.4f}")
        print("-" * 40)


def analyze_layer_weights(model, title="Layer Weight Statistics"):
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    for name, module in model.named_modules():
        # Skip container modules (Sequential, VGG, etc.)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                # For pruned layers (torch.nn.utils.prune)
                weight_orig = module.weight_orig
                weight_mask = module.weight_mask
                pruned_weight = weight_orig * weight_mask
                nonzero_weights = pruned_weight[pruned_weight != 0]

                if nonzero_weights.numel() > 0:
                    mean_val = nonzero_weights.mean().item()
                    std_val = nonzero_weights.std().item()
                    min_val = nonzero_weights.min().item()
                    max_val = nonzero_weights.max().item()
                    non_zero_ratio = nonzero_weights.numel() / pruned_weight.numel()
                else:
                    mean_val = std_val = min_val = max_val = 0.0
                    non_zero_ratio = 0.0

                print(f"Layer: {name}")
                print(f"  Shape: {pruned_weight.shape}")
                print(f"  Remaining Weights: {nonzero_weights.numel()}")
                print(f"  Mean: {mean_val:.6f}")
                print(f"  Std:  {std_val:.6f}")
                print(f"  Min:  {min_val:.6f}")
                print(f"  Max:  {max_val:.6f}")
                print(f"  Non-zero ratio: {non_zero_ratio:.4f}")
                print("-" * 50)

            elif hasattr(module, 'weight'):
                # For unpruned layers
                weight = module.weight
                nonzero_weights = weight[weight != 0]

                mean_val = nonzero_weights.mean().item()
                std_val = nonzero_weights.std().item()
                min_val = nonzero_weights.min().item()
                max_val = nonzero_weights.max().item()
                non_zero_ratio = nonzero_weights.numel() / weight.numel()

                print(f"Layer: {name}")
                print(f"  Shape: {weight.shape}")
                print(f"  Weights: {nonzero_weights.numel()}")
                print(f"  Mean: {mean_val:.6f}")
                print(f"  Std:  {std_val:.6f}")
                print(f"  Min:  {min_val:.6f}")
                print(f"  Max:  {max_val:.6f}")
                print(f"  Non-zero ratio: {non_zero_ratio:.4f}")
                print("-" * 50)


def analyze_layer_weights_blockwise(model, blocks, title="Blockwise Layer Weight Statistics"):
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    block_weights = {block_name: [] for block_name in blocks}
    block_total_weights = {block_name: 0 for block_name in blocks}
    block_nonzero_counts = {block_name: 0 for block_name in blocks}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Determine if the layer is pruned
            if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                weight = module.weight_orig * module.weight_mask
            elif hasattr(module, 'weight'):
                weight = module.weight
            else:
                continue

            total_weights = weight.numel()
            nonzero_weights = weight[weight != 0]
            nonzero_count = nonzero_weights.numel()

            if nonzero_count == 0:
                continue

            for block_name, layers in blocks.items():
                if name in layers:
                    block_weights[block_name].append(nonzero_weights.flatten())
                    block_total_weights[block_name] += total_weights
                    block_nonzero_counts[block_name] += nonzero_count
                    break

    for block_name, weights_list in block_weights.items():
        if not weights_list:
            print(f"Block: {block_name} - No weights found")
            continue

        block_tensor = torch.cat(weights_list)
        mean_val = block_tensor.mean().item()
        std_val = block_tensor.std().item()
        min_val = block_tensor.min().item()
        max_val = block_tensor.max().item()
        non_zero_ratio = block_nonzero_counts[block_name] / block_total_weights[block_name]

        print(f"Block: {block_name}")
        print(f"  Total Weights: {block_total_weights[block_name]}")
        print(f"  Remaining Weights: {block_nonzero_counts[block_name]}")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std:  {std_val:.6f}")
        print(f"  Min:  {min_val:.6f}")
        print(f"  Max:  {max_val:.6f}")
        print(f"  Non-zero ratio: {non_zero_ratio:.4f}")
        print("-" * 50)

    return block_nonzero_counts


def count_pruned_params(model):
    total_params = 0
    surviving_params = 0

    for m in model.modules():
        if hasattr(m, 'weight_orig') and hasattr(m, 'weight_mask'):
            w_orig = m.weight_orig.data
            mask = m.weight_mask.data

            total_params += w_orig.numel()
            surviving_params += mask.sum().item()

    pruned_params = total_params - surviving_params
    print(f"Total params: {total_params}")
    print(f"Surviving (unpruned) params: {int(surviving_params)}")
    print(f"Pruned params: {int(pruned_params)}")
    print(f"Sparsity: {pruned_params / total_params:.4f}")

    return total_params, surviving_params, pruned_params


def sample_one_per_class(data_loader, num_classes=10):
    """Sample one image from each class.

    Args:
        data_loader: DataLoader yielding (inputs, labels) pairs
        num_classes: Number of classes in the dataset

    Returns:
        Tuple of (images_tensor, labels_tensor) with one sample per class
    """
    class_samples = {}
    class_labels = {}

    # Iterate through dataset to find one sample per class
    for inputs, labels in data_loader:
        for i in range(len(labels)):
            label = labels[i].item()
            if label not in class_samples and len(class_samples) < num_classes:
                class_samples[label] = inputs[i]
                class_labels[label] = label

        # Stop once we have one sample from each class
        if len(class_samples) == num_classes:
            break

    # Sort by class label and stack into tensors
    sorted_labels = sorted(class_samples.keys())
    images = torch.stack([class_samples[label] for label in sorted_labels])
    labels = torch.tensor([class_labels[label] for label in sorted_labels])

    return images, labels


def sample_recovery_cases(data_loader, model_baseline, model_improved, device='cuda',
                          max_samples_per_class=5, num_classes=10):
    """Sample images that are misclassified by baseline but correctly classified by improved model.

    This function identifies "recovery cases" where the improved model fixes mistakes made by
    the baseline model. Useful for analyzing what the improved model learned.

    Args:
        data_loader: DataLoader yielding (inputs, labels) pairs
        model_baseline: Baseline model (e.g., pruned model) to compare against
        model_improved: Improved model (e.g., regrown model) that should perform better
        device: Device to run inference on ('cuda' or 'cpu')
        max_samples_per_class: Maximum number of recovery cases to collect per class
        num_classes: Number of classes in the dataset

    Returns:
        Tuple of (images, labels, baseline_preds, improved_preds):
        - images: Tensor of sampled images (N, C, H, W)
        - labels: Tensor of ground truth labels (N,)
        - baseline_preds: Tensor of baseline model predictions (N,)
        - improved_preds: Tensor of improved model predictions (N,)
    """
    model_baseline.eval()
    model_improved.eval()

    # Storage for recovery cases per class
    recovery_cases = {i: [] for i in range(num_classes)}
    recovery_labels = {i: [] for i in range(num_classes)}
    recovery_baseline_preds = {i: [] for i in range(num_classes)}
    recovery_improved_preds = {i: [] for i in range(num_classes)}

    # Check if we have enough samples for all classes
    samples_found = {i: 0 for i in range(num_classes)}
    all_classes_satisfied = False

    with torch.no_grad():
        for inputs, labels in data_loader:
            if all_classes_satisfied:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get predictions from both models
            outputs_baseline = model_baseline(inputs)
            outputs_improved = model_improved(inputs)

            preds_baseline = outputs_baseline.argmax(dim=1)
            preds_improved = outputs_improved.argmax(dim=1)

            # Find recovery cases: baseline wrong AND improved correct
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_baseline = preds_baseline[i].item()
                pred_improved = preds_improved[i].item()

                # Check if this is a recovery case
                is_recovery = (pred_baseline != true_label) and (pred_improved == true_label)

                if is_recovery and samples_found[true_label] < max_samples_per_class:
                    recovery_cases[true_label].append(inputs[i].cpu())
                    recovery_labels[true_label].append(true_label)
                    recovery_baseline_preds[true_label].append(pred_baseline)
                    recovery_improved_preds[true_label].append(pred_improved)
                    samples_found[true_label] += 1

            # Check if we have enough samples for all classes
            all_classes_satisfied = all(count >= max_samples_per_class for count in samples_found.values())

    # Flatten and stack all recovery cases
    all_images = []
    all_labels = []
    all_baseline_preds = []
    all_improved_preds = []

    for class_id in sorted(recovery_cases.keys()):
        if recovery_cases[class_id]:  # Only add if we found samples for this class
            all_images.extend(recovery_cases[class_id])
            all_labels.extend(recovery_labels[class_id])
            all_baseline_preds.extend(recovery_baseline_preds[class_id])
            all_improved_preds.extend(recovery_improved_preds[class_id])

    if not all_images:
        print("Warning: No recovery cases found! The improved model may not be fixing any mistakes.")
        return None, None, None, None

    # Convert to tensors
    images = torch.stack(all_images)
    labels = torch.tensor(all_labels)
    baseline_preds = torch.tensor(all_baseline_preds)
    improved_preds = torch.tensor(all_improved_preds)

    print(f"\nRecovery Cases Summary:")
    print(f"Total recovery cases found: {len(images)}")
    for class_id in range(num_classes):
        count = sum(1 for l in all_labels if l == class_id)
        if count > 0:
            print(f"  Class {class_id}: {count} samples")

    return images, labels, baseline_preds, improved_preds


class FeatureExtractor:
    """Helper class to extract features from specific layers"""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        """Register forward hooks to capture intermediate features"""

        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output.detach().cpu()

            return hook

        # Clear previous hooks
        self.clear_hooks()

        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_features(self, data_loader, num_batches=1):
        """Extract features from registered layers"""
        self.model.eval()
        all_features = {}

        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= num_batches:
                    break

                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                _ = self.model(inputs)  # Forward pass triggers hooks

                # Collect features from this batch
                for layer_name, features in self.features.items():
                    if layer_name not in all_features:
                        all_features[layer_name] = []
                    all_features[layer_name].append(features)

        # Concatenate features from all batches
        for layer_name in all_features:
            all_features[layer_name] = torch.cat(all_features[layer_name], dim=0)

        return all_features

    def extract_features_per_class(self, images, labels, num_classes=10):
        """Extract features for one sample from each class.

        Args:
            data_loader: DataLoader yielding (inputs, labels) pairs
            num_classes: Number of classes in the dataset

        Returns:
            Dict mapping layer_name -> dict mapping class_id -> features tensor
        """
        self.model.eval()

        # Sample one image per class
        # images, labels = self.sample_one_per_class(data_loader, num_classes)
        images = images.cuda() if torch.cuda.is_available() else images

        # Extract features
        with torch.no_grad():
            _ = self.model(images)  # Forward pass triggers hooks

        # Organize features by class
        features_per_class = {}
        for layer_name, features in self.features.items():
            features_per_class[layer_name] = {}
            for i, label in enumerate(labels):
                features_per_class[layer_name][label.item()] = features[i]

        return features_per_class

    def extract_attention_features(self, data_loader, num_batches=1):
        """Extract spatial attention maps (where network focuses)

        Returns attention maps with shape (H, W) per layer, averaged across all batches.
        Values are normalized to [0, 1] range as a probability distribution.
        """
        self.model.eval()
        all_features = {}

        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= num_batches:
                    break

                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                _ = self.model(inputs)

                for layer_name, features in self.features.items():
                    # features: (B, C, H, W) for conv layers
                    if features.dim() != 4:
                        # Skip non-convolutional layers (e.g., linear layers)
                        continue

                    # Spatial attention: sum across channels to get (B, H, W)
                    attention = features.sum(dim=1)  # (B, H, W)

                    # Normalize each sample to [0, 1] probability distribution
                    B, H, W = attention.shape
                    attention_flat = attention.view(B, -1)  # (B, H*W)
                    attention_flat = F.normalize(attention_flat, p=1, dim=1)  # L1 normalize to sum=1
                    attention = attention_flat.view(B, H, W)  # Back to (B, H, W)

                    if layer_name not in all_features:
                        all_features[layer_name] = []
                    all_features[layer_name].append(attention)

        # Average attention maps across all batches → (H, W)
        for layer_name in all_features:
            # Concatenate all batches: [(B1, H, W), (B2, H, W), ...] → (B_total, H, W)
            all_batches = torch.cat(all_features[layer_name], dim=0)
            # Average across batch dimension → (H, W)
            all_features[layer_name] = all_batches.mean(dim=0)

        return all_features

    def extract_attention_features_per_class(self, images, labels, num_classes=10):
        """Extract spatial attention maps for one sample from each class.

        Args:
            data_loader: DataLoader yielding (inputs, labels) pairs
            num_classes: Number of classes in the dataset

        Returns:
            Dict mapping layer_name -> dict mapping class_id -> attention map (H, W)
        """
        self.model.eval()

        # Sample one image per class
        # images, labels = self.sample_one_per_class(data_loader, num_classes)
        images = images.cuda() if torch.cuda.is_available() else images

        # Extract features
        with torch.no_grad():
            _ = self.model(images)  # Forward pass triggers hooks

        # Organize attention maps by class
        attention_per_class = {}
        for layer_name, features in self.features.items():
            # features: (num_classes, C, H, W) for conv layers
            if features.dim() != 4:
                # Skip non-convolutional layers
                continue

            attention_per_class[layer_name] = {}

            # Compute attention for each class sample
            for i, label in enumerate(labels):
                # features[i]: (C, H, W)
                attention = features[i].sum(dim=0)  # (H, W)

                # Normalize to [0, 1] probability distribution
                attention_flat = attention.view(-1)  # (H*W,)
                attention_flat = F.normalize(attention_flat, p=1, dim=0)  # L1 normalize
                attention = attention_flat.view(attention.shape)  # Back to (H, W)

                attention_per_class[layer_name][label.item()] = attention

        return attention_per_class


class BlockwiseFeatureExtractor(FeatureExtractor):
    """Extract and aggregate features per logical block.
    block_dict: dict mapping block_name -> list of layer names (as in model.named_modules())

    Returns features in format [num_layers, channels] for each block, where:
    - num_layers is the number of layers in that block
    - channels is the number of channels/features for each layer
    """

    def __init__(self, model, block_dict):
        super().__init__(model)
        self.block_dict = block_dict

    def register_block_hooks(self):
        """Register hooks for all layers mentioned in block_dict."""
        layer_names = set()
        for layers in self.block_dict.values():
            layer_names.update(layers)
        # reuse base helper
        self.register_hooks(layer_names)

    def extract_block_features(self, data_loader, num_batches=1):
        """Run model and aggregate features per block.
        Args:
            data_loader: DataLoader yielding (inputs, _) pairs
            num_batches: Number of batches to process

        Returns: dict block_name -> dict[layer_name -> Tensor]
        For each block, returns a dictionary mapping layer names to their feature tensors
        If reshape_channels=True: Features are reshaped to (C, H', W') for SSIM computation
        If reshape_channels=False: Features remain as 1D tensors of shape (C,)
        """
        self.clear_hooks()
        self.register_block_hooks()
        # layer_features = self.extract_features(data_loader, num_batches=num_batches)
        layer_features = self.extract_attention_features(data_loader, num_batches=num_batches)

        # Aggregate per block
        block_features = {}
        for block_name, layers in self.block_dict.items():
            layer_dict = {}

            for lname in layers:
                if lname not in layer_features:
                    continue

                t = layer_features[lname]  # For attention features: already (H, W) - no batch dim!

                # extract_attention_features already returns mean over batch dimension
                # So t is already (H, W) for attention maps, not (B, H, W)
                # extract_attention_features already returns mean over batch dimension
                # So t is already (H, W) for attention maps, not (B, H, W)

                if t.dim() == 2:  # For attention maps: (H, W)
                    # Attention maps are already normalized to [0, 1] during extraction
                    # But we re-normalize here to ensure consistency for SSIM
                    t_min, t_max = t.min(), t.max()
                    if t_max > t_min:  # Avoid division by zero
                        t = (t - t_min) / (t_max - t_min + 1e-8)
                    # If t_max == t_min, leave as is (uniform attention)

                    # Attention map is already 2D (H, W), directly usable for SSIM
                    # No need to reshape - already in spatial format

                elif t.dim() == 3:  # Should not happen with attention features, but handle anyway
                    # This would be for raw activation features (C, H, W)
                    C, H, W = t.shape
                    grid_size = int(math.ceil(math.sqrt(C)))

                    # Pad if C is not a perfect square
                    if grid_size ** 2 > C:
                        padding = torch.zeros((grid_size ** 2 - C, H, W), device=t.device)
                        t = torch.cat([t, padding], dim=0)

                    # Reshape to grid
                    t = t.view(grid_size, grid_size, H, W)  # (G, G, H, W)
                    t = t.permute(0, 2, 1, 3)  # (G, H, G, W)
                    t = t.reshape(grid_size * H, grid_size * W)  # (H*G, W*G)

                elif t.dim() == 1:  # For linear layers - should not happen with attention
                    # Skip 1D features for attention-based SSIM
                    continue

                layer_dict[lname] = t

            if layer_dict:
                block_features[block_name] = layer_dict

        return block_features

    def extract_block_features_per_class(self, images, labels, num_classes=10, return_images=False):
        """Extract features per block for one sample from each class.

        Args:
            data_loader: DataLoader yielding (inputs, labels) pairs
            num_classes: Number of classes in the dataset
            return_images: If True, also return the sampled input images

        Returns:
            - block_features: dict block_name -> dict[layer_name -> dict[class_id -> Tensor]]
              For each block and layer, returns a dictionary mapping class IDs to feature tensors.
              Features are attention maps in (H, W) format for convolutional layers.
            - input_images (optional): dict mapping class_id -> input image tensor (C, H, W)
              Only returned if return_images=True
        """
        self.clear_hooks()
        self.register_block_hooks()

        # Sample input images
        # images, labels = self.sample_one_per_class(data_loader, num_classes)

        # Extract attention features per class
        layer_features = self.extract_attention_features_per_class(images, labels, num_classes=num_classes)

        # Aggregate per block
        block_features = {}
        for block_name, layers in self.block_dict.items():
            layer_dict = {}

            for lname in layers:
                if lname not in layer_features:
                    continue

                # layer_features[lname] is a dict: class_id -> attention map (H, W)
                class_dict = {}
                for class_id, attention in layer_features[lname].items():
                    t = attention

                    if t.dim() == 2:  # For attention maps: (H, W)
                        # Normalize to [0, 1] range
                        t_min, t_max = t.min(), t.max()
                        if t_max > t_min:
                            t = (t - t_min) / (t_max - t_min + 1e-8)

                        class_dict[class_id] = t
                    else:
                        # Skip non-2D features
                        continue

                if class_dict:
                    layer_dict[lname] = class_dict

            if layer_dict:
                block_features[block_name] = layer_dict

        if return_images:
            # Create dict mapping class_id to image tensor
            input_images = {labels[i].item(): images[i] for i in range(len(labels))}
            return block_features, input_images
        else:
            return block_features


def compute_block_ssim(features1, features2):
    """Compute SSIM between corresponding layers in two feature sets.
    Args:
        features1, features2: Output from extract_block_features()

    Returns: dict block_name -> dict of layer similarities
    """
    block_ssim = {}
    for block_name in features1:
        if block_name not in features2:
            continue

        layer_ssim = {}
        layers1, layers2 = features1[block_name], features2[block_name]

        for lname in layers1:
            if lname not in layers2:
                continue

            feat1 = layers1[lname]
            feat2 = layers2[lname]

            if feat1.shape != feat2.shape:
                print(f"Warning: Shape mismatch in {block_name}/{lname}")
                continue

            if feat1.dim() == 2:  # (H, W) format
                feat1_np = feat1.cpu().numpy()
                feat2_np = feat2.cpu().numpy()
                H, W = feat1_np.shape
                win_size = min(7, H, W)
                if win_size % 2 == 0:
                    win_size -= 1
                if win_size < 3:
                    # score = 0.5
                    # Cosine similarity alternative:
                    v1 = feat1_np.flatten()
                    v2 = feat2_np.flatten()
                    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                    score = float(np.dot(v1, v2) / denom) if denom > 1e-8 else 0.0
                else:
                    score = ssim(feat1_np, feat2_np,
                                 data_range=1.0,
                                 win_size=win_size)
            else:
                # score = 0.5
                # Cosine similarity alternative:
                v1 = feat1.cpu().numpy().flatten()
                v2 = feat2.cpu().numpy().flatten()
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                score = float(np.dot(v1, v2) / denom) if denom > 1e-8 else 0.0

            layer_ssim[lname] = score
        #add
        if layer_ssim:
            block_ssim[block_name] = layer_ssim

    return block_ssim


def compute_block_ssim_per_class(features1, features2):
    """Compute SSIM between corresponding layers for each class separately.

    Args:
        features1, features2: Output from extract_block_features_per_class()
                             Structure: block_name -> layer_name -> class_id -> Tensor(H, W)

    Returns: dict block_name -> dict layer_name -> dict class_id -> SSIM score
             Also returns average SSIM per layer: block_name -> layer_name -> avg_ssim
    """
    block_ssim_per_class = {}
    block_ssim_avg = {}

    for block_name in features1:
        if block_name not in features2:
            continue

        layer_ssim_per_class = {}
        layer_ssim_avg = {}
        layers1, layers2 = features1[block_name], features2[block_name]

        for lname in layers1:
            if lname not in layers2:
                continue

            # layers1[lname] and layers2[lname] are dicts: class_id -> Tensor(H, W)
            class_dict1 = layers1[lname]
            class_dict2 = layers2[lname]

            class_ssim = {}

            # Compute SSIM for each class
            for class_id in class_dict1:
                if class_id not in class_dict2:
                    continue

                feat1 = class_dict1[class_id]
                feat2 = class_dict2[class_id]

                if feat1.shape != feat2.shape:
                    print(f"Warning: Shape mismatch in {block_name}/{lname} class {class_id}")
                    continue

                if feat1.dim() == 2:  # (H, W) format
                    feat1_np = feat1.cpu().numpy()
                    feat2_np = feat2.cpu().numpy()
                    H, W = feat1_np.shape
                    win_size = min(7, H, W)
                    if win_size % 2 == 0:  # SSIM requires odd window size
                        win_size -= 1
                    if win_size < 3:  # If features too small for SSIM
                        continue
                    else:
                        score = ssim(feat1_np, feat2_np,
                                     data_range=1.0,
                                     win_size=win_size)
                        class_ssim[class_id] = score
                else:
                    print(f'Error: non-2D feature for layer {lname} class {class_id}')
                    continue

            if class_ssim:
                layer_ssim_per_class[lname] = class_ssim
                # Compute average SSIM across classes for this layer
                avg_ssim = np.mean(list(class_ssim.values()))
                layer_ssim_avg[lname] = avg_ssim

        if layer_ssim_per_class:
            block_ssim_per_class[block_name] = layer_ssim_per_class
        if layer_ssim_avg:
            block_ssim_avg[block_name] = layer_ssim_avg

    return block_ssim_per_class, block_ssim_avg


def compare_layer_features(features1, features2, block_name, layer_indices=None,
                           labels=('Original', 'Pruned'), plot_size=3, title=None,
                           save_path=None, show_diff=True):
    """Compare features from two models/states side by side.

    Works with any 2D feature representation:
    - Attention maps: (H, W) showing where network focuses
    - Reshaped activations: (H', W') grid of channels
    - Any other 2D feature representation

    Args:
        features1, features2: Outputs from extract_block_features() for two models
        block_name: Name of the block to visualize
        layer_indices: List of indices to visualize. If None, show all layers
        labels: Tuple of strings to label the two feature sets
        plot_size: Size of each feature plot in inches
        title: Custom title for the plot (e.g., 'Attention Map Comparison')
        save_path: If provided, save the plot to this path
        show_diff: If True, also show the difference map

    Note: Features must be 2D (H, W). If using attention features, they're
          already in the correct format. Bright regions = high attention/activation.
    """
    if block_name not in features1 or block_name not in features2:
        print(f"Block {block_name} not found in one or both feature sets")
        return

    # Get layer features for the block
    layers1 = list(features1[block_name].keys())
    layers2 = list(features2[block_name].keys())
    common_layers = [l for l in layers1 if l in layers2]

    if layer_indices is not None:
        # Filter layers by indices
        common_layers = [common_layers[i] for i in layer_indices if i < len(common_layers)]

    num_layers = len(common_layers)
    if num_layers == 0:
        print("No common layers to compare")
        return

    # Create subplot grid: each row shows original, pruned, and difference
    cols = 3 if show_diff else 2
    fig = plt.figure(figsize=(plot_size * cols, plot_size * num_layers))
    if title:
        plt.suptitle(title, fontsize=14)

    for i, layer_name in enumerate(common_layers):
        feat1 = features1[block_name][layer_name]
        feat2 = features2[block_name][layer_name]

        if feat1.shape != feat2.shape:
            print(f"Warning: Shape mismatch in layer {layer_name}")
            continue

        if feat1.dim() != 2:
            print(f"Warning: Unexpected feature dimensions in layer {layer_name}")
            continue

        # Convert to numpy for plotting
        feat1_np = feat1.cpu().numpy()
        feat2_np = feat2.cpu().numpy()

        # Plot original features
        ax1 = plt.subplot(num_layers, cols, i * cols + 1)
        im1 = ax1.imshow(feat1_np, cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        if i == 0:  # Only show title for first row
            ax1.set_title(f'{labels[0]}')
        ax1.set_ylabel(f'Layer: {layer_name}')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot pruned features
        ax2 = plt.subplot(num_layers, cols, i * cols + 2)
        im2 = ax2.imshow(feat2_np, cmap='viridis')
        plt.colorbar(im2, ax=ax2)
        if i == 0:  # Only show title for first row
            ax2.set_title(f'{labels[1]}')
        ax2.set_xticks([])
        ax2.set_yticks([])

        if show_diff:
            # Plot difference map using diverging colormap
            diff = feat1_np - feat2_np
            vmax = max(abs(diff.min()), abs(diff.max()))
            ax3 = plt.subplot(num_layers, cols, i * cols + 3)
            im3 = ax3.imshow(diff, cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
            plt.colorbar(im3, ax=ax3)
            if i == 0:  # Only show title for first row
                ax3.set_title('Difference')
            ax3.set_xticks([])
            ax3.set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    plt.show()


def compare_layer_features_per_class(features1, features2, block_name, layer_name,
                                     class_ids=None, labels=('Original', 'Pruned'),
                                     plot_size=3, title=None, save_path=None, show_diff=True,
                                     class_names=None, input_images=None,
                                     model1=None, model2=None, device='cuda'):
    """Compare features from two models for each class separately.

    Args:
        features1, features2: Outputs from extract_block_features_per_class()
                             Structure: block_name -> layer_name -> class_id -> Tensor(H, W)
        block_name: Name of the block to visualize
        layer_name: Name of the layer within the block
        class_ids: List of class IDs to visualize. If None, show all classes
        labels: Tuple of strings to label the two feature sets
        plot_size: Size of each feature plot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
        show_diff: If True, also show the difference map
        class_names: Optional list/dict of class names for labeling (e.g., CIFAR-10 classes)
        input_images: Optional dict mapping class_id -> input image tensor (C, H, W)
                     If provided, shows original images in first column
        model1, model2: Optional models for prediction checking
                       If provided, checks if predictions are correct
        device: Device to run predictions on (default: 'cuda')
    """
    if block_name not in features1 or block_name not in features2:
        print(f"Block {block_name} not found in one or both feature sets")
        return

    if layer_name not in features1[block_name] or layer_name not in features2[block_name]:
        print(f"Layer {layer_name} not found in block {block_name}")
        return

    # Get class dictionaries
    class_dict1 = features1[block_name][layer_name]
    class_dict2 = features2[block_name][layer_name]

    # Get common classes
    common_classes = sorted([c for c in class_dict1.keys() if c in class_dict2])

    if class_ids is not None:
        common_classes = [c for c in common_classes if c in class_ids]

    num_classes = len(common_classes)
    if num_classes == 0:
        print("No common classes to compare")
        return

    # Get predictions if models are provided
    predictions1 = {}
    predictions2 = {}
    if model1 is not None and model2 is not None and input_images is not None:
        model1.eval()
        model2.eval()
        with torch.no_grad():
            for class_id in common_classes:
                if class_id in input_images:
                    img = input_images[class_id].unsqueeze(0).to(device)

                    # Get predictions
                    out1 = model1(img)
                    pred1 = out1.argmax(dim=1).item()
                    predictions1[class_id] = (pred1, pred1 == class_id)

                    out2 = model2(img)
                    pred2 = out2.argmax(dim=1).item()
                    predictions2[class_id] = (pred2, pred2 == class_id)

    # Determine number of columns
    # Base: attention1 + attention2 + diff (if show_diff)
    # Additional: input_image (if provided)
    base_cols = 3 if show_diff else 2
    cols = base_cols + (1 if input_images is not None else 0)

    fig = plt.figure(figsize=(plot_size * cols, plot_size * num_classes))
    if title:
        plt.suptitle(title, fontsize=14)
    else:
        plt.suptitle(f'Layer: {layer_name} - Per-Class Comparison', fontsize=14)

    for i, class_id in enumerate(common_classes):
        feat1 = class_dict1[class_id]
        feat2 = class_dict2[class_id]

        if feat1.shape != feat2.shape:
            print(f"Warning: Shape mismatch for class {class_id}")
            continue

        # Convert to numpy for plotting
        feat1_np = feat1.cpu().numpy()
        feat2_np = feat2.cpu().numpy()

        # Get class name for label
        if class_names is not None:
            if isinstance(class_names, dict):
                class_label = class_names.get(class_id, f"Class {class_id}")
            else:
                class_label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        else:
            class_label = f"Class {class_id}"

        # Add prediction info to label if available
        if class_id in predictions1 and class_id in predictions2:
            pred1, correct1 = predictions1[class_id]
            pred2, correct2 = predictions2[class_id]
            status1 = "✓" if correct1 else "✗"
            status2 = "✓" if correct2 else "✗"
            class_label = f"{class_label}\n{labels[0]}:{status1} {labels[1]}:{status2}"

        col_offset = 0

        # Plot original input image if provided
        if input_images is not None and class_id in input_images:
            ax_img = plt.subplot(num_classes, cols, i * cols + 1)
            img = input_images[class_id].cpu()

            # Convert from (C, H, W) to (H, W, C) for display
            if img.shape[0] == 3:  # RGB
                img_display = img.permute(1, 2, 0).numpy()
                # Denormalize if needed (assuming ImageNet normalization)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_display = img_display * std + mean
                img_display = np.clip(img_display, 0, 1)
            else:  # Grayscale
                img_display = img.squeeze().numpy()

            ax_img.imshow(img_display, cmap='gray' if img.shape[0] == 1 else None)
            if i == 0:  # Only show title for first row
                ax_img.set_title('Input Image')
            ax_img.set_ylabel(class_label, fontsize=9)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            col_offset = 1
        else:
            # No input image, just use label on first attention map
            col_offset = 0

        # Plot model1 attention features
        ax1 = plt.subplot(num_classes, cols, i * cols + col_offset + 1)
        im1 = ax1.imshow(feat1_np, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        if i == 0:  # Only show title for first row
            ax1.set_title(f'{labels[0]} Attention')
        if col_offset == 0:  # No input image column
            ax1.set_ylabel(class_label, fontsize=9)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot model2 attention features
        ax2 = plt.subplot(num_classes, cols, i * cols + col_offset + 2)
        im2 = ax2.imshow(feat2_np, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        if i == 0:  # Only show title for first row
            ax2.set_title(f'{labels[1]} Attention')
        ax2.set_xticks([])
        ax2.set_yticks([])

        if show_diff:
            # Plot difference map using diverging colormap
            diff = feat1_np - feat2_np
            vmax_diff = max(abs(diff.min()), abs(diff.max()))
            ax3 = plt.subplot(num_classes, cols, i * cols + col_offset + 3)
            im3 = ax3.imshow(diff, cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax_diff, vmax=vmax_diff))
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            if i == 0:  # Only show title for first row
                ax3.set_title('Difference')
            ax3.set_xticks([])
            ax3.set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class comparison to {save_path}")

    plt.show()


def compare_recovery_cases(features_baseline, features_improved, block_name, layer_name,
                           sample_indices, true_labels, baseline_preds, improved_preds,
                           input_images, labels=('Pruned (Wrong)', 'Regrown (Fixed)'),
                           plot_size=3, title=None, save_path=None, show_diff=True,
                           class_names=None):
    """Compare features for recovery cases: samples misclassified by baseline but fixed by improved model.

    This visualization shows side-by-side comparisons of attention maps for specific samples
    that demonstrate improvement from baseline to improved model.

    Args:
        features_baseline, features_improved: Outputs from extract_block_features_per_class()
                                             Structure: block_name -> layer_name -> sample_idx -> Tensor(H, W)
        block_name: Name of the block to visualize
        layer_name: Name of the layer within the block
        sample_indices: List of sample indices to visualize (from recovery case collection)
        true_labels: Tensor or list of ground truth labels for each sample
        baseline_preds: Tensor or list of baseline model predictions
        improved_preds: Tensor or list of improved model predictions
        input_images: Dict or tensor mapping sample_idx -> input image (C, H, W)
        labels: Tuple of strings to label baseline vs improved
        plot_size: Size of each subplot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
        show_diff: If True, show difference map
        class_names: Optional list/dict of class names for labeling

    Note: This function expects features extracted with sample indices as keys, not class IDs.
          Use extract_block_features_per_class() on recovery case samples.
    """
    if block_name not in features_baseline or block_name not in features_improved:
        print(f"Block {block_name} not found in one or both feature sets")
        return

    if layer_name not in features_baseline[block_name] or layer_name not in features_improved[block_name]:
        print(f"Layer {layer_name} not found in block {block_name}")
        return

    # Get feature dictionaries
    layer_features_baseline = features_baseline[block_name][layer_name]
    layer_features_improved = features_improved[block_name][layer_name]

    # Filter to requested sample indices
    available_indices = [idx for idx in sample_indices if
                         idx in layer_features_baseline and idx in layer_features_improved]

    if not available_indices:
        print("No valid samples found with extracted features")
        return

    num_samples = len(available_indices)

    # Determine columns
    base_cols = 3 if show_diff else 2
    cols = base_cols + 1  # Always include input image for recovery cases

    fig = plt.figure(figsize=(plot_size * cols, plot_size * num_samples))
    if title:
        plt.suptitle(title, fontsize=14)
    else:
        plt.suptitle(f'Recovery Cases - Layer: {layer_name}', fontsize=14)

    for i, sample_idx in enumerate(available_indices):
        feat_baseline = layer_features_baseline[sample_idx]
        feat_improved = layer_features_improved[sample_idx]

        if feat_baseline.shape != feat_improved.shape:
            print(f"Warning: Shape mismatch for sample {sample_idx}")
            continue

        # Convert to numpy
        feat_baseline_np = feat_baseline.cpu().numpy()
        feat_improved_np = feat_improved.cpu().numpy()

        # Get labels for this sample - FIXED: use sample_idx to index into lists
        # The sample_indices list contains the indices in the original recovery_images batch
        # So we need to use sample_idx to access the corresponding labels/predictions
        true_label = true_labels[sample_idx] if isinstance(true_labels, (list, tuple)) else true_labels[sample_idx]
        baseline_pred = baseline_preds[sample_idx] if isinstance(baseline_preds, (list, tuple)) else baseline_preds[
            sample_idx]
        improved_pred = improved_preds[sample_idx] if isinstance(improved_preds, (list, tuple)) else improved_preds[
            sample_idx]

        # Convert tensors to int if needed
        if isinstance(true_label, torch.Tensor):
            true_label = true_label.item()
        if isinstance(baseline_pred, torch.Tensor):
            baseline_pred = baseline_pred.item()
        if isinstance(improved_pred, torch.Tensor):
            improved_pred = improved_pred.item()

        # Create label text
        if class_names is not None:
            if isinstance(class_names, dict):
                true_name = class_names.get(true_label, f"Class {true_label}")
                baseline_name = class_names.get(baseline_pred, f"Class {baseline_pred}")
                improved_name = class_names.get(improved_pred, f"Class {improved_pred}")
            else:
                true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
                baseline_name = class_names[baseline_pred] if baseline_pred < len(
                    class_names) else f"Class {baseline_pred}"
                improved_name = class_names[improved_pred] if improved_pred < len(
                    class_names) else f"Class {improved_pred}"
        else:
            true_name = f"Class {true_label}"
            baseline_name = f"Class {baseline_pred}"
            improved_name = f"Class {improved_pred}"

        sample_label = f"True: {true_name}\n{labels[0]}: {baseline_name} ✗\n{labels[1]}: {improved_name} ✓"

        # Plot input image
        ax_img = plt.subplot(num_samples, cols, i * cols + 1)
        if isinstance(input_images, dict):
            img = input_images[sample_idx].cpu()
        else:  # Assume it's a tensor with batch dimension
            img = input_images[i].cpu()

        # Convert from (C, H, W) to (H, W, C) for display
        if img.shape[0] == 3:  # RGB
            img_display = img.permute(1, 2, 0).numpy()
            # Denormalize (ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = img_display * std + mean
            img_display = np.clip(img_display, 0, 1)
        else:  # Grayscale
            img_display = img.squeeze().numpy()

        ax_img.imshow(img_display, cmap='gray' if img.shape[0] == 1 else None)
        if i == 0:
            ax_img.set_title('Input Image')
        ax_img.set_ylabel(sample_label, fontsize=8)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Plot baseline attention
        ax1 = plt.subplot(num_samples, cols, i * cols + 2)
        im1 = ax1.imshow(feat_baseline_np, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        if i == 0:
            ax1.set_title(f'{labels[0]}\nAttention')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot improved attention
        ax2 = plt.subplot(num_samples, cols, i * cols + 3)
        im2 = ax2.imshow(feat_improved_np, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        if i == 0:
            ax2.set_title(f'{labels[1]}\nAttention')
        ax2.set_xticks([])
        ax2.set_yticks([])

        if show_diff:
            # Plot difference map
            diff = feat_improved_np - feat_baseline_np
            vmax_diff = max(abs(diff.min()), abs(diff.max()))
            ax3 = plt.subplot(num_samples, cols, i * cols + 4)
            im3 = ax3.imshow(diff, cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax_diff, vmax=vmax_diff))
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            if i == 0:
                ax3.set_title('Improvement\n(Regrown - Pruned)')
            ax3.set_xticks([])
            ax3.set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved recovery cases comparison to {save_path}")

    plt.show()


def compare_single_sample_all_layers(features_baseline, features_improved, block_dict,
                                     sample_idx, true_label, baseline_pred, improved_pred,
                                     input_image, labels=('Pruned (Wrong)', 'Regrown (Fixed)'),
                                     plot_size=3, title=None, save_path=None, show_diff=True,
                                     class_names=None, layer_indices=None):
    """Compare features across all layers for a single recovery case sample.

    This visualization shows how a single sample's features evolve through all layers,
    comparing baseline (wrong prediction) vs improved (correct prediction) models.

    Args:
        features_baseline, features_improved: Outputs from extract_block_features_per_class()
                                             Structure: block_name -> layer_name -> sample_idx -> Tensor(H, W)
        block_dict: Dictionary mapping block names to lists of layer names
                   e.g., {'block1': ['conv1', 'conv2'], 'block2': ['conv3', 'conv4']}
        sample_idx: Index of the sample to visualize
        true_label: Ground truth label for the sample
        baseline_pred: Baseline model prediction
        improved_pred: Improved model prediction
        input_image: Input image tensor (C, H, W)
        labels: Tuple of strings to label baseline vs improved
        plot_size: Size of each subplot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
        show_diff: If True, show difference map
        class_names: Optional list/dict of class names for labeling
        layer_indices: Optional list of layer indices to show (if None, show all)

    Returns:
        None (displays and optionally saves the plot)
    """
    # Collect all layers in order from block_dict
    all_layers = []
    all_blocks = []
    for block_name, layer_list in block_dict.items():
        for layer_name in layer_list:
            all_layers.append(layer_name)
            all_blocks.append(block_name)

    # Filter to requested layer indices if provided
    if layer_indices is not None:
        all_layers = [all_layers[i] for i in layer_indices if i < len(all_layers)]
        all_blocks = [all_blocks[i] for i in layer_indices if i < len(all_blocks)]

    # Check which layers have features for this sample
    available_layers = []
    available_blocks = []
    for block_name, layer_name in zip(all_blocks, all_layers):
        if (block_name in features_baseline and
                layer_name in features_baseline[block_name] and
                sample_idx in features_baseline[block_name][layer_name] and
                block_name in features_improved and
                layer_name in features_improved[block_name] and
                sample_idx in features_improved[block_name][layer_name]):
            available_layers.append(layer_name)
            available_blocks.append(block_name)

    if not available_layers:
        print(f"No features found for sample {sample_idx}")
        return

    num_layers = len(available_layers)

    # Convert tensors to int if needed
    if isinstance(true_label, torch.Tensor):
        true_label = true_label.item()
    if isinstance(baseline_pred, torch.Tensor):
        baseline_pred = baseline_pred.item()
    if isinstance(improved_pred, torch.Tensor):
        improved_pred = improved_pred.item()

    # Create label text
    if class_names is not None:
        if isinstance(class_names, dict):
            true_name = class_names[true_label]
            baseline_name = class_names[baseline_pred]
            improved_name = class_names[improved_pred]
        else:
            true_name = class_names[true_label]
            baseline_name = class_names[baseline_pred]
            improved_name = class_names[improved_pred]
    else:
        true_name = str(true_label)
        baseline_name = str(baseline_pred)
        improved_name = str(improved_pred)

    sample_label = f"True: {true_name}\n{labels[0]}: {baseline_name} ✗\n{labels[1]}: {improved_name} ✓"

    # Determine columns
    base_cols = 3 if show_diff else 2
    cols = base_cols + 1  # Always include input image

    fig = plt.figure(figsize=(plot_size * cols, plot_size * (num_layers + 1)))
    if title:
        plt.suptitle(title, fontsize=14)
    else:
        plt.suptitle(f'Single Sample Layer-wise Evolution (Sample {sample_idx})', fontsize=14)

    # First row: Show input image and sample info
    ax_img = plt.subplot(num_layers + 1, cols, 1)

    # Convert from (C, H, W) to (H, W, C) for display
    img = input_image.cpu() if isinstance(input_image, torch.Tensor) else input_image
    if img.shape[0] == 3:  # RGB
        img_display = img.permute(1, 2, 0).numpy()
        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img_display = img_display * std + mean
        img_display = np.clip(img_display, 0, 1)
    else:  # Grayscale
        img_display = img.squeeze().numpy()

    ax_img.imshow(img_display, cmap='gray' if img.shape[0] == 1 else None)
    ax_img.set_title('Input Image')
    ax_img.set_ylabel(sample_label, fontsize=8)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Add text info in the remaining first-row cells
    ax_info1 = plt.subplot(num_layers + 1, cols, 2)
    ax_info1.text(0.5, 0.5, f'{labels[0]}\n\nPredicts: {baseline_name}\n(Wrong)',
                  ha='center', va='center', fontsize=10, color='red')
    ax_info1.axis('off')

    ax_info2 = plt.subplot(num_layers + 1, cols, 3)
    ax_info2.text(0.5, 0.5, f'{labels[1]}\n\nPredicts: {improved_name}\n(Correct!)',
                  ha='center', va='center', fontsize=10, color='green')
    ax_info2.axis('off')

    if show_diff:
        ax_info3 = plt.subplot(num_layers + 1, cols, 4)
        ax_info3.text(0.5, 0.5, 'Improvement\n(Difference)',
                      ha='center', va='center', fontsize=10)
        ax_info3.axis('off')

    # Subsequent rows: Show each layer's features
    for i, (block_name, layer_name) in enumerate(zip(available_blocks, available_layers)):
        feat_baseline = features_baseline[block_name][layer_name][sample_idx]
        feat_improved = features_improved[block_name][layer_name][sample_idx]

        if feat_baseline.shape != feat_improved.shape:
            print(f"Warning: Shape mismatch for layer {layer_name}")
            continue

        # Convert to numpy
        feat_baseline_np = feat_baseline.cpu().numpy()
        feat_improved_np = feat_improved.cpu().numpy()

        row = i + 2  # +2 because first row is input image/info

        # Column 1: Layer name label
        ax_label = plt.subplot(num_layers + 1, cols, (row - 1) * cols + 1)
        layer_display_name = f"{block_name}\n{layer_name}"
        ax_label.text(0.5, 0.5, layer_display_name,
                      ha='center', va='center', fontsize=8, wrap=True)
        ax_label.axis('off')

        # Column 2: Baseline attention
        ax1 = plt.subplot(num_layers + 1, cols, (row - 1) * cols + 2)
        im1 = ax1.imshow(feat_baseline_np, cmap='viridis', vmin=0, vmax=1)
        if i == 0:
            ax1.set_title(labels[0])
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Column 3: Improved attention
        ax2 = plt.subplot(num_layers + 1, cols, (row - 1) * cols + 3)
        im2 = ax2.imshow(feat_improved_np, cmap='viridis', vmin=0, vmax=1)
        if i == 0:
            ax2.set_title(labels[1])
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # Column 4: Difference (if requested)
        if show_diff:
            ax3 = plt.subplot(num_layers + 1, cols, (row - 1) * cols + 4)
            diff = feat_improved_np - feat_baseline_np
            vmax_diff = max(abs(diff.min()), abs(diff.max()))
            if vmax_diff == 0:
                vmax_diff = 1
            im3 = ax3.imshow(diff, cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax_diff, vmax=vmax_diff))
            if i == 0:
                ax3.set_title('Improvement')
            ax3.set_xticks([])
            ax3.set_yticks([])
            plt.colorbar(im3, ax=ax3, fraction=0.046)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved single-sample all-layers comparison to {save_path}")

    plt.show()


def visualize_single_image_all_layers(model1, model2, image, label, layer_names=None,
                                      labels=('Model 1', 'Model 2'), device='cuda',
                                      plot_size=3, title=None, save_path=None,
                                      show_diff=True, class_names=None, max_channels=16):
    """
    Visualize feature maps from all specified layers for a single input image across two models.

    This function extracts and displays feature activations layer-by-layer for both models,
    showing how each model processes the same input through its layers.

    Args:
        model1: First model to compare
        model2: Second model to compare
        image: Input image tensor (C, H, W) or (1, C, H, W)
        label: Ground truth label for the image
        layer_names: List of layer names to visualize (as they appear in model.named_modules())
                    If None, automatically extracts all Conv2d and Linear layers
        labels: Tuple of strings to label the two models
        device: Device to run inference on ('cuda' or 'cpu')
        plot_size: Size of each subplot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
        show_diff: If True, show difference maps between models
        class_names: Optional list/dict of class names for labeling
        max_channels: Maximum number of channels to display per layer (default: 16)
                     Shows the first N channels with highest activation

    Returns:
        predictions: Dict with 'model1' and 'model2' predictions
    """
    model1.eval()
    model2.eval()

    # Auto-detect layer names if not provided
    if layer_names is None:
        layer_names = []
        for name, module in model1.named_modules():
            # Include Conv2d and Linear layers (main feature-producing layers)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_names.append(name)
        print(f"Auto-detected {len(layer_names)} layers to visualize")

    # Prepare image
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Get predictions
    with torch.no_grad():
        pred1 = model1(image).argmax(dim=1).item()
        pred2 = model2(image).argmax(dim=1).item()

    # Format labels
    if isinstance(label, torch.Tensor):
        label = label.item()

    if class_names is not None:
        if isinstance(class_names, dict):
            true_name = class_names.get(label, f"Class {label}")
            pred1_name = class_names.get(pred1, f"Class {pred1}")
            pred2_name = class_names.get(pred2, f"Class {pred2}")
        else:
            true_name = class_names[label] if label < len(class_names) else f"Class {label}"
            pred1_name = class_names[pred1] if pred1 < len(class_names) else f"Class {pred1}"
            pred2_name = class_names[pred2] if pred2 < len(class_names) else f"Class {pred2}"
    else:
        true_name = f"Class {label}"
        pred1_name = f"Class {pred1}"
        pred2_name = f"Class {pred2}"

    # Check if predictions are correct
    pred1_correct = "✓" if pred1 == label else "✗"
    pred2_correct = "✓" if pred2 == label else "✗"

    # Extract features from both models
    extractor1 = FeatureExtractor(model1)
    extractor2 = FeatureExtractor(model2)

    extractor1.register_hooks(layer_names)
    extractor2.register_hooks(layer_names)

    # Forward pass to extract features
    with torch.no_grad():
        _ = model1(image)
        features1 = extractor1.features.copy()
        _ = model2(image)
        features2 = extractor2.features.copy()

    extractor1.clear_hooks()
    extractor2.clear_hooks()

    # Filter to common layers
    common_layers = [l for l in layer_names if l in features1 and l in features2]

    if not common_layers:
        print("No common layers found with features")
        return {'model1': pred1, 'model2': pred2}

    # Determine grid layout
    base_cols = 3 if show_diff else 2
    cols = base_cols + 1  # Add column for input image info
    num_layers = len(common_layers)

    fig = plt.figure(figsize=(plot_size * cols, plot_size * (num_layers + 1)))

    if title:
        plt.suptitle(title, fontsize=14)
    else:
        plt.suptitle(f'Layer-by-Layer Feature Comparison\nTrue: {true_name}', fontsize=14)

    # Row 0: Show input image and prediction info
    ax_img = plt.subplot(num_layers + 1, cols, 1)
    img_display = image[0].cpu()
    if img_display.shape[0] == 3:
        img_display = img_display.permute(1, 2, 0)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    else:
        img_display = img_display.squeeze()

    ax_img.imshow(img_display, cmap='gray' if image.shape[1] == 1 else None)
    ax_img.set_title('Input Image', fontsize=10)
    ax_img.set_ylabel(f'True: {true_name}', fontsize=9)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Model 1 prediction info
    ax_pred1 = plt.subplot(num_layers + 1, cols, 2)
    ax_pred1.text(0.5, 0.5, f'{labels[0]}\n\nPredicts:\n{pred1_name}\n{pred1_correct}',
                  ha='center', va='center', fontsize=10,
                  color='green' if pred1 == label else 'red')
    ax_pred1.axis('off')

    # Model 2 prediction info
    ax_pred2 = plt.subplot(num_layers + 1, cols, 3)
    ax_pred2.text(0.5, 0.5, f'{labels[1]}\n\nPredicts:\n{pred2_name}\n{pred2_correct}',
                  ha='center', va='center', fontsize=10,
                  color='green' if pred2 == label else 'red')
    ax_pred2.axis('off')

    if show_diff:
        ax_diff_header = plt.subplot(num_layers + 1, cols, 4)
        ax_diff_header.text(0.5, 0.5, 'Difference\n(Model2 - Model1)',
                            ha='center', va='center', fontsize=10)
        ax_diff_header.axis('off')

    # Subsequent rows: Show each layer's features
    for i, layer_name in enumerate(common_layers):
        feat1 = features1[layer_name][0]  # Remove batch dimension
        feat2 = features2[layer_name][0]

        if feat1.shape != feat2.shape:
            print(f"Warning: Shape mismatch for {layer_name}: {feat1.shape} vs {feat2.shape}")
            continue

        # For conv layers with multiple channels, aggregate to 2D
        if feat1.dim() == 3:  # (C, H, W)
            # Select top channels by activation magnitude
            channel_importance1 = feat1.abs().mean(dim=(1, 2))
            channel_importance2 = feat2.abs().mean(dim=(1, 2))

            # Get top channels (union of both models)
            top_channels1 = channel_importance1.argsort(descending=True)[:max_channels]
            top_channels2 = channel_importance2.argsort(descending=True)[:max_channels]
            top_channels = torch.unique(torch.cat([top_channels1, top_channels2]))[:max_channels]

            # Average over selected channels
            feat1_2d = feat1[top_channels].mean(dim=0).cpu().numpy()
            feat2_2d = feat2[top_channels].mean(dim=0).cpu().numpy()
        elif feat1.dim() == 2:  # (H, W)
            feat1_2d = feat1.cpu().numpy()
            feat2_2d = feat2.cpu().numpy()
        elif feat1.dim() == 1:  # (C,) - fully connected
            # Reshape to square-ish grid for visualization
            size = int(np.ceil(np.sqrt(feat1.shape[0])))
            pad_size = size * size - feat1.shape[0]
            feat1_padded = torch.cat([feat1, torch.zeros(pad_size, device=feat1.device)])
            feat2_padded = torch.cat([feat2, torch.zeros(pad_size, device=feat2.device)])
            feat1_2d = feat1_padded.reshape(size, size).cpu().numpy()
            feat2_2d = feat2_padded.reshape(size, size).cpu().numpy()
        else:
            print(f"Unsupported feature dimension for {layer_name}: {feat1.shape}")
            continue

        row_idx = i + 1

        # Plot Model 1 features
        ax1 = plt.subplot(num_layers + 1, cols, row_idx * cols + 1)
        im1 = ax1.imshow(feat1_2d, cmap='viridis', aspect='auto')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        if i == 0:
            ax1.set_title(labels[0], fontsize=10)
        ax1.set_ylabel(f'{layer_name}\n{feat1.shape}', fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot Model 2 features
        ax2 = plt.subplot(num_layers + 1, cols, row_idx * cols + 2)
        im2 = ax2.imshow(feat2_2d, cmap='viridis', aspect='auto')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        if i == 0:
            ax2.set_title(labels[1], fontsize=10)
        ax2.set_xticks([])
        ax2.set_yticks([])

        if show_diff:
            # Plot difference
            ax3 = plt.subplot(num_layers + 1, cols, row_idx * cols + 3)
            diff = feat2_2d - feat1_2d
            max_abs_diff = np.abs(diff).max()
            im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto',
                             vmin=-max_abs_diff, vmax=max_abs_diff)
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            if i == 0:
                ax3.set_title('Difference', fontsize=10)
            ax3.set_xticks([])
            ax3.set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved single-image all-layers visualization to {save_path}")

    plt.show()

    return {'model1': pred1, 'model2': pred2}


def visualize_blockwise_features_pca(features1, features2=None, block_names=None,
                                     labels=('Original', 'Pruned'), n_components=2,
                                     method='pca', plot_size=4, title=None,
                                     save_path=None, show_variance=True):
    """Visualize each block separately with layers as scatter points in PCA space.

    Args:
        features1: Output from extract_block_features() for first model
        features2: Optional output from extract_block_features() for second model
        block_names: List of block names to visualize. If None, show all blocks
        labels: Tuple of strings to label the two feature sets
        n_components: Number of dimensions for reduction (2 or 3)
        method: 'pca', 'tsne', or 'both'
        plot_size: Size of each subplot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path (will create separate files for each block)
        show_variance: If True, show explained variance for PCA
    """
    if block_names is None:
        block_names = list(features1.keys())
        if features2:
            block_names = [b for b in block_names if b in features2]

    # Filter available blocks
    available_blocks = [b for b in block_names if b in features1]
    if features2:
        available_blocks = [b for b in available_blocks if b in features2]

    if not available_blocks:
        print("No valid blocks found for visualization")
        return

    methods = ['pca'] if method == 'pca' else (['tsne'] if method == 'tsne' else ['pca', 'tsne'])

    # Create separate figure for each block
    for block_name in available_blocks:
        print(f"\nProcessing block: {block_name}")

        # Collect all layer features for this block
        block_features1 = []
        layer_names = []

        for layer_name, features in features1[block_name].items():
            if features.dim() == 2:  # (H, W) format
                flattened = features.flatten()
                block_features1.append(flattened)
                layer_names.append(layer_name)
            elif features.dim() == 1:  # Already 1D
                block_features1.append(features)
                layer_names.append(layer_name)

        if not block_features1:
            print(f"No valid features found for block {block_name}")
            continue

        # Handle different feature sizes by padding/truncating
        max_len = max(feat.shape[0] for feat in block_features1)

        def normalize_features(feat_list, target_len):
            normalized = []
            for feat in feat_list:
                if feat.shape[0] > target_len:
                    normalized.append(feat[:target_len])
                elif feat.shape[0] < target_len:
                    padding = torch.zeros(target_len - feat.shape[0])
                    normalized.append(torch.cat([feat, padding]))
                else:
                    normalized.append(feat)
            return normalized

        block_features1 = normalize_features(block_features1, max_len)
        X1 = torch.stack(block_features1).cpu().numpy()

        # Get features from second model if available
        if features2 and block_name in features2:
            block_features2 = []
            for layer_name in layer_names:
                if layer_name in features2[block_name]:
                    features = features2[block_name][layer_name]
                    if features.dim() == 2:
                        flattened = features.flatten()
                        block_features2.append(flattened)
                    elif features.dim() == 1:
                        block_features2.append(features)

            if len(block_features2) == len(block_features1):
                block_features2 = normalize_features(block_features2, max_len)
                X2 = torch.stack(block_features2).cpu().numpy()
                X_combined = np.vstack([X1, X2])
                colors = ['blue'] * len(X1) + ['red'] * len(X2)
                markers = ['o'] * len(X1) + ['^'] * len(X2)
                point_labels = [f"{labels[0]}_{ln}" for ln in layer_names] + \
                               [f"{labels[1]}_{ln}" for ln in layer_names]
            else:
                X_combined = X1
                colors = ['blue'] * len(X1)
                markers = ['o'] * len(X1)
                point_labels = [f"Layer_{ln}" for ln in layer_names]
        else:
            X_combined = X1
            colors = ['blue'] * len(X1)
            markers = ['o'] * len(X1)
            point_labels = [f"Layer_{ln}" for ln in layer_names]

        # Create figure for this block
        num_methods = len(methods)
        if num_methods == 1:
            fig, ax = plt.subplots(1, 1, figsize=(plot_size, plot_size))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, num_methods, figsize=(plot_size * num_methods, plot_size))

        block_title = f"{title} - Block: {block_name}" if title else f"Block: {block_name}"
        fig.suptitle(block_title, fontsize=14)

        for method_idx, reduction_method in enumerate(methods):
            ax = axes[method_idx] if num_methods > 1 else axes[0]

            if reduction_method == 'pca':
                # Apply PCA
                pca = PCA(n_components=n_components)
                X_reduced = pca.fit_transform(X_combined)

                if n_components == 2:
                    # Use different markers for different models
                    for i, (x, y, color, marker) in enumerate(zip(X_reduced[:, 0], X_reduced[:, 1],
                                                                  colors, markers)):
                        ax.scatter(x, y, c=color, marker=marker, alpha=0.7, s=120, edgecolors='black', linewidths=0.5)

                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

                    # Add layer name annotations
                    for i, txt in enumerate(point_labels):
                        # Clean up the label for better readability
                        clean_label = txt.replace(f"{labels[0]}_", "").replace(f"{labels[1]}_", "")
                        ax.annotate(clean_label, (X_reduced[i, 0], X_reduced[i, 1]),
                                    xytext=(8, 8), textcoords='offset points',
                                    fontsize=9, alpha=0.9,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                elif n_components == 3:
                    ax.remove()
                    ax = fig.add_subplot(1, num_methods, method_idx + 1, projection='3d')
                    for i, (x, y, z, color, marker) in enumerate(zip(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                                                                     colors, markers)):
                        ax.scatter(x, y, z, c=color, marker=marker, alpha=0.7, s=120)

                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')

                method_title = f'PCA'
                if show_variance:
                    total_var = pca.explained_variance_ratio_[:n_components].sum()
                    method_title += f' (Var: {total_var:.1%})'

            elif reduction_method == 'tsne':
                # Apply t-SNE
                perplexity = min(30, max(5, len(X_combined) // 3))
                tsne = TSNE(n_components=n_components, perplexity=perplexity,
                            random_state=42, n_iter=1000)
                X_reduced = tsne.fit_transform(X_combined)

                if n_components == 2:
                    for i, (x, y, color, marker) in enumerate(zip(X_reduced[:, 0], X_reduced[:, 1],
                                                                  colors, markers)):
                        ax.scatter(x, y, c=color, marker=marker, alpha=0.7, s=120, edgecolors='black', linewidths=0.5)

                    ax.set_xlabel('t-SNE 1')
                    ax.set_ylabel('t-SNE 2')

                    # Add layer name annotations
                    for i, txt in enumerate(point_labels):
                        clean_label = txt.replace(f"{labels[0]}_", "").replace(f"{labels[1]}_", "")
                        ax.annotate(clean_label, (X_reduced[i, 0], X_reduced[i, 1]),
                                    xytext=(8, 8), textcoords='offset points',
                                    fontsize=9, alpha=0.9,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                method_title = 't-SNE'

            ax.set_title(method_title, fontsize=12)
            ax.grid(True, alpha=0.3)

        # Add legend if comparing two models
        if features2 and block_name in features2:
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                                  markeredgecolor='black', markersize=10, label=f'{labels[0]} Layers'),
                       plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                                  markeredgecolor='black', markersize=10, label=f'{labels[1]} Layers')]
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))

        plt.tight_layout()

        # Save individual block plots
        if save_path:
            base_dir = os.path.dirname(save_path)
            base_name = os.path.splitext(os.path.basename(save_path))[0]
            block_save_path = os.path.join(base_dir, f"{base_name}_{block_name}.png")
            os.makedirs(base_dir, exist_ok=True)
            plt.savefig(block_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved {block_name} visualization to {block_save_path}")

        plt.show()

    print(f"\nCompleted visualization for {len(available_blocks)} blocks")


def visualize_combined_blocks_pca(features_dict_list, block_names=None, labels=None,
                                  colors=None, markers=None, n_components=2, method='pca',
                                  plot_size=10, title=None, save_path=None, show_variance=True,
                                  legend_cols=2):
    """Visualize all blocks in one figure with different colors for blocks and markers for sparsity levels.

    Args:
        features_dict_list: List of feature dictionaries from extract_block_features() for different models/sparsity
        block_names: List of block names to visualize. If None, show all blocks
        labels: List of strings to label each feature set (e.g., ['Original', '50% Pruned', '90% Pruned'])
        colors: List of colors for each block. If None, uses default color palette
        markers: List of markers for each sparsity level. If None, uses default markers
        n_components: Number of dimensions for reduction (2 or 3)
        method: 'pca', 'tsne', or 'both'
        plot_size: Size of the plot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
        show_variance: If True, show explained variance for PCA
        legend_cols: Number of columns in legend
    """
    if not features_dict_list:
        print("No feature dictionaries provided")
        return

    # Default labels if not provided
    if labels is None:
        labels = [f'Model_{i}' for i in range(len(features_dict_list))]

    # Get common blocks across all feature dictionaries
    if block_names is None:
        all_blocks = set(features_dict_list[0].keys())
        for features in features_dict_list[1:]:
            all_blocks = all_blocks.intersection(set(features.keys()))
        block_names = sorted(list(all_blocks))

    # Filter available blocks
    available_blocks = []
    for block_name in block_names:
        if all(block_name in features for features in features_dict_list):
            available_blocks.append(block_name)

    if not available_blocks:
        print("No common blocks found across all feature sets")
        return

    # Default colors for blocks (using a colorful palette)
    if colors is None:
        import matplotlib.cm as cm
        colors = cm.Set3(np.linspace(0, 1, len(available_blocks)))

    # Default markers for different sparsity levels
    if markers is None:
        markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', 'h', '*'][:len(features_dict_list)]

    methods = ['pca'] if method == 'pca' else (['tsne'] if method == 'tsne' else ['pca', 'tsne'])

    # Create figure
    num_methods = len(methods)
    if num_methods == 1:
        fig, ax = plt.subplots(1, 1, figsize=(plot_size, plot_size))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, num_methods, figsize=(plot_size * num_methods, plot_size))

    if title:
        fig.suptitle(title, fontsize=16)

    # Collect all features from all blocks and all models
    all_features = []
    all_labels = []
    all_colors = []
    all_markers = []

    for model_idx, features_dict in enumerate(features_dict_list):
        for block_idx, block_name in enumerate(available_blocks):
            if block_name not in features_dict:
                continue

            # Collect all layer features for this block
            block_features = []
            layer_names = []

            for layer_name, features in features_dict[block_name].items():
                if features.dim() == 2:  # (H, W) format
                    flattened = features.flatten()
                    block_features.append(flattened)
                    layer_names.append(layer_name)
                elif features.dim() == 1:  # Already 1D
                    block_features.append(features)
                    layer_names.append(layer_name)

            if not block_features:
                continue

            # Handle different feature sizes by padding/truncating
            max_len = max(feat.shape[0] for feat in block_features)

            def normalize_features(feat_list, target_len):
                normalized = []
                for feat in feat_list:
                    if feat.shape[0] > target_len:
                        normalized.append(feat[:target_len])
                    elif feat.shape[0] < target_len:
                        padding = torch.zeros(target_len - feat.shape[0])
                        normalized.append(torch.cat([feat, padding]))
                    else:
                        normalized.append(feat)
                return normalized

            block_features = normalize_features(block_features, max_len)

            # Add each layer as a separate point
            for layer_idx, layer_feat in enumerate(block_features):
                all_features.append(layer_feat.cpu().numpy())
                all_labels.append(f"{labels[model_idx]}_{block_name}_{layer_names[layer_idx]}")
                all_colors.append(colors[block_idx])
                all_markers.append(markers[model_idx])

    if not all_features:
        print("No valid features found for visualization")
        return

    # Handle different feature sizes across all features
    max_global_len = max(feat.shape[0] for feat in all_features)

    def pad_features(feat, target_len):
        if feat.shape[0] < target_len:
            padding = np.zeros(target_len - feat.shape[0])
            return np.concatenate([feat, padding])
        elif feat.shape[0] > target_len:
            return feat[:target_len]
        else:
            return feat

    all_features = [pad_features(feat, max_global_len) for feat in all_features]
    X_combined = np.vstack(all_features)

    for method_idx, reduction_method in enumerate(methods):
        ax = axes[method_idx] if num_methods > 1 else axes[0]

        if reduction_method == 'pca':
            # Apply PCA
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_combined)

            if n_components == 2:
                # Group points by model and block for legend
                plotted_combinations = set()

                for i, (x, y, color, marker, label) in enumerate(zip(X_reduced[:, 0], X_reduced[:, 1],
                                                                     all_colors, all_markers, all_labels)):
                    # Extract model and block info for legend
                    model_name = label.split('_')[0]
                    block_name = '_'.join(label.split('_')[1:-1])  # Handle multi-word block names

                    # Create legend label
                    legend_key = f"{model_name}_{block_name}"

                    # Plot point
                    scatter = ax.scatter(x, y, c=[color], marker=marker, alpha=0.7, s=100,
                                         edgecolors='black', linewidths=0.5,
                                         label=legend_key if legend_key not in plotted_combinations else "")

                    plotted_combinations.add(legend_key)

                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

            elif n_components == 3:
                ax.remove()
                ax = fig.add_subplot(1, num_methods, method_idx + 1, projection='3d')

                plotted_combinations = set()
                for i, (x, y, z, color, marker, label) in enumerate(
                        zip(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                            all_colors, all_markers, all_labels)):
                    model_name = label.split('_')[0]
                    block_name = '_'.join(label.split('_')[1:-1])
                    legend_key = f"{model_name}_{block_name}"

                    ax.scatter(x, y, z, c=[color], marker=marker, alpha=0.7, s=100,
                               label=legend_key if legend_key not in plotted_combinations else "")

                    plotted_combinations.add(legend_key)

                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')

            method_title = 'Combined Blocks PCA'
            if show_variance:
                total_var = pca.explained_variance_ratio_[:n_components].sum()
                method_title += f' (Total Var: {total_var:.1%})'

        elif reduction_method == 'tsne':
            # Apply t-SNE
            perplexity = min(30, max(5, len(X_combined) // 3))
            tsne = TSNE(n_components=n_components, perplexity=perplexity,
                        random_state=42, n_iter=1000)
            X_reduced = tsne.fit_transform(X_combined)

            if n_components == 2:
                plotted_combinations = set()

                for i, (x, y, color, marker, label) in enumerate(zip(X_reduced[:, 0], X_reduced[:, 1],
                                                                     all_colors, all_markers, all_labels)):
                    model_name = label.split('_')[0]
                    block_name = '_'.join(label.split('_')[1:-1])
                    legend_key = f"{model_name}_{block_name}"

                    ax.scatter(x, y, c=[color], marker=marker, alpha=0.7, s=100,
                               edgecolors='black', linewidths=0.5,
                               label=legend_key if legend_key not in plotted_combinations else "")

                    plotted_combinations.add(legend_key)

                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')

            method_title = 'Combined Blocks t-SNE'

        ax.set_title(method_title, fontsize=14)
        ax.grid(True, alpha=0.3)

    # Create custom legend
    legend_elements = []

    # Add block color legend
    for block_idx, block_name in enumerate(available_blocks):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=colors[block_idx], markersize=10,
                                          label=f'Block: {block_name}', markeredgecolor='black'))

    # Add separator
    if len(available_blocks) > 0 and len(labels) > 0:
        legend_elements.append(plt.Line2D([0], [0], color='none', label=''))  # Spacer

    # Add sparsity marker legend
    for model_idx, label in enumerate(labels):
        legend_elements.append(plt.Line2D([0], [0], marker=markers[model_idx], color='w',
                                          markerfacecolor='gray', markersize=10,
                                          label=f'Sparsity: {label}', markeredgecolor='black'))

    # Position legend outside the plot
    if num_methods == 1:
        ax = axes[0]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
                  ncol=legend_cols, fontsize=10)
    else:
        fig.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left',
                   ncol=legend_cols, fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined blocks visualization to {save_path}")

    plt.show()

    return X_reduced, all_labels


def visualize_individual_blocks_pca(features1, features2=None, block_names=None,
                                    labels=('Original', 'Pruned'), n_components=2,
                                    method='pca', plot_size=4, title=None,
                                    save_path=None, show_variance=True):
    """Visualize each block separately using PCA/t-SNE, showing layer-level clustering within blocks.

    Args:
        features1: Output from extract_block_features() for first model
        features2: Optional output from extract_block_features() for second model
        block_names: List of block names to visualize. If None, show all blocks
        labels: Tuple of strings to label the two feature sets
        n_components: Number of dimensions for reduction (2 or 3)
        method: 'pca', 'tsne', or 'both'
        plot_size: Size of each subplot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
        show_variance: If True, show explained variance for PCA
    """
    if block_names is None:
        block_names = list(features1.keys())
        if features2:
            block_names = [b for b in block_names if b in features2]

    # Filter available blocks
    available_blocks = [b for b in block_names if b in features1]
    if features2:
        available_blocks = [b for b in available_blocks if b in features2]

    if not available_blocks:
        print("No valid blocks found for visualization")
        return

    num_blocks = len(available_blocks)
    methods = ['pca'] if method == 'pca' else (['tsne'] if method == 'tsne' else ['pca', 'tsne'])
    num_methods = len(methods)

    # Create subplot grid - separate plot for each block
    fig, axes = plt.subplots(num_blocks, num_methods,
                             figsize=(plot_size * num_methods, plot_size * num_blocks))
    if title:
        fig.suptitle(title, fontsize=16)

    # Handle single row/column cases
    if num_blocks == 1 and num_methods == 1:
        axes = [[axes]]
    elif num_blocks == 1:
        axes = [axes]
    elif num_methods == 1:
        axes = [[ax] for ax in axes]

    for block_idx, block_name in enumerate(available_blocks):
        # Collect all layer features for this block
        block_features1 = []
        layer_names = []

        for layer_name, features in features1[block_name].items():
            if features.dim() == 2:  # (H, W) format
                flattened = features.flatten()
                block_features1.append(flattened)
                layer_names.append(layer_name)
            elif features.dim() == 1:  # Already 1D
                block_features1.append(features)
                layer_names.append(layer_name)

        if not block_features1:
            print(f"No valid features found for block {block_name}")
            continue

        # Handle different feature sizes by padding/truncating
        max_len = max(feat.shape[0] for feat in block_features1)

        def normalize_features(feat_list, target_len):
            normalized = []
            for feat in feat_list:
                if feat.shape[0] > target_len:
                    normalized.append(feat[:target_len])
                elif feat.shape[0] < target_len:
                    padding = torch.zeros(target_len - feat.shape[0])
                    normalized.append(torch.cat([feat, padding]))
                else:
                    normalized.append(feat)
            return normalized

        block_features1 = normalize_features(block_features1, max_len)
        X1 = torch.stack(block_features1).cpu().numpy()

        # Get features from second model if available
        if features2 and block_name in features2:
            block_features2 = []
            for layer_name in layer_names:
                if layer_name in features2[block_name]:
                    features = features2[block_name][layer_name]
                    if features.dim() == 2:
                        flattened = features.flatten()
                        block_features2.append(flattened)
                    elif features.dim() == 1:
                        block_features2.append(features)

            if len(block_features2) == len(block_features1):
                block_features2 = normalize_features(block_features2, max_len)
                X2 = torch.stack(block_features2).cpu().numpy()
                X_combined = np.vstack([X1, X2])
                colors = ['blue'] * len(X1) + ['red'] * len(X2)
                markers = ['o'] * len(X1) + ['^'] * len(X2)
                point_labels = [f"{labels[0]}_{ln}" for ln in layer_names] + \
                               [f"{labels[1]}_{ln}" for ln in layer_names]
            else:
                X_combined = X1
                colors = ['blue'] * len(X1)
                markers = ['o'] * len(X1)
                point_labels = [f"Layer_{ln}" for ln in layer_names]
        else:
            X_combined = X1
            colors = ['blue'] * len(X1)
            markers = ['o'] * len(X1)
            point_labels = [f"Layer_{ln}" for ln in layer_names]

        for method_idx, reduction_method in enumerate(methods):
            ax = axes[block_idx][method_idx]

            if reduction_method == 'pca':
                # Apply PCA
                pca = PCA(n_components=n_components)
                X_reduced = pca.fit_transform(X_combined)

                if n_components == 2:
                    # Use different markers for different models
                    for i, (x, y, color, marker) in enumerate(zip(X_reduced[:, 0], X_reduced[:, 1],
                                                                  colors, markers)):
                        ax.scatter(x, y, c=color, marker=marker, alpha=0.7, s=100)

                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

                    # Add layer name annotations
                    for i, txt in enumerate(layer_names * (2 if features2 and block_name in features2 else 1)):
                        ax.annotate(txt, (X_reduced[i, 0], X_reduced[i, 1]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, alpha=0.8)

                method_title = f'PCA - {block_name}'
                if show_variance:
                    total_var = pca.explained_variance_ratio_[:n_components].sum()
                    method_title += f' ({total_var:.1%})'

            elif reduction_method == 'tsne':
                # Apply t-SNE
                perplexity = min(30, max(5, len(X_combined) // 3))
                tsne = TSNE(n_components=n_components, perplexity=perplexity,
                            random_state=42, n_iter=1000)
                X_reduced = tsne.fit_transform(X_combined)

                if n_components == 2:
                    for i, (x, y, color, marker) in enumerate(zip(X_reduced[:, 0], X_reduced[:, 1],
                                                                  colors, markers)):
                        ax.scatter(x, y, c=color, marker=marker, alpha=0.7, s=100)

                    ax.set_xlabel('t-SNE 1')
                    ax.set_ylabel('t-SNE 2')

                    # Add layer name annotations
                    for i, txt in enumerate(layer_names * (2 if features2 and block_name in features2 else 1)):
                        ax.annotate(txt, (X_reduced[i, 0], X_reduced[i, 1]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, alpha=0.8)

                method_title = f't-SNE - {block_name}'

            ax.set_title(method_title, fontsize=11)
            ax.grid(True, alpha=0.3)

    # Add legend if comparing two models
    if features2:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                              markersize=8, label=f'{labels[0]} Layers'),
                   plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                              markersize=8, label=f'{labels[1]} Layers')]
        fig.legend(handles=handles, loc='upper right')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved individual blocks PCA visualization to {save_path}")

    plt.show()


def visualize_block_feature_distributions(features1, features2=None, block_names=None,
                                          labels=('Original', 'Pruned'), plot_size=4,
                                          title=None, save_path=None, show_stats=True):
    """Visualize feature value distributions across blocks using histograms and box plots.

    Args:
        features1: Output from extract_block_features() for first model
        features2: Optional output from extract_block_features() for second model
        block_names: List of block names to visualize. If None, show all blocks
        labels: Tuple of strings to label the two feature sets
        plot_size: Size of each subplot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
        show_stats: If True, print statistical summaries
    """
    if block_names is None:
        block_names = list(features1.keys())
        if features2:
            block_names = [b for b in block_names if b in features2]

    # Filter available blocks
    available_blocks = [b for b in block_names if b in features1]
    if features2:
        available_blocks = [b for b in available_blocks if b in features2]

    if not available_blocks:
        print("No valid blocks found for visualization")
        return

    num_blocks = len(available_blocks)

    # Create subplot grid: histograms and box plots
    fig, axes = plt.subplots(num_blocks, 2, figsize=(plot_size * 2, plot_size * num_blocks))
    if title:
        fig.suptitle(title, fontsize=16)

    # Handle single row case
    if num_blocks == 1:
        axes = [axes]

    for block_idx, block_name in enumerate(available_blocks):
        # Aggregate all features for this block
        all_features1 = []
        for layer_name, features in features1[block_name].items():
            if features.dim() >= 1:
                all_features1.extend(features.flatten().cpu().numpy())

        all_features2 = []
        if features2 and block_name in features2:
            for layer_name, features in features2[block_name].items():
                if features.dim() >= 1:
                    all_features2.extend(features.flatten().cpu().numpy())

        # Plot histograms
        ax_hist = axes[block_idx][0]
        ax_hist.hist(all_features1, bins=50, alpha=0.7, label=labels[0],
                     color='blue', density=True)
        if all_features2:
            ax_hist.hist(all_features2, bins=50, alpha=0.7, label=labels[1],
                         color='red', density=True)
        ax_hist.set_title(f'Distribution - {block_name}')
        ax_hist.set_xlabel('Feature Value')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # Plot box plots
        ax_box = axes[block_idx][1]
        data_to_plot = [all_features1]
        box_labels = [labels[0]]
        if all_features2:
            data_to_plot.append(all_features2)
            box_labels.append(labels[1])

        bp = ax_box.boxplot(data_to_plot, labels=box_labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax_box.set_title(f'Box Plot - {block_name}')
        ax_box.set_ylabel('Feature Value')
        ax_box.grid(True, alpha=0.3)

        # Print statistics if requested
        if show_stats:
            print(f"\n=== {block_name} Statistics ===")
            print(f"{labels[0]}: Mean={np.mean(all_features1):.4f}, "
                  f"Std={np.std(all_features1):.4f}, "
                  f"Min={np.min(all_features1):.4f}, "
                  f"Max={np.max(all_features1):.4f}")
            if all_features2:
                print(f"{labels[1]}: Mean={np.mean(all_features2):.4f}, "
                      f"Std={np.std(all_features2):.4f}, "
                      f"Min={np.min(all_features2):.4f}, "
                      f"Max={np.max(all_features2):.4f}")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution visualization to {save_path}")

    plt.show()


def visualize_layer_features(block_features, block_name, layer_indices=None, plot_size=3, title=None, save_path=None):
    """Visualize features from specific layers in a block.

    Args:
        block_features: Output from extract_block_features()
        block_name: Name of the block to visualize
        layer_indices: List of indices to visualize. If None, show all layers
        plot_size: Size of each feature plot in inches
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
    """
    if block_name not in block_features:
        print(f"Block {block_name} not found in features")
        return

    # Get layer features for the block
    layers = list(block_features[block_name].keys())
    if layer_indices is not None:
        # Filter layers by indices
        layers = [layers[i] for i in layer_indices if i < len(layers)]

    num_layers = len(layers)
    if num_layers == 0:
        print("No layers to visualize")
        return

    # Create subplot grid
    fig = plt.figure(figsize=(plot_size * 4, plot_size * num_layers))
    if title:
        plt.suptitle(title, fontsize=14)

    for i, layer_name in enumerate(layers):
        features = block_features[block_name][layer_name]

        # Create subplot for this layer
        ax = plt.subplot(num_layers, 1, i + 1)

        if features.dim() == 2:  # Already in (H, W) format
            # Plot the 2D feature map directly
            im = ax.imshow(features.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Layer: {layer_name}')
        else:
            print(f"Warning: Unexpected feature dimensions {features.shape} for layer {layer_name}")
            continue

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def visualize_layer_similarity_heatmap(features1, features2=None, block_names=None,
                                       labels=('Original', 'Pruned'), similarity_metric='cosine',
                                       plot_size=8, title=None, save_path=None):
    """Visualize similarity between layers within and across blocks using heatmaps.

    Args:
        features1: Output from extract_block_features() for first model
        features2: Optional output from extract_block_features() for second model
        block_names: List of block names to analyze. If None, use all blocks
        labels: Tuple of strings to label the two feature sets
        similarity_metric: 'cosine', 'correlation', or 'euclidean'
        plot_size: Size of the heatmap
        title: Custom title for the plot
        save_path: If provided, save the plot to this path
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import pdist, squareform

    if block_names is None:
        block_names = list(features1.keys())
        if features2:
            block_names = [b for b in block_names if b in features2]

    # Collect all layer features
    all_layers = []
    layer_labels = []
    block_labels = []

    # Process first model
    for block_name in block_names:
        if block_name not in features1:
            continue
        for layer_name, features in features1[block_name].items():
            if features.dim() >= 1:
                flattened = features.flatten().cpu().numpy()
                all_layers.append(flattened)
                layer_labels.append(f"{labels[0]}_{block_name}_{layer_name}")
                block_labels.append(f"{labels[0]}_{block_name}")

    # Process second model if provided
    if features2:
        for block_name in block_names:
            if block_name not in features2:
                continue
            for layer_name, features in features2[block_name].items():
                if features.dim() >= 1:
                    flattened = features.flatten().cpu().numpy()
                    all_layers.append(flattened)
                    layer_labels.append(f"{labels[1]}_{block_name}_{layer_name}")
                    block_labels.append(f"{labels[1]}_{block_name}")

    if not all_layers:
        print("No layers found for similarity analysis")
        return

    # Stack all features
    X = np.vstack(all_layers)

    # Compute similarity matrix
    if similarity_metric == 'cosine':
        sim_matrix = cosine_similarity(X)
    elif similarity_metric == 'correlation':
        sim_matrix = np.corrcoef(X)
    elif similarity_metric == 'euclidean':
        # Convert distances to similarities
        distances = squareform(pdist(X, metric='euclidean'))
        sim_matrix = 1 / (1 + distances)  # Convert to similarity
    else:
        raise ValueError("similarity_metric must be 'cosine', 'correlation', or 'euclidean'")

    # Create the heatmap
    plt.figure(figsize=(plot_size, plot_size))

    # Use seaborn for better heatmap
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)  # Mask upper triangle

    sns.heatmap(sim_matrix,
                mask=mask,
                xticklabels=[label.replace(f"{labels[0]}_", "").replace(f"{labels[1]}_", "")
                             for label in layer_labels],
                yticklabels=[label.replace(f"{labels[0]}_", "").replace(f"{labels[1]}_", "")
                             for label in layer_labels],
                cmap='RdYlBu_r',
                center=0 if similarity_metric == 'correlation' else None,
                annot=False,
                fmt='.2f',
                square=True)

    plt.title(f'Layer {similarity_metric.capitalize()} Similarity' +
              (f' - {title}' if title else ''))
    plt.xlabel('Layers')
    plt.ylabel('Layers')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add block boundaries
    block_changes = []
    current_block = None
    for i, block_label in enumerate(block_labels):
        if block_label != current_block:
            block_changes.append(i)
            current_block = block_label

    for change_idx in block_changes[1:]:  # Skip first boundary
        plt.axhline(y=change_idx, color='white', linewidth=2)
        plt.axvline(x=change_idx, color='white', linewidth=2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity heatmap to {save_path}")

    plt.show()

    return sim_matrix, layer_labels


def plot_feature_maps(features_dict, title="Feature Maps", target_folder=None):
    """
    Plot the first channel of each feature map layer.
    If target_folder is specified, saves the image instead of just displaying.
    """
    num_layers = len(features_dict)
    fig, axes = plt.subplots(num_layers, 4, figsize=(5, 4 * num_layers))  # One column per layer
    fig.suptitle(title, fontsize=16)

    # Handle the case when num_layers == 1 (axes is not a list)
    if num_layers == 1:
        axes = [axes]

    for i, (layer_name, features) in enumerate(features_dict.items()):
        ax = axes[i]
        sample_features = features[0]  # [C, H, W] or [C]

        if len(sample_features.shape) == 3:  # Conv
            feature_map = sample_features[0].detach().cpu().numpy()  # First channel
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'{layer_name} - Channel 0', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)

        elif len(sample_features.shape) == 1:  # FC layer
            ax.plot(sample_features.detach().cpu().numpy())
            ax.set_title(f'{layer_name} (FC layer)', fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if target_folder:
        os.makedirs(target_folder, exist_ok=True)
        save_path = os.path.join(target_folder, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    plt.show()


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def compute_ssim_mapwise(features_dict1, features_dict2):
    ssim_scores = {}

    for layer in features_dict1:
        feat1 = features_dict1[layer]
        feat2 = features_dict2[layer]

        if feat1.shape != feat2.shape:
            print(f"[Skip] Shape mismatch at layer {layer}: {feat1.shape} vs {feat2.shape}")
            continue

        if feat1.dim() != 4:
            print(f"[Skip] Non-4D tensor at layer {layer}: shape {feat1.shape}")
            continue

        B, C, H, W = feat1.shape
        score_sum = 0.0
        count = 0

        for b in range(B):
            for c in range(C):
                img1 = feat1[b, c].detach().cpu().numpy()
                img2 = feat2[b, c].detach().cpu().numpy()

                if img1.shape[0] < 3 or img1.shape[1] < 3:
                    # print(f"[Skip] Layer {layer}, image too small for SSIM: {img1.shape}")
                    # score = 1 - mse(img1, img2)
                    # score_sum += score
                    # count += 1
                    continue

                else:
                    # Normalize if needed
                    if img1.max() > 1 or img1.min() < 0:
                        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
                    if img2.max() > 1 or img2.min() < 0:
                        img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)

                    try:
                        score, _ = ssim(img1, img2, full=True, win_size=3)
                        score_sum += score
                        count += 1
                    except:
                        continue

        ssim_scores[layer] = score_sum / count if count > 0 else np.nan

    return ssim_scores


def plot_ssim_scores(ssim_scores, model_name, sparsity1, sparsity2, save_path="./feature_SSIM", title="SSIM per Layer"):
    """
    Plot SSIM values as a polyline chart.
    """
    target_folder = os.path.join(save_path, model_name)
    os.makedirs(target_folder, exist_ok=True)

    layers = list(ssim_scores.keys())
    scores = [ssim_scores[l] for l in layers]

    plt.figure(figsize=(8, 4))
    plt.plot(layers, scores, marker='o', color='blue', linewidth=2)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Layer")
    plt.ylabel("SSIM")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    filename = f"{model_name}_{sparsity1}_{sparsity2}.png"
    target_path = os.path.join(target_folder, filename)
    plt.savefig(target_path, dpi=300)
    plt.close()


def count_params_block(model, block_dict):
    total_params = 0
    surviving_params = 0

    for b in block_dict.values():
        for m in b:
            m = get_layer_by_name(model, m)
            if hasattr(m, 'weight_orig') and hasattr(m, 'weight_mask'):
                w_orig = m.weight_orig.data
                mask = m.weight_mask.data

                total_params += w_orig.numel()
                surviving_params += mask.sum().item()
                layer_pruned_param = w_orig.numel() - mask.sum().item()

                print(f"Layer {m} params: {m.weight_orig.data.numel()}")
                print(f"Surviving (unpruned) params: {mask.sum().item()}")
                print(f"Pruned params: {layer_pruned_param}")
                print(f"Sparsity: {layer_pruned_param / m.weight_orig.data.numel():.4f}")
            else:
                if hasattr(m, 'weight'):
                    total_params += m.weight.data.numel()

                    print(f"Layer {m} params: {m.weight.data.numel()}")

    pruned_params = total_params - surviving_params
    print(f"Total params: {total_params}")
    print(f"Surviving (unpruned) params: {int(surviving_params)}")
    print(f"Pruned params: {int(pruned_params)}")
    print(f"Sparsity: {pruned_params / total_params:.4f}")

    return total_params, surviving_params, pruned_params


def get_resnet18_layer_names():
    """Get important layer names from ResNet-18"""
    return [
        'conv1',
        'bn1',
        'layer1.0.conv1',
        'layer1.0.conv2',
        'layer1.1.conv1',
        'layer1.1.conv2',
        'layer2.0.conv1',
        'layer2.0.conv2',
        'layer2.0.downsample.0',
        'layer2.1.conv1',
        'layer2.1.conv2',
        'layer3.0.conv1',
        'layer3.0.conv2',
        'layer3.0.downsample.0',
        'layer3.1.conv1',
        'layer3.1.conv2',
        'layer4.0.conv1',
        'layer4.0.conv2',
        'layer4.0.downsample.0',
        'layer4.1.conv1',
        'layer4.1.conv2',
        'avgpool',
        'fc'
    ]


def get_resnet20_layer_names():
    """Get key layer names from CifarResNet (ResNet-20), including shortcut paths"""
    return [
        'conv1',
        # 'bn1',
        # Layer1 (no shortcuts)
        'layer1.0.conv1', 'layer1.0.conv2',
        'layer1.1.conv1', 'layer1.1.conv2',
        'layer1.2.conv1', 'layer1.2.conv2',
        # Layer2
        'layer2.0.conv1', 'layer2.0.conv2',
        'layer2.0.shortcut.0',  # Conv2d
        'layer2.1.conv1', 'layer2.1.conv2',
        'layer2.2.conv1', 'layer2.2.conv2',
        # Layer3
        'layer3.0.conv1', 'layer3.0.conv2',
        'layer3.0.shortcut.0',  # Conv2d
        'layer3.1.conv1', 'layer3.1.conv2',
        'layer3.2.conv1', 'layer3.2.conv2',
        # Final classifier
        'linear'
    ]


def get_cifar_resnet20_blocks():
    """Distribute CifarResNet20 layers into logical residual blocks"""
    return {
        'block1': [
            'conv1',
            'layer1.0.conv1', 'layer1.0.conv2',
            'layer1.1.conv1', 'layer1.1.conv2',
            'layer1.2.conv1', 'layer1.2.conv2',
        ],
        'block2': [
            'layer2.0.conv1', 'layer2.0.conv2',
            'layer2.0.shortcut.0',
            'layer2.1.conv1', 'layer2.1.conv2',
            'layer2.2.conv1', 'layer2.2.conv2',
        ],
        'block3': [
            'layer3.0.conv1', 'layer3.0.conv2',
            'layer3.0.shortcut.0',
            'layer3.1.conv1', 'layer3.1.conv2',
            'layer3.2.conv1', 'layer3.2.conv2',
        ],
        'block_fc': ['linear']
    }


def get_alexnet_blocks():
    """Distribute AlexNet layers into logical blocks"""
    return {
        'block1': ['features.0', 'features.3', 'features.6', 'features.8', 'features.10'],
        'block_fc': ['classifier.1']
    }


def get_cifar_resnet32_blocks():
    """Distribute CifarResNet20 layers into logical residual blocks"""
    return {
        'block1': [
            'conv1',
            'layer1.0.conv1', 'layer1.0.conv2',
            'layer1.1.conv1', 'layer1.1.conv2',
            'layer1.2.conv1', 'layer1.2.conv2',
            'layer1.3.conv1', 'layer1.3.conv2',
            'layer1.4.conv1', 'layer1.4.conv2',
        ],
        'block2': [
            'layer2.0.conv1', 'layer2.0.conv2',
            'layer2.0.shortcut.0',
            'layer2.1.conv1', 'layer2.1.conv2',
            'layer2.2.conv1', 'layer2.2.conv2',
            'layer2.3.conv1', 'layer2.3.conv2',
            'layer2.4.conv1', 'layer2.4.conv2',
        ],
        'block3': [
            'layer3.0.conv1', 'layer3.0.conv2',
            'layer3.0.shortcut.0',
            'layer3.1.conv1', 'layer3.1.conv2',
            'layer3.2.conv1', 'layer3.2.conv2',
            'layer3.3.conv1', 'layer3.3.conv2',
            'layer3.4.conv1', 'layer3.4.conv2',
        ],
        'block_fc': ['linear']
    }


def get_vgg16_layer_names():
    """Get important layer names from VGG16"""
    return [
        'features.0',
        'features.3',
        'features.7',
        'features.10',
        'features.14',
        'features.17',
        'features.20',
        'features.24',
        'features.27',
        'features.30',
        'features.34',
        'features.37',
        'features.40',
        'classifier',
    ]


def get_vgg_block():
    """distribute layers into blocks"""
    return {
        'block1': ['features.0', 'features.3'],
        'block2': ['features.7', 'features.10'],
        'block3': ['features.14', 'features.17', 'features.20'],
        'block4': ['features.24', 'features.27', 'features.30'],
        'block5': ['features.34', 'features.37', 'features.40'],
        'block_fc': ['classifier']
    }


# def get_densenet121_layer_names():
#     """Get important layer names from DenseNet121"""
#     return [
#         'features.conv0',
#         'features.norm0',
#         'features.denseblock1.denselayer1',
#         'features.denseblock1.denselayer2',
#         'features.denseblock1.denselayer6',
#         'features.transition1.conv',
#         'features.denseblock2.denselayer1',
#         'features.denseblock2.denselayer12',
#         'features.transition2.conv',
#         'features.denseblock3.denselayer1',
#         'features.denseblock3.denselayer24',
#         'features.transition3.conv',
#         'features.denseblock4.denselayer1',
#         'features.denseblock4.denselayer16',
#         'features.norm5',
#         'classifier'  # FC layer
#     ]

def get_densenet_block():
    """distribute layers into blocks"""
    return {
        'conv1': ['conv1'],
        'block1': [
            # dense1 conv layers
            'dense1.0.conv1', 'dense1.0.conv2',
            'dense1.1.conv1', 'dense1.1.conv2',
            'dense1.2.conv1', 'dense1.2.conv2',
            'dense1.3.conv1', 'dense1.3.conv2',
            'dense1.4.conv1', 'dense1.4.conv2',
            'dense1.5.conv1', 'dense1.5.conv2',
            # trans1 conv layer
            'trans1.conv',
        ],
        'block2': [
            # dense2 conv layers
            'dense2.0.conv1', 'dense2.0.conv2',
            'dense2.1.conv1', 'dense2.1.conv2',
            'dense2.2.conv1', 'dense2.2.conv2',
            'dense2.3.conv1', 'dense2.3.conv2',
            'dense2.4.conv1', 'dense2.4.conv2',
            'dense2.5.conv1', 'dense2.5.conv2',
            'dense2.6.conv1', 'dense2.6.conv2',
            'dense2.7.conv1', 'dense2.7.conv2',
            'dense2.8.conv1', 'dense2.8.conv2',
            'dense2.9.conv1', 'dense2.9.conv2',
            'dense2.10.conv1', 'dense2.10.conv2',
            'dense2.11.conv1', 'dense2.11.conv2',
            # trans2 conv layer
            'trans2.conv',
        ],
        'block3': [
            # dense3 conv layers
            'dense3.0.conv1', 'dense3.0.conv2',
            'dense3.1.conv1', 'dense3.1.conv2',
            'dense3.2.conv1', 'dense3.2.conv2',
            'dense3.3.conv1', 'dense3.3.conv2',
            'dense3.4.conv1', 'dense3.4.conv2',
            'dense3.5.conv1', 'dense3.5.conv2',
            'dense3.6.conv1', 'dense3.6.conv2',
            'dense3.7.conv1', 'dense3.7.conv2',
            'dense3.8.conv1', 'dense3.8.conv2',
            'dense3.9.conv1', 'dense3.9.conv2',
            'dense3.10.conv1', 'dense3.10.conv2',
            'dense3.11.conv1', 'dense3.11.conv2',
            'dense3.12.conv1', 'dense3.12.conv2',
            'dense3.13.conv1', 'dense3.13.conv2',
            'dense3.14.conv1', 'dense3.14.conv2',
            'dense3.15.conv1', 'dense3.15.conv2',
            'dense3.16.conv1', 'dense3.16.conv2',
            'dense3.17.conv1', 'dense3.17.conv2',
            'dense3.18.conv1', 'dense3.18.conv2',
            'dense3.19.conv1', 'dense3.19.conv2',
            'dense3.20.conv1', 'dense3.20.conv2',
            'dense3.21.conv1', 'dense3.21.conv2',
            'dense3.22.conv1', 'dense3.22.conv2',
            'dense3.23.conv1', 'dense3.23.conv2',
            # trans3 conv layer
            'trans3.conv',
        ],
        'block4': [
            # dense4 conv layers
            'dense4.0.conv1', 'dense4.0.conv2',
            'dense4.1.conv1', 'dense4.1.conv2',
            'dense4.2.conv1', 'dense4.2.conv2',
            'dense4.3.conv1', 'dense4.3.conv2',
            'dense4.4.conv1', 'dense4.4.conv2',
            'dense4.5.conv1', 'dense4.5.conv2',
            'dense4.6.conv1', 'dense4.6.conv2',
            'dense4.7.conv1', 'dense4.7.conv2',
            'dense4.8.conv1', 'dense4.8.conv2',
            'dense4.9.conv1', 'dense4.9.conv2',
            'dense4.10.conv1', 'dense4.10.conv2',
            'dense4.11.conv1', 'dense4.11.conv2',
            'dense4.12.conv1', 'dense4.12.conv2',
            'dense4.13.conv1', 'dense4.13.conv2',
            'dense4.14.conv1', 'dense4.14.conv2',
            'dense4.15.conv1', 'dense4.15.conv2',
        ],
        'linear': ['linear']
    }


# def get_efficientnet_b0_layer_names():
#     """Get important layer names from EfficientNet-B0"""
#     return [
#         'features.0.0',      # Conv stem
#         'features.1.0.block.0',  # First MBConv
#         'features.2.0.block.0',
#         'features.3.0.block.0',
#         'features.4.0.block.0',
#         'features.5.0.block.0',
#         'features.6.0.block.0',
#         'features.7.0.block.0',
#         'features.7.0.block.2',  # Last MBConv block
#         'features.8.0',      # Head conv
#         'classifier.1'       # Final FC layer
#     ]

def get_effnetb0_layer_names():
    """Get key EfficientNet-B0 layer names, grouped by blocks with same output channels, excluding SE layers"""
    return {
        'block1': ['conv1'],

        # Output: 16
        'block2': ['layers.0.conv1', 'layers.0.conv2', 'layers.0.se.se1', 'layers.0.se.se2', 'layers.0.conv3'],

        # Output: 24
        'block3': ['layers.1.conv1', 'layers.1.conv2', 'layers.1.se.se1', 'layers.1.se.se2', 'layers.1.conv3',
                   'layers.2.conv1', 'layers.2.conv2', 'layers.2.se.se1', 'layers.2.se.se2', 'layers.2.conv3'],

        # Output: 40
        'block4': ['layers.3.conv1', 'layers.3.conv2', 'layers.3.se.se1', 'layers.3.se.se2', 'layers.3.conv3',
                   'layers.4.conv1', 'layers.4.conv2', 'layers.4.se.se1', 'layers.4.se.se2', 'layers.4.conv3'],

        # Output: 80
        'block5': ['layers.5.conv1', 'layers.5.conv2', 'layers.5.se.se1', 'layers.5.se.se2', 'layers.5.conv3',
                   'layers.6.conv1', 'layers.6.conv2', 'layers.6.se.se1', 'layers.6.se.se2', 'layers.6.conv3',
                   'layers.7.conv1', 'layers.7.conv2', 'layers.7.se.se1', 'layers.7.se.se2', 'layers.7.conv3'],

        # Output: 112
        'block6': ['layers.8.conv1', 'layers.8.conv2', 'layers.8.se.se1', 'layers.8.se.se2', 'layers.8.conv3',
                   'layers.9.conv1', 'layers.9.conv2', 'layers.9.se.se1', 'layers.9.se.se2', 'layers.9.conv3',
                   'layers.10.conv1', 'layers.10.conv2', 'layers.10.se.se1', 'layers.10.se.se2', 'layers.10.conv3'],

        # Output: 192
        'block7': ['layers.11.conv1', 'layers.11.conv2', 'layers.11.se.se1', 'layers.11.se.se2', 'layers.11.conv3',
                   'layers.12.conv1', 'layers.12.conv2', 'layers.12.se.se1', 'layers.12.se.se2', 'layers.12.conv3',
                   'layers.13.conv1', 'layers.13.conv2', 'layers.13.se.se1', 'layers.13.se.se2', 'layers.13.conv3',
                   'layers.14.conv1', 'layers.14.conv2', 'layers.14.se.se1', 'layers.14.se.se2', 'layers.14.conv3'],

        # Output: 320
        'block8': ['layers.15.conv1', 'layers.15.conv2', 'layers.15.se.se1', 'layers.15.se.se2', 'layers.15.conv3'],

        # Classifier
        'classifier': ['linear']
    }

