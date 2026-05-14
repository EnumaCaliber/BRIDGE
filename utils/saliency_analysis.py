"""
Saliency Analysis for FairPrune-style Parameter Importance

This module implements saliency computation based on second-order derivatives (Hessian)
as described in "FairPrune: Achieving Fairness Through Pruning for Dermatological Disease Diagnosis"
(Wu et al., 2022, https://arxiv.org/abs/2203.02110)

The key idea is to use the second derivative of model parameters to quantify each parameter's
importance with respect to model accuracy for different groups (e.g., different classes, demographics).

Key Features:
- FairPrune saliency formula: S(θ) = H_ii * θ² (Hessian diagonal * parameter squared)
  - H_ii ≈ (∂L/∂θ)² (Fisher Information approximation of Hessian diagonal)
  - Combines curvature (how sensitive loss is) with magnitude (how large parameter is)
- Per-class importance: Measures parameter importance for each class separately
- Layer filtering: `conv_only=True` to compute only for Conv2d/Linear layers (excludes BatchNorm)
- Multiple visualizations: Heatmaps, per-layer plots, mean saliency plots

Usage:
    # Compute saliency for Conv/Linear layers only (recommended for pruning)
    saliency_dict = compute_parameter_saliency_per_class(
        model=model,
        data_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        num_classes=10,
        device='cuda',
        conv_only=True  # Filter to Conv2d and Linear only
    )
    
    # Visualize mean saliency per layer
    stats = visualize_mean_saliency_per_layer(
        saliency_dict=saliency_dict,
        title="Mean Saliency per Layer",
        save_path='./saliency.png'
    )
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from tqdm import tqdm


def compute_parameter_saliency_per_class(model, data_loader, criterion, num_classes=10, 
                                         device='cuda', use_second_order=True, conv_only=False):
    """
    Compute parameter saliency scores for each class using second-order derivatives.
    
    The saliency score measures how important each parameter is for each class's accuracy.
    Higher saliency = more important for that class.
    
    Algorithm (from FairPrune):
    1. For each class c:
       2. For each sample x in class c:
          3. Compute loss L(θ) where θ are model parameters
          4. If use_second_order:
               Compute saliency: s_c(θ_i) = H_ii * θ_i²
               where H_ii ≈ (∂L/∂θ_i)² (Hessian diagonal approximation)
             Else:
               Compute first derivative: s_c(θ_i) = |∂L/∂θ_i|
       5. Aggregate saliency across samples: S_c = mean(s_c)
    
    The formula s = H * θ² comes from Taylor expansion analysis:
    - H_ii measures curvature (how sensitive loss is to parameter changes)
    - θ_i² measures magnitude (how large the parameter value is)
    - Together they estimate the impact of removing this parameter
    
    Args:
        model: Neural network model
        data_loader: DataLoader yielding (inputs, labels)
        criterion: Loss function (e.g., CrossEntropyLoss)
        num_classes: Number of classes in dataset
        device: Device to run computation on
        use_second_order: If True, use H*θ² formula (recommended)
                         If False, use first derivative (gradient magnitude)
        conv_only: If True, only compute saliency for convolutional layers (Conv2d)
                   and fully connected layers (Linear). Excludes BatchNorm, LayerNorm, etc.
    
    Returns:
        saliency_dict: Dict mapping layer_name -> (num_classes, num_params) tensor
                      saliency_dict[layer][c, i] = importance of param i for class c
    """
    model.eval()
    model.to(device)
    
    # Storage for saliency scores per class
    # Structure: {layer_name: {class_id: [list of saliency vectors]}}
    class_saliencies = defaultdict(lambda: defaultdict(list))
    
    print(f"Computing {'second-order' if use_second_order else 'first-order'} saliency scores per class...")
    
    # Process each class separately
    for class_id in range(num_classes):
        print(f"\nProcessing class {class_id}...")
        class_sample_count = 0
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"Class {class_id}")):
            # Filter samples belonging to current class
            mask = labels == class_id
            if not mask.any():
                continue
            
            inputs = inputs[mask].to(device)
            labels = labels[mask].to(device)
            class_sample_count += len(labels)
            
            # Compute loss for this batch
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Compute first derivative (gradient)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=use_second_order)
            
            if use_second_order:
                # Compute second derivative (Hessian diagonal approximation)
                # For each parameter, compute ∂²L/∂θ² ≈ sum of gradients of gradients
                for param, grad in zip(model.parameters(), grads):
                    if grad is None:
                        continue
                    
                    # Get layer name
                    layer_name = get_param_name(model, param)
                    if not layer_name:
                        continue
                    
                    # Filter by layer type if conv_only is True
                    if conv_only and not is_conv_or_linear_param(model, layer_name):
                        continue
                    
                    # Approximate Hessian diagonal: sum(∂g_i/∂θ_i) for each gradient component
                    # We use the squared gradient as an approximation
                    # This is computationally cheaper than full Hessian
                    
                    # For efficiency, we approximate using gradient magnitude
                    # True second derivative would be: ∂²L/∂θ² = ∂(∂L/∂θ)/∂θ
                    # Approximation: H_ii ≈ g_i²  (Fisher Information approximation)
                    hessian_diag = grad.detach().pow(2)
                    
                    # FairPrune formula: Importance = H_ii * θ_i²
                    # This combines curvature (Hessian) with parameter magnitude
                    param_squared = param.detach().pow(2)
                    saliency = (hessian_diag * param_squared).cpu()
                    
                    class_saliencies[layer_name][class_id].append(saliency.flatten())
            else:
                # Use first-order gradient as saliency
                for param, grad in zip(model.parameters(), grads):
                    if grad is None:
                        continue
                    
                    # Get layer name
                    layer_name = get_param_name(model, param)
                    if not layer_name:
                        continue
                    
                    # Filter by layer type if conv_only is True
                    if conv_only and not is_conv_or_linear_param(model, layer_name):
                        continue
                    
                    saliency = grad.detach().abs().cpu()
                    class_saliencies[layer_name][class_id].append(saliency.flatten())
        
        print(f"Class {class_id}: Processed {class_sample_count} samples")
    
    # Aggregate saliencies across batches for each class
    print("\nAggregating saliency scores...")
    saliency_dict = {}
    
    for layer_name, class_dict in class_saliencies.items():
        # Get number of parameters in this layer
        num_params = class_dict[0][0].shape[0]
        
        # Create tensor: (num_classes, num_params)
        saliency_tensor = torch.zeros(num_classes, num_params)
        
        for class_id in range(num_classes):
            if class_id in class_dict and len(class_dict[class_id]) > 0:
                # Average saliency across all batches for this class
                class_saliency = torch.stack(class_dict[class_id]).mean(dim=0)
                saliency_tensor[class_id] = class_saliency
        
        saliency_dict[layer_name] = saliency_tensor
    
    return saliency_dict


def get_param_name(model, param):
    """Get the name of a parameter in the model."""
    for name, p in model.named_parameters():
        if p is param:
            return name
    return None


def is_conv_or_linear_param(model, param_name):
    """
    Check if a parameter belongs to a Conv2d or Linear layer.
    
    Args:
        model: Neural network model
        param_name: Name of the parameter (e.g., 'features.0.weight')
    
    Returns:
        True if parameter belongs to Conv2d or Linear layer, False otherwise
    """
    # Extract module name from parameter name (remove '.weight' or '.bias')
    if param_name.endswith('.weight') or param_name.endswith('.bias'):
        module_name = param_name.rsplit('.', 1)[0]
    else:
        module_name = param_name
    
    # Get the module
    try:
        module_dict = dict(model.named_modules())
        module = module_dict.get(module_name)
        
        if module is None:
            return False
        
        # Check if it's Conv2d or Linear
        return isinstance(module, (nn.Conv2d, nn.Linear))
    except:
        return False


def visualize_saliency_distribution(saliency_dict, layer_names=None, class_names=None,
                                   title="Saliency Distribution", save_path=None,
                                   figsize=(12, 8), style='heatmap'):
    """
    Visualize saliency distribution similar to FairPrune Figure 3.
    
    The visualization shows how parameter importance (saliency) differs across classes.
    Each row represents a class (or group), and bright regions indicate high-importance parameters.
    
    Args:
        saliency_dict: Output from compute_parameter_saliency_per_class()
                      Dict mapping layer_name -> (num_classes, num_params) tensor
        layer_names: List of layer names to visualize (if None, use all)
        class_names: List of class names for labeling
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        style: 'heatmap' (FairPrune-style) or 'line' (line plot per class)
    
    Creates visualization like FairPrune Figure 3:
    - Top panel: Shows computation flow
    - Bottom panel: Saliency heatmap for different groups
    """
    if layer_names is None:
        layer_names = list(saliency_dict.keys())
    
    # Filter to requested layers
    layer_names = [name for name in layer_names if name in saliency_dict]
    
    if not layer_names:
        print("No valid layers found in saliency_dict")
        return
    
    # Concatenate saliencies from all layers
    # Structure: (num_classes, total_params)
    all_saliencies = []
    layer_boundaries = [0]  # Track where each layer starts
    
    for layer_name in layer_names:
        saliency = saliency_dict[layer_name]
        all_saliencies.append(saliency)
        layer_boundaries.append(layer_boundaries[-1] + saliency.shape[1])
    
    # Concatenate along parameter dimension
    saliency_matrix = torch.cat(all_saliencies, dim=1).numpy()  # (num_classes, total_params)
    num_classes, num_params = saliency_matrix.shape
    
    # Normalize saliency to [0, 1] for better visualization
    saliency_matrix = (saliency_matrix - saliency_matrix.min()) / (saliency_matrix.max() - saliency_matrix.min() + 1e-8)
    
    if style == 'heatmap':
        # FairPrune-style heatmap visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(saliency_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        
        # Set labels
        if class_names is not None:
            ax.set_yticks(range(num_classes))
            ax.set_yticklabels(class_names)
        else:
            ax.set_ylabel('Class ID')
        
        ax.set_xlabel('Parameter Index')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Saliency (Normalized)', rotation=270, labelpad=20)
        
        # Add layer boundaries as vertical lines
        for boundary in layer_boundaries[1:-1]:
            ax.axvline(x=boundary, color='cyan', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Add layer labels at the top
        ax2 = ax.twiny()
        layer_centers = [(layer_boundaries[i] + layer_boundaries[i+1]) / 2 
                        for i in range(len(layer_boundaries)-1)]
        ax2.set_xticks(layer_centers)
        ax2.set_xticklabels([name.split('.')[-1] for name in layer_names], 
                           rotation=45, ha='left', fontsize=8)
        ax2.set_xlim(ax.get_xlim())
        
    elif style == 'line':
        # Line plot showing saliency distribution for each class
        fig, ax = plt.subplots(figsize=figsize)
        
        for class_id in range(num_classes):
            label = class_names[class_id] if class_names else f'Class {class_id}'
            ax.plot(saliency_matrix[class_id], label=label, alpha=0.7)
        
        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('Saliency Score')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add layer boundaries
        for boundary in layer_boundaries[1:-1]:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved saliency visualization to {save_path}")
    
    plt.show()


def visualize_saliency_per_layer(saliency_dict, layer_name, class_names=None,
                                 title=None, save_path=None, figsize=(10, 6)):
    """
    Visualize saliency distribution for a single layer across classes.
    
    Args:
        saliency_dict: Output from compute_parameter_saliency_per_class()
        layer_name: Name of the layer to visualize
        class_names: List of class names for labeling
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if layer_name not in saliency_dict:
        print(f"Layer {layer_name} not found in saliency_dict")
        return
    
    saliency = saliency_dict[layer_name].numpy()  # (num_classes, num_params)
    num_classes, num_params = saliency.shape
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: Heatmap
    im = ax1.imshow(saliency, aspect='auto', cmap='hot', interpolation='nearest')
    
    if class_names is not None:
        ax1.set_yticks(range(num_classes))
        ax1.set_yticklabels(class_names)
    else:
        ax1.set_ylabel('Class ID')
    
    ax1.set_xlabel('Parameter Index')
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title(f'Saliency Distribution for Layer: {layer_name}')
    
    plt.colorbar(im, ax=ax1, label='Saliency')
    
    # Bottom: Line plot overlay
    for class_id in range(num_classes):
        label = class_names[class_id] if class_names else f'Class {class_id}'
        ax2.plot(saliency[class_id], label=label, alpha=0.7)
    
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Saliency')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer saliency visualization to {save_path}")
    
    plt.show()


def compute_saliency_statistics(saliency_dict, class_names=None):
    """
    Compute statistics about saliency distributions across classes.
    
    This helps identify which parameters have the most divergent importance
    across different classes/groups - key for fairness analysis.
    
    Args:
        saliency_dict: Output from compute_parameter_saliency_per_class()
        class_names: List of class names for labeling
    
    Returns:
        stats_dict: Dictionary containing:
            - 'mean_per_class': Mean saliency for each class
            - 'std_per_class': Std of saliency for each class
            - 'max_difference': Max saliency difference between any two classes
            - 'fairness_score': How evenly distributed saliency is across classes
    """
    print(f"\n{'='*60}")
    print("Saliency Statistics Across Classes")
    print(f"{'='*60}\n")
    
    stats_dict = {}
    
    for layer_name, saliency in saliency_dict.items():
        saliency_np = saliency.numpy()
        num_classes, num_params = saliency_np.shape
        
        # Per-class statistics
        mean_per_class = saliency_np.mean(axis=1)  # (num_classes,)
        std_per_class = saliency_np.std(axis=1)    # (num_classes,)
        
        # Cross-class statistics
        # For each parameter, compute difference between max and min class saliency
        param_max_diff = saliency_np.max(axis=0) - saliency_np.min(axis=0)  # (num_params,)
        avg_max_diff = param_max_diff.mean()
        
        # Fairness score: lower is more fair (less variation across classes)
        # Computed as coefficient of variation across classes
        fairness_score = mean_per_class.std() / (mean_per_class.mean() + 1e-8)
        
        stats_dict[layer_name] = {
            'mean_per_class': mean_per_class,
            'std_per_class': std_per_class,
            'avg_max_diff': avg_max_diff,
            'fairness_score': fairness_score
        }
        
        # Print summary
        print(f"Layer: {layer_name}")
        print(f"  Shape: {saliency_np.shape}")
        print(f"  Avg Max Difference: {avg_max_diff:.6f}")
        print(f"  Fairness Score: {fairness_score:.6f} (lower = more fair)")
        
        for class_id in range(num_classes):
            class_label = class_names[class_id] if class_names else f"Class {class_id}"
            print(f"    {class_label}: mean={mean_per_class[class_id]:.6f}, "
                  f"std={std_per_class[class_id]:.6f}")
        print()
    
    return stats_dict


def identify_biased_parameters(saliency_dict, threshold_percentile=90, layer_names=None):
    """
    Identify parameters that show high saliency difference across classes.
    
    These parameters are "biased" - they are much more important for some classes
    than others. In FairPrune, these are candidates for pruning.
    
    Args:
        saliency_dict: Output from compute_parameter_saliency_per_class()
        threshold_percentile: Consider top X% most different parameters as biased
        layer_names: List of layer names to analyze (if None, use all)
    
    Returns:
        biased_params: Dict mapping layer_name -> list of (param_idx, max_diff, class_with_max)
    """
    if layer_names is None:
        layer_names = list(saliency_dict.keys())
    
    biased_params = {}
    
    print(f"\n{'='*60}")
    print(f"Identifying Biased Parameters (top {100-threshold_percentile}%)")
    print(f"{'='*60}\n")
    
    for layer_name in layer_names:
        if layer_name not in saliency_dict:
            continue
        
        saliency = saliency_dict[layer_name].numpy()  # (num_classes, num_params)
        
        # Compute saliency difference for each parameter
        # max_diff[i] = max saliency - min saliency for parameter i
        max_saliency = saliency.max(axis=0)  # (num_params,)
        min_saliency = saliency.min(axis=0)  # (num_params,)
        saliency_diff = max_saliency - min_saliency  # (num_params,)
        
        # Find class with maximum saliency for each parameter
        class_with_max = saliency.argmax(axis=0)  # (num_params,)
        
        # Threshold: top X% most different parameters
        threshold = np.percentile(saliency_diff, threshold_percentile)
        biased_indices = np.where(saliency_diff >= threshold)[0]
        
        # Store results
        biased_list = []
        for idx in biased_indices:
            biased_list.append({
                'param_idx': int(idx),
                'max_diff': float(saliency_diff[idx]),
                'class_with_max_saliency': int(class_with_max[idx]),
                'max_saliency': float(max_saliency[idx]),
                'min_saliency': float(min_saliency[idx])
            })
        
        # Sort by max_diff descending
        biased_list.sort(key=lambda x: x['max_diff'], reverse=True)
        biased_params[layer_name] = biased_list
        
        print(f"Layer: {layer_name}")
        print(f"  Total parameters: {len(saliency_diff)}")
        print(f"  Biased parameters (>{threshold_percentile}th percentile): {len(biased_list)}")
        print(f"  Threshold: {threshold:.6f}")
        if biased_list:
            print(f"  Top 3 most biased:")
            for i, param_info in enumerate(biased_list[:3]):
                print(f"    {i+1}. Param {param_info['param_idx']}: "
                      f"diff={param_info['max_diff']:.6f}, "
                      f"most important for class {param_info['class_with_max_saliency']}")
        print()
    
    return biased_params


def visualize_fairprune_style(saliency_dict, layer_names=None, class_names=None,
                              groups=None, save_path=None, figsize=(14, 8)):
    """
    Create FairPrune Figure 3 style visualization with both light/dark group comparison.
    
    Args:
        saliency_dict: Output from compute_parameter_saliency_per_class()
        layer_names: List of layer names to visualize
        class_names: List of class names
        groups: Dict mapping group_name -> list of class indices
                e.g., {'light': [0, 1, 2, 5], 'dark': [3, 4, 6, 7, 8, 9]}
        save_path: Path to save figure
        figsize: Figure size
    """
    if layer_names is None:
        layer_names = list(saliency_dict.keys())
    
    # Concatenate saliencies
    all_saliencies = []
    layer_boundaries = [0]
    
    for layer_name in layer_names:
        if layer_name in saliency_dict:
            saliency = saliency_dict[layer_name]
            all_saliencies.append(saliency)
            layer_boundaries.append(layer_boundaries[-1] + saliency.shape[1])
    
    saliency_matrix = torch.cat(all_saliencies, dim=1).numpy()
    
    # Normalize
    saliency_matrix = (saliency_matrix - saliency_matrix.min()) / \
                     (saliency_matrix.max() - saliency_matrix.min() + 1e-8)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[20, 1],
                         hspace=0.3, wspace=0.05)
    
    # Top: Computation diagram (placeholder)
    ax_diagram = fig.add_subplot(gs[0, :])
    ax_diagram.text(0.5, 0.5, 
                   'Saliency Computation:\n'
                   '(1) Compute second-order derivatives for each class\n'
                   '(2) Aggregate saliency scores across samples',
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax_diagram.axis('off')
    
    if groups is not None:
        # Middle & Bottom: Separate heatmaps for each group
        for i, (group_name, class_indices) in enumerate(groups.items()):
            ax = fig.add_subplot(gs[i+1, 0])
            cax = fig.add_subplot(gs[i+1, 1])
            
            # Extract saliency for this group
            group_saliency = saliency_matrix[class_indices, :]
            
            # Plot heatmap
            im = ax.imshow(group_saliency, aspect='auto', cmap='hot', 
                          interpolation='nearest', vmin=0, vmax=1)
            
            ax.set_ylabel(f'{group_name.capitalize()} Group')
            ax.set_xlabel('Parameter Index')
            
            # Add layer boundaries
            for boundary in layer_boundaries[1:-1]:
                ax.axvline(x=boundary, color='cyan', linestyle='--', 
                          linewidth=1, alpha=0.7)
            
            # Add colorbar
            plt.colorbar(im, cax=cax, label='Saliency')
            
            # Add class labels
            if class_names is not None:
                yticks = range(len(class_indices))
                yticklabels = [class_names[idx] for idx in class_indices]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
    else:
        # Single heatmap for all classes
        ax = fig.add_subplot(gs[1:, 0])
        cax = fig.add_subplot(gs[1:, 1])
        
        im = ax.imshow(saliency_matrix, aspect='auto', cmap='hot',
                      interpolation='nearest', vmin=0, vmax=1)
        
        ax.set_ylabel('Class')
        ax.set_xlabel('Parameter Index')
        
        for boundary in layer_boundaries[1:-1]:
            ax.axvline(x=boundary, color='cyan', linestyle='--',
                      linewidth=1, alpha=0.7)
        
        plt.colorbar(im, cax=cax, label='Saliency')
        
        if class_names is not None:
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names)
    
    plt.suptitle('FairPrune-style Saliency Distribution', fontsize=14, y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved FairPrune-style visualization to {save_path}")
    
    plt.show()


def visualize_mean_saliency_per_layer(saliency_dict, layer_names=None, 
                                       title="Mean Saliency Score per Layer",
                                       save_path=None, figsize=(12, 6),
                                       plot_std=True, log_scale=False):
    """
    Visualize mean saliency scores (without normalization) for each layer.
    
    This function computes and plots the average saliency across all weights
    in each layer, providing insight into which layers have the most important
    parameters overall.
    
    Args:
        saliency_dict: Output from compute_parameter_saliency_per_class()
                      Dict mapping layer_name -> (num_classes, num_params) tensor
        layer_names: List of layer names to visualize (if None, use all)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size (width, height)
        plot_std: If True, plot error bars showing standard deviation
        log_scale: If True, use log scale for y-axis (useful for large range)
    
    Returns:
        layer_stats: Dict with layer names as keys and statistics as values
                    {'layer_name': {'mean': float, 'std': float, 'min': float, 'max': float}}
    """
    if layer_names is None:
        layer_names = list(saliency_dict.keys())
    
    # Filter to requested layers
    layer_names = [name for name in layer_names if name in saliency_dict]
    
    if not layer_names:
        print("No valid layers found in saliency_dict")
        return None
    
    # Compute statistics for each layer
    layer_stats = {}
    mean_saliencies = []
    std_saliencies = []
    
    for layer_name in layer_names:
        saliency = saliency_dict[layer_name]  # (num_classes, num_params)
        
        # Compute mean across all classes and all parameters
        mean_sal = saliency.mean().item()
        std_sal = saliency.std().item()
        min_sal = saliency.min().item()
        max_sal = saliency.max().item()
        
        mean_saliencies.append(mean_sal)
        std_saliencies.append(std_sal)
        
        layer_stats[layer_name] = {
            'mean': mean_sal,
            'std': std_sal,
            'min': min_sal,
            'max': max_sal,
            'num_params': saliency.shape[1]
        }
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(layer_names))
    
    if plot_std:
        # Bar plot with error bars
        bars = ax.bar(x_pos, mean_saliencies, yerr=std_saliencies, 
                     capsize=5, alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # Color bars by magnitude
        colors = plt.cm.viridis(np.array(mean_saliencies) / max(mean_saliencies))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    else:
        # Simple bar plot
        bars = ax.bar(x_pos, mean_saliencies, alpha=0.7, 
                     edgecolor='black', linewidth=1.2)
        
        # Color bars by magnitude
        colors = plt.cm.viridis(np.array(mean_saliencies) / max(mean_saliencies))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    # Set labels and title
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Saliency Score (Unnormalized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Shorten layer names for display
    display_names = [name.split('.')[-1] if len(name) > 15 else name 
                    for name in layer_names]
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Use log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Mean Saliency Score (Unnormalized, Log Scale)', fontsize=12)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars (only if not too many layers)
    if len(layer_names) <= 20:
        for i, (bar, mean_val) in enumerate(zip(bars, mean_saliencies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.2e}' if mean_val < 0.01 else f'{mean_val:.4f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved mean saliency per layer plot to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*70}")
    print("Mean Saliency Statistics per Layer (Unnormalized)")
    print(f"{'='*70}")
    print(f"{'Layer':<30} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'#Params':>10}")
    print(f"{'-'*70}")
    
    for layer_name in layer_names:
        stats = layer_stats[layer_name]
        print(f"{layer_name:<30} "
              f"{stats['mean']:>12.6f} "
              f"{stats['std']:>12.6f} "
              f"{stats['min']:>12.6f} "
              f"{stats['max']:>12.6f} "
              f"{stats['num_params']:>10d}")
    
    print(f"{'='*70}\n")
    
    return layer_stats


# Example usage
if __name__ == "__main__":
    print("FairPrune Saliency Analysis Module")
    print("=" * 60)
    print("\nThis module implements parameter saliency computation using")
    print("second-order derivatives (Hessian) as described in:")
    print("'FairPrune: Achieving Fairness Through Pruning for")
    print("Dermatological Disease Diagnosis' (Wu et al., 2022)")
    print("\nKey functions:")
    print("  1. compute_parameter_saliency_per_class()")
    print("  2. visualize_saliency_distribution()")
    print("  3. visualize_saliency_per_layer()")
    print("  4. compute_saliency_statistics()")
    print("  5. identify_biased_parameters()")
    print("  6. visualize_fairprune_style()")
    print("  7. visualize_mean_saliency_per_layer()  [NEW]")
