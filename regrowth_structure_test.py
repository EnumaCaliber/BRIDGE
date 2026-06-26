"""
test_regrow.py
python test_regrow.py --m_name resnet20 --layer conv1 --dense_idx 2 5 8
"""
import torch
import torch.nn as nn
import torch_pruning as tp
import argparse
import copy

# ═══════════════════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument('--m_name',    type=str, default='resnet20')
parser.add_argument('--ckpt',      type=str,
                    default='./structured_rl_ckpts/resnet20/structured_iterative/iter0_sp0.8750/iter1_ep48_rwd+0.0385.pth')
parser.add_argument('--dense_ckpt', type=str,
                    default='./resnet20/ckpt_structured_iterative/dense_resnet20.pth')
parser.add_argument('--layer',     type=str, default='conv1',
                    help='target conv layer name to regrow channels')
parser.add_argument('--dense_idx', type=int, nargs='+', default=[2, 5, 8],
                    help='dense indices to regrow')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ═══════════════════════════════════════════════════════════════════════════════
# Load checkpoints
# ═══════════════════════════════════════════════════════════════════════════════

ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
net            = ckpt['model'].to(device)
index_map      = ckpt['index_map']        # {layer_name: [survived_dense_idx, ...]}
pruned_dense_map = ckpt['pruned_dense_map']  # {layer_name: [pruned_dense_idx, ...]}

dense_model = torch.load(args.dense_ckpt, map_location=device, weights_only=False)
dense_model = dense_model.to(device)
dense_model.eval()

example_inputs = torch.randn(1, 3, 32, 32).to(device)

# ═══════════════════════════════════════════════════════════════════════════════
# Pre-flight checks
# ═══════════════════════════════════════════════════════════════════════════════

assert args.layer in pruned_dense_map, \
    f"Layer [{args.layer}] not found in pruned_dense_map. " \
    f"Available: {list(pruned_dense_map.keys())}"

available = pruned_dense_map[args.layer]
for idx in args.dense_idx:
    assert idx in available, \
        f"dense index {idx} is not in the pruned list {available} of layer [{args.layer}]"

# ShuffleNet: channel split requires even num_grow
if 'shuffle' in args.m_name.lower():
    assert len(args.dense_idx) % 2 == 0, \
        "ShuffleNet uses channel split — num_grow must be even"

print(f"\nTarget layer   : {args.layer}")
print(f"Dense indices  : {args.dense_idx}")
print(f"Survived so far: {index_map[args.layer]}")

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: extract specific channels from dense model
# ═══════════════════════════════════════════════════════════════════════════════

def _get_dense_weight(dense_model, layer_name, dense_idx, dim):
    """
    Extract specific channel slices from the dense model weight tensor.

    Args:
        dense_model : original unpruned model
        layer_name  : dot-separated module name (e.g. 'layer1.0.conv1')
        dense_idx   : list of channel indices in the dense model
        dim         : 0 → out_channels axis,  1 → in_channels axis

    Returns:
        Tensor of shape [len(dense_idx), ...] or None if layer not found.
    """
    named = dict(dense_model.named_modules())
    if layer_name not in named:
        return None
    w = named[layer_name].weight.data
    idx_t = torch.tensor(dense_idx, device=w.device)
    return w.index_select(dim, idx_t)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: regrow_channels
# ═══════════════════════════════════════════════════════════════════════════════

def regrow_channels(model, dense_model, example_inputs,
                    target_conv_name, dense_idx_to_grow,
                    index_map, pruned_dense_map, device):
    """
    Regrow pruned channels back into the model using DependencyGraph.

    Weights are initialized from the corresponding dense model channels
    (falls back to zeros / identity values when unavailable).

    Supported layer types
    ─────────────────────
    Conv2d  depthwise   groups == in_channels == out_channels
    Conv2d  out         expanding out_channels
    Conv2d  in          expanding in_channels
    BatchNorm2d         num_features expansion
    Linear  in          expanding in_features  (VGG/EfficientNet head)
    Linear  out         expanding out_features (SE excitation)

    Args:
        model               : pruned model (modified in-place)
        dense_model         : original dense model (read-only reference)
        example_inputs      : single-sample tensor for DG tracing
        target_conv_name    : name of the conv layer to regrow
        dense_idx_to_grow   : list of dense channel indices to restore
        index_map           : {layer: [survived_dense_idx]}  ← updated in-place
        pruned_dense_map    : {layer: [pruned_dense_idx]}    ← updated in-place
        device              : torch device string
    """
    named_modules = dict(model.named_modules())
    if target_conv_name not in named_modules:
        raise KeyError(f"Layer [{target_conv_name}] not found in model")

    target_conv = named_modules[target_conv_name]
    num_grow    = len(dense_idx_to_grow)

    # ── Build DependencyGraph and get the coupled group
    DG    = tp.DependencyGraph().build_dependency(model, example_inputs)
    group = DG.get_pruning_group(
        target_conv, tp.prune_conv_out_channels, idxs=[0]
    )

    print("\nConnected layers in dependency group:")

    # Track depthwise layers already processed to avoid double-expansion
    already_processed = set()

    for dep, _ in group:
        layer   = dep.target.module
        handler = dep.handler
        name    = dep.target.name

        # ── Case 1: Conv2d — depthwise (groups == in_channels == out_channels)
        #
        # Must be checked BEFORE the regular out/in cases.
        # DG emits both prune_conv_out_channels and prune_conv_in_channels for
        # the same depthwise layer, so we guard with already_processed.
        if not list(layer.parameters(recurse=False)):
            print(f"  ·       [{name}]: {type(layer).__name__} (无参数)")
            continue

        if (isinstance(layer, nn.Conv2d)
                and layer.groups > 1
                and layer.groups == layer.in_channels):

            if name in already_processed:
                continue
            already_processed.add(name)

            old_w   = layer.weight.data              # [C, 1, kH, kW]
            dense_w = _get_dense_weight(dense_model, name, dense_idx_to_grow, dim=0)
            new_w   = dense_w if dense_w is not None else \
                      torch.zeros(num_grow, 1, *old_w.shape[2:], device=device)

            layer.weight      = nn.Parameter(torch.cat([old_w, new_w], dim=0))
            layer.out_channels += num_grow
            layer.in_channels  += num_grow
            layer.groups       += num_grow

            if layer.bias is not None:
                dense_layer = dict(dense_model.named_modules()).get(name)
                new_b = (dense_layer.bias.data[dense_idx_to_grow]
                         if dense_layer is not None and dense_layer.bias is not None
                         else torch.zeros(num_grow, device=device))
                layer.bias = nn.Parameter(torch.cat([layer.bias.data, new_b]))

            print(f"  ✓ conv dw   [{name}]: out/in/groups → {layer.out_channels}")

        # ── Case 2: Conv2d — regular, expanding out_channels
        elif (isinstance(layer, nn.Conv2d)
              and handler == tp.prune_conv_out_channels):

            old_w   = layer.weight.data              # [out, in, kH, kW]
            dense_w = _get_dense_weight(dense_model, name, dense_idx_to_grow, dim=0)

            if dense_w is not None:
                # Align in_channels: sparse in_ch may be < dense in_ch
                min_in = min(old_w.shape[1], dense_w.shape[1])
                new_w  = torch.zeros(num_grow, old_w.shape[1],
                                     *old_w.shape[2:], device=device)
                new_w[:, :min_in] = dense_w[:, :min_in]
            else:
                new_w = torch.zeros(num_grow, old_w.shape[1],
                                    *old_w.shape[2:], device=device)

            layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=0))
            layer.out_channels += num_grow

            if layer.bias is not None:
                dense_layer = dict(dense_model.named_modules()).get(name)
                new_b = (dense_layer.bias.data[dense_idx_to_grow]
                         if dense_layer is not None and dense_layer.bias is not None
                         else torch.zeros(num_grow, device=device))
                layer.bias = nn.Parameter(torch.cat([layer.bias.data, new_b]))

            print(f"  ✓ conv out  [{name}]: {old_w.shape[0]} → {layer.out_channels}")

        # ── Case 3: Conv2d — regular, expanding in_channels
        elif (isinstance(layer, nn.Conv2d)
              and handler == tp.prune_conv_in_channels):

            old_w   = layer.weight.data              # [out, in, kH, kW]
            dense_w = _get_dense_weight(dense_model, name, dense_idx_to_grow, dim=1)

            if dense_w is not None:
                # Align out_channels: sparse out_ch may be < dense out_ch
                min_out = min(old_w.shape[0], dense_w.shape[0])
                new_w   = torch.zeros(old_w.shape[0], num_grow,
                                      *old_w.shape[2:], device=device)
                new_w[:min_out] = dense_w[:min_out]
            else:
                new_w = torch.zeros(old_w.shape[0], num_grow,
                                    *old_w.shape[2:], device=device)

            layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=1))
            layer.in_channels += num_grow

            print(f"  ✓ conv in   [{name}]: {old_w.shape[1]} → {layer.in_channels}")

        # ── Case 4: BatchNorm2d
        elif isinstance(layer, nn.BatchNorm2d):
            old_ch   = layer.num_features
            dense_bn = dict(dense_model.named_modules()).get(name)

            # Learnable affine params
            for attr, default_fill in [('weight', 1.0), ('bias', 0.0)]:
                old = getattr(layer, attr).data
                new = (getattr(dense_bn, attr).data[dense_idx_to_grow]
                       if dense_bn is not None
                       else torch.full((num_grow,), default_fill, device=device))
                setattr(layer, attr, nn.Parameter(torch.cat([old, new])))

            # Running statistics
            for attr, default_fill in [('running_mean', 0.0), ('running_var', 1.0)]:
                old = getattr(layer, attr)
                new = (getattr(dense_bn, attr)[dense_idx_to_grow]
                       if dense_bn is not None
                       else torch.full((num_grow,), default_fill, device=device))
                setattr(layer, attr, torch.cat([old, new]))

            layer.num_features += num_grow
            print(f"  ✓ bn        [{name}]: {old_ch} → {layer.num_features}")

        # ── Case 5: Linear — expanding in_features
        # Triggered when the last conv before a flatten feeds into a Linear
        # (VGG16 classifier, EfficientNet/ShuffleNet head, etc.)
        elif (isinstance(layer, nn.Linear)
              and handler == tp.prune_linear_in_channels):

            old_w        = layer.weight.data          # [out_features, in_features]
            dense_linear = dict(dense_model.named_modules()).get(name)

            if dense_linear is not None:
                new_w   = dense_linear.weight.data[:, dense_idx_to_grow]
                min_out = min(old_w.shape[0], new_w.shape[0])
                pad_w   = torch.zeros(old_w.shape[0], num_grow, device=device)
                pad_w[:min_out] = new_w[:min_out]
                new_w = pad_w
            else:
                new_w = torch.zeros(old_w.shape[0], num_grow, device=device)

            layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=1))
            layer.in_features += num_grow

            print(f"  ✓ linear in [{name}]: {old_w.shape[1]} → {layer.in_features}")

        # ── Case 6: Linear — expanding out_features
        # Triggered inside SE blocks (EfficientNet) where the squeeze Linear
        # output feeds into the excitation Linear whose width == channel count.
        elif (isinstance(layer, nn.Linear)
              and handler == tp.prune_linear_out_channels):

            old_w        = layer.weight.data          # [out_features, in_features]
            dense_linear = dict(dense_model.named_modules()).get(name)

            if dense_linear is not None:
                new_w   = dense_linear.weight.data[dense_idx_to_grow]
                min_in  = min(old_w.shape[1], new_w.shape[1])
                pad_w   = torch.zeros(num_grow, old_w.shape[1], device=device)
                pad_w[:, :min_in] = new_w[:, :min_in]
                new_w = pad_w
            else:
                new_w = torch.zeros(num_grow, old_w.shape[1], device=device)

            layer.weight = nn.Parameter(torch.cat([old_w, new_w], dim=0))
            layer.out_features += num_grow

            if layer.bias is not None:
                dense_linear = dict(dense_model.named_modules()).get(name)
                new_b = (dense_linear.bias.data[dense_idx_to_grow]
                         if dense_linear is not None and dense_linear.bias is not None
                         else torch.zeros(num_grow, device=device))
                layer.bias = nn.Parameter(torch.cat([layer.bias.data, new_b]))

            print(f"  ✓ linear out[{name}]: {old_w.shape[0]} → {layer.out_features}")

        # ── Fallback: unhandled combination → hard fail so nothing is silently skipped
        else:
            raise NotImplementedError(
                f"Unhandled combination — "
                f"layer type: '{type(layer).__name__}', "
                f"handler: '{handler}', "
                f"name: [{name}]. "
                f"Please add a case for this."
            )

    # ── Update bookkeeping maps (in-place)
    index_map[target_conv_name] = sorted(
        index_map[target_conv_name] + list(dense_idx_to_grow)
    )
    pruned_dense_map[target_conv_name] = sorted(
        set(pruned_dense_map[target_conv_name]) - set(dense_idx_to_grow)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

net_copy           = copy.deepcopy(net)
index_map_copy     = copy.deepcopy(index_map)
pruned_map_copy    = copy.deepcopy(pruned_dense_map)

regrow_channels(
    model            = net_copy,
    dense_model      = dense_model,
    example_inputs   = example_inputs,
    target_conv_name = args.layer,
    dense_idx_to_grow= args.dense_idx,
    index_map        = index_map_copy,
    pruned_dense_map = pruned_map_copy,
    device           = device,
)

# ── Verify forward pass
with torch.no_grad():
    out = net_copy(example_inputs)
print(f"\nForward OK — output shape: {out.shape}")

# ── Print channel counts after regrow
print("\nChannel counts after regrow:")
for name, m in net_copy.named_modules():
    if isinstance(m, nn.Conv2d):
        print(f"  {name}: out={m.out_channels}  in={m.in_channels}  groups={m.groups}")
    elif isinstance(m, nn.Linear):
        print(f"  {name}: in={m.in_features}  out={m.out_features}")

print(f"\n[{args.layer}] survived dense indices : {index_map_copy[args.layer]}")
print(f"[{args.layer}] remaining pruned indices: {pruned_map_copy[args.layer]}")