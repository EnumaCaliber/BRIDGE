import torch, os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from models.model_loader import model_loader

MODELS = {
    "vgg16": {
        "pretrained":  "vgg16/checkpoint/pretrain_vgg16_ckpt.pth",
        "pruned-high": "vgg16/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth",
        "pruned-low":  "vgg16/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9424.pth",
    },
}

TAGS       = ["pretrained", "pruned-high", "pruned-low"]
COL_LABELS = ["Baseline", "High Sparsity", "Low Sparsity"]
COL_COLORS = ["steelblue", "tomato", "orange"]

HIGHLIGHT_COLOR = np.array([0xf7/255, 0xeb/255, 0xc6/255], dtype=np.float32)  # #F7EBC6 米黄 异号
SAME_SIGN_COLOR = np.array([0x9b/255, 0xb8/255, 0x9c/255], dtype=np.float32)  # #9BB89C 鼠尾草绿 同号
POSITIVE_COLOR  = np.array([0.0, 0.0, 0.0], dtype=np.float32)                 # 黑色 正值
NEGATIVE_COLOR  = np.array([1.0, 1.0, 1.0], dtype=np.float32)                 # 白色 负值
ZERO_COLOR      = np.array([0.5, 0.5, 0.5], dtype=np.float32)                 # 灰色 零值

IMG_IDX  = 0
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES    = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
OUT_BASE = "feature_comparison"


# ── model loading ─────────────────────────────────────────────────────────────
def load_pruned(ckpt_path, device, model_name):
    sd = torch.load(ckpt_path, map_location=device)
    merged, done = {}, set()
    for k in sd:
        if k in done:
            continue
        if k.endswith("_orig"):
            base = k[:-5]
            merged[base] = sd[k] * sd[base + "_mask"]
            done.update([k, base + "_mask"])
        elif not k.endswith("_mask"):
            merged[k] = sd[k]
    model = model_loader(model_name, device)
    model.load_state_dict(merged)
    return model.eval()


def load_pretrained(ckpt_path, device, model_name):
    model = model_loader(model_name, device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    return model.eval()


# ── hooks ─────────────────────────────────────────────────────────────────────
def make_hook(name, storage, order):
    def fn(module, inp, out):
        storage[name] = out.detach()
        if name not in order:
            order.append(name)
    return fn


def register_hooks(model, storage, order):
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(name, storage, order)))
    return hooks


def extract_features(model, img, device):
    storage, order = {}, []
    hooks = register_hooks(model, storage, order)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    for h in hooks:
        h.remove()
    return storage, order, pred


# ── helpers ───────────────────────────────────────────────────────────────────
def all_tags_present(feat_dict):
    return all(t in feat_dict for t in TAGS)


def safe_name(layer_name):
    return layer_name.replace(".", "_")


def make_dirs(base, model_name):
    dirs = {}
    for tag in TAGS:
        d = os.path.join(base, model_name, tag)
        os.makedirs(d, exist_ok=True)
        dirs[tag] = d
    return dirs


def make_highlight_mask(base: torch.Tensor, pruned: torch.Tensor) -> torch.Tensor:
    return (base * pruned) < 0


def to_sign_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.cpu().float().numpy()
    out = np.full((*arr.shape, 3), 0.5, dtype=np.float32)
    out[arr > 0] = POSITIVE_COLOR
    out[arr < 0] = NEGATIVE_COLOR
    return out


def apply_highlight(base: torch.Tensor, pruned: torch.Tensor) -> np.ndarray:
    mask = make_highlight_mask(base, pruned).cpu().numpy()
    out  = np.where(mask[..., None], HIGHLIGHT_COLOR, SAME_SIGN_COLOR)
    return out.astype(np.float32)


def add_legend(ax, tag):
    if tag == "pretrained":
        handles = [
            mpatches.Patch(facecolor=POSITIVE_COLOR, label='positive', edgecolor='lightgray'),
            mpatches.Patch(facecolor=NEGATIVE_COLOR, label='negative', edgecolor='lightgray'),
            mpatches.Patch(facecolor=ZERO_COLOR,     label='zero'),
        ]
    else:
        handles = [
            mpatches.Patch(facecolor=SAME_SIGN_COLOR, label='same sign as baseline'),
            mpatches.Patch(facecolor=HIGHLIGHT_COLOR, label='opposite sign'),
        ]
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.75, handlelength=1.2)


# ── per-layer per-tag saves ───────────────────────────────────────────────────
def save_conv_per_tag(feat_dict, layer_name, dirs, max_ch=64):
    if not all_tags_present(feat_dict):
        return
    feats = {t: feat_dict[t][0] for t in TAGS}
    if feats["pretrained"].dim() != 3:
        return

    total = min(feats["pretrained"].shape[0], max_ch)
    n     = int(total ** 0.5)
    fname = safe_name(layer_name) + ".pdf"
    base  = feats["pretrained"]
    h, w  = base[0].shape[-2:]

    for tag, label, color in zip(TAGS, COL_LABELS, COL_COLORS):
        pruned = feats[tag]
        rows   = []
        for r in range(n):
            row_imgs = []
            for c in range(n):
                ch = r * n + c
                if tag == "pretrained":
                    row_imgs.append(to_sign_rgb(base[ch]))
                else:
                    row_imgs.append(apply_highlight(base[ch], pruned[ch]))
            rows.append(np.concatenate(row_imgs, axis=1))
        grid = np.concatenate(rows, axis=0)

        changed_pct = None
        if tag != "pretrained":
            flat_mask   = make_highlight_mask(base[:total].reshape(-1),
                                              pruned[:total].reshape(-1))
            changed_pct = flat_mask.float().mean().item() * 100

        fig_w = max(w * n / 50, 4) + 0.3
        fig_h = max(h * n / 50, 4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(grid, interpolation='nearest', aspect='equal')
        ax.axis('off')
        title = f"{label} — {layer_name}"
        if changed_pct is not None:
            title += f"\n{changed_pct:.1f}% pixels flipped sign"
        ax.set_title(title, fontsize=8, color=color)
        add_legend(ax, tag)
        plt.tight_layout()
        plt.savefig(os.path.join(dirs[tag], fname), bbox_inches="tight", dpi=150)
        plt.close()


def save_linear_per_tag(feat_dict, layer_name, dirs):
    if not all_tags_present(feat_dict):
        return
    feats = {t: feat_dict[t][0] for t in TAGS}
    if feats["pretrained"].dim() != 1:
        return

    fname = safe_name(layer_name) + ".pdf"
    base  = feats["pretrained"]
    g_min = min(feats[t].min().item() for t in TAGS)
    g_max = max(feats[t].max().item() for t in TAGS)

    for tag, label, color in zip(TAGS, COL_LABELS, COL_COLORS):
        pruned = feats[tag]
        norm   = ((pruned - g_min) / (g_max - g_min + 1e-8)).cpu().numpy()

        if tag == "pretrained":
            def sign_color(v):
                if v > 0: return tuple(POSITIVE_COLOR)
                if v < 0: return tuple(NEGATIVE_COLOR)
                return tuple(ZERO_COLOR)
            bar_colors  = [sign_color(v) for v in pruned.cpu().numpy()]
            changed_pct = None
        else:
            mask        = make_highlight_mask(base, pruned).cpu().numpy()
            bar_colors  = [tuple(HIGHLIGHT_COLOR) if m else tuple(SAME_SIGN_COLOR)
                           for m in mask]
            changed_pct = mask.mean() * 100

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.bar(range(len(norm)), norm, color=bar_colors, width=1.0, edgecolor='none')
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0, color='k', linewidth=0.4)
        ax.tick_params(labelbottom=False)
        title = f"{label} — {layer_name}"
        if changed_pct is not None:
            title += f"\n{changed_pct:.1f}% neurons flipped sign"
        ax.set_title(title, fontsize=8, color=color)
        add_legend(ax, tag)
        plt.tight_layout()
        plt.savefig(os.path.join(dirs[tag], fname), bbox_inches="tight", dpi=150)
        plt.close()


# ── combined multi-layer figure ───────────────────────────────────────────────
def plot_combined(records, save_path):
    """
    records: list of dict with keys:
        layer_name, dense_map (H,W), flip_maps [(H,W,3), ...], pcts [float, ...]
    Rows = layers, cols = [dense sign | pruned-high flip | pruned-low flip]
    """
    compare_tags  = TAGS[1:]          # skip pretrained
    compare_labels = COL_LABELS[1:]
    compare_colors = COL_COLORS[1:]
    n_cols = 1 + len(compare_tags)
    n_rows = len(records)
    if n_rows == 0:
        return

    cell_h, cell_w = 2.2, 2.5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_w, n_rows * cell_h),
                             squeeze=False)

    for row, rec in enumerate(records):
        lname     = rec["layer_name"]
        dense_map = rec["dense_map"]    # (H, W) sign values in [-1,0,1]
        flip_maps = rec["flip_maps"]    # list of (H, W, 3) RGB
        pcts      = rec["pcts"]         # list of float

        # col 0: dense sign map
        ax0 = axes[row][0]
        ax0.imshow(dense_map, cmap='gray', vmin=-1, vmax=1,
                   interpolation='nearest', aspect='auto')
        ax0.set_ylabel(lname, fontsize=6, rotation=0,
                       labelpad=70, va='center')
        ax0.axis('off')
        if row == 0:
            ax0.set_title("Baseline", fontsize=8, color=COL_COLORS[0])

        # cols 1+: flip maps
        for col_idx, (rgb, pct, label, color) in enumerate(
                zip(flip_maps, pcts, compare_labels, compare_colors)):
            ax = axes[row][col_idx + 1]
            ax.imshow(rgb, interpolation='nearest', aspect='auto')
            ax.axis('off')
            title = f"{pct:.1f}% pixels flipped sign"
            if row == 0:
                title = f"{label}\n{title}"
            ax.set_title(title, fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Combined sign-flip figure → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def run_comparison(model_name, ckpts, img, device):
    dirs = make_dirs(OUT_BASE, model_name)

    models = {
        "pretrained":  load_pretrained(ckpts["pretrained"],  device, model_name),
        "pruned-high": load_pruned(ckpts["pruned-high"], device, model_name),
        "pruned-low":  load_pruned(ckpts["pruned-low"],  device, model_name),
    }

    features, orders = {}, {}
    for tag, model in models.items():
        storage, order, pred = extract_features(model, img, device)
        features[tag] = storage
        orders[tag]   = order
        print(f"[{model_name}/{tag}] pred: {NAMES[pred]}")

    compare_tags = TAGS[1:]
    records = []   # for combined figure

    for layer_name in orders["pretrained"]:
        feat_dict = {t: features[t][layer_name]
                     for t in TAGS if layer_name in features[t]}
        if not all_tags_present(feat_dict):
            continue

        sample = feat_dict["pretrained"][0]
        if sample.dim() == 3:
            save_conv_per_tag(feat_dict, layer_name, dirs)

            # collect mean-channel sign map for combined figure
            base = feat_dict["pretrained"][0]           # (C, H, W)
            dense_mean = base.mean(0)                   # (H, W)
            dense_sign = torch.sign(dense_mean).cpu().numpy()

            flip_maps, pcts = [], []
            for tag in compare_tags:
                pruned      = feat_dict[tag][0].mean(0)
                mask        = ((dense_mean * pruned) < 0).cpu().numpy()
                rgb         = np.where(mask[..., None],
                                       HIGHLIGHT_COLOR, SAME_SIGN_COLOR)
                flip_maps.append(rgb.astype(np.float32))
                pcts.append(mask.mean() * 100)

            records.append(dict(layer_name=layer_name,
                                dense_map=dense_sign,
                                flip_maps=flip_maps,
                                pcts=pcts))

        elif sample.dim() == 1:
            save_linear_per_tag(feat_dict, layer_name, dirs)

    # combined figure (conv layers only)
    if records:
        combined_path = os.path.join(OUT_BASE, model_name, "sign_flip_combined.pdf")
        plot_combined(records, combined_path)

    print(f"[{model_name}] saved to {OUT_BASE}/{model_name}/")


if __name__ == "__main__":
    dataset = datasets.CIFAR10(
        "./data", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    )
    img, label = dataset[IMG_IDX]
    print(f"[INFO] image {IMG_IDX}, label: {NAMES[label]}")

    os.makedirs(OUT_BASE, exist_ok=True)
    raw = np.clip(
        img.permute(1, 2, 0).numpy() * [.247, .2435, .2616]
        + [.4914, .4822, .4465], 0, 1
    )
    plt.figure(figsize=(3, 3))
    plt.imshow(raw)
    plt.axis('off')
    plt.savefig(os.path.join(OUT_BASE, "original.pdf"), bbox_inches="tight")
    plt.close()

    for model_name, ckpts in MODELS.items():
        print(f"\n{'=' * 50}\nmodel: {model_name}\n{'=' * 50}")
        run_comparison(model_name, ckpts, img, DEVICE)
