import torch, os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from models.model_loader import model_loader

MODELS = {
    "effnet": {
        "pretrained": "effnet/checkpoint/pretrain_effnet_ckpt.pth",
        "pruned":     "effnet/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth",
    },
}

SAME_COLOR = np.array([0x9b/255, 0xb8/255, 0x9c/255], dtype=np.float32)  # #9BB89C 同号
FLIP_COLOR = np.array([0xf7/255, 0xeb/255, 0xc6/255], dtype=np.float32)  # #F7EBC6 异号

IMG_IDX     = 0
NUM_BATCHES = 4
BATCH_SIZE  = 64
MAX_CH      = 16
LAYERS_PER_FIG = 20
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES       = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
OUT_BASE    = "comparison_results"


# ── model loading ─────────────────────────────────────────────────────────────
def load_pretrained(ckpt_path, device, model_name):
    model = model_loader(model_name, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['net'])
    return model.eval()


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


def extract_single(model, img, device):
    storage, order = {}, []
    hooks = register_hooks(model, storage, order)
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    for h in hooks:
        h.remove()
    return storage, order, pred


def extract_batched(model, loader, device, num_batches):
    storage, order, hooks = {}, [], []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            def _fn(mod, inp, out, n=name):
                if n not in storage:
                    storage[n] = []
                    order.append(n)
                t = out.detach()
                if t.dim() == 4:          # (N,C,H,W) → (N,C) via GAP
                    t = t.mean(dim=(2, 3))
                storage[n].append(t.cpu())
            hooks.append(m.register_forward_hook(_fn))
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(x.to(device))
    for h in hooks:
        h.remove()
    return {n: torch.cat(ts, dim=0) for n, ts in storage.items()}, order


# ── CKA ──────────────────────────────────────────────────────────────────────
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    dot_XY = (X.T @ Y).norm(p='fro').pow(2)
    dot_XX = (X.T @ X).norm(p='fro')
    dot_YY = (Y.T @ Y).norm(p='fro')
    denom  = dot_XX * dot_YY
    return (dot_XY / denom).item() if denom > 1e-10 else 0.0


def flatten(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.size(0), -1).float()


# ── visualisation helpers ─────────────────────────────────────────────────────
def make_sign_grid(feat: torch.Tensor, max_ch: int = MAX_CH) -> np.ndarray:
    """(C,H,W) → tiled channel grid, black=positive / white=negative."""
    total = min(feat.shape[0], max_ch)
    n     = int(total ** 0.5)
    tiles = []
    for i in range(n * n):
        ch  = feat[i].cpu().float().numpy()
        rgb = np.full((*ch.shape, 3), 0.5, dtype=np.float32)
        rgb[ch > 0] = 0.0   # black
        rgb[ch < 0] = 1.0   # white
        tiles.append(rgb)
    rows = [np.concatenate(tiles[r*n:(r+1)*n], axis=1) for r in range(n)]
    return np.concatenate(rows, axis=0)


def make_flip_grid(base: torch.Tensor, pruned: torch.Tensor,
                   max_ch: int = MAX_CH) -> tuple:
    """Returns (grid np.ndarray, flip_pct float)."""
    total = min(base.shape[0], max_ch)
    n     = int(total ** 0.5)
    tiles = []
    flip_count, total_px = 0, 0
    for i in range(n * n):
        b = base[i].cpu().float().numpy()
        p = pruned[i].cpu().float().numpy()
        flip = (b * p) < 0
        rgb  = np.where(flip[..., None], FLIP_COLOR, SAME_COLOR)
        tiles.append(rgb.astype(np.float32))
        flip_count += flip.sum()
        total_px   += flip.size
    rows = [np.concatenate(tiles[r*n:(r+1)*n], axis=1) for r in range(n)]
    grid = np.concatenate(rows, axis=0)
    return grid, 100.0 * flip_count / max(total_px, 1)


# ── combined figure ───────────────────────────────────────────────────────────
def plot_combined(records, save_path):
    """
    rows = conv layers
    col 0: pretrained sign map  (black/white)
    col 1: pruned flip map      (green/cream) + flip% + CKA in title
    """
    n_rows = len(records)
    if n_rows == 0:
        return

    gh, gw = records[0]["sign_grid"].shape[:2]
    cell_h = max(2.0, gh / 40)
    cell_w = max(2.0, gw / 40)

    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(2 * (cell_w + 0.2), n_rows * (cell_h + 0.4)),
                             squeeze=False)

    for row, rec in enumerate(records):
        lname     = rec["layer_name"]
        sign_grid = rec["sign_grid"]
        flip_grid = rec["flip_grid"]
        flip_pct  = rec["flip_pct"]
        cka       = rec["cka"]

        ax0 = axes[row][0]
        ax0.imshow(sign_grid, interpolation='nearest', aspect='equal')
        ax0.axis('off')
        ax0.set_ylabel(lname, fontsize=6, rotation=0, labelpad=65, va='center')
        if row == 0:
            ax0.set_title("Baseline", fontsize=8, color="steelblue")

        ax1 = axes[row][1]
        ax1.imshow(flip_grid, interpolation='nearest', aspect='equal')
        ax1.axis('off')
        title = f"{flip_pct:.1f}% flipped  |  CKA={cka:.3f}"
        if row == 0:
            title = f"Pruned\n{title}"
        ax1.set_title(title, fontsize=7, color="tomato")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Combined figure → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def run(model_name, ckpts, img, loader, device):
    out_dir = os.path.join(OUT_BASE, model_name)
    os.makedirs(out_dir, exist_ok=True)

    model_pre    = load_pretrained(ckpts["pretrained"], device, model_name)
    model_pruned = load_pruned(ckpts["pruned"],         device, model_name)
    print(f"Models loaded.")

    # single-image features for visualisation
    vis_pre,    order, pred_pre    = extract_single(model_pre,    img, device)
    vis_pruned, _,     pred_pruned = extract_single(model_pruned, img, device)
    print(f"  pretrained pred: {NAMES[pred_pre]}  |  pruned pred: {NAMES[pred_pruned]}")

    # batched features for CKA
    print(f"Extracting batched features ({NUM_BATCHES} batches)…")
    batch_pre,    _ = extract_batched(model_pre,    loader, device, NUM_BATCHES)
    batch_pruned,  _ = extract_batched(model_pruned, loader, device, NUM_BATCHES)

    records = []
    for layer_name in order:
        f_pre    = vis_pre.get(layer_name)
        f_pruned = vis_pruned.get(layer_name)
        if f_pre is None or f_pruned is None:
            continue

        feat_pre    = f_pre[0]     # (C, H, W) or (C,)
        feat_pruned = f_pruned[0]

        if feat_pre.dim() != 3:    # skip linear layers
            continue

        sign_grid             = make_sign_grid(feat_pre)
        flip_grid, flip_pct   = make_flip_grid(feat_pre, feat_pruned)

        # CKA from batched features
        bd = batch_pre.get(layer_name)
        bp = batch_pruned.get(layer_name)
        cka = linear_cka(flatten(bd), flatten(bp)) if (bd is not None and bp is not None) else 0.0

        print(f"  {layer_name}: flip={flip_pct:.1f}%  CKA={cka:.4f}")
        records.append(dict(layer_name=layer_name,
                            sign_grid=sign_grid, flip_grid=flip_grid,
                            flip_pct=flip_pct, cka=cka))

    for chunk_i, start in enumerate(range(0, len(records), LAYERS_PER_FIG)):
        chunk = records[start:start + LAYERS_PER_FIG]
        path  = os.path.join(out_dir, f"combined_{chunk_i:02d}.pdf")
        plot_combined(chunk, path)

    print(f"[{model_name}] done → {out_dir}/")


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    dataset = datasets.CIFAR10("./data", train=False, download=True,
                               transform=transform)
    img, label = dataset[IMG_IDX]
    print(f"[INFO] image {IMG_IDX}  label: {NAMES[label]}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=4)
    os.makedirs(OUT_BASE, exist_ok=True)

    for model_name, ckpts in MODELS.items():
        print(f"\n{'=' * 50}\nmodel: {model_name}\n{'=' * 50}")
        run(model_name, ckpts, img, loader, DEVICE)
