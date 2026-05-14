# Breaking the Compression Barrier: Device-Aware Boundary Estimation via Reverse Regrowth
This folder contains code and experiment artifacts for both **one-shot magnitude pruning(structure and unstructure)**, **iterative magnitude pruning(structure and unstructure)**. We implement our **regrowth(for both structured and unstructured)** strategy applying to models and evaluated on the CIFAR-10 and Tiny-imagenet dataset.

The four main regrowth implementations are:

- **regrowth_iterative.py, regrowth_oneshot.py for unstructure prune**
- **regrowth_structure_oneshot.py, regrowth_structure_iterative.py for unstructure prune** 



---

## What “regrowth” means in this repo

You start from a **highly sparse** (pruned) network with masks (PyTorch pruning reparameterization):

- Each prunable module has
  - `weight_orig` (trainable dense weights)
  - `weight_mask` (binary mask, 1=active, 0=pruned)
  - effective weight is `weight = weight_orig * weight_mask`

**Regrowth** updates the mask by turning some pruned connections back on:

- pick $K$ indices among currently pruned weights (`weight_mask == 0`)
- set `weight_mask[idx] = 1`
- Initialize the newly regrown weights (copy from reference, zeros, Kaiming, etc.)
- optionally mini-finetune / finetune to recover accuracy

Regrowth is usually done to move from **98% sparsity → 97% sparsity** (i.e., regrow 1% of all weights).

---

## Folder Overview

### Training/pruning
- `pretrain.py`, `prune_stucture.py`，`prune_unstucture.py`: CIFAR10 pretraining + iterative pruning loop (saves pruned checkpoints)
- `models/model_loader.py`: constructs models (`resnet20`, `vgg16`, `effnet`, …)
- `data/data_loader.py`: CIFAR-10 dataloaders
- `utils/analysis_tools.py`: pruning reparam helpers, SSIM feature extraction, mask stats

### Regrowth
- `regrowth_iterative.py`, `regrowth_oneshot.py`: for regrowth unstructure prune.
- `regrowth_structure_oneshot.py`: for regrowth structure prune.

---

## Setup

### Environment
Please check `requirements.txt` for details, and you can use `pip install -r requirements.txt` to install all required packages.

### Data
`data/data_loader.py` uses `data_dir='./data'` by default, which will download the required dataset if it is not provided.

---

## Workflow overview

Most experiments follow this pipeline:

1. **Pretrain** a dense model
2. **Prune + finetune** to a target sparsity (e.g., 0.99)
3. **Regrow** some weights (e.g., 2% of total, **adjustable**) to a less sparse target (0.97)
4. **Finetune selectively or fully** to recover accuracy

Artifacts are usually saved under:
- `./{model_name}/checkpoint/ckpt.pth` (dense / baseline)
- `./{model_name}/iterative/{model_name}_{sparsity}.pth` (sparse checkpoint for iterative)
- `./{model_name}/oneshot/{model_name}_{sparsity}.pth` (sparse checkpoint for oneshot)
- For iterative regrowth: After regrowth, the final best model will be saved to `{save_dir}/{m_name}/{method}/iter_{iter_idx}/best_grown_model.pth.`
- For oneshot regrowth: After regrowth, the final best model will be saved to `{save_dir}/{m_name}/oneshot/{model_sparsity}/baseline_exceeded_ep{epoch+1}_acc{accuracy:.2f}.pth.`

---

### 1) Pretraining + Pruning

Pretraining and pruning are combined in `main.py`.

Key args (from `main.py`):
- `--m_name`: `resnet20`, `vgg16`, `effnet`, ...,
- If your dataset is tiny-imagenet. --`m_name` should be: VGGTinyImageNet, EfficientNetB0TinyImageNet...
- And you can define your model in `./models/model_loader.py.`
- `--pruner`: pruning method (passed into `weight_pruner_loader(args.pruner)`)
- `--iter_start`, `--iter_end`: pruning iterations

### 2) Regrowth methods

- Implemented in `regrowth_iterative.py` for iterative regrowth and `regrowth_oneshot.py` for oneshot regrowth
- There is some difference in iterative regrowth and one-shot regrowth
- In iterative regrowth, you can use `--budget_space_size`,` --min_budget_frac`, and `--max_budget_frac` to define how much sparsity regrowth occurs in each iteration.
- In oneshot regrowth, you can use `--regrow_step` to set how much sparsity you need for regrowth.
- In both oneshot regrowth and iterative regrowth, you can use `--ssim_threshold` to set the threshold for SSIM. If ssim_threshold = 0.1, the process will choose the layer for which SSIM <= 0.1 for search.


Core ideas:
1. **Layer priority** is computed once using **SSIM** between feature maps from:
   - pretrained model (`./{m_name}/checkpoint/ckpt.pth`)
   - pruned model (typically `pruned_finetuned_mask_0.99.pth`)

   Lower SSIM → more feature drift → higher priority for regrowth.

2. **Allocation**: an LSTM controller samples an allocation ratio per target layer.
3. **Selection**: within each layer, select weights to regrow using **saliency sore**.
4. **(Mini)Finetune** to evaluate reward.

$$\text{saliency}(\theta_i) \approx \left(\frac{\partial L}{\partial \theta_i}\right)^2 \cdot \theta_i^2$$

This is a Fisher/Hessian-diagonal approximation plus magnitude scaling.

5. `SaliencyBasedRegrowth.apply_regrowth(...)` selects the top-K pruned weights by saliency and updates the mask.
6. Newly regrown weights can be initialized via `--init_strategy`.
---

### 3) Growth metrics (what to report)

This repo contains several “metrics” used to analyze regrowth and results.
#### A. Final regrowth model sparsity and accuracy test
- Use `test_unstructure_regrowth_model.py` to test the final regrowth model sparsity and accuracy
#### B. Sparsity and regrowth budget

`utils/analysis_tools.py` exposes `count_pruned_params(model)` which reports:
- total parameters in prunable layers
- surviving (unmasked) parameters
- pruned parameters

Common derived metrics:
- **Sparsity**: $s = \frac{\text{pruned}}{\text{total}}$
- **Regrowth budget** (global):
  $$K = (s_{\text{start}} - s_{\text{target}})\cdot N_{\text{total}}$$


#### C. Layer capacity
When regrowing, each layer has a **capacity**:

- capacity(layer) = number of currently pruned weights = `sum(weight_mask == 0)`

This bounds how many weights you can regrow in that layer.

#### D. SSIM-based “growth metric” / priority
In the reference-based RL approach, each layer gets an SSIM score computed from features.

- Feature extraction: `BlockwiseFeatureExtractor` (in `utils/analysis_tools.py`)
- Similarity: `compute_block_ssim(features_pretrained, features_pruned)`

Interpretation:
- **Lower SSIM** ⇒ larger feature drift from baseline ⇒ layer is more damaged by pruning ⇒ regrowing there is more likely to help.

`single_layer_regrowth_analysis.py` is designed to measure the correlation between:
- SSIM(layer)
- accuracy improvement when regrowing only that layer

#### E. Saliency-based importance
1. `utils/saliency_analysis.py` provides a more general “per-class” saliency analyzer:
   - supports second-order approximation (`use_second_order=True`)
   - can compute per-class importance tensors and visualize distributions

Interpretation:
- higher saliency ⇒ parameter is important for loss/accuracy
- regrow highest-saliency weights among currently pruned positions (RigL-inspired)

## Quickstart recipes

### 0. Produce the required sparse checkpoint(s)
Most regrowth and benchmark scripts assume the 99% sparse checkpoint exists:

```bash
python main.py --m_name resnet20 --pruner magnitude --iter_end 1
```

### 1. Run a single-episode timing comparison

```bash
python quick_benchmark.py --m_name resnet20
```

### 2. Test whether SSIM actually predicts “good layers to regrow”

`single_layer_regrowth_analysis.py` applies the *full* global regrowth budget to one layer at a time, then finetunes and measures the recovered accuracy.

```bash
python single_layer_regrowth_analysis.py --m_name resnet20
```

### 3. Compute and visualize FairPrune-style saliency (analysis-only)

Use `utils/saliency_analysis.py` if you want diagnostic plots (per-layer / per-class distributions). This is separate from the RL saliency regrowth code.

---
## Experiment results (VGG16, CIFAR-10)
### PART1: Regrowth

The table below summarizes the VGG16 results from your tracking sheet.

**Setting A: Iterative magnitude pruning (10 iterations)**

| Sparsity (%) | Test acc (mean±std) |
|---:|---:|
| 40.00 | 92.09 ± 0.28 |
| 64.00 | 92.43 ± 0.12 |
| 78.40 | 92.60 ± 0.15 |
| 87.04 | 92.68 ± 0.18 |
| 92.22 | 92.68 ± 0.19 |
| 95.33 | 92.49 ± 0.26 |
| 97.20 | 92.35 ± 0.02 |
| 98.32 | 91.92 ± 0.09 |
| 98.99 | 91.01 ± 0.05 |
| 99.40 | 90.31 ± 0.18 |

**Setting B: One-shot magnitude pruning**

| Sparsity (%) | Test acc (mean±std) |
|---:|---:|
| 95.0 | 92.43 ± 0.10 |
| 96.0 | 92.41 ± 0.22 |
| 97.0 | 92.27 ± 0.06 |
| 98.0 | 91.93 ± 0.17 |
| 99.0 | 91.03 ± 0.17 |
| 99.5 | 89.53 ± 0.10 |
| 99.9 | 81.08 ± 0.85 |

### Regrowth Outcome (start from 99% sparsity)

| Baseline | Step | Result | Improvement |
|---:|---:|---:|---:|
| 91.93 | step1 | 92.33 | +0.40 |
| 92.27 | step2 | 92.41 | +0.14 |
| 92.41 | step3 | 92.57 | +0.16 |
| 92.43 | step4 | 92.73 | +0.30 |

---
### PART2: Growth Metric Evaluation

Below is a **single-episode** timing breakdown (VGG16 @ 99% sparsity, regrow $K\approx 147{,}155$ weights, CUDA).

### Weight selection time comparison

| Method | Weight selection total (s) | Notes |
|---|---:|---|
| Reference-based (SSIM + reference masks/weights) | 0.0178 | Includes loading reference masks/weights (0.0012s) and selecting 147,148 weights (0.0166s) |
| Saliency-based (gradient saliency) | 0.0041 | Selects 147,148 weights by direct saliency ranking |

In this run, saliency-based selection is $\approx 4.38\times$ faster (absolute savings: 0.0137s). 


## Citation

If you use this codebase, results, or figures in your work, please refer:
```bibtex
@article{liu2025beyond,
  title={Beyond One-Way Pruning: Bidirectional Pruning-Regrowth for Extreme Accuracy-Sparsity Tradeoff},
  author={Liu, Junchen and Sheng, Yi},
  journal={arXiv preprint arXiv:2511.11675},
  year={2025}
}
```

We also build on the public magnitude-pruning implementation from LAMP (ICLR 2021):
```bibtex
@article{lee2020layer,
  title={Layer-adaptive sparsity for the magnitude-based pruning},
  author={Lee, Jaeho and Park, Sejun and Mo, Sangwoo and Ahn, Sungsoo and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2010.07611},
  year={2020}
}
```

