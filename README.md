# MicroCLR: Contrastive Learning for Microbiome Disease Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MicroCLR** is a contrastive self-supervised learning framework for gut microbiome data. It pre-trains a neural encoder using biologically-motivated data augmentations and NT-Xent loss, then evaluates learned representations via linear probing and fine-tuning on four disease classification tasks (CRC, T2D, IBD, PD).

This repository accompanies the paper:

> Guzzi P.H., Perri C.F., Defilippo A., Lomoio U., Veltri P.
> **What Contrastive Learning Can and Cannot Do for Microbiome Disease Classification: A Systematic Augmentation Benchmark**
> *Bioinformatics Advances*, 2026.

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Augmentations](#augmentations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Results Summary](#results-summary)
- [Citation](#citation)

---

## Overview

MicroCLR applies contrastive learning (SimCLR-style) to CLR-transformed gut microbiome profiles. The framework:

1. **Pre-trains** a 3-layer MLP encoder (426→512→256→128) using NT-Xent loss (τ=0.07) with biologically-motivated augmentations
2. **Evaluates** representations via:
   - Within-disease 5-fold stratified CV (linear probe)
   - Leave-One-Cohort-Out (LOCO) generalization
   - Low-label fine-tuning (10%, 25%, 50% of labels)
3. **Benchmarks** against Random Forest, Logistic Regression, XGBoost, SCARF, VAE, and ComBat+RF

Eight augmentation strategies are implemented, including two novel biologically-inspired augmentations:
- **EcologicalAug**: Spearman co-occurrence network-guided community perturbation
- **CohortStyleAug**: Inter-cohort technical variation simulation via mean-shift transfer

---

## Datasets

### Dataset 1: IGC2/HGMA (restricted access)

The HGMA (Human Gut Microbiome Atlas) dataset from the ML4Microbiome course is **not redistributable** under the ML4Microbiome data sharing terms.

**To obtain the data**, contact the ML4Microbiome consortium:
- Aldert Zomer: a.l.zomer@uu.nl
- Website: https://www.ml4microbiome.eu/

**Dataset statistics:**
- 1,286 samples × 426 metagenomic species (MSP) features
- 7 cohorts, 4 disease labels: CRC, T2D, IBD, PD
- Features: CLR (centered log-ratio) transformed relative abundances
- **Note:** Participant identifiers are not available in the provided metadata. IBD results
  on this dataset may be inflated by participant leakage in longitudinal cohorts; they are
  reported separately as exploratory (manuscript Section 3.13) and are excluded from the
  primary comparative analyses.

### Dataset 2: curatedMetagenomicData (open access)

The CMD dataset is publicly available via Bioconductor:

```r
# In R:
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("curatedMetagenomicData")

library(curatedMetagenomicData)
# List available datasets:
curatedMetagenomicData(pattern=".*", dryrun=TRUE)
```

**Dataset statistics:**
- 4,101 samples × 351 MetaPhlAn3 species features
- 16 cohorts across 4 diseases (CRC: 9, IBD: 3, T2D: 3, PD: 1)
- Features: CLR-transformed with pseudocount 1e-6
- HMP_2019_ibdmdb subsampled to 500 samples (stratified by label)
- **Participant identifiers (`subject_id`) are available** in the CMD metadata

Once you have the raw data, use `scripts/preprocess.py` to reproduce the CLR transformation and train/test splits used in this study.

---

## Augmentations

All augmentations are implemented in `augmentations/augmentations_v2.py`.

| Augmentation | Description | Biologically Motivated |
|---|---|---|
| `PhyloAug` | Phylogenetic clade-level abundance smoothing | Yes |
| `CompAug` | Compositional Augmentation (Gaussian noise + re-centering) | Yes |
| `TaxDropAug` | Random taxon dropout | Yes |
| `FeatureMask` | Random feature masking | No |
| `GaussianNoise` | Additive Gaussian noise | No |
| `MixupAug` | Linear interpolation between samples | No |
| `EcologicalAug` | Co-occurrence network-guided perturbation | **Yes (novel)** |
| `CohortStyleAug` | Inter-cohort mean-shift style transfer | **Yes (novel)** |

---

## Repository Structure

```
MicroCLR/
├── README.md
├── LICENSE
├── requirements.txt              # Python dependencies with pinned versions
│
├── augmentations/
│   ├── augmentations_v2.py        # All 8 augmentation classes (main)
│   └── augmentations.py           # Original 6 augmentations (v1, for reference)
│
├── models/
│   ├── microclr_PhyloAug.pt       # Pre-trained encoder checkpoint
│   ├── microclr_CompAug.pt
│   ├── microclr_TaxDropAug.pt
│   ├── microclr_FeatureMask.pt
│   ├── microclr_GaussianNoise.pt
│   ├── microclr_MixupAug.pt
│   ├── microclr_EcologicalAug.pt  # Novel augmentation checkpoint
│   └── microclr_CohortStyleAug.pt # Novel augmentation checkpoint
│
├── data/
│   ├── clade_groups.json          # 88 genus-level clade groups for PhyloAug
│   ├── cooccurrence_network.pkl   # Spearman co-occurrence network (426×426)
│   ├── cohort_means.npy           # Per-cohort mean CLR profiles (7 cohorts)
│   ├── cv_splits_verified.json    # Frozen 5-fold stratified CV splits (seed=42)
│   └── loco_splits_verified.json  # LOCO splits (CRC:2, T2D:2, IBD:2)
│   [NOTE: features_clr_verified.csv.gz and metadata_verified.csv are NOT included
│    for Dataset 1 — contact ML4Microbiome for access.
│    Dataset 2 (CMD) is available via Bioconductor — see above.]
│
├── evaluation/
│   ├── cv_results.csv             # Within-disease 5-fold CV AUROC (all methods)
│   ├── cv_results_novel_augs.csv  # Within-disease CV, novel augmentations
│   ├── loco_results.csv           # LOCO AUROC (all methods)
│   ├── loco_results_novel_augs.csv# LOCO AUROC, novel augmentations
│   ├── lowlabel_results.csv       # Low-label LP AUROC (10/25/50/100%)
│   ├── lowlabel_results_novel_augs.csv
│   ├── ft_cv_results.csv          # Fine-tuning CV results
│   ├── ft_loco_results.csv        # Fine-tuning LOCO results
│   ├── ft_lowlabel_results.csv    # Fine-tuning low-label results
│   ├── ft_lowlabel_errors.csv     # Fine-tuning low-label error log
│   ├── silhouette_scores.csv      # Disease vs. cohort silhouette scores
│   ├── statistical_tests.csv      # Wilcoxon test results
│   ├── pretrain_convergence_novel_augs.csv
│   ├── recentering_results.json   # Re-centering sensitivity (FeatureMask)
│   ├── masking_rate_cv_results.json # Masking-rate sensitivity (FeatureMask)
│   ├── lowlabel_10pct_summary.csv
│   ├── cv_results_cmd.csv         # CMD within-disease CV
│   ├── loco_results_cmd.csv       # CMD LOCO
│   ├── baseline_results_cmd.csv   # CMD baseline results
│   ├── ibdmdb_participant_cv_results.csv # CMD participant-grouped CV (IBD)
│   ├── novel_aug_cv_results.json  # CMD novel-augmentation CV
│   ├── novel_aug_loco_results.json# CMD novel-augmentation LOCO
│   ├── temporal_ablation_results.csv
│   ├── training_losses_cmd.json   # CMD pre-training losses
│   └── cmd/                       # CMD analysis summaries (A2-A6)
│       ├── A2_participant_grouped_cv.json
│       ├── A3_inductive_evaluation.json
│       ├── A4_hyperparameter_tuning.json
│       ├── A5_mixup_alternatives.json
│       └── A6_rf_on_embeddings.json
│
├── figures/
│   ├── fig1_within_cohort_benchmark_v2.{png,svg}   # Fig 1: within-disease heatmap
│   ├── fig3_loco_benchmark_v2.{png,svg}            # Fig 2: IGC2 LOCO heatmap
│   ├── fig_loco_heatmap_cmd_v4.{png,svg}           # Fig 3: CMD LOCO heatmap
│   ├── fig_loco_bar_cmd_v4.{png,svg}               # Fig 4: CMD LOCO bars
│   ├── fig4_lowlabel_curves_v2.{png,svg}           # Fig 5: low-label curves (CRC)
│   ├── fig2_umap_embeddings.{png,svg}              # Fig 6: UMAP
│   ├── fig5_silhouette_scores_v2.{png,svg}         # Fig 7: silhouette
│   ├── fig6_finetuning_comparison_v2.{png,svg}     # Fig 8: LP/FT/RF
│   ├── fig_recentering_sensitivity_v2.{png,svg}    # Fig 9: re-centering
│   ├── fig_masking_sensitivity_v2.{png,svg}        # Fig 10: masking rate
│   ├── figS1_training_curves.{png,svg}             # Fig S1: pre-training loss
│   └── figS2_lowlabel_ibd_v2.{png,svg}             # Fig S2: exploratory IBD low-label
│
├── manuscripts/
│   ├── manuscript_MicroCLR_bioinf_v6.pdf  # Bioinformatics Advances revision 2
│   └── supplementary_microclr_v6.pdf      # Supplementary material
│
└── scripts/
    ├── preprocess.py              # CLR transformation + split generation
    ├── pretrain.py                # Pre-training script (NT-Xent, 150 epochs)
    ├── evaluate_lp.py             # Linear probe evaluation (CV + LOCO)
    ├── ft_lowlabel_experiment.py  # Low-label fine-tuning experiment
    ├── ft_lowlabel_fix.py         # Low-label fine-tuning (10/25/50%)
    ├── make_fig7.py               # Figure generation (legacy)
    ├── run_cmd_analyses.py        # CMD analyses (participant-grouped CV, inductive,
    │                              #  Mixup alternatives, RF-on-embeddings, LOCO/CV)
    ├── run_a4_hyperparameter_tuning.py  # Nested CV hyperparameter tuning (CMD CRC)
    ├── make_fig_cmd_loco.py       # Figures 3-4 (CMD LOCO)
    ├── make_figs_v6.py            # Figures 1, 2, 5, 8
    ├── make_figs_sensitivity_v2.py# Figures 7, 9, 10
    └── make_figS2_ibd_lowlabel.py # Figure S2 (exploratory IBD low-label)
```

---

## Installation

```bash
git clone https://github.com/hguzzi/MicroCLR.git
cd MicroCLR
pip install -r requirements.txt
```

**Requirements** (see `requirements.txt` for pinned versions):
- Python 3.8+
- PyTorch 1.12+
- scikit-learn 1.0+
- numpy, pandas, scipy, matplotlib, seaborn, umap-learn

---

## Quick Start

### Load a pre-trained encoder and extract embeddings

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Define the encoder architecture
class MicroCLREncoder(nn.Module):
    def __init__(self, input_dim=426):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    def forward(self, x):
        return self.encoder(x)

# Load pre-trained checkpoint
model = MicroCLREncoder(input_dim=426)
checkpoint = torch.load("models/microclr_FeatureMask.pt", map_location="cpu")

# Load encoder weights (key is "encoder_state_dict")
model.load_state_dict(checkpoint["encoder_state_dict"])
model.eval()

# Load your CLR-transformed features
# X_clr should be a numpy array of shape (n_samples, 426)
# Replace with your actual data loading:
# feat_df = pd.read_csv("data/features_clr_verified.csv.gz", index_col=0)
# X_clr = feat_df.values.astype(np.float32)
X_clr = np.random.randn(10, 426).astype(np.float32)  # placeholder

# Extract 128-dim embeddings
with torch.no_grad():
    embeddings = model(torch.tensor(X_clr, dtype=torch.float32))
print(f"Embeddings shape: {embeddings.shape}")  # (10, 128)
```

### Run linear probe evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# embeddings: (n_samples, 128), y: disease labels (0/1)
# Split into train/test using the provided CV splits
import json
with open("data/cv_splits_verified.json") as f:
    splits = json.load(f)

disease = "CRC"
fold = 0
train_ids = splits[disease][fold]["train"]
test_ids = splits[disease][fold]["test"]

# Assuming embeddings and y are aligned with sample IDs
# clf = LogisticRegression(max_iter=1000, C=1.0)
# clf.fit(embeddings[train_idx], y[train_idx])
# auroc = roc_auc_score(y[test_idx], clf.predict_proba(embeddings[test_idx])[:, 1])
# print(f"AUROC: {auroc:.3f}")
```

---

## Reproducibility Checklist

### Software versions
- Python 3.8+
- PyTorch 1.12+ (CPU or CUDA)
- scikit-learn 1.0+
- numpy 1.21+, pandas 1.3+, scipy 1.7+
- matplotlib 3.5+, seaborn 0.12+, umap-learn 0.5+

See `requirements.txt` for exact pinned versions.

### Random seeds
All experiments use seed=42 (set in `pretrain.py`, `evaluate_lp.py`, `evaluate_ft.py`).

### Data acquisition
- **Dataset 1 (IGC2/HGMA):** Contact ML4Microbiome consortium (a.l.zomer@uu.nl). NOT redistributable.
- **Dataset 2 (CMD):** Available via Bioconductor `curatedMetagenomicData` package. Openly accessible.

### Preprocessing
Run `scripts/preprocess.py` with the raw data to reproduce:
- CLR transformation (pseudocount 1e-6)
- Feature filtering (≥10% prevalence)
- 5-fold stratified CV splits (seed=42)
- LOCO splits

### Model configurations
- Encoder: 3-layer MLP (input→512→256→128), BatchNorm, ReLU
- Projector: 128→64→64
- NT-Xent loss, temperature τ=0.07
- Adam optimizer, lr=3e-4, weight_decay=1e-4
- Cosine annealing scheduler, 150 epochs
- Batch size 128, drop_last=True

### Commands to reproduce principal results

```bash
# Pre-train each augmentation (Dataset 1)
for aug in PhyloAug CompAug TaxDropAug FeatureMask GaussianNoise MixupAug EcologicalAug CohortStyleAug; do
    python scripts/pretrain.py --features data/features_clr_verified.csv.gz --meta data/metadata_verified.csv --aug $aug --outdir models/ --epochs 150
done

# Linear probe evaluation (within-disease CV)
python scripts/evaluate_lp.py --features data/features_clr_verified.csv.gz --meta data/metadata_verified.csv --splits data/cv_splits_verified.json --models models/

# Fine-tuning evaluation
python scripts/evaluate_ft.py --features data/features_clr_verified.csv.gz --meta data/metadata_verified.csv --splits data/cv_splits_verified.json --models models/

# Low-label experiments
python scripts/ft_lowlabel_fix.py --features data/features_clr_verified.csv.gz --meta data/metadata_verified.csv --splits data/cv_splits_verified.json --models models/
```

### Fold and LOCO assignments
Frozen splits are in `data/cv_splits_verified.json` and `data/loco_splits_verified.json`.

---

## Results Summary

### Within-Disease 5-Fold CV AUROC (Linear Probe, Dataset 1)

| Method | CRC | T2D | PD |
|---|---|---|---|
| MicroCLR-FeatureMask | 0.838±0.028 | 0.601±0.054 | 0.979±0.029 |
| MicroCLR-EcologicalAug | 0.789±0.085 | **0.672±0.045** | 0.954±0.032 |
| Random Forest | **0.864±0.051** | 0.658±0.062 | **0.999±0.003** |

**Note:** IBD results on Dataset 1 are reported separately as **exploratory** in the
manuscript (Section 3.13) because participant identifiers are unavailable and the results
may be inflated by participant leakage; they are excluded from the primary comparative
analyses. Full exploratory IBD values (within-disease, LOCO, low-label, fine-tuning) are
tabulated in manuscript Section 3.13 and Supplementary Section S6. IBD results on Dataset 2
(CMD) are part of the primary analyses: participant identifiers are available there and the
participant-grouped CV directly quantifies the leakage effect (see
`evaluation/ibdmdb_participant_cv_results.csv` and `evaluation/cmd/A2_participant_grouped_cv.json`).

### CMD LOCO Generalization AUROC

| Method | CRC | T2D | IBD |
|---|---|---|---|
| MicroCLR-FeatureMask | **0.724±0.085** | 0.664±0.056 | **0.766±0.043** |
| MicroCLR-ComboAug | 0.713±0.089 | **0.664±0.056** | 0.739±0.065 |
| Random Forest | 0.756±0.119 | 0.644±0.084 | 0.778±0.042 |

---

## Version History

### v1.1.0 (2026-10-18) — Second revision of BIOADV-2026-484

- Added all CMD (Dataset 2) analysis scripts and results: participant-grouped CV,
  inductive pre-training evaluation, nested hyperparameter tuning, Mixup alternative
  configurations, RF-on-embeddings, CMD LOCO/CV result files.
- Added re-centering and masking-rate sensitivity result files
  (`evaluation/recentering_results.json`, `evaluation/masking_rate_cv_results.json`).
- Regenerated all main-text figures with a harmonized color scheme
  (blue = MicroCLR variants, gray = no-pre-training controls, orange = Random Forest only)
  and fully defined labels (Figures 3-4: all eight CMD configurations defined in caption
  and manuscript Section 3.4; MixupAug row excluded — did not converge).
- IBD results on Dataset 1 (IGC2/HGMA) are reported separately as exploratory
  (manuscript Section 3.13, Supplementary Section S6) and excluded from primary
  comparative analyses; `figures/figS2_lowlabel_ibd_v2` provides the exploratory IBD
  low-label curves.
- Updated README (both datasets documented, quick-start fixed), `requirements.txt`
  added, reproducibility checklist extended.

### v1.0.0 (2026-06-18)

- Initial release: augmentation implementations, pre-trained encoders, IGC2/HGMA
  evaluation results, figures.

---

## Citation

If you use MicroCLR in your research, please cite:

```bibtex
@article{guzzi2026microclr,
  title     = {What Contrastive Learning Can and Cannot Do for Microbiome Disease Classification:
               A Systematic Augmentation Benchmark},
  author    = {Guzzi, Pietro Hiram and Perri, Caterina Francesca and Defilippo, Annamaria
               and Lomoio, Ugo and Veltri, Pierangelo},
  journal   = {Bioinformatics Advances},
  year      = {2026},
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Model weights and evaluation results are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The raw microbiome data (Dataset 1) is **not included** and is subject to ML4Microbiome data sharing terms. Contact a.l.zomer@uu.nl for access. Dataset 2 (CMD) is available via Bioconductor.

---

## Contact

Pietro Hiram Guzzi — hguzzi@unicz.it
GitHub: https://github.com/hguzzi/MicroCLR
