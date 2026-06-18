# MicroCLR: Contrastive Learning for Microbiome Disease Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MicroCLR** is a contrastive self-supervised learning framework for gut microbiome data. It pre-trains a neural encoder using biologically-motivated data augmentations and NT-Xent loss, then evaluates learned representations via linear probing and fine-tuning on four disease classification tasks (CRC, T2D, IBD, PD).

This repository accompanies the paper:

> Defilippo A., Lomoio U., Perri C.F., Veltri P., Guzzi P.H.  
> **MicroCLR: Contrastive Learning for Microbiome-Based Disease Classification**  
> *Bioinformatics* (submitted), 2025.  
> WIVACE 2025 extended abstract also available (see `manuscripts/`).

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Augmentations](#augmentations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results Summary](#results-summary)
- [Citation](#citation)

---

## Overview

MicroCLR applies contrastive learning (SimCLR-style) to CLR-transformed gut microbiome profiles. The framework:

1. **Pre-trains** a 3-layer MLP encoder (426→512→256→128) using NT-Xent loss (τ=0.07) with biologically-motivated augmentations
2. **Evaluates** representations via:
   - Within-cohort 5-fold stratified CV (linear probe)
   - Leave-One-Cohort-Out (LOCO) generalization
   - Low-label fine-tuning (10%, 25%, 50% of labels)
3. **Benchmarks** against Random Forest, Logistic Regression, XGBoost, and SCARF

Eight augmentation strategies are implemented, including two novel biologically-inspired augmentations:
- **EcologicalAug**: Spearman co-occurrence network-guided community perturbation
- **CohortStyleAug**: Inter-cohort technical variation simulation via mean-shift transfer

---

## Dataset

The dataset is derived from the **HGMA (Human Gut Microbiome Atlas)** and is **not redistributable** under the ML4Microbiome data sharing terms.

**To obtain the data**, contact the ML4Microbiome consortium:
- Aldert Zomer: a.l.zomer@uu.nl
- Website: https://www.ml4microbiome.eu/

**Dataset statistics:**
- 1,286 samples × 426 metagenomic species (MSP) features
- 7 cohorts, 4 disease labels: CRC (colorectal cancer), T2D (type 2 diabetes), IBD (inflammatory bowel disease), PD (Parkinson's disease)
- Features: CLR (centered log-ratio) transformed relative abundances

Once you have the raw data, use `scripts/preprocess.py` to reproduce the CLR transformation and train/test splits used in this study.

---

## Augmentations

All augmentations are implemented in `augmentations/augmentations_v2.py`.

| Augmentation | Description | Biologically Motivated |
|---|---|---|
| `PhyloAug` | Phylogenetic clade-level abundance smoothing | Yes |
| `CompAug` | Compositional resampling (Dirichlet) | Yes |
| `TaxDropAug` | Random taxon dropout | Yes |
| `FeatureMask` | Random feature masking | No |
| `GaussianNoise` | Additive Gaussian noise | No |
| `MixupAug` | Linear interpolation between samples | No |
| `EcologicalAug` | Co-occurrence network-guided perturbation | **Yes (novel)** |
| `CohortStyleAug` | Inter-cohort mean-shift style transfer | **Yes (novel)** |

### EcologicalAug

Perturbs a random set of seed taxa and propagates the perturbation through a Spearman co-occurrence network (|ρ| > 0.3). Positive co-occurrers are reduced; negative co-occurrers are increased. The CLR zero-sum constraint is preserved by re-centering.

```python
from augmentations.augmentations_v2 import EcologicalAug
import pickle

with open("data/cooccurrence_network.pkl", "rb") as f:
    network = pickle.load(f)

aug = EcologicalAug(
    pos_neighbors=network["pos_neighbors"],
    neg_neighbors=network["neg_neighbors"],
    k_seeds_range=(1, 5),
    alpha_range=(0.1, 0.4),
    seed=42
)
x_aug = aug(x)  # x: numpy array of shape (426,)
```

### CohortStyleAug

Simulates inter-cohort technical variation by transferring a fraction of the mean CLR profile difference between a randomly selected target cohort and the sample's own cohort.

```python
from augmentations.augmentations_v2 import CohortStyleAug
import numpy as np

cohort_means = np.load("data/cohort_means.npy", allow_pickle=True).item()
aug = CohortStyleAug(
    cohort_means=cohort_means,
    sample_cohorts=sample_cohorts,  # dict: {sample_id: cohort_id}
    beta_max=0.5,
    noise_std=0.05,
    seed=42
)
x_aug = aug(x, sample_id=sid)
```

---

## Repository Structure

```
MicroCLR/
├── README.md
├── LICENSE
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
│   [NOTE: features_clr_verified.csv.gz and metadata_verified.csv not included
│    — contact ML4Microbiome for raw data access]
│
├── evaluation/
│   ├── cv_results.csv             # Within-cohort 5-fold CV AUROC (all methods)
│   ├── loco_results.csv           # LOCO AUROC (all methods)
│   ├── lowlabel_results.csv       # Low-label LP AUROC (10/25/50%)
│   ├── ft_cv_results.csv          # Fine-tuning CV results
│   ├── ft_loco_results.csv        # Fine-tuning LOCO results
│   ├── ft_lowlabel_results.csv    # Fine-tuning low-label results
│   ├── silhouette_scores.csv      # Disease vs. cohort silhouette scores
│   └── statistical_tests.csv      # Wilcoxon test results
│
├── figures/
│   ├── fig1_within_cohort_benchmark.{png,svg}
│   ├── fig2_umap_embeddings.{png,svg}
│   ├── fig3_loco_benchmark.{png,svg}
│   ├── fig4_lowlabel_curves.{png,svg}
│   ├── fig5_silhouette_scores.{png,svg}
│   ├── fig6_finetuning_comparison.{png,svg}
│   ├── fig7_lowlabel_finetuning.{png,svg}
│   └── figS1_training_curves.{png,svg}
│
├── manuscripts/
│   ├── manuscript_MicroCLR_bioinf.pdf    # OUP Bioinformatics submission
│   ├── manuscript_MicroCLR_bioinf.tex    # LaTeX source
│   ├── supplementary_MicroCLR.pdf        # Supplementary material
│   ├── supplementary_MicroCLR.tex
│   ├── manuscript_MicroCLR_wivace.pdf    # WIVACE 2025 extended abstract
│   ├── manuscript_MicroCLR_wivace.tex
│   └── references_bioinf.bib             # Bibliography
│
└── scripts/
    ├── preprocess.py              # CLR transformation + split generation
    ├── pretrain.py                # Pre-training script (NT-Xent, 150 epochs)
    ├── evaluate_lp.py             # Linear probe evaluation (CV + LOCO)
    ├── evaluate_ft.py             # Fine-tuning evaluation
    ├── ft_lowlabel_fix.py         # Low-label fine-tuning (10/25/50%)
    └── make_fig7.py               # Figure generation script
```

---

## Installation

```bash
git clone https://github.com/hguzziunicz/MicroCLR.git
cd MicroCLR
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn umap-learn
```

**Requirements:**
- Python 3.8+
- PyTorch 1.12+
- scikit-learn 1.0+
- numpy, pandas, scipy, matplotlib, seaborn, umap-learn

---

## Quick Start

### Load a pre-trained encoder

```python
import torch
import torch.nn as nn

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

model = MicroCLREncoder()
checkpoint = torch.load("models/microclr_EcologicalAug.pt", map_location="cpu")
model.load_state_dict(checkpoint["encoder_state_dict"])
model.eval()

# Extract 128-dim embeddings
with torch.no_grad():
    embeddings = model(torch.tensor(X_clr, dtype=torch.float32))
```

### Run linear probe evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# embeddings: (n_samples, 128), y: disease labels
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(embeddings[train_idx], y[train_idx])
auroc = roc_auc_score(y[test_idx], clf.predict_proba(embeddings[test_idx])[:, 1])
```

---

## Results Summary

### Within-Cohort 5-Fold CV AUROC (Linear Probe)

| Method | CRC | T2D | IBD | PD |
|---|---|---|---|---|
| MicroCLR-EcologicalAug | 0.789±0.085 | **0.672±0.045** | 0.889±0.017 | 0.954±0.032 |
| MicroCLR-CohortStyleAug | 0.820±0.047 | 0.570±0.035 | 0.917±0.023 | 0.980±0.022 |
| MicroCLR-FeatureMask | 0.838±0.028 | 0.601±0.054 | 0.958±0.019 | 0.979±0.029 |
| Random Forest | 0.864±0.051 | 0.658±0.062 | 0.993±0.006 | 0.999±0.003 |

**EcologicalAug achieves AUROC=0.672 on T2D — the first MicroCLR variant to surpass Random Forest (0.658) on this task.**

### LOCO Generalization AUROC

| Method | CRC | T2D | IBD |
|---|---|---|---|
| MicroCLR-EcologicalAug | **0.611±0.141** | **0.575±0.107** | 0.410±0.042 |
| MicroCLR-CohortStyleAug | 0.539±0.013 | 0.449±0.031 | **0.590±0.110** |
| Best existing (SCARF/XGBoost/FeatureMask) | 0.605 | 0.524 | 0.578 |

EcologicalAug sets new best LOCO on CRC and T2D; CohortStyleAug sets new best LOCO on IBD.

### Pre-Training Convergence

| Augmentation | Final Loss | Min Loss |
|---|---|---|
| CohortStyleAug | 0.1120 | **0.0867** |
| EcologicalAug | 0.1209 | 0.0902 |
| TaxDropAug | 0.1229 | 0.0951 |
| GaussianNoise | 0.1354 | 0.1056 |
| CompAug | 0.1363 | 1.036 |
| PhyloAug | 0.1506 | 0.1208 |
| FeatureMask | 0.3385 | 0.3017 |
| MixupAug | 4.5841 | 4.3957 |

---

## Citation

If you use MicroCLR in your research, please cite:

```bibtex
@article{defilippo2025microclr,
  title     = {MicroCLR: Contrastive Learning for Microbiome-Based Disease Classification},
  author    = {Defilippo, Annamaria and Lomoio, Ugo and Perri, Caterina Francesca
               and Veltri, Pierangelo and Guzzi, Pietro Hiram},
  journal   = {Bioinformatics},
  year      = {2025},
  note      = {Submitted}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Model weights and evaluation results are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The raw microbiome data is **not included** and is subject to ML4Microbiome data sharing terms. Contact a.l.zomer@uu.nl for access.

---

## Contact

Pietro Hiram Guzzi — pietro.guzzi@unicz.it  
GitHub: https://github.com/hguzziunicz/MicroCLR
