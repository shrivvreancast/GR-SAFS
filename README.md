# GR-SAFS: Graph-Regularized Stacking with Adaptive Feature Selection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)

Official implementation of the paper:

> **GR-SAFS: A Graph-Regularized Stacking Framework with Adaptive Feature Selection for High-Dimensional Prognostic Biomarker Discovery**
> *Bioinformatics and Biomedicine*, 2026 (under review).

GR-SAFS unifies (1) a Graph-Lasso engine with gene co-expression Laplacian priors, (2) a Random Forest engine for nonlinear interactions, (3) an empirical CDF distribution alignment layer, and (4) a diversity-penalized quadratic programming (QP) router with Ledoit-Wolf shrinkage, into a single pipeline for high-dimensional prognostic biomarker discovery.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Reproducing the Paper](#reproducing-the-paper)
- [Using GR-SAFS on Your Own Data](#using-gr-safs-on-your-own-data)
- [Repository Layout](#repository-layout)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<YOUR-USERNAME>/GR-SAFS.git
cd GR-SAFS

# 2. Create a clean environment (conda recommended)
conda create -n grsafs python=3.10 -y
conda activate grsafs
pip install -r requirements.txt
pip install -e .

# 3. Run the demo on the included synthetic dataset
python scripts/demo_synthetic.py
```

If everything is set up correctly, the demo will print the top-20 selected features and produce a Kaplan-Meier plot in `outputs/demo/`.

---

## Installation

### Option A: pip (recommended)

```bash
pip install -r requirements.txt
pip install -e .
```

### Option B: conda environment file

```bash
conda env create -f environment.yml
conda activate grsafs
```

### System requirements

- Python ≥ 3.9 (tested on 3.9, 3.10, 3.11)
- Linux / macOS / Windows
- ~4 GB RAM for the default TCGA-LUAD setting (n=504, p=10,000)
- No GPU required

---

## Reproducing the Paper

All paper results can be reproduced from a single Makefile target:

```bash
# Reproduce all main-paper figures and tables (~2 hours on 8-core CPU)
make reproduce-all

# Or run individual stages
make data         # Download & preprocess TCGA-LUAD + GEO cohorts
make simulation   # Run the 3-scenario benchmark (Section 3.1)
make ablation     # Run the component ablation (Section 3.2)
make luad         # Run the TCGA-LUAD signature discovery (Section 3.3)
make external     # Run the frozen-signature validation (Section 3.4)
make figures      # Regenerate all figures into figures/
```

Outputs land in `outputs/<run-name>/` and can be diffed against `outputs/paper-frozen/` (provided in the release tarball) to verify reproducibility.

### Hardware notes

The simulation benchmark (20 runs × 9 methods × 3 scenarios) is the only step that benefits from parallelism; pass `JOBS=8` to `make simulation` to use 8 cores.

---

## Using GR-SAFS on Your Own Data

```python
from grsafs import GRSAFS
import pandas as pd

# X: (n_samples, n_genes) expression matrix, log2-transformed
# time: (n_samples,) follow-up time
# event: (n_samples,) 1=event, 0=censored
X = pd.read_csv("my_expression.csv", index_col=0)
time = pd.read_csv("my_clinical.csv")["OS_time"].values
event = pd.read_csv("my_clinical.csv")["OS_status"].values

model = GRSAFS(
    n_top=20,                # number of features to select
    lambda1=0.2,             # L1 sparsity
    lambda2=0.05,            # graph smoothness
    gamma=10.0,              # diversity penalty
    cv_folds=5,
    random_state=42,
)

model.fit(X, time, event)

# Top-20 selected genes with fused scores
print(model.signature_)

# Risk score on new samples
risk = model.predict_risk(X_new)
```

Full API reference is in [`docs/api.md`](docs/api.md).

---

## Repository Layout

```
GR-SAFS/
├── grsafs/                     # Main Python package
│   ├── __init__.py
│   ├── engines/
│   │   ├── graph_lasso.py      # Graph-Lasso PGD solver (§2.2)
│   │   └── random_forest.py    # RF importance engine (§2.2)
│   ├── alignment.py            # eCDF distribution alignment (§2.3)
│   ├── fusion.py               # Diversity-penalized QP router (§2.4)
│   ├── survival_utils.py       # Deviance residuals, Nelson-Aalen
│   ├── graph_utils.py          # Co-expression network construction
│   └── grsafs.py               # Top-level GRSAFS class
├── scripts/                    # Reproducibility scripts
│   ├── 00_download_data.py
│   ├── 01_preprocess_tcga.py
│   ├── 02_run_simulation.py
│   ├── 03_train_luad.py
│   ├── 04_external_validation.py
│   └── demo_synthetic.py
├── tests/                      # Unit tests (pytest)
├── data/                       # Data placeholders (NOT versioned)
│   ├── raw/                    # Raw downloads (gitignored)
│   └── processed/              # Cleaned tables (gitignored)
├── outputs/                    # Run outputs (gitignored)
├── figures/                    # Paper figures (versioned)
├── notebooks/                  # Jupyter exploration notebooks
├── docs/                       # Extended documentation
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── Makefile
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## Datasets

The training data are **publicly available** but not redistributed in this repo:

| Cohort | Source | Accession | Used in |
|---|---|---|---|
| TCGA-LUAD | UCSC Xena | [HiSeqV2](https://xenabrowser.net/) | Training (n=504) |
| GSE31210 | NCBI GEO | [GSE31210](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31210) | External validation (n=226) |
| GSE50081 | NCBI GEO | [GSE50081](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50081) | External validation (n=181) |

Run `make data` to download and preprocess all three cohorts (~3 GB; requires internet).

---

## Citation

If you use GR-SAFS in your research, please cite:

```bibtex
@article{grsafs2026,
  title   = {GR-SAFS: A Graph-Regularized Stacking Framework with Adaptive Feature Selection for High-Dimensional Prognostic Biomarker Discovery},
  author  = {[Author Names]},
  conference = {Bioinformatics and Biomedicine},
  year    = {2026},
  doi     = {PLACEHOLDER}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

We thank TCGA, UCSC Xena, and GEO for providing the data; STRING, KEGG, and Gene Ontology consortia for the functional annotation resources used in the biological validation.

## Contact

For questions or bug reports, please open an [Issue](https://github.com/<YOUR-USERNAME>/GR-SAFS/issues).
