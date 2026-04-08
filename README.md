# LEAD: Linear Encoder with Dependency Expert

LEAD is an unsupervised anomaly detection method that combines a linear encoder with a TabPFN-based dependency expert. It learns a compact latent representation in which each dimension is predictable from the others under normal conditions; anomalies are identified as points where these learned dependencies break down.

## Architecture

```
X (d dims)
  └─► Linear Encoder ──► z (p dims) ──► Dependency Expert ──► Anomaly Score
         (trained with                    (TabPFN regressor,
          dependency loss)                 one per latent dim)
```

**Three-step pipeline:**
1. **Context generation** – select a representative set of normal training samples via clustering-based downsampling.
2. **Encoder training** – train a linear encoder end-to-end using a dependency deviation loss (TabPFN predicts each latent dimension from the others; the encoder is pushed to minimise prediction error on normal data).
3. **Anomaly scoring** – at test time, encode each sample and compute how much each latent dimension deviates from what the fitted TabPFN regressors predict; aggregate deviations into a scalar anomaly score.

Optional extensions: Distributional Dependency Modeling (DDM) replaces point predictions with a Gaussian NLL loss; post-hoc Mixture-of-Experts (MoE) fits separate experts per cluster for multi-modal data.

## Requirements

- Python 3.10+
- CUDA-capable GPU (strongly recommended; CPU inference is very slow for TabPFN)

Install dependencies:

```bash
pip install -r requirements.txt
```

> **TabPFN note:** `tabpfn>=2.0.0` is required. If the version on PyPI differs, install directly:
> ```bash
> pip install tabpfn
> ```

## Data Preparation

LEAD is evaluated on [ADBench](https://github.com/Minqi824/ADBench). Download the benchmark datasets and note the directory that contains the `Classical/`, `CV_by_ResNet18/`, and `NLP_by_BERT/` subdirectories.

Set the path **before running** via the environment variable:

```bash
export ADBENCH_DATA_PATH=/path/to/adbench/datasets/
```

Alternatively, open `config.py`, find `_detect_data_path()`, and add a `return` statement:

```python
def _detect_data_path():
    return Path('/path/to/adbench/datasets/')
```

## Quick Start

```bash
# Run LEAD on all ADBench datasets (results saved to results/results.csv)
python run.py

# Run on a single dataset (dataset ID 26 = optdigits)
python run.py --dataset-id 26

# Override the data path without editing config.py
python run.py --data-path /path/to/adbench/datasets/

# Force retraining from scratch (ignore cached artifacts)
python run.py --from-scratch

# Customise output location
python run.py --output-csv my_results.csv --save-dir my_artifacts/
```

All command-line flags override the corresponding value in `config.py`:

| Flag | Description |
|---|---|
| `--data-path PATH` | ADBench dataset directory |
| `--dataset-id INT` | Run a single dataset by numeric ID |
| `--output-csv PATH` | Output CSV file for results |
| `--save-dir PATH` | Directory for models and cached artifacts |
| `--epochs INT` | Encoder training epochs (default: 20) |
| `--seeds INT [INT ...]` | Random seeds (default: 0 1 2 3 4) |
| `--from-scratch` | Retrain from scratch, ignore cache |
| `--scoring-method STR` | Scoring method (see below) |
| `--device cuda\|cpu` | Compute device |

## Configuration

All hyperparameters live in `config.py`. Key settings:

### Scoring methods (`scoring_method`)

| Value | Description |
|---|---|
| `dependency` | Dependency deviation in latent space (default) |
| `reconstruction` | Reconstruction error in input space |
| `proximity` | SemiTabPFN proximity score (synthetic anomalies) |
| `dep-recon` | Fusion of dependency + reconstruction |
| `dep-prox` | Fusion of dependency + proximity |

### Latent dimensionality (`latent_dim_strategy`)

| Value | Description |
|---|---|
| `adaptive` | Identity mapping for low-dim (≤35 features), sqrt-linear compression for high-dim (default) |
| `match_features` | `min(d, 50)` |
| `quarter` | `d / 4` |

### Caching (`from_scratch`)

LEAD caches the expensive TabPFN predictions (`dep_dev/`) and fitted models (`model/`) under `save_path`. On subsequent runs with `from_scratch=False` (the default), the cached artifacts are reloaded and only the final scoring step is re-executed. Set `from_scratch=True` or use `--from-scratch` to force full retraining.

## Output

Results are appended to `results/results.csv` (configurable). Each row corresponds to one dataset × seed combination and includes:

| Column | Description |
|---|---|
| `dataset` | Dataset name |
| `seed` | Random seed |
| `roc_auc` | Area under the ROC curve |
| `ap` | Average precision |
| `pr_auc` | Area under the precision-recall curve |
| `f1_at_anomaly_pct` | F1 score at the true anomaly percentage threshold |
| `t_train` | Training time (seconds) |
| `t_predict` | Prediction time (seconds) |

Intermediate artifacts are saved under `save_path/` (default: `results/lead/`):

```
results/lead/
├── config.json          # Experiment configuration snapshot
├── scripts/             # Copy of source files used for this run
├── model/               # Trained encoder models (.pkl)
├── dep_dev/             # Cached dependency deviations (.npy)
├── labels/              # Test labels (.npy)
├── recon_loss/          # Reconstruction errors (.npy)
└── context_set/         # Context set indices (.npy)
```

## File Overview

| File | Description |
|---|---|
| `lead.py` | Core `LeadTabPFN` model (encoder training, anomaly scoring, DDM, MoE) |
| `run.py` | Evaluation script (dataset iteration, caching, result logging) |
| `config.py` | Centralised hyperparameter configuration |
| `semitabpfn.py` | `SemiTabPFN` proximity detector (synthetic anomaly generation) |
| `datasets_files_name.json` | ADBench dataset catalogue |

## ADBench Interface

`LeadTabPFN` follows the ADBench detector interface:

```python
from lead import LeadTabPFN

model = LeadTabPFN(seed=42)
model.fit(X_train, y_train)   # y_train: 0 = normal, 1 = anomaly (used only to build context set)
scores = model.predict_score(X_test)  # higher score = more anomalous
```

## Reproducibility

Each run snapshots `config.json` and copies all source files into `save_path/scripts/` so that results can be traced back to the exact configuration and code used.
