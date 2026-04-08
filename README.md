# uLEAD-TabPFN: Uncertainty-aware Dependency-based Anomaly Detection with TabPFN

Anomaly detection in tabular data is challenging due to high dimensionality,
complex feature dependencies, and heterogeneous noise.
Many existing methods rely on proximity-based cues and may
miss anomalies caused by violations of complex feature dependencies.
Dependency-based anomaly detection provides a principled
alternative by identifying anomalies as violations of dependencies
among features. However, existing methods often struggle to
model such dependencies robustly and to scale to high-dimensional
data with complex dependency structures. To address these challenges,
we propose uLEAD-TabPFN, a dependency-based anomaly
detection framework built on Prior-Data Fitted Networks (PFNs).
uLEAD-TabPFN identifies anomalies as violations of conditional
dependencies in a learned latent space, leveraging frozen PFNs for
dependency estimation. Combined with uncertainty-aware scoring,
the proposed framework enables robust and scalable anomaly detection.
Experiments on 57 tabular datasets from ADBench show
that uLEAD-TabPFN achieves particularly strong performance in
medium- and high-dimensional settings, where it attains the top
average rank. On high-dimensional datasets, uLEAD-TabPFN improves
the average ROC-AUC by nearly 20% over the average baseline
and by approximately 2.8% over the best-performing baseline,
while maintaining overall superior performance compared to state-of-
the-art methods. Further analysis shows that uLEAD-TabPFN
provides complementary anomaly detection capability, achieving
strong performance on datasets where many existing methods
struggle.

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


