# uLEAD-TabPFN: Anonymous Code and Supplementary Material

This repository contains the anonymous implementation and supplementary material for the ICDM submission **"Uncertainty-aware Dependency-based Anomaly Detection with TabPFN"**.

## Main Idea

uLEAD-TabPFN is an unsupervised tabular anomaly detector. It first selects a representative context set from normal samples, learns a linear latent representation, and then uses TabPFN regressors to model dependencies among latent dimensions. A test sample receives a high anomaly score when its latent dimensions violate the conditional dependencies learned from normal data. The final method also uses uncertainty-aware distributional dependency modeling, where variance networks convert point prediction deviations into Gaussian negative log-likelihood scores.

The implementation follows the paper pipeline:

1. Build a representative context set from normal training samples.
2. Train a linear encoder with a dependency loss computed from frozen TabPFN regressors.
3. Fit uncertainty-aware variance networks on normal data.
4. Score test samples by aggregating latent dependency violations.

## Repository Contents

```text
.
|-- README.md
|-- requirements.txt
|-- config.py
|-- lead.py
|-- run.py
|-- datasets_example.json
|-- datasets_adbench.json
|-- datasets_files_name.json
|-- data/
|   `-- Classical/
|       `-- 0_synthetic_dependency.npz
`-- supplementary/
    |-- ulead_icdm_supplementary.pdf
    |-- ulead_icdm_supplementary.tex
    |-- reference.bib
    `-- figures/
```

`lead.py` contains the model. `run.py` contains the ADBench-style evaluation loop, caching, metrics, and result logging. `config.py` stores the default method settings used by the paper implementation. The bundled dataset is a small synthetic ADBench-format example so the repository can be downloaded and run directly.

## Installation

Python 3.10 or newer is recommended. A CUDA GPU is strongly recommended for full experiments because TabPFN can be slow on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `tabpfn` installation differs on your platform, install it following the official TabPFN package instructions, then rerun the commands below.

## Quick Smoke Test

The repository is self-contained for a direct smoke test. The default dataset catalog points to the bundled synthetic example in `data/Classical/`.

```bash
python run.py --epochs 2 --seeds 0 --no-ddm --device cpu --from-scratch
```

This quick command is only a functionality check. It reduces the number of encoder epochs and disables uncertainty-aware DDM to keep runtime short on CPU. Results are written to `results/results.csv`, and cached artifacts are written under `results/lead/`.

For a closer run of the paper method on the bundled example, use:

```bash
python run.py --seeds 0 --from-scratch
```

This keeps the default paper settings in `config.py`, including context budget 500, encoder training with 100 epochs, and DDM enabled.

## Reproducing Paper-Scale Experiments

The paper-scale evaluation uses the ADBench dataset layout with category directories such as:

```text
ADBench/
|-- Classical/
|-- CV_by_ResNet18/
`-- NLP_by_BERT/
```

To run the full catalog used for the paper:

```bash
export ADBENCH_DATA_PATH=/path/to/ADBench
python run.py \
  --datasets-json datasets_adbench.json \
  --data-path "$ADBENCH_DATA_PATH" \
  --seeds 0 1 2 3 4 \
  --from-scratch
```

To run one classical ADBench dataset by numeric ID:

```bash
python run.py \
  --datasets-json datasets_adbench.json \
  --data-path "$ADBENCH_DATA_PATH" \
  --dataset-id 26 \
  --seeds 0
```

The main paper reports averages over the evaluation protocol described in the manuscript. The bundled synthetic dataset is not part of the benchmark table; it is included only to make the anonymous repository runnable without downloading external data.

## Important Configuration

The main paper settings are in `config.py`:

| Setting | Default | Meaning |
|---|---:|---|
| `context_fraction` | `0.5` | Initial fraction of normal samples used before downsampling |
| `context_cap` | `500` | Representative context set budget |
| `context_clustering_n_clusters` | `100` | KMeans clusters used for representative context selection |
| `test_set_exclude` | `initial_normals` | Exclude the initial normal training pool from test evaluation |
| `latent_dim` | `None` | Adaptive latent dimension |
| `latent_dim_max_cap` | `100` | Maximum latent dimension after compression |
| `ae_epochs` | `100` | Encoder training epochs |
| `use_ddm` | `True` | Enable uncertainty-aware distributional dependency modeling |
| `ddm_train_on_context` | `False` | Train variance networks on the initial normal pool |

Command-line flags override the config values for a run. Useful flags include:

```bash
python run.py --help
```

## Data Format

Each dataset is an `.npz` file with:

- `X`: numeric feature matrix with shape `(n_samples, n_features)`.
- `y`: binary labels with `0` for normal and `1` for anomaly.

The evaluation code uses `y` only to simulate the semi-supervised ADBench protocol: normal samples are used to construct the context/training set, and anomalies are held out for testing.

## Outputs

By default, outputs are written under:

```text
results/
|-- results.csv
`-- lead/
    |-- config.json
    |-- scripts/
    |-- context_set/
    |-- context_initial/
    |-- context_dev/
    `-- ddm_nll/ or dep_dev/
```

`results/results.csv` records ROC-AUC, average precision, PR-AUC, F1 at the anomaly percentage threshold, training time, prediction time, context size, and key configuration fields.

## Supplementary Material

The `supplementary/` directory contains the supplementary PDF, LaTeX source, bibliography file, and all figures required to rebuild the supplementary document.

To rebuild the supplementary PDF from this repository:

```bash
cd supplementary
pdflatex -interaction=nonstopmode -halt-on-error ulead_icdm_supplementary.tex
bibtex ulead_icdm_supplementary
pdflatex -interaction=nonstopmode -halt-on-error ulead_icdm_supplementary.tex
pdflatex -interaction=nonstopmode -halt-on-error ulead_icdm_supplementary.tex
```

## Notes

- The code is anonymized for review and does not include local machine paths.
- Full benchmark reproduction requires downloading the external ADBench datasets.
- The first TabPFN run may download or initialize model assets depending on the local TabPFN installation.
