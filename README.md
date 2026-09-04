# uLEAD-TabPFN

**Uncertainty-aware Dependency-based Anomaly Detection with TabPFN**

Accepted at the **2026 IEEE International Conference on Data Mining (ICDM 2026)**.

Sha Lu, Jixue Liu, Stefan Peters, Thuc Duy Le, Yongzheng Xie, Lin Liu, and Jiuyong Li

uLEAD-TabPFN is an unsupervised detector for numerical tabular data. It learns a dependency-aligned linear latent representation from normal training samples, uses frozen TabPFN regressors to estimate conditional means, and fits lightweight variance networks to estimate input-dependent conditional residual scales. Test samples are ranked with a composite conditional Gaussian negative log-likelihood (NLL).

## Method overview

The implementation follows four stages:

1. Construct a Representative Context Set (RCS) from normal training samples.
2. Train a linear encoder so that each latent dimension is predictable from the others.
3. Fit a residual-scale network for each latent conditional model while keeping the encoder and TabPFN regressors frozen.
4. Average the per-dimension conditional Gaussian NLL contributions to obtain the anomaly score.

Here, *uncertainty* means input-dependent conditional residual variability learned from normal data. It is aleatoric residual-scale uncertainty, not TabPFN epistemic uncertainty or a calibrated predictive distribution.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── config.py
├── lead.py
├── run.py
├── datasets_example.json
├── datasets_adbench.json
├── data/
│   └── Classical/
│       └── 0_synthetic_dependency.npz
└── supplementary/
    ├── ulead_icdm_supplementary.pdf
    ├── ulead_icdm_supplementary.tex
    ├── reference.bib
    └── figures/
```

- `lead.py` implements the detector.
- `run.py` provides the ADBench-style evaluation, metrics, caching, and result logging.
- `config.py` contains the paper settings.
- `data/` contains a small synthetic example in ADBench `.npz` format.
- `supplementary/` contains the camera-ready supplementary material and its source.

## Installation

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended for full experiments; the smoke test can run on CPU.

```bash
git clone https://github.com/ShaLu-ML/uLEAD-TabPFN.git
cd uLEAD-TabPFN
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first TabPFN run may download model assets. If installation differs on your platform, follow the [official TabPFN instructions](https://github.com/PriorLabs/TabPFN).

## Quick start

Run the bundled synthetic example with a short CPU smoke test:

```bash
python run.py --epochs 2 --seeds 0 --no-ddm --device cpu --from-scratch
```

This command checks the data-loading, context-construction, encoder, scoring, and result-writing pipeline. It disables the residual-scale networks and reduces encoder training to two epochs for speed; it is not the full paper configuration.

Run the complete uLEAD-TabPFN method on the example with:

```bash
python run.py --seeds 0 --from-scratch
```

By default, results are written to `results/results.csv`, and intermediate artifacts are cached under `results/lead/`.

## Reproducing the paper protocol

Download the [ADBench datasets](https://github.com/Minqi824/ADBench) and retain their category directories, for example:

```text
ADBench/
├── Classical/
├── CV_by_ResNet18/
└── NLP_by_BERT/
```

Then run the complete catalog:

```bash
export ADBENCH_DATA_PATH=/path/to/ADBench
python run.py \
  --datasets-json datasets_adbench.json \
  --data-path "$ADBENCH_DATA_PATH" \
  --seeds 0 1 2 3 4 \
  --from-scratch
```

To run one classical dataset by its numeric filename prefix:

```bash
python run.py \
  --datasets-json datasets_adbench.json \
  --data-path "$ADBENCH_DATA_PATH" \
  --dataset-id 26 \
  --seeds 0
```

All preprocessing statistics, RCS prototypes, encoder parameters, and variance-network parameters are computed exclusively from the normal training split. Test samples and labels are used only for evaluation. The bundled synthetic dataset is a functionality example and is not included in the paper's benchmark tables.

## Important settings

The defaults in `config.py` match the paper configuration.

| Setting | Default | Description |
|---|---:|---|
| `context_fraction` | `0.5` | Fraction of normal samples in the initial training pool |
| `context_cap` | `500` | Maximum RCS size |
| `context_clustering_n_clusters` | `100` | Clusters used for representative context selection |
| `test_set_exclude` | `initial_normals` | Exclude the normal training pool from evaluation |
| `latent_dim` | `None` | Apply the paper rule $p=\min(d,100)$ |
| `latent_dim_max_cap` | `100` | Maximum latent dimension |
| `ae_epochs` | `20` | Encoder-training epochs |
| `use_ddm` | `True` | Enable conditional Gaussian NLL scoring |
| `ddm_train_on_context` | `False` | Fit variance networks on the initial normal pool |
| `ddm_aggregation` | `mean` | Average per-dimension NLL contributions |

Use `python run.py --help` to see the supported command-line overrides.

## Data format

Each dataset is an `.npz` file containing:

- `X`: numeric feature matrix with shape `(n_samples, n_features)`;
- `y`: binary labels with `0` for normal and `1` for anomaly.

At least two input features and one normal training sample are required. Categorical or multimodal fields must first be converted to numerical features.

## Outputs and caching

```text
results/
├── results.csv
└── lead/
    ├── config.json
    ├── scripts/
    ├── context_set/
    ├── context_initial/
    ├── context_dev/
    ├── ddm_nll/
    └── dep_dev/
```

`results.csv` records ROC-AUC, average precision, PR-AUC, F1 at the anomaly-percentage threshold, timing, split sizes, and configuration values. Cached DDM matrices are aggregated directly as conditional NLL contributions; non-DDM dependency deviations are normalized against their cached context deviations.

## Checks

Run the deterministic core checks without downloading model assets:

```bash
python -m unittest discover -s tests
```

For an end-to-end check, run the CPU smoke-test command from the Quick start section.

## Supplementary material

The camera-ready supplementary PDF is included in `supplementary/`. To rebuild it:

```bash
cd supplementary
pdflatex -interaction=nonstopmode -halt-on-error ulead_icdm_supplementary.tex
bibtex ulead_icdm_supplementary
pdflatex -interaction=nonstopmode -halt-on-error ulead_icdm_supplementary.tex
pdflatex -interaction=nonstopmode -halt-on-error ulead_icdm_supplementary.tex
```

## Citation

If you use this repository, please cite the ICDM 2026 paper:

```bibtex
@inproceedings{lu2026ulead,
  title     = {Uncertainty-aware Dependency-based Anomaly Detection with TabPFN},
  author    = {Lu, Sha and Liu, Jixue and Peters, Stefan and Le, Thuc Duy and
               Xie, Yongzheng and Liu, Lin and Li, Jiuyong},
  booktitle = {2026 IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

The repository also provides GitHub-compatible citation metadata in `CITATION.cff`.
