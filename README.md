# tabular-synth-calibration

A research framework for **calibrating synthetic tabular data quality** so that model-selection rankings on synthetic data faithfully reflect rankings on real holdout data. The core contribution is **BPR Calibration** (Bayesian Personalized Ranking), a rank-preserving correction layer that wraps any generative model and significantly improves models' rank preservation between synthetic and real model-evaluation losses.

---

## Motivation

Synthetic tabular data is increasingly used as a cheap proxy for real data in model selection and hyperparameter search. However, the *relative ordering* of models by loss on synthetic data is often poorly correlated with the ordering on real data. This project formalises the calibration problem and proposes methods to close that gap without requiring access to a large real test set.

---

## Methods

| Method | Class | Description |
|---|---|---|
| **BPR** | `SyntheticBPRCalibrator` | Pairwise ranking loss with temperature, margin penalty (β), and L2 regularisation (λ). Primary contribution. |
| **Standard** | `SyntheticCalibrator` | Linear/ridge correction baseline. |
| **Density** | `SyntheticDensityCalibration` | XGBoost-based density-ratio re-weighting. |
| **PPI** | `PPICalibration` | Prediction-powered inference calibration. |

All methods share a common k-fold evaluation harness that trains calibrators on a held-out calibration split, then measures Spearman rank correlation on a disjoint test set.

---

## Generative Backends

Five pretrained generative models are supported and pre-trained checkpoints are stored under `notebooks/gan_models/`:

- **CTGAN** — Conditional GAN for tabular data (`sdv` / `ctgan`)
- **TVAE** — Variational autoencoder for tabular data (`sdv`)
- **TabDDPM** — Denoising diffusion model for tabular data (`synthcity`)
- **TabPFGen** — Prior-data fitted network generative model (`tabpfgen`)
- **Gaussian Copula** — Parametric copula baseline (`sdv`)

Each checkpoint is keyed as `{dataset}_{method}_{timestamp}/`.

---

## Datasets

Ten benchmark datasets are used by default, covering both classification and regression:

| Dataset | Task |
|---|---|
| `heart_disease` | Classification |
| `diabetes` | Classification |
| `german_credit` | Classification |
| `mushroom` | Classification |
| `obesity` | Classification |
| `wine_quality` | Regression |
| `abalone` | Regression |
| `california_housing` | Regression |
| `concrete_strength` | Regression |
| `diabetes_regression` | Regression |

Datasets are loaded automatically via `ucimlrepo` on first run.

---

## Project Structure

```
tabular-synth-calibration/
├── src/synth_validation/
│   ├── runner.py          # ExperimentRunner — main entry point
│   ├── calibrator.py      # BPR, Standard, Density, PPI calibrators
│   ├── generation.py      # SyntheticDataGenerator (CTGAN / TVAE / TabDDPM)
│   ├── models.py          # Model pool + ModelSelector
│   ├── metrics.py         # Spearman, NDCG, hit-rate, rank preservation
│   ├── confidence.py      # CI estimation and aggregation
│   ├── data_loader.py     # UCI dataset loading and preprocessing
│   ├── shap_analizer.py   # SHAP-based feature importance
│   ├── theory.py          # Theoretical bounds and guarantees
│   └── utils.py           # Shared utilities
├── notebooks/
│   ├── gan_models/        # Pretrained generative model checkpoints
│   └── *.ipynb            # Experiment notebooks (Parts 1–3)
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from src.synth_validation.runner import ExperimentRunner

runner = ExperimentRunner(
    dataset_name="heart_disease",
    synth_method="ctgan",          # "ctgan" | "tvae" | "tabddpm" | "tabpfgen" | "gaussian_copula"
    task_type="classification",
    gan_model_dir="notebooks/gan_models",
)

results_standard = runner.run_kfold_calibration_experiment(
    n_folds=5,
    M_calibration=15,
    synth_size_multiplier=1.0,
    calib_test_ratio=0.2,
)

results_bpr = runner.run_kfold_bpr_calibration_experiment(
    n_folds=5,
    M_calibration=15,
    synth_size_multiplier=1.0,
    calib_test_ratio=0.2,
)
```

---

## Key Hyperparameters (BPR)

| Parameter | Default | Effect |
|---|---|---|
| `bpr_tau` | 0.1 | Score distribution temperature — keep ≤ 1.0 for regression |
| `bpr_beta` | 10.0 | Pairwise margin penalty — values < 1 harm regression |
| `bpr_lambda_reg` | 0.8 | L2 regularisation — higher is better for BPR |
| `M_calibration` | 25 | Calibration pool size — minimum viable is 10 |

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{filatov2026when,
  title={When Synthetic Data Is Enough: Calibration for Tabular Model Ranking},
  author={Gennadii Filatov and Irina Deeva},
  booktitle={Towards Trustworthy Predictions: Theory and Applications of Calibration for Modern AI},
  year={2026},
  url={https://openreview.net/forum?id=YECegW8nBY}
}
```
