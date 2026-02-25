# tabular-synth-calibration

---

## Overview

This repository contains the implementation and experimental framework for my master's research work on **calibrated synthetic data validation for machine learning model selection**. The core problem we tackle: **how can we reliably select ML models when we have limited labeled validation data?**

### The Problem

In tabular data domains (healthcare, finance, science), getting labeled validation data is expensive and time-consuming. Synthetic data could be a solution, but current approaches lack theoretical guarantees and often produce misleading model rankings. This leads to:
- Poor model selection decisions
- Wasted computational resources
- Unreliable performance estimates

### Our Solution

We develop a framework based on **constrained optimization calibration for tabular data with interpretability** that:
1. Generates synthetic validation data using multiple generators (CTGAN, TVAE, TabPFGen, TabDDPM, Gaussian Copula)
2. Calibrates synthetic data by learning per-sample weights that align synthetic errors with real validation errors
3. Provides theoretical guarantees on rank preservation through total variation analysis
4. Delivers confidence intervals for model performance estimates

---

## Repository Structure

```
tabular-synth-calibration/
├── src/
│   └── synth_validation/          # Main Python package
│       ├── __init__.py            # Package exports
│       ├── runner.py              # ExperimentRunner - main orchestration
│       ├── calibrator.py          # SyntheticDataCalibrator - sample-level calibration
│       ├── shap_analizer.py       # SHAPWeightAnalizer - SHAP interpretability
│       ├── data_loader.py         # DataLoader - UCI dataset loading
│       ├── generation.py          # SyntheticDataGenerator - CTGAN/TVAE/etc.
│       ├── models.py              # ModelSelectionFramework - 44 architectures
│       ├── metrics.py             # EvaluationMetrics - Spearman, rank preservation
│       ├── confidence.py          # ConfidenceIntervalEstimator - bootstrap/analytical CI
│       ├── theory.py              # TheoreticalFramework - total variation analysis
│       └── utils.py               # Constants and utilities
├── notebooks/
│   ├── experiment_demo.ipynb      # Demo notebook showing package usage
│   ├── gan_models                 # Tuned and saved models
│   └── experiment_figures         # Figures of each exepriment
└── README.md                      # This file
```

### Package Modules

| Module | Main Class | Description |
|--------|------------|-------------|
| `runner.py` | `ExperimentRunner` | K-fold calibration experiment orchestration |
| `calibrator.py` | `SyntheticDataCalibrator` | L-BFGS-B constrained optimization for sample weights |
| `shap_analizer.py` | `SHAPWeightAnalizer` | CatBoost-based dual SHAP analysis |
| `data_loader.py` | `DataLoader` | UCI dataset loading and preprocessing |
| `generation.py` | `SyntheticDataGenerator` | Synthetic data generation (CTGAN, TVAE, etc.) |
| `models.py` | `ModelSelectionFramework` | 44 ML model architectures for evaluation |
| `metrics.py` | `EvaluationMetrics` | Spearman correlation, rank preservation metrics |
| `confidence.py` | `ConfidenceIntervalEstimator` | analytical confidence intervals |
| `theory.py` | `TheoreticalFramework` | Total variation and rank preservation theory |
| `utils.py` | - | Random seeds and utility functions |

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended for CTGAN/TVAE training (but not required)

### Installation

```bash
# Clone the repository
git clone https://github.com/ITMO-NSS-team/tabular-synth-calibration.git
cd tabular-synth-calibration

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas torch scikit-learn scipy matplotlib seaborn
pip install sdv  # Synthetic Data Vault (CTGAN, TVAE)
pip install optuna  # Hyperparameter optimization
pip install shap catboost  # Interpretability
pip install tabpfn  # TabPFN foundation model (optional but recommended)
pip install synthcity # TabDDPM generative model
pip install tabpfgen # Implementation of TabPFN as a generator of synthetic samples
```

### Basic Usage

**Using the Package**

```python
import sys
sys.path.insert(0, './src')

from synth_validation import ExperimentRunner

# Initialize experiment
runner = ExperimentRunner(
    dataset_name='adult',           # UCI dataset
    synth_method='ctgan',           # or 'tvae', 'gaussian_copula' etc.
    task_type='classification',
    lambda_reg=0.5,                 # Calibration regularization
    verbose=True
)

# Run K-fold calibration experiment
results = runner.run_kfold_calibration_experiment(
    n_folds=5,
    M_calibration=15,               # Models for calibration
    synth_size_multiplier=1.0,
    analyze_shap=True
)

# Visualize results
runner.visualize_correlation_results()
runner.print_summary_table()
runner.plot_weight_histograms()
```

**Using the Demo Notebook**

1. Open `notebooks/experiment_demo.ipynb` in Jupyter
2. Execute cells sequentially
3. The notebook demonstrates:
   - Package imports and setup
   - Running experiments
   - Visualizing correlation results
   - SHAP weight analysis

---

## Methodology Details

### Calibration Algorithm (Constrained Optimization)

The `SyntheticDataCalibrator` solves a constrained optimization problem to find per-sample weights that align synthetic losses with real validation losses.

**Input:**
- M calibration models {h₁, ..., h_M}
- Synthetic data (X_synth, y_synth) with N samples
- Real validation data (X_real, y_real)
- Regularization λ (default: 0.1)

**Optimization Problem:**

```
w* = argmin_w ||l_r - L^T @ w||² + λ||w||²

Subject to: w >= 0 (non-negativity constraint)
```

Where:
- `L[i, m]` = loss of model m on synthetic sample i (shape: N × M)
- `l_r[m]` = average loss of model m on real validation data (shape: M,)
- `w[i]` = weight for synthetic sample i (shape: N,)


**Output:** Non-negative weights w for all synthetic samples
