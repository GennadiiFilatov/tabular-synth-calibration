"""
Synthetic Data Validation Package.

A framework for validating machine learning models using synthetic data
with calibration techniques to improve rank preservation.
"""

from .utils import RANDOM_SEED, CV_RANDOM_STATE, setup_random_seeds
from .data_loader import DataLoader
from .generation import SyntheticDataGenerator, plot_sdv_training_loss
from .models import ModelSelectionFramework, ModelConfig
from .metrics import EvaluationMetrics
from .confidence import ConfidenceIntervalEstimator
from .runner import ExperimentRunner
from .calibrator import SyntheticDataCalibrator
from .shap_analizer import SHAPWeightsAnalyzer
from .family_composition import run_family_composition_experiment, run_mcal_sweep_experiment

# Try importing optional modules
try:
    from .theory import TheoreticalFramework
except ImportError:
    TheoreticalFramework = None

__all__ = [
    # Core classes
    'ExperimentRunner',
    'DataLoader',
    'SyntheticDataGenerator',
    'ModelSelectionFramework',
    'ModelConfig',
    'SyntheticDataCalibrator',
    'DualSHAPAnalyzer',
    'EvaluationMetrics',
    'ConfidenceIntervalEstimator',
    'TheoreticalFramework',
    'run_family_composition_experiment',
    'run_mcal_sweep_experiment',
    
    # Utilities
    'RANDOM_SEED',
    'CV_RANDOM_STATE',
    'setup_random_seeds',
    'plot_sdv_training_loss',
]

__version__ = '0.1.0'
