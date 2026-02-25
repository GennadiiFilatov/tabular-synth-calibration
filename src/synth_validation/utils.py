"""
Utility constants and helper functions for synthetic data validation.

This module contains shared configuration values and utility functions
used across the synthetic validation framework.
"""

import os
import warnings
import logging

# ============================================================
# Environment Configuration
# ============================================================

# Disable TorchDynamo to prevent circular import issues with SDV/CTGAN
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Allow TabPFN/TabPFGen to run on CPU with large datasets (>1000 samples)
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

# Configure warnings
warnings.filterwarnings('ignore')

# Configure logging for external libraries
logging.getLogger('sdv').setLevel(logging.WARNING)
logging.getLogger('rdt').setLevel(logging.WARNING)
logging.getLogger('ctgan').setLevel(logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)

# ============================================================
# Global Constants
# ============================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation random state
CV_RANDOM_STATE = 42

# Default figure output directory
DEFAULT_FIGURES_DIR = './experiment_figures'

# Default GAN model cache directory
DEFAULT_GAN_CACHE_DIR = './gan_models'


def setup_random_seeds(seed: int = RANDOM_SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    import numpy as np
    import torch
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set torch threads for stability
    torch.set_num_threads(1)


# Initialize seeds on module import
setup_random_seeds(RANDOM_SEED)


def check_optional_dependencies():
    """
    Check availability of optional dependencies and return status dict.
    
    Returns:
        Dict with availability status for each optional dependency
    """
    status = {}
    
    # Check TabPFGen
    try:
        from tabpfgen import TabPFGen
        status['tabpfgen'] = True
    except ImportError:
        status['tabpfgen'] = False
        warnings.warn("TabPFGen not available. Install with: pip install tabpfgen")
    
    # Check synthcity for TabDDPM
    try:
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader
        status['synthcity'] = True
    except (ImportError, AttributeError) as e:
        status['synthcity'] = False
        warnings.warn(f"synthcity not available: {str(e)[:100]}. TabDDPM will be disabled.")
    
    return status


# Check dependencies on import
OPTIONAL_DEPS = check_optional_dependencies()
TABPFGEN_AVAILABLE = OPTIONAL_DEPS.get('tabpfgen', False)
SYNTHCITY_AVAILABLE = OPTIONAL_DEPS.get('synthcity', False)
