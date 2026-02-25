"""
Synthetic Data Generation.

Implementation of multiple synthetic data generators:
- CTGAN (Conditional Tabular GAN)
- TVAE (Tabular Variational Autoencoder)
- GaussianCopula (Statistical method)
- TabPFGen (TabPFN-based generation with SGLD)
- TabDDPM (Denoising Diffusion Probabilistic Model)
"""

import warnings
import traceback
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

from .utils import RANDOM_SEED, TABPFGEN_AVAILABLE, SYNTHCITY_AVAILABLE


def plot_sdv_training_loss(synthesizer, method: str = 'ctgan'):
    """Plot training loss for SDV synthesizers (CTGAN, TVAE)."""
    loss_data = None

    if hasattr(synthesizer, "get_loss_values"):
        try:
            loss_data = synthesizer.get_loss_values()
        except Exception:
            loss_data = None

    if loss_data is None and hasattr(synthesizer, "loss_values"):
        loss_data = synthesizer.loss_values

    if loss_data is None or len(loss_data) == 0:
        print("Loss history is not available for this synthesizer / SDV version.")
        return

    if isinstance(loss_data, dict):
        keys = list(loss_data.keys())
        plt.figure(figsize=(10, 5))
        for k in keys:
            if k != 'Epoch':
                plt.plot(loss_data[k], label=k)
    else:
        try:
            plt.figure(figsize=(10, 5))
            for col in loss_data.columns:
                if col != 'Epoch':
                    plt.plot(loss_data[col], label=col)
        except Exception:
            print("Cannot interpret loss history format.")
            return

    plt.title(f"{method.upper()} Training Loss")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


class SyntheticDataGenerator:
    """
    Handles generation of synthetic tabular data using multiple methods with integrated Optuna tuning.
    
    Supported methods:
        - 'ctgan': CTGAN from SDV library
        - 'tvae': TVAE from SDV library
        - 'gaussian_copula': Gaussian Copula from SDV library
        - 'tabpfgen': TabPFGen - Synthetic data generation with TabPFN (energy-based SGLD)
        - 'tabddpm': TabDDPM diffusion model from synthcity library
    """

    def __init__(self, method: str = 'ctgan', task_type: str = 'classification'):
        """
        Initialize synthetic data generator.

        Args:
            method: 'ctgan', 'tvae', 'gaussian_copula', 'tabpfgen', 'tabddpm'
            task_type: 'classification' or 'regression'
        """
        self.method = method
        self.task_type = task_type
        self.synthesizer = None
        self.metadata = None
        self.best_hyperparams = None
        self.study = None

        # TabPFGen specific attributes
        self._tabpfgen = None
        self.X_train_data = None
        self.y_train_data = None
        self.feature_names = None
        
        # TabDDPM specific attributes
        self._tabddpm_plugin = None
        self._tabddpm_data_loader = None
        self._tabddpm_config = None
        
        # Validate method
        valid_methods = ['ctgan', 'tvae', 'gaussian_copula', 'tabpfgen', 'tabddpm']
        if method not in valid_methods:
            raise ValueError(f"Unknown method '{method}'. Choose from: {valid_methods}")
        
        # Check synthcity availability for tabddpm
        if method == 'tabddpm' and not SYNTHCITY_AVAILABLE:
            raise ImportError("TabDDPM requires synthcity library. Install with: pip install synthcity")
        
        # Validate TabPFGen availability
        if method == 'tabpfgen' and not TABPFGEN_AVAILABLE:
            raise ImportError("TabPFGen is not available. Install with: pip install tabpfgen")

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            epochs: int = 300,
            verbose: bool = False,
            batch_size: int = 500,
            embedding_dim: int = 128,
            generator_dim: Tuple[int, int] = (256, 256),
            discriminator_dim: Tuple[int, int] = (256, 256),
            generator_lr: float = 2e-4,
            discriminator_lr: float = 2e-4,
            pac: int = 10,
            tune_hyperparams: bool = False,
            n_trials: int = 20,
            quality_metric: str = 'mse',
            use_tuned_params: bool = False,
            n_sgld_steps: int = 1000,
            sgld_step_size: float = 0.01,
            sgld_noise_scale: float = 0.01,
            ddpm_n_iter: int = 1000,
            ddpm_lr: float = 2e-3,
            ddpm_num_timesteps: int = 1000,
            ddpm_dim_hidden: int = 256,
            ddpm_depth: int = 2,
            ddpm_dropout: float = 0.0) -> None:
        """
        Fit the synthetic data generator with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (required with tune_hyperparams=True)
            y_val: Validation labels (required with tune_hyperparams=True)
            epochs: Number of training epochs (for SDV methods)
            verbose: Print training progress
            batch_size: Batch size
            embedding_dim: Embedding dimension
            generator_dim: Generator layer dimensions
            discriminator_dim: Discriminator layer dimensions
            generator_lr: Generator learning rate
            discriminator_lr: Discriminator learning rate
            pac: PAC parameter for CTGAN
            tune_hyperparams: If True, run Optuna before general training
            n_trials: Number of Optuna trials
            quality_metric: Quality metric for tuning
            use_tuned_params: Use already found best_hyperparams
            n_sgld_steps: Number of SGLD iterations for TabPFGen
            sgld_step_size: Step size for SGLD updates
            sgld_noise_scale: Scale of noise in SGLD
            ddpm_n_iter: Number of training iterations for TabDDPM
            ddpm_lr: Learning rate for TabDDPM
            ddpm_num_timesteps: Number of diffusion timesteps for TabDDPM
            ddpm_dim_hidden: Hidden dimension for TabDDPM MLP
            ddpm_depth: Number of hidden layers for TabDDPM
            ddpm_dropout: Dropout rate for TabDDPM
        """

        # TabPFGen doesn't support Optuna tuning
        if self.method == 'tabpfgen' and tune_hyperparams:
            warnings.warn("Hyperparameter tuning with Optuna is not supported for TabPFGen.")
            tune_hyperparams = False

        if tune_hyperparams:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val are required when tune_hyperparams=True.")
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"Stage 1: Hyperparameter Tuning ({self.method.upper()})")
                print(f"{'='*80}")
            
            self.best_hyperparams = self._run_optuna_optimization(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                n_trials=n_trials,
                quality_metric=quality_metric,
                verbose=verbose
            )
            
            if verbose:
                print(f"Best hyperparameters: {self.best_hyperparams}\n")
    
        if verbose:
            print(f"Stage 2: Synthesizer training ({self.method.upper()})")
        
        # Combine features and target
        train_data = X_train.copy()
        train_data['target'] = y_train.values
        
        # Store feature names for all methods
        self.feature_names = list(X_train.columns)

        # ============== TabPFGen ==============
        if self.method == 'tabpfgen':
            if not TABPFGEN_AVAILABLE:
                raise ImportError("TabPFGen is not available.")
            
            from tabpfgen import TabPFGen
                
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Store training data for generation
            if len(X_train) > 10000:
                warnings.warn("TabPFGen is optimized for datasets <10k samples. Sampling 10000 rows.")
                indices = np.random.choice(len(X_train), 10000, replace=False)
                self.X_train_data = X_train.iloc[indices].values
                self.y_train_data = y_train.iloc[indices].values
            else:
                self.X_train_data = X_train.values
                self.y_train_data = y_train.values
            
            if verbose:
                print(f"Initializing TabPFGen (Energy-based SGLD generator)...")
                print(f"  n_sgld_steps: {n_sgld_steps}")
                print(f"  device: {device}")
            
            self._tabpfgen = TabPFGen(
                n_sgld_steps=n_sgld_steps,
                sgld_step_size=sgld_step_size,
                sgld_noise_scale=sgld_noise_scale,
                device=device
            )
            
            if verbose:
                print(f"TabPFGen Ready. Training data: {len(self.X_train_data)} samples.")
            return

        # ============== TabDDPM (synthcity) ==============
        if self.method == 'tabddpm':
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import GenericDataLoader
            
            if verbose:
                print("Initializing TabDDPM...")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Get hyperparameters
            if use_tuned_params and self.best_hyperparams:
                n_iter = self.best_hyperparams.get('training_iterations', ddpm_n_iter)
                lr = self.best_hyperparams.get('learning_rate', ddpm_lr)
                batch_sz = self.best_hyperparams.get('batch_size', batch_size)
                num_timesteps = self.best_hyperparams.get('diffusion_timesteps', ddpm_num_timesteps)
            else:
                n_iter = ddpm_n_iter
                lr = ddpm_lr
                batch_sz = batch_size
                num_timesteps = ddpm_num_timesteps
            
            self._tabddpm_config = {
                'n_iter': n_iter,
                'lr': lr,
                'batch_size': batch_sz,
                'num_timesteps': num_timesteps,
                'device': device,
                'is_classification': self.task_type == 'classification'
            }
            
            self._tabddpm_data_loader = GenericDataLoader(
                train_data, target_column='target', sensitive_columns=[]
            )
            
            if verbose:
                print(f"TabDDPM Configuration: {self._tabddpm_config}")
            
            try:
                self._tabddpm_plugin = Plugins().get(
                    "ddpm",
                    n_iter=n_iter, lr=lr, batch_size=batch_sz,
                    num_timesteps=num_timesteps,
                    is_classification=(self.task_type == 'classification'),
                    device=device
                )
                
                if verbose:
                    print(f"Training TabDDPM on {len(train_data)} samples...")
                
                self._tabddpm_plugin.fit(self._tabddpm_data_loader)
                
                if verbose:
                    print("TabDDPM training complete!")
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Falling back to basic DDPM config: {str(e)[:100]}")
                
                self._tabddpm_plugin = Plugins().get(
                    "ddpm", n_iter=n_iter, lr=lr, batch_size=batch_sz,
                    is_classification=(self.task_type == 'classification')
                )
                self._tabddpm_plugin.fit(self._tabddpm_data_loader)
                
                if verbose:
                    print("TabDDPM training complete (fallback config)!")
            
            return

        # ============== SDV Methods (CTGAN, TVAE, GaussianCopula) ==============
        
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_data)

        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            metadata.update_column(column_name=col, sdtype='categorical')

        bool_cols = X_train.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            metadata.update_column(column_name=col, sdtype='boolean')

        numeric_cols = X_train.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            n_unique = X_train[col].nunique()
            if n_unique < 10 and pd.api.types.is_integer_dtype(X_train[col]):
                metadata.update_column(column_name=col, sdtype='categorical')
        
        if self.task_type == 'classification':
            metadata.update_column(column_name='target', sdtype='categorical')
        else:
            metadata.update_column(column_name='target', sdtype='numerical')
        self.metadata = metadata

        cuda_available = torch.cuda.is_available()

        hyperparams = self._get_hyperparameters(
            use_tuned=use_tuned_params or tune_hyperparams,
            default_epochs=epochs,
            default_batch_size=batch_size,
            default_embedding_dim=embedding_dim,
            default_generator_dim=generator_dim,
            default_discriminator_dim=discriminator_dim,
            default_generator_lr=generator_lr,
            default_discriminator_lr=discriminator_lr,
            default_pac=pac
        )

        if self.method == 'ctgan':
            self.synthesizer = CTGANSynthesizer(
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=False,
                epochs=hyperparams['epochs'],
                verbose=verbose,
                cuda=cuda_available,
                batch_size=hyperparams['batch_size'],
                embedding_dim=hyperparams['embedding_dim'],
                generator_dim=hyperparams['generator_dim'],
                discriminator_dim=hyperparams['discriminator_dim'],
                generator_lr=hyperparams['generator_lr'],
                discriminator_lr=hyperparams['discriminator_lr'],
                pac=hyperparams['pac']
            )

        elif self.method == 'tvae':
            self.synthesizer = TVAESynthesizer(
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=False,
                epochs=hyperparams['epochs'],
                verbose=verbose,
                cuda=cuda_available,
                batch_size=hyperparams['batch_size'],
                embedding_dim=hyperparams['embedding_dim'],
                compress_dims=hyperparams['generator_dim'],
                decompress_dims=hyperparams['discriminator_dim']
            )

        elif self.method == 'gaussian_copula':
            self.synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if verbose:
            print(f"Training {self.method.upper()} synthesizer...")
        
        self.synthesizer.fit(train_data)
        
        if verbose:
            print(f"{self.method.upper()} training complete!\n")

        if self.method in ['ctgan', 'tvae']:
            self.plot_training_loss()

    def _run_optuna_optimization(self,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 X_val: pd.DataFrame,
                                 y_val: pd.Series,
                                 n_trials: int,
                                 quality_metric: str,
                                 verbose: bool) -> Dict[str, Any]:
        """Run Optuna optimization for hyperparameter tuning."""

        train_data = X_train.copy()
        train_data['target'] = y_train.values

        def objective(trial: optuna.Trial) -> float:
            try:
                if self.method == 'ctgan':
                    hyperparams = self._suggest_ctgan_hyperparams(trial)
                elif self.method == 'tvae':
                    hyperparams = self._suggest_tvae_hyperparams(trial)
                elif self.method == 'gaussian_copula':
                    hyperparams = self._suggest_gaussian_copula_hyperparams(trial)
                elif self.method == 'tabddpm':
                    hyperparams = self._suggest_tabddpm_hyperparams(trial)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

                temp_synthesizer = self._train_with_hyperparams(train_data, hyperparams)

                n_val_samples = len(X_val)
                
                if self.method == 'tabddpm':
                    synthetic_data = temp_synthesizer.generate(count=n_val_samples).dataframe()
                else:
                    synthetic_data = temp_synthesizer.sample(num_rows=n_val_samples)

                X_synth = synthetic_data.drop(columns=['target'])
                y_synth = synthetic_data['target']

                score = self._calculate_quality_metric(
                    X_val, y_val, X_synth, y_synth, metric=quality_metric
                )

                trial.report(score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except Exception as e:
                if verbose:
                    print(f"  Trial {trial.number} failed: {str(e)}")
                return float('inf')

        sampler = TPESampler(seed=RANDOM_SEED)
        pruner = MedianPruner()

        self.study = optuna.create_study(
            direction='minimize', sampler=sampler, pruner=pruner
        )

        if verbose:
            print(f"Running {n_trials} trials of Optuna optimization...\n")

        self.study.optimize(
            objective, n_trials=n_trials,
            show_progress_bar=verbose, gc_after_trial=True
        )

        return self.study.best_params

    def _suggest_ctgan_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest CTGAN hyperparameters."""
        epochs = trial.suggest_int('epochs', 300, 1500, step=100)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
        pac = trial.suggest_categorical('pac', [1, 2, 4, 8, 16, 32, 64])
        generator_lr = trial.suggest_float("generator_lr", 1e-4, 5e-4, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 1e-4, 5e-4, log=True)
        embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        gen_width = trial.suggest_categorical("gen_width", [128, 256, 512])
        disc_width = trial.suggest_categorical("disc_width", [128, 256, 512])

        return {
            'epochs': epochs, 'batch_size': batch_size, 'pac': pac,
            'generator_lr': generator_lr, 'discriminator_lr': discriminator_lr,
            'embedding_dim': embedding_dim,
            'generator_dim': (gen_width, gen_width),
            'discriminator_dim': (disc_width, disc_width)
        }
    
    def _suggest_tvae_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest TVAE hyperparameters."""
        epochs = trial.suggest_int('epochs', 300, 1500, step=100)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
        embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        compress_width = trial.suggest_categorical("compress_width", [128, 256, 512])
        decompress_width = trial.suggest_categorical("decompress_width", [128, 256, 512])

        return {
            'epochs': epochs, 'batch_size': batch_size, 'embedding_dim': embedding_dim,
            'generator_dim': (compress_width, compress_width),
            'discriminator_dim': (decompress_width, decompress_width),
        }

    def _suggest_gaussian_copula_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Gaussian Copula hyperparameters."""
        return {}

    def _suggest_tabddpm_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest TabDDPM hyperparameters."""
        n_iter = trial.suggest_int('training_iterations', 500, 1000, step=100)
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
        num_timesteps = trial.suggest_categorical('diffusion_timesteps', [50, 100, 200, 500, 1000])

        return {
            'training_iterations': n_iter, 'learning_rate': lr,
            'batch_size': batch_size, 'diffusion_timesteps': num_timesteps
        }

    def _train_with_hyperparams(self, train_data: pd.DataFrame, hyperparams: Dict[str, Any]):
        """Train a synthesizer with specific hyperparameters."""

        if self.method == 'tabddpm':
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import GenericDataLoader
            
            data_loader = GenericDataLoader(
                train_data, target_column='target', sensitive_columns=[]
            )
            
            try:
                plugin = Plugins().get(
                    "ddpm",
                    n_iter=hyperparams.get('training_iterations', 1000),
                    lr=hyperparams.get('learning_rate', 1e-3),
                    batch_size=hyperparams.get('batch_size', 256),
                    num_timesteps=hyperparams.get('diffusion_timesteps', 100),
                    is_classification=(self.task_type == 'classification')
                )
            except Exception:
                plugin = Plugins().get(
                    "ddpm",
                    n_iter=hyperparams.get('training_iterations', 1000),
                    lr=hyperparams.get('learning_rate', 1e-3),
                    batch_size=hyperparams.get('batch_size', 256),
                    is_classification=(self.task_type == 'classification')
                )
            
            plugin.fit(data_loader)
            return plugin
        
        # SDV methods
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_data)
        
        X_cols = [c for c in train_data.columns if c != 'target']
        for col in X_cols:
            if train_data[col].dtype == 'object' or train_data[col].dtype.name == 'category':
                metadata.update_column(column_name=col, sdtype='categorical')
            elif train_data[col].dtype == 'bool':
                metadata.update_column(column_name=col, sdtype='boolean')
        
        if self.task_type == 'classification':
            metadata.update_column(column_name='target', sdtype='categorical')
        else:
            metadata.update_column(column_name='target', sdtype='numerical')
            
        cuda_available = torch.cuda.is_available()
        
        if self.method == 'ctgan':
            synthesizer = CTGANSynthesizer(
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=False,
                epochs=hyperparams['epochs'],
                verbose=False,
                cuda=cuda_available,
                batch_size=hyperparams['batch_size'],
                embedding_dim=hyperparams['embedding_dim'],
                generator_dim=hyperparams['generator_dim'],
                discriminator_dim=hyperparams['discriminator_dim'],
                generator_lr=hyperparams['generator_lr'],
                discriminator_lr=hyperparams['discriminator_lr'],
                pac=hyperparams['pac']
            )
        elif self.method == 'tvae':
            synthesizer = TVAESynthesizer(
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=False,
                epochs=hyperparams['epochs'],
                verbose=False,
                cuda=cuda_available,
                batch_size=hyperparams['batch_size'],
                embedding_dim=hyperparams['embedding_dim'],
                compress_dims=hyperparams['generator_dim'],
                decompress_dims=hyperparams['discriminator_dim']
            )
        elif self.method == 'gaussian_copula':
            synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        synthesizer.fit(train_data)
        return synthesizer

    def _get_hyperparameters(self, use_tuned: bool, default_epochs: int,
                            default_batch_size: int, default_embedding_dim: int,
                            default_generator_dim: Tuple[int, int],
                            default_discriminator_dim: Tuple[int, int],
                            default_generator_lr: float,
                            default_discriminator_lr: float,
                            default_pac: int) -> Dict[str, Any]:
        """Get hyperparameters, using tuned values if available."""
        
        default_params = {
            'epochs': default_epochs, 'batch_size': default_batch_size,
            'embedding_dim': default_embedding_dim,
            'generator_dim': default_generator_dim,
            'discriminator_dim': default_discriminator_dim,
            'generator_lr': default_generator_lr,
            'discriminator_lr': default_discriminator_lr,
            'pac': default_pac
        }

        if use_tuned and self.best_hyperparams:
            merged = default_params.copy()
            
            for key, value in self.best_hyperparams.items():
                if key in ['gen_width']:
                    merged['generator_dim'] = (value, value)
                elif key in ['disc_width']:
                    merged['discriminator_dim'] = (value, value)
                elif key in ['compress_width']:
                    merged['generator_dim'] = (value, value)
                elif key in ['decompress_width']:
                    merged['discriminator_dim'] = (value, value)
                elif key in merged:
                    merged[key] = value
            
            return merged
        
        return default_params

    def _calculate_quality_metric(self, X_real: pd.DataFrame, y_real: pd.Series,
                                  X_synth: pd.DataFrame, y_synth: pd.Series,
                                  metric: str) -> float:
        """Calculate synthetic data quality metric."""
        
        X_real_np = X_real.values.astype(np.float64)
        X_synth_np = X_synth.values.astype(np.float64)
        
        if metric == 'mse':
            return np.mean((X_real_np.mean(axis=0) - X_synth_np.mean(axis=0))**2)
            
        elif metric == 'correlation':
            corr_real = np.corrcoef(X_real_np, rowvar=False)
            corr_synth = np.corrcoef(X_synth_np, rowvar=False)
            corr_real = np.nan_to_num(corr_real, nan=0.0)
            corr_synth = np.nan_to_num(corr_synth, nan=0.0)
            return np.mean(np.abs(corr_real - corr_synth))
            
        elif metric == 'ks_statistic':
            ks_values = []
            for i in range(X_real_np.shape[1]):
                ks_stat, _ = ks_2samp(X_real_np[:, i], X_synth_np[:, i])
                ks_values.append(ks_stat)
            return np.mean(ks_values)

        elif metric == 'swd':
            n_projections = 256
            n_features = X_real_np.shape[1]
            directions = np.random.randn(n_projections, n_features)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            
            proj_real = X_real_np @ directions.T
            proj_synth = X_synth_np @ directions.T
            
            swd_values = []
            for i in range(n_projections):
                swd_values.append(wasserstein_distance(proj_real[:, i], proj_synth[:, i]))
            return np.mean(swd_values)

        elif metric == 'mmd':
            gamma = 1.0 / X_real_np.shape[1]
            
            n_sample = min(500, len(X_real_np), len(X_synth_np))
            idx_real = np.random.choice(len(X_real_np), n_sample, replace=False)
            idx_synth = np.random.choice(len(X_synth_np), n_sample, replace=False)
            
            X_r = X_real_np[idx_real]
            X_s = X_synth_np[idx_synth]
            
            K_rr = np.exp(-gamma * cdist(X_r, X_r, 'sqeuclidean'))
            K_ss = np.exp(-gamma * cdist(X_s, X_s, 'sqeuclidean'))
            K_rs = np.exp(-gamma * cdist(X_r, X_s, 'sqeuclidean'))
            
            mmd = K_rr.mean() + K_ss.mean() - 2 * K_rs.mean()
            return mmd

        elif metric == 'swd_mmd':
            swd = self._calculate_quality_metric(X_real, y_real, X_synth, y_synth, 'swd')
            mmd = self._calculate_quality_metric(X_real, y_real, X_synth, y_synth, 'mmd')
            return 0.5 * swd + 0.5 * mmd

        elif metric == 'fidelity_full':
            swd = self._calculate_quality_metric(X_real, y_real, X_synth, y_synth, 'swd')
            ks = self._calculate_quality_metric(X_real, y_real, X_synth, y_synth, 'ks_statistic')
            corr = self._calculate_quality_metric(X_real, y_real, X_synth, y_synth, 'correlation')
            return (swd + ks + corr) / 3
            
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def generate(self, n_samples: int, balance_classes: bool = True, 
                 use_quantiles: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data.
        
        Args:
            n_samples: Number of synthetic samples to generate
            balance_classes: For classification, whether to generate balanced class distributions
            use_quantiles: For regression, whether to use quantile-based sampling
        
        Returns:
            Tuple of (X_synth DataFrame, y_synth Series)
        """

        # TabPFGen
        if self.method == 'tabpfgen':
            if self._tabpfgen is None:
                raise ValueError("Generator not fitted. Call fit() first.")
            
            if self.task_type == 'classification':
                X_synth_np, y_synth_np = self._tabpfgen.generate_classification(
                    self.X_train_data, self.y_train_data,
                    n_samples=n_samples, balance_classes=balance_classes
                )
            else:
                X_synth_np, y_synth_np = self._tabpfgen.generate_regression(
                    self.X_train_data, self.y_train_data,
                    n_samples=n_samples, use_quantiles=use_quantiles
                )
            
            X_synth = pd.DataFrame(X_synth_np, columns=self.feature_names)
            y_synth = pd.Series(y_synth_np, name='target')
            
            return X_synth, y_synth
        
        # TabDDPM
        if self.method == 'tabddpm':
            if self._tabddpm_plugin is None:
                raise ValueError("Generator not fitted. Call fit() first.")
            
            synthetic_data = self._tabddpm_plugin.generate(count=n_samples).dataframe()
            
            if self.task_type == 'classification':
                y_synth = synthetic_data['target'].astype(int)
            else:
                y_synth = synthetic_data['target'].astype(float)
            X_synth = synthetic_data.drop(columns=['target'])
            
            return X_synth, y_synth
        
        # SDV Methods
        if self.synthesizer is None:
            raise ValueError("Generator not fitted. Call fit() first.")
            
        synthetic_data = self.synthesizer.sample(num_rows=n_samples)
        if self.task_type == 'classification':
            y_synth = synthetic_data['target'].astype(int)
        else:
            y_synth = synthetic_data['target'].astype(float)
        X_synth = synthetic_data.drop(columns=['target'])
        
        return X_synth, y_synth

    def plot_training_loss(self):
        """Plot training loss for CTGAN, TVAE, or TabDDPM."""
        
        if self.method == 'tabpfgen':
            print("TabPFGen uses SGLD sampling - no training loss to display.")
            return
        
        if self.method == 'tabddpm':
            print("TabDDPM training metrics not directly available from synthcity.")
            return
        
        if self.synthesizer is None:
            raise ValueError("Synthesizer not trained. Call fit() first.")
        
        if self.method in ['ctgan', 'tvae']:
            loss_df = self.synthesizer.get_loss_values()
            if loss_df is None or loss_df.empty:
                print("No training loss data available.")
                return
            
            plt.figure(figsize=(10, 6))
            
            if self.method == 'ctgan':
                plt.plot(loss_df['Epoch'], loss_df['Generator Loss'],
                        label='Generator Loss', linewidth=2, marker='o', alpha=0.7)
                plt.plot(loss_df['Epoch'], loss_df['Discriminator Loss'],
                        label='Discriminator Loss', linewidth=2, marker='s', alpha=0.7)
                plt.title('CTGAN Training Loss', fontsize=13, fontweight='bold')
            else:
                plt.plot(loss_df['Epoch'], loss_df['Loss'],
                        label='Reconstruction Loss', linewidth=2, marker='o', alpha=0.7)
                plt.title('TVAE Training Loss', fontsize=13, fontweight='bold')
            
            plt.xlabel('Epoch', fontsize=11)
            plt.ylabel('Loss', fontsize=11)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of the generator."""
        config = {
            'method': self.method,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'best_hyperparams': self.best_hyperparams
        }
        
        if self.method == 'tabddpm' and self._tabddpm_config:
            config['tabddpm_config'] = self._tabddpm_config
            
        return config

    @staticmethod
    def compare_distributions(real_data: pd.DataFrame, synth_data: pd.DataFrame,
                            n_features: int = 5) -> None:
        """Visualize comparison between real and synthetic data distributions."""
        
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns[:n_features]

        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 4))
        if len(numeric_cols) == 1:
            axes = [axes]

        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(real_data[col], bins=30, alpha=0.5, label='Real', density=True)
            axes[idx].hist(synth_data[col], bins=30, alpha=0.5, label='Synthetic', density=True)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].set_title(f'{col} Distribution')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def evaluate_synthetic_quality(X_real: pd.DataFrame, X_synth: pd.DataFrame,
                                   verbose: bool = True) -> Dict[str, float]:
        """Evaluate synthetic data quality with multiple metrics."""
        X_real_np = X_real.values.astype(np.float64)
        X_synth_np = X_synth.values.astype(np.float64)
        
        results = {}
        
        # SWD
        n_projections = 256
        n_features = X_real_np.shape[1]
        directions = np.random.randn(n_projections, n_features)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        proj_real = X_real_np @ directions.T
        proj_synth = X_synth_np @ directions.T
        
        swd_vals = []
        for i in range(n_projections):
            swd_vals.append(wasserstein_distance(proj_real[:, i], proj_synth[:, i]))
        results['swd'] = np.mean(swd_vals)
        
        # KS statistic
        ks_vals = []
        for i in range(n_features):
            ks_stat, _ = ks_2samp(X_real_np[:, i], X_synth_np[:, i])
            ks_vals.append(ks_stat)
        results['ks_mean'] = np.mean(ks_vals)
        
        # Correlation MAE
        corr_real = np.corrcoef(X_real_np, rowvar=False)
        corr_synth = np.corrcoef(X_synth_np, rowvar=False)
        corr_real = np.nan_to_num(corr_real, nan=0.0)
        corr_synth = np.nan_to_num(corr_synth, nan=0.0)
        
        mask = ~np.eye(n_features, dtype=bool)
        results['corr_mae'] = np.mean(np.abs((corr_real - corr_synth)[mask]))
        
        if verbose:
            print("Synthetic Data Quality Evaluation:")
            print(f"  SWD: {results['swd']:.4f} (lower is better)")
            print(f"  KS mean: {results['ks_mean']:.4f} (lower is better)")
            print(f"  Corr MAE: {results['corr_mae']:.4f} (lower is better)")
        
        return results
