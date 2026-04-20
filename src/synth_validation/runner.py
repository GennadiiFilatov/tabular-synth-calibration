"""
Experiment Runner Module.

Main orchestration class for K-fold calibration experiments with synthetic data.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from .utils import RANDOM_SEED, CV_RANDOM_STATE
from .data_loader import DataLoader
from .generation import SyntheticDataGenerator
from .models import ModelSelectionFramework, ModelConfig
from .confidence import ConfidenceIntervalEstimator
from .metrics import EvaluationMetrics
from .calibrator import SyntheticDataCalibrator, SyntheticBPRCalibrator
from .shap_analizer import SHAPWeightsAnalyzer
from typing import Any, Optional, Union, List, Dict, Tuple, Callable


class ExperimentRunner:
    """
    Main orchestration class for K-fold calibration experiments.
    """

    def __init__(self, 
                 dataset_name: str,
                 synth_method: str = 'ctgan',
                 task_type: str = 'classification',
                 loss_type: Optional[str] = None,
                 lambda_reg: float = 0.1,
                 verbose: bool = True,
                 save_figures: bool = False,
                 figures_dir: str = None,
                 gan_cache_dir: str = 'gan_cache',
                 cl_type: str = 'straight',
                 regression_n_bins: int = 10,
                 bpr_eps: float = 0.0,
                 bpr_beta: float = 1.0,
                 bpr_lambda_reg: float = 0.5,
                 bpr_mu: float = 0.0,
                 bpr_tau: float = 1.0):
        """
        Initialize experiment runner.
        
        Args:
            dataset_name: Name of dataset to load
            synth_method: Synthesis method ('ctgan', 'tvae', 'gaussian_copula', 'tabpfgen', 'tabddpm')
            task_type: 'classification' or 'regression'
            loss_type: Loss type for evaluation (auto-selected if None)
            lambda_reg: Regularization for calibration
            verbose: Print progress information
            save_figures: Whether to save generated figures
            figures_dir: Directory for saving figures
            gan_cache_dir: Directory for caching GAN models
            cl_type: Calibration type ('straight', 'per_class')
            regression_n_bins: Number of bins for regression calibration
            bpr_eps: Preference threshold for BPR calibration
            bpr_beta: Sigmoid temperature for BPR calibration
            bpr_lambda_reg: L2 regularization for BPR calibration
            bpr_mu: KL regularization strength for BPR calibration
            bpr_tau: Temperature parameter for BPR calibration
        """
        self.dataset_name = dataset_name
        self.synth_method = synth_method
        self.task_type = task_type
        self.verbose = verbose
        self.save_figures = save_figures
        self.figures_dir = figures_dir
        self.gan_cache_dir = gan_cache_dir
        self.bpr_eps = bpr_eps
        self.bpr_beta = bpr_beta
        self.bpr_lambda_reg = bpr_lambda_reg
        self.bpr_mu = bpr_mu
        self.bpr_tau = bpr_tau
        # Auto-select loss_type based on task
        if loss_type is None:
            loss_type = 'log_loss' if task_type == 'classification' else 'mae'
        self.loss_type = loss_type
        
        # Initialize components
        self.data_loader = DataLoader()
        self.model_selector = ModelSelectionFramework(task_type=task_type, loss_type=loss_type)
        self.calibrator = SyntheticDataCalibrator(
            lambda_reg=lambda_reg,
            verbose=verbose,
            task_type=task_type,
            loss_type=loss_type,
            cl_type=cl_type,
            regression_n_bins=regression_n_bins
        )
        self.ci_estimator = ConfidenceIntervalEstimator()
        self.metrics = EvaluationMetrics()
        
        # State
        self._synthesizer = None
        self._synth_generator = None
        self._last_bpr_calibrator = None
        self._best_hyperparams = None
        self._synth_data_cached = None
        self._figure_counter = 0

        if self.save_figures:
            self._setup_figures_dir()
        
        # Results storage
        self.results = {}
        self.xreal = None
        self.yreal = None
        self.xsynth = None
        self.ysynth = None
    
    def _setup_figures_dir(self):
        """Create figures directory with timestamp."""
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.figures_dir is None:
            self.figures_dir = f"./experiment_figures/{self.dataset_name}_{self.synth_method}_{timestamp}"
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"Figures will be saved to: {self.figures_dir}")

    def _save_figure(self, fig, name: str, dpi: int = 150):
        """Save figure if save_figures is enabled."""
        if not self.save_figures:
            return
        self._figure_counter += 1
        filename = f"{self._figure_counter:02d}_{name}.pdf"
        filepath = Path(self.figures_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, format="pdf", bbox_inches='tight', facecolor='white')
        if self.verbose:
            print(f"Saved: {filepath}")

    def load_gan_model(self, model_dir: str, verbose: bool = True) -> 'ExperimentRunner':
        """Load a trained GAN model."""
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            gan_metadata = json.load(f)
        
        self._best_hyperparams = gan_metadata.get('tune_info')

        if verbose:
            print(f"\nGAN LOADED: {gan_metadata.get('dataset', 'unknown')}")
            print(f"  Method: {gan_metadata.get('method', 'unknown')}")

        if self.synth_method != "tabddpm":
            model_path = model_dir / "synthesizer.pkl"
            with open(model_path, 'rb') as f:
                self._synthesizer = pickle.load(f)
        
        return self

    def save_gan_model(self, model_name: str, verbose: bool = True) -> str:
        """Save a trained GAN model."""
        if self._synthesizer is None:
            raise ValueError("No synthesizer available!")
        
        gan_dir = Path(self.gan_cache_dir)
        gan_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = gan_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'dataset': self.dataset_name,
            'method': self.synth_method,
            'timestamp': timestamp,
            'tune_info': self._best_hyperparams
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.synth_method != "tabddpm":
            model_path = model_dir / "synthesizer.pkl"
            if hasattr(self._synthesizer, 'save'):
                self._synthesizer.save(str(model_path))
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(self._synthesizer, f)
        
        if verbose:
            print(f"Model saved: {model_dir}")
        
        return str(model_dir)

    def _train_generative_model_for_fold(self,
                                          X_train_calib: pd.DataFrame,
                                          y_train_calib: pd.Series,
                                          use_cached_hyperparams: bool = True) -> SyntheticDataGenerator:
        """Train generative model on fold data with fixed hyperparameters."""
        synth_generator = SyntheticDataGenerator(method=self.synth_method, task_type=self.task_type)
        
        if use_cached_hyperparams and self._best_hyperparams:
            synth_generator.best_hyperparams = self._best_hyperparams
            use_tuned = True
        else:
            use_tuned = False
        
        synth_generator.fit(
            X_train=X_train_calib,
            y_train=y_train_calib,
            X_val=None,
            y_val=None,
            tune_hyperparams=False,
            use_tuned_params=use_tuned,
            verbose=self.verbose
        )
        
        return synth_generator

    def run_kfold_calibration_experiment_perclass(self,
                                          n_folds: int = 5,
                                          M_calibration: int = 10,
                                          synth_size_multiplier: float = 1.0,
                                          calib_test_ratio: float = 0.2,
                                          tune_synthetic: bool = False,
                                          n_tune_trials: int = 40,
                                          analyze_shap: bool = True,
                                          shap_plot_types: List[str] = ["dot"],
                                          shap_max_display: int = 15) -> Dict:
        """
        Main experiment pipeline with K-fold cross-validation.
        
        Pipeline for each fold:

        1. Split data: D_train (fold train), D_test (fold test)
        2. Further split D_train -> D_train_calib, D_test_calib
        3. Fit preprocessing on D_train_calib, apply to D_test_calib and D_test
        4. Train generative model on D_train_calib (fixed hyperparams)
        5. Split all architectures -> M_train (calibration), M_test (evaluation)
        6. Train M_train and M_test models on D_train_calib
        7. Generate D_synth_test (size = len(D_test))
        8. Compute losses for M_train on D_test_calib, train calibrator
        9. For M_test: compute losses on D_test, D_synth (uncalibrated), D_synth (calibrated)
        10. Compute Spearman correlation for uncalibrated and calibrated rankings

        Args:
            n_folds: Number of cross-validation folds
            M_calibration: Number of models for calibration
            synth_size_multiplier: Multiplier for synthetic data size
            calib_test_ratio: Ratio of train data for calibration test
            tune_synthetic: Whether to tune GAN hyperparameters
            n_tune_trials: Number of Optuna trials for tuning
            analyze_shap: Whether to perform SHAP analysis
            shap_plot_types: SHAP plot types
            shap_max_display: Max features for SHAP plots
            
        Returns:
            Dictionary with comprehensive experiment results
        """
        # ============================================================
        # [1] LOAD RAW DATA
        # ============================================================
        if self.verbose:
            print("\n" + "="*80)
            print("K-FOLD CALIBRATION EXPERIMENT")
            print("="*80)
            print(f"\n[1/6] Loading dataset: {self.dataset_name}...")
        
        df = self.data_loader.load_uci_dataset(self.dataset_name)
        
        target_col = None
        for col in ['income', 'target', 'class']:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]
        
        X_full = df.drop(columns=[target_col]).copy()
        y_full = df[target_col].copy()
        
        if self.verbose:
            print(f"   Samples: {len(X_full)}, Features: {X_full.shape[1]}")
        
        # ============================================================
        # [2] OPTIONAL: TUNE GAN ON REFERENCE SPLIT
        # ============================================================
        if tune_synthetic and self._synthesizer is None:
            if self.verbose:
                print(f"\n[2/6] Tuning GAN hyperparameters...")
            
            if self.task_type == 'classification':
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE, stratify=y_full
                )
            else:
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )
            
            X_ref_train_proc, _, y_ref_train_proc, _, _ = self.data_loader.prepare_data(
                X_ref_train, X_ref_test, y_ref_train, y_ref_test, task_type=self.task_type
            )

            if self.task_type == 'classification':
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, 
                    random_state=CV_RANDOM_STATE, stratify=y_ref_train_proc
                )
            else:
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )
            
            ref_generator = SyntheticDataGenerator(method=self.synth_method, task_type=self.task_type)
            ref_generator.fit(
                X_train=X_ref_train_proc,
                y_train=y_ref_train_proc,
                X_val=X_ref_val_proc,
                y_val=y_ref_val_proc,
                tune_hyperparams=True,
                n_trials=n_tune_trials,
                quality_metric='swd',
                verbose=self.verbose
            )
            
            self._best_hyperparams = ref_generator.best_hyperparams
            self._synthesizer = ref_generator.synthesizer
            self._synth_generator = ref_generator
            
            if self.verbose:
                print(f"   Best hyperparameters: {self._best_hyperparams}")
        else:
            if self.verbose:
                print(f"\n[2/6] Skipping GAN tuning")
        
        # ============================================================
        # [3] GET ALL MODEL ARCHITECTURES
        # ============================================================
        if self.verbose:
            print(f"\n[3/6] Setting up model architectures...")
        
        all_architectures = self.model_selector.get_model_architectures()
        n_total_models = len(all_architectures)
        
        if M_calibration >= n_total_models:
            raise ValueError(f"M_calibration ({M_calibration}) must be < total architectures ({n_total_models})")
        
        if self.verbose:
            print(f"   Total: {n_total_models}, M_train: {M_calibration}, M_test: {n_total_models - M_calibration}")
        
        # ============================================================
        # [4] SETUP K-FOLD CROSS-VALIDATION
        # ============================================================
        if self.verbose:
            print(f"\n[4/6] Setting up {n_folds}-fold cross-validation...")
        
        if self.task_type == 'classification':
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full, y_full)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full)
        
        fold_results = []
        all_uncalibrated_spearmans = []
        all_calibrated_spearmans = []
        
        # ============================================================
        # [5] RUN EXPERIMENT FOR EACH FOLD
        # ============================================================
        if self.verbose:
            print(f"\n[5/6] Running {n_folds}-fold experiment...")
        
        for fold_idx, (train_index, test_index) in enumerate(fold_iterator):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"FOLD {fold_idx + 1}/{n_folds}")
                print(f"{'='*70}")
            
            # Split data
            X_train_fold = X_full.iloc[train_index].reset_index(drop=True)
            y_train_fold = y_full.iloc[train_index].reset_index(drop=True)
            X_test_fold = X_full.iloc[test_index].reset_index(drop=True)
            y_test_fold = y_full.iloc[test_index].reset_index(drop=True)

            X_train_fold_proc, X_test_proc, y_train_fold_proc, y_test_proc, _ = self.data_loader.prepare_data(
                X_train_fold, X_test_fold, y_train_fold, y_test_fold, task_type=self.task_type
            )

            if self.verbose:
                print(f"   D_train size: {len(X_train_fold_proc)}")
                print(f"   D_test size: {len(X_test_proc)}")
            
            # Split for calibration
            if self.task_type == 'classification':
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx,
                    stratify=y_train_fold_proc
                )
            else:
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx
                )
            
            # Split architectures
            rng = np.random.RandomState(CV_RANDOM_STATE + fold_idx)
            shuffled_architectures = all_architectures.copy()
            rng.shuffle(shuffled_architectures)
            
            architectures_M_train = shuffled_architectures[:M_calibration]
            architectures_M_test = shuffled_architectures[M_calibration:]

            m_train_names = [a.name for a in architectures_M_train]
            m_test_names = [a.name for a in architectures_M_test]
            
            if self.verbose:
                print(f"   M_train models: {m_train_names[:3]}... ({len(m_train_names)} total)")
                print(f"   M_test models: {m_test_names[:3]}... ({len(m_test_names)} total)")
            
            # Train generative model
            if self.verbose:
                print(f"   Training {self.synth_method}...")
            
            fold_generator = self._train_generative_model_for_fold(
                X_train_fold_proc, y_train_fold_proc,
                use_cached_hyperparams=(self._best_hyperparams is not None)
            )
            
            # Generate synthetic data
            n_synth = int(len(X_test_proc) * synth_size_multiplier)
            X_synth, y_synth = fold_generator.generate(n_samples=n_synth)
            
            if self.task_type == 'classification':
                y_synth = y_synth.astype(int)
            else:
                y_synth = y_synth.astype(float)
            
            # Train M_train models
            trained_m_train_models = []
            for config in architectures_M_train:
                model = self.model_selector.train_model(config, X_train_calib, y_train_calib)
                trained_m_train_models.append(model)
            
            # Fit calibrator
            self.calibrator.fit(
                calibration_models=trained_m_train_models,
                X_synth=X_synth,
                y_synth=y_synth,
                X_real_val=X_test_calib,
                y_real_val=y_test_calib
            )
            
            # Train M_test models
            trained_m_test_models = []
            for config in architectures_M_test:
                model = self.model_selector.train_model(config, X_train_fold_proc, y_train_fold_proc)
                trained_m_test_models.append(model)
            
            # Compute losses
            fold_real_losses = []
            fold_synth_losses = []
            fold_calib_losses = []
            fold_model_names = []

            fold_per_sample_real = []
            fold_per_sample_synth = []
            fold_per_sample_calib = []
            
            for config, model in zip(architectures_M_test, trained_m_test_models):
                real_eval = self.model_selector.evaluate_model(model, X_test_proc, y_test_proc)
                synth_eval = self.model_selector.evaluate_model(model, X_synth, y_synth)
                calib_loss = self.calibrator.evaluate_calibrated_loss(model, X_synth, y_synth)
                
                fold_real_losses.append(real_eval['loss'])
                fold_synth_losses.append(synth_eval['loss'])
                fold_calib_losses.append(calib_loss)
                fold_model_names.append(config.name)

                per_sample_real = self.calibrator._compute_sample_losses(model, X_test_proc, y_test_proc)
                per_sample_synth = self.calibrator._compute_sample_losses(model, X_synth, y_synth)
                per_sample_calib = per_sample_synth * self.calibrator.compute_weights_for_samples(y_synth)

                fold_per_sample_real.append(per_sample_real)
                fold_per_sample_synth.append(per_sample_synth)
                fold_per_sample_calib.append(per_sample_calib)
            
            fold_real_losses = np.array(fold_real_losses)
            fold_synth_losses = np.array(fold_synth_losses)
            fold_calib_losses = np.array(fold_calib_losses)
            
            # Compute correlations
            uncalib_spearman, uncalib_pvalue = self.ci_estimator.compute_spearman(
                fold_real_losses, fold_synth_losses
            )
            calib_spearman, calib_pvalue = self.ci_estimator.compute_spearman(
                fold_real_losses, fold_calib_losses
            )
            
            all_uncalibrated_spearmans.append(uncalib_spearman)
            all_calibrated_spearmans.append(calib_spearman)

                        # Rank preservation analysis
            rank_analysis_uncalib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_synth_losses
            )
            rank_analysis_calib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_calib_losses
            )
            
            # Store fold results
            fold_result = {
                'fold': fold_idx + 1,
                'n_train_calib': len(X_train_calib),
                'n_test_calib': len(X_test_calib),
                'n_test': len(X_test_proc),
                'n_synth': n_synth,
                'm_train_architectures': m_train_names,
                'm_test_architectures': m_test_names,
                'real_losses': fold_real_losses,
                'synth_losses': fold_synth_losses,
                'calibrated_synth_losses': fold_calib_losses,
                'uncalibrated_spearman': uncalib_spearman,
                'uncalibrated_pvalue': uncalib_pvalue,
                'calibrated_spearman': calib_spearman,
                'calibrated_pvalue': calib_pvalue,
                'rank_analysis_uncalibrated': rank_analysis_uncalib,
                'rank_analysis_calibrated': rank_analysis_calib,
                'model_names': fold_model_names,
                'per_sample_real_losses': fold_per_sample_real,
                'per_sample_synth_losses': fold_per_sample_synth,
                'per_sample_calibrated_losses': fold_per_sample_calib,
                'weights': self.calibrator.compute_weights_for_samples(y_synth)
            }
            fold_results.append(fold_result)
            
            if self.verbose:
                print(f"\n   Uncalibrated ρ: {uncalib_spearman:.3f}")
                print(f"   Calibrated ρ:   {calib_spearman:.3f} ({calib_spearman - uncalib_spearman:+.3f})")
        
        # ============================================================
        # [6] AGGREGATE RESULTS
        # ============================================================
        if self.verbose:
            print(f"\n[6/6] Computing aggregate statistics...")
        
        uncalib_stats = self.ci_estimator.aggregate_ci_from_samples(all_uncalibrated_spearmans)
        calib_stats = self.ci_estimator.aggregate_ci_from_samples(all_calibrated_spearmans)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"AGGREGATE RESULTS ({n_folds} folds)")
            print(f"{'='*80}")
            print(f"Uncalibrated: {uncalib_stats['mean']:.3f} ± {uncalib_stats['std']:.3f}")
            print(f"  95% CI: [{uncalib_stats['ci_lower']:.3f}, {uncalib_stats['ci_upper']:.3f}]")
            print(f"Calibrated:   {calib_stats['mean']:.3f} ± {calib_stats['std']:.3f}")
            print(f"  95% CI: [{calib_stats['ci_lower']:.3f}, {calib_stats['ci_upper']:.3f}]")
        
        # Cache data
        self.xreal = X_test_proc
        self.yreal = y_test_proc
        self.xsynth = X_synth
        self.ysynth = y_synth
        self._synth_data_cached = (X_synth, y_synth)

        if self._synthesizer is None and fold_generator is not None:
            # Store the underlying synthesizer for SDK methods (CTGAN, TVAE, GaussianCopula)
            if hasattr(fold_generator, 'synthesizer') and fold_generator.synthesizer is not None:
                self._synthesizer = fold_generator.synthesizer
            # For TabPFGen, store the generator itself
            elif hasattr(fold_generator, '_tabpfgen') and fold_generator._tabpfgen is not None:
                self._synthesizer = fold_generator
            # For TabDDPM, store the plugin
            elif hasattr(fold_generator, '_tabddpm_plugin') and fold_generator._tabddpm_plugin is not None:
                self._synthesizer = fold_generator
            else:
                # Fallback: store the whole generator
                self._synthesizer = fold_generator
            self._synth_generator = fold_generator
        
        # SHAP analysis
        shap_analyzer = None
        if analyze_shap and self._synth_data_cached is not None:
            calibration_weights = self.calibrator.compute_weights_for_samples(y_synth)
            
            if calibration_weights is not None and len(calibration_weights) > 0:

                y_real_values = y_test_proc.values if hasattr(y_test_proc, 'values') else np.array(y_test_proc)
                
                shap_analyzer = self.analyze_calibration_weights_with_shap(
                    X_synth=X_synth,
                    y_synth=y_synth,
                    calibration_weights=calibration_weights,
                    X_real=X_test_proc,
                    y_real=y_real_values,
                    plot_types=shap_plot_types,
                    max_display=shap_max_display,
                    verbose=self.verbose
                )

        # Store results
        self.results = {
            'dataset': self.dataset_name,
            'synth_method': self.synth_method,
            'task_type': self.task_type,
            'n_folds': n_folds,
            'M_calibration': M_calibration,
            'n_evaluation_models': n_total_models - M_calibration,
            'fold_results': fold_results,
            'iteration_results': fold_results,  # Backward compatibility
            'uncalibrated_stats': uncalib_stats,
            'calibrated_stats': calib_stats,
            'uncalibrated_spearmans': all_uncalibrated_spearmans,
            'calibrated_spearmans': all_calibrated_spearmans,
            'shap_analyzer': shap_analyzer
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("EXPERIMENT COMPLETE!")
            print(f"{'='*80}")
        
        return self.results
    
    def run_kfold_calibration_experiment(self,
                                          n_folds: int = 5,
                                          M_calibration: int = 10,
                                          synth_size_multiplier: float = 1.0,
                                          calib_test_ratio: float = 0.2,
                                          tune_synthetic: bool = False,
                                          n_tune_trials: int = 40,
                                          analyze_shap: bool = True,
                                          shap_plot_types: List[str] = ["dot"],
                                          shap_max_display: int = 15) -> Dict:
        """
        Main experiment pipeline with K-fold cross-validation.
        
        Pipeline for each fold:

        1. Split data: D_train (fold train), D_test (fold test)
        2. Further split D_train -> D_train_calib, D_test_calib
        3. Fit preprocessing on D_train_calib, apply to D_test_calib and D_test
        4. Train generative model on D_train_calib (fixed hyperparams)
        5. Split all architectures -> M_train (calibration), M_test (evaluation)
        6. Train M_train and M_test models on D_train_calib
        7. Generate D_synth_test (size = len(D_test))
        8. Compute losses for M_train on D_test_calib, train calibrator
        9. For M_test: compute losses on D_test, D_synth (uncalibrated), D_synth (calibrated)
        10. Compute Spearman correlation for uncalibrated and calibrated rankings

        Args:
            n_folds: Number of cross-validation folds
            M_calibration: Number of models for calibration
            synth_size_multiplier: Multiplier for synthetic data size
            calib_test_ratio: Ratio of train data for calibration test
            tune_synthetic: Whether to tune GAN hyperparameters
            n_tune_trials: Number of Optuna trials for tuning
            analyze_shap: Whether to perform SHAP analysis
            shap_plot_types: SHAP plot types
            shap_max_display: Max features for SHAP plots
            
        Returns:
            Dictionary with comprehensive experiment results
        """
        # ============================================================
        # [1] LOAD RAW DATA
        # ============================================================
        if self.verbose:
            print("\n" + "="*80)
            print("K-FOLD CALIBRATION EXPERIMENT")
            print("="*80)
            print(f"\n[1/6] Loading dataset: {self.dataset_name}...")
        
        df = self.data_loader.load_uci_dataset(self.dataset_name)
        
        target_col = None
        for col in ['income', 'target', 'class']:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]
        
        X_full = df.drop(columns=[target_col]).copy()
        y_full = df[target_col].copy()
        
        if self.verbose:
            print(f"   Samples: {len(X_full)}, Features: {X_full.shape[1]}")
        
        # ============================================================
        # [2] OPTIONAL: TUNE GAN ON REFERENCE SPLIT
        # ============================================================
        if tune_synthetic and self._synthesizer is None:
            if self.verbose:
                print(f"\n[2/6] Tuning GAN hyperparameters...")
            
            if self.task_type == 'classification':
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE, stratify=y_full
                )
            else:
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )
            
            X_ref_train_proc, _, y_ref_train_proc, _, _ = self.data_loader.prepare_data(
                X_ref_train, X_ref_test, y_ref_train, y_ref_test, task_type=self.task_type
            )

            if self.task_type == 'classification':
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, 
                    random_state=CV_RANDOM_STATE, stratify=y_ref_train_proc
                )
            else:
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )
            
            ref_generator = SyntheticDataGenerator(method=self.synth_method, task_type=self.task_type)
            ref_generator.fit(
                X_train=X_ref_train_proc,
                y_train=y_ref_train_proc,
                X_val=X_ref_val_proc,
                y_val=y_ref_val_proc,
                tune_hyperparams=True,
                n_trials=n_tune_trials,
                quality_metric='swd',
                verbose=self.verbose
            )
            
            self._best_hyperparams = ref_generator.best_hyperparams
            self._synthesizer = ref_generator.synthesizer
            self._synth_generator = ref_generator
            
            if self.verbose:
                print(f"   Best hyperparameters: {self._best_hyperparams}")
        else:
            if self.verbose:
                print(f"\n[2/6] Skipping GAN tuning")
        
        # ============================================================
        # [3] GET ALL MODEL ARCHITECTURES
        # ============================================================
        if self.verbose:
            print(f"\n[3/6] Setting up model architectures...")
        
        all_architectures = self.model_selector.get_model_architectures()
        n_total_models = len(all_architectures)
        
        if M_calibration >= n_total_models:
            raise ValueError(f"M_calibration ({M_calibration}) must be < total architectures ({n_total_models})")
        
        if self.verbose:
            print(f"   Total: {n_total_models}, M_train: {M_calibration}, M_test: {n_total_models - M_calibration}")
        
        # ============================================================
        # [4] SETUP K-FOLD CROSS-VALIDATION
        # ============================================================
        if self.verbose:
            print(f"\n[4/6] Setting up {n_folds}-fold cross-validation...")
        
        if self.task_type == 'classification':
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full, y_full)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full)
        
        fold_results = []
        all_uncalibrated_spearmans = []
        all_calibrated_spearmans = []
        
        # ============================================================
        # [5] RUN EXPERIMENT FOR EACH FOLD
        # ============================================================
        if self.verbose:
            print(f"\n[5/6] Running {n_folds}-fold experiment...")
        
        for fold_idx, (train_index, test_index) in enumerate(fold_iterator):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"FOLD {fold_idx + 1}/{n_folds}")
                print(f"{'='*70}")
            
            # Split data
            X_train_fold = X_full.iloc[train_index].reset_index(drop=True)
            y_train_fold = y_full.iloc[train_index].reset_index(drop=True)
            X_test_fold = X_full.iloc[test_index].reset_index(drop=True)
            y_test_fold = y_full.iloc[test_index].reset_index(drop=True)

            X_train_fold_proc, X_test_proc, y_train_fold_proc, y_test_proc, _ = self.data_loader.prepare_data(
                X_train_fold, X_test_fold, y_train_fold, y_test_fold, task_type=self.task_type
            )

            if self.verbose:
                print(f"   D_train size: {len(X_train_fold_proc)}")
                print(f"   D_test size: {len(X_test_proc)}")
            
            # Split for calibration
            if self.task_type == 'classification':
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx,
                    stratify=y_train_fold_proc
                )
            else:
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx
                )
            
            # Split architectures
            rng = np.random.RandomState(CV_RANDOM_STATE + fold_idx)
            shuffled_architectures = all_architectures.copy()
            rng.shuffle(shuffled_architectures)
            
            architectures_M_train = shuffled_architectures[:M_calibration]
            architectures_M_test = shuffled_architectures[M_calibration:]

            m_train_names = [a.name for a in architectures_M_train]
            m_test_names = [a.name for a in architectures_M_test]
            
            if self.verbose:
                print(f"   M_train models: {m_train_names[:3]}... ({len(m_train_names)} total)")
                print(f"   M_test models: {m_test_names[:3]}... ({len(m_test_names)} total)")
            
            # Train generative model
            if self.verbose:
                print(f"   Training {self.synth_method}...")
            
            fold_generator = self._train_generative_model_for_fold(
                X_train_fold_proc, y_train_fold_proc,
                use_cached_hyperparams=(self._best_hyperparams is not None)
            )
            
            # Generate synthetic data
            n_synth = int(len(X_test_proc) * synth_size_multiplier)
            X_synth, y_synth = fold_generator.generate(n_samples=n_synth)
            
            if self.task_type == 'classification':
                y_synth = y_synth.astype(int)
            else:
                y_synth = y_synth.astype(float)
            
            # Train M_train models
            trained_m_train_models = []
            for config in architectures_M_train:
                model = self.model_selector.train_model(config, X_train_calib, y_train_calib)
                trained_m_train_models.append(model)
            
            # Fit calibrator
            self.calibrator.fit(
                calibration_models=trained_m_train_models,
                X_synth=X_synth,
                y_synth=y_synth,
                X_real_val=X_test_calib,
                y_real_val=y_test_calib
            )
            
            # Train M_test models
            trained_m_test_models = []
            for config in architectures_M_test:
                model = self.model_selector.train_model(config, X_train_fold_proc, y_train_fold_proc)
                trained_m_test_models.append(model)
            
            # Compute losses
            fold_real_losses = []
            fold_synth_losses = []
            fold_calib_losses = []
            fold_model_names = []

            fold_per_sample_real = []
            fold_per_sample_synth = []
            fold_per_sample_calib = []
            
            for config, model in zip(architectures_M_test, trained_m_test_models):
                real_eval = self.model_selector.evaluate_model(model, X_test_proc, y_test_proc)
                synth_eval = self.model_selector.evaluate_model(model, X_synth, y_synth)
                calib_loss = self.calibrator.evaluate_calibrated_loss(model, X_synth, y_synth)
                
                fold_real_losses.append(real_eval['loss'])
                fold_synth_losses.append(synth_eval['loss'])
                fold_calib_losses.append(calib_loss)
                fold_model_names.append(config.name)

                per_sample_real = self.calibrator._compute_sample_losses(model, X_test_proc, y_test_proc)
                per_sample_synth = self.calibrator._compute_sample_losses(model, X_synth, y_synth)
                per_sample_calib = per_sample_synth * self.calibrator.weights

                fold_per_sample_real.append(per_sample_real)
                fold_per_sample_synth.append(per_sample_synth)
                fold_per_sample_calib.append(per_sample_calib)
            
            fold_real_losses = np.array(fold_real_losses)
            fold_synth_losses = np.array(fold_synth_losses)
            fold_calib_losses = np.array(fold_calib_losses)
            
            # Compute correlations
            uncalib_spearman, uncalib_pvalue = self.ci_estimator.compute_spearman(
                fold_real_losses, fold_synth_losses
            )
            calib_spearman, calib_pvalue = self.ci_estimator.compute_spearman(
                fold_real_losses, fold_calib_losses
            )
            
            all_uncalibrated_spearmans.append(uncalib_spearman)
            all_calibrated_spearmans.append(calib_spearman)

                        # Rank preservation analysis
            rank_analysis_uncalib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_synth_losses
            )
            rank_analysis_calib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_calib_losses
            )
            
            # Store fold results
            fold_result = {
                'fold': fold_idx + 1,
                'n_train_calib': len(X_train_calib),
                'n_test_calib': len(X_test_calib),
                'n_test': len(X_test_proc),
                'n_synth': n_synth,
                'm_train_architectures': m_train_names,
                'm_test_architectures': m_test_names,
                'real_losses': fold_real_losses,
                'synth_losses': fold_synth_losses,
                'calibrated_synth_losses': fold_calib_losses,
                'uncalibrated_spearman': uncalib_spearman,
                'uncalibrated_pvalue': uncalib_pvalue,
                'calibrated_spearman': calib_spearman,
                'calibrated_pvalue': calib_pvalue,
                'rank_analysis_uncalibrated': rank_analysis_uncalib,
                'rank_analysis_calibrated': rank_analysis_calib,
                'model_names': fold_model_names,
                'per_sample_real_losses': fold_per_sample_real,
                'per_sample_synth_losses': fold_per_sample_synth,
                'per_sample_calibrated_losses': fold_per_sample_calib,
                'weights': self.calibrator.compute_weights_for_samples(y_synth)
            }
            fold_results.append(fold_result)
            
            if self.verbose:
                print(f"\n   Uncalibrated ρ: {uncalib_spearman:.3f}")
                print(f"   Calibrated ρ:   {calib_spearman:.3f} ({calib_spearman - uncalib_spearman:+.3f})")
        
        # ============================================================
        # [6] AGGREGATE RESULTS
        # ============================================================
        if self.verbose:
            print(f"\n[6/6] Computing aggregate statistics...")
        
        uncalib_stats = self.ci_estimator.aggregate_ci_from_samples(all_uncalibrated_spearmans)
        calib_stats = self.ci_estimator.aggregate_ci_from_samples(all_calibrated_spearmans)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"AGGREGATE RESULTS ({n_folds} folds)")
            print(f"{'='*80}")
            print(f"Uncalibrated: {uncalib_stats['mean']:.3f} ± {uncalib_stats['std']:.3f}")
            print(f"  95% CI: [{uncalib_stats['ci_lower']:.3f}, {uncalib_stats['ci_upper']:.3f}]")
            print(f"Calibrated:   {calib_stats['mean']:.3f} ± {calib_stats['std']:.3f}")
            print(f"  95% CI: [{calib_stats['ci_lower']:.3f}, {calib_stats['ci_upper']:.3f}]")
        
        # Cache data
        self.xreal = X_test_proc
        self.yreal = y_test_proc
        self.xsynth = X_synth
        self.ysynth = y_synth
        self._synth_data_cached = (X_synth, y_synth)

        if self._synthesizer is None and fold_generator is not None:
            # Store the underlying synthesizer for SDK methods (CTGAN, TVAE, GaussianCopula)
            if hasattr(fold_generator, 'synthesizer') and fold_generator.synthesizer is not None:
                self._synthesizer = fold_generator.synthesizer
            # For TabPFGen, store the generator itself
            elif hasattr(fold_generator, '_tabpfgen') and fold_generator._tabpfgen is not None:
                self._synthesizer = fold_generator
            # For TabDDPM, store the plugin
            elif hasattr(fold_generator, '_tabddpm_plugin') and fold_generator._tabddpm_plugin is not None:
                self._synthesizer = fold_generator
            else:
                # Fallback: store the whole generator
                self._synthesizer = fold_generator
            self._synth_generator = fold_generator
        
        # SHAP analysis
        shap_analyzer = None
        if analyze_shap and self._synth_data_cached is not None:
            calibration_weights = self.calibrator.weights
            
            if calibration_weights is not None and len(calibration_weights) > 0:

                y_real_values = y_test_proc.values if hasattr(y_test_proc, 'values') else np.array(y_test_proc)
                
                shap_analyzer = self.analyze_calibration_weights_with_shap(
                    X_synth=X_synth,
                    y_synth=y_synth,
                    calibration_weights=calibration_weights,
                    X_real=X_test_proc,
                    y_real=y_real_values,
                    plot_types=shap_plot_types,
                    max_display=shap_max_display,
                    verbose=self.verbose
                )

        # Store results
        self.results = {
            'dataset': self.dataset_name,
            'synth_method': self.synth_method,
            'task_type': self.task_type,
            'n_folds': n_folds,
            'M_calibration': M_calibration,
            'n_evaluation_models': n_total_models - M_calibration,
            'fold_results': fold_results,
            'iteration_results': fold_results,  # Backward compatibility
            'uncalibrated_stats': uncalib_stats,
            'calibrated_stats': calib_stats,
            'uncalibrated_spearmans': all_uncalibrated_spearmans,
            'calibrated_spearmans': all_calibrated_spearmans,
            'shap_analyzer': shap_analyzer
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("EXPERIMENT COMPLETE!")
            print(f"{'='*80}")
        
        return self.results

    def _select_diverse_calibration_models(
        self,
        trained_models: List[Any],
        architectures: List[Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        M_calibration: int
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        """
        Select M_calibration models with maximally equal gaps in loss value space.
        Guarantees: all selected models have distinct losses (no duplicates by value).
        """
        all_losses = np.array([
            self.model_selector.evaluate_model(m, X_val, y_val)['loss']
            for m in trained_models
        ])

        n_total = len(trained_models)
        M = min(M_calibration, n_total)

        # Работаем только с уникальными значениями лоссов
        sorted_indices = np.argsort(all_losses)
        sorted_losses  = all_losses[sorted_indices]

        # Убрать дубликаты по значению: оставить первое вхождение каждого уникального лосса
        unique_mask = np.concatenate([[True], np.diff(sorted_losses) > 1e-9])
        unique_sorted_indices = sorted_indices[unique_mask]
        unique_sorted_losses  = sorted_losses[unique_mask]
        n_unique = len(unique_sorted_losses)

        if self.verbose:
            print(f"  Unique loss values: {n_unique}/{n_total}")

        M = min(M, n_unique)

        if M == 1:
            mid = n_unique // 2
            best_selected = [unique_sorted_indices[mid]]

        elif M >= n_unique:
            best_selected = list(unique_sorted_indices)

        else:
            # Перебираем шаги от минимального до максимального возможного
            min_possible_gap = np.diff(unique_sorted_losses).min()
            max_possible_gap = (unique_sorted_losses[-1] - unique_sorted_losses[0]) / (M - 1)

            # Если пул слишком кластеризован — предупредить
            if self.verbose and max_possible_gap < 10 * min_possible_gap:
                print(f"  WARNING: loss pool is highly clustered "
                    f"(range={unique_sorted_losses[-1]-unique_sorted_losses[0]:.4f}, "
                    f"min_gap={min_possible_gap:.4f}). "
                    f"Equal spacing may be limited by pool diversity.")

            best_std = np.inf
            best_selected = None

            for step in np.linspace(min_possible_gap, max_possible_gap, num=200):
                for start_pos in range(n_unique):
                    targets = unique_sorted_losses[start_pos] + step * np.arange(M)

                    # Быстрая проверка: последний target не выходит за пул
                    if targets[-1] > unique_sorted_losses[-1] + 1e-9:
                        break

                    # Snap каждого target к ближайшей ещё не выбранной уникальной модели
                    chosen_pos = []
                    used = set()
                    valid = True

                    for t in targets:
                        dists = np.abs(unique_sorted_losses - t)
                        for u in used:
                            dists[u] = np.inf
                        best_pos = int(np.argmin(dists))

                        if np.isinf(dists[best_pos]):
                            valid = False
                            break

                        # Отклонение snap от цели не должно превышать половину шага
                        # — иначе это не равноудалённость, а произвол
                        if abs(unique_sorted_losses[best_pos] - t) > step * 0.5 + 1e-9:
                            valid = False
                            break

                        chosen_pos.append(best_pos)
                        used.add(best_pos)

                    if not valid or len(chosen_pos) < M:
                        continue

                    gaps = np.diff(unique_sorted_losses[chosen_pos])
                    std = gaps.std()
                    if std < best_std:
                        best_std = std
                        best_selected = [unique_sorted_indices[p] for p in chosen_pos]

            # Fallback: взять равномерно по уникальным позициям (rank-based)
            if best_selected is None:
                positions = np.round(
                    np.linspace(0, n_unique - 1, M)
                ).astype(int)
                positions = list(dict.fromkeys(positions))  # дедупликация позиций
                best_selected = [unique_sorted_indices[p] for p in positions]

                if self.verbose:
                    print("  FALLBACK: rank-based selection (pool too clustered for value-based equal spacing)")

        selected  = best_selected
        remaining = [i for i in range(n_total) if i not in selected]

        if self.verbose:
            sel_losses = np.sort(all_losses[selected])
            gaps = np.diff(sel_losses)
            print(f"  Selected losses : {sel_losses.round(4).tolist()}")
            print(f"  Gaps            : {gaps.round(4).tolist()}")
            print(f"  std(gaps)       : {gaps.std():.5f}  mean(gaps): {gaps.mean():.4f}")

        archs_train  = [architectures[i] for i in selected]
        archs_test   = [architectures[i] for i in remaining]
        models_train = [trained_models[i] for i in selected]
        models_test  = [trained_models[i] for i in remaining]

        return archs_train, archs_test, models_train, models_test

    def run_kfold_bpr_calibration_experiment(self,
                                             n_folds: int = 5,
                                             M_calibration: int = 10,
                                             synth_size_multiplier: float = 1.0,
                                             calib_test_ratio: float = 0.2,
                                             tune_synthetic: bool = False,
                                             n_tune_trials: int = 40,
                                             analyze_shap: bool = True,
                                             shap_plot_types: List[str] = ["dot"],
                                             shap_max_display: int = 15) -> Dict:
        """
        Main experiment pipeline with K-fold cross-validation using BPR calibration.

        Pipeline for each fold:

        1. Split data: D_train (fold train), D_test (fold test)
        2. Further split D_train -> D_train_calib, D_test_calib
        3. Fit preprocessing on D_train_calib, apply to D_test_calib and D_test
        4. Train generative model on D_train_calib (fixed hyperparams)
        5. Split all architectures -> M_train (calibration), M_test (evaluation)
        6. Train M_train and M_test models on D_train_calib
        7. Generate D_synth_test (size = len(D_test))
        8. Compute losses for M_train on D_test_calib, train BPR calibrator
        9. For M_test: compute losses on D_test, D_synth (uncalibrated), D_synth (calibrated)
        10. Compute Spearman correlation for uncalibrated and calibrated rankings

        Args:
            n_folds: Number of cross-validation folds
            M_calibration: Number of models for calibration
            synth_size_multiplier: Multiplier for synthetic data size
            calib_test_ratio: Ratio of train data for calibration test
            tune_synthetic: Whether to tune GAN hyperparameters
            n_tune_trials: Number of Optuna trials for tuning
            analyze_shap: Whether to perform SHAP analysis
            shap_plot_types: SHAP plot types
            shap_max_display: Max features for SHAP plots

        Returns:
            Dictionary with comprehensive experiment results
        """
        bpr_calibrator = SyntheticBPRCalibrator(
            eps=self.bpr_eps,
            beta=self.bpr_beta,
            lambda_reg=self.bpr_lambda_reg,
            mu=self.bpr_mu,
            tau=self.bpr_tau,
            verbose=self.verbose,
            task_type=self.task_type,
            loss_type=self.loss_type
        )

        # ============================================================
        # [1] LOAD RAW DATA
        # ============================================================
        if self.verbose:
            print("\n" + "="*80)
            print("K-FOLD BPR CALIBRATION EXPERIMENT")
            print("="*80)
            print(f"\n[1/6] Loading dataset: {self.dataset_name}...")

        df = self.data_loader.load_uci_dataset(self.dataset_name)

        target_col = None
        for col in ['income', 'target', 'class']:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]

        X_full = df.drop(columns=[target_col]).copy()
        y_full = df[target_col].copy()

        if self.verbose:
            print(f"   Samples: {len(X_full)}, Features: {X_full.shape[1]}")

        # ============================================================
        # [2] OPTIONAL: TUNE GAN ON REFERENCE SPLIT
        # ============================================================
        if tune_synthetic and self._synthesizer is None:
            if self.verbose:
                print(f"\n[2/6] Tuning GAN hyperparameters...")

            if self.task_type == 'classification':
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE, stratify=y_full
                )
            else:
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )

            X_ref_train_proc, _, y_ref_train_proc, _, _ = self.data_loader.prepare_data(
                X_ref_train, X_ref_test, y_ref_train, y_ref_test, task_type=self.task_type
            )

            if self.task_type == 'classification':
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE, stratify=y_ref_train_proc
                )
            else:
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )

            ref_generator = SyntheticDataGenerator(method=self.synth_method, task_type=self.task_type)
            ref_generator.fit(
                X_train=X_ref_train_proc,
                y_train=y_ref_train_proc,
                X_val=X_ref_val_proc,
                y_val=y_ref_val_proc,
                tune_hyperparams=True,
                n_trials=n_tune_trials,
                quality_metric='swd',
                verbose=self.verbose
            )

            self._best_hyperparams = ref_generator.best_hyperparams
            self._synthesizer = ref_generator.synthesizer
            self._synth_generator = ref_generator

            if self.verbose:
                print(f"   Best hyperparameters: {self._best_hyperparams}")
        else:
            if self.verbose:
                print(f"\n[2/6] Skipping GAN tuning")

        # ============================================================
        # [3] GET ALL MODEL ARCHITECTURES
        # ============================================================
        if self.verbose:
            print(f"\n[3/6] Setting up model architectures...")
            print(
                f"   BPR config: eps={self.bpr_eps}, beta={self.bpr_beta}, "
                f"lambda={self.bpr_lambda_reg}, mu={self.bpr_mu}, tau={self.bpr_tau}"
            )

        all_architectures = self.model_selector.get_model_architectures()
        n_total_models = len(all_architectures)

        if M_calibration >= n_total_models:
            raise ValueError(f"M_calibration ({M_calibration}) must be < total architectures ({n_total_models})")

        if self.verbose:
            print(f"   Total: {n_total_models}, M_train: {M_calibration}, M_test: {n_total_models - M_calibration}")

        # ============================================================
        # [4] SETUP K-FOLD CROSS-VALIDATION
        # ============================================================
        if self.verbose:
            print(f"\n[4/6] Setting up {n_folds}-fold cross-validation...")

        if self.task_type == 'classification':
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full, y_full)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full)

        fold_results = []
        all_uncalibrated_spearmans = []
        all_calibrated_spearmans = []

        # ============================================================
        # [5] RUN EXPERIMENT FOR EACH FOLD
        # ============================================================
        if self.verbose:
            print(f"\n[5/6] Running {n_folds}-fold experiment...")

        for fold_idx, (train_index, test_index) in enumerate(fold_iterator):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"FOLD {fold_idx + 1}/{n_folds}")
                print(f"{'='*70}")

            # Split data
            X_train_fold = X_full.iloc[train_index].reset_index(drop=True)
            y_train_fold = y_full.iloc[train_index].reset_index(drop=True)
            X_test_fold = X_full.iloc[test_index].reset_index(drop=True)
            y_test_fold = y_full.iloc[test_index].reset_index(drop=True)

            X_train_fold_proc, X_test_proc, y_train_fold_proc, y_test_proc, _ = self.data_loader.prepare_data(
                X_train_fold, X_test_fold, y_train_fold, y_test_fold, task_type=self.task_type
            )

            if self.verbose:
                print(f"   D_train size: {len(X_train_fold_proc)}")
                print(f"   D_test size: {len(X_test_proc)}")

            # Split for calibration
            if self.task_type == 'classification':
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx,
                    stratify=y_train_fold_proc
                )
            else:
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx
                )

            # Split architectures
            rng = np.random.RandomState(CV_RANDOM_STATE + fold_idx)
            shuffled_architectures = all_architectures.copy()
            rng.shuffle(shuffled_architectures)
            
            architectures_M_train = shuffled_architectures[:M_calibration]
            architectures_M_test = shuffled_architectures[M_calibration:]

            m_train_names = [a.name for a in architectures_M_train]
            m_test_names = [a.name for a in architectures_M_test]
            
            if self.verbose:
                print(f"   M_train models: {m_train_names[:3]}... ({len(m_train_names)} total)")
                print(f"   M_test models: {m_test_names[:3]}... ({len(m_test_names)} total)")
            
            # Train generative model
            if self.verbose:
                print(f"   Training {self.synth_method}...")
            
            fold_generator = self._train_generative_model_for_fold(
                X_train_fold_proc, y_train_fold_proc,
                use_cached_hyperparams=(self._best_hyperparams is not None)
            )
            
            # Generate synthetic data
            n_synth = int(len(X_test_proc) * synth_size_multiplier)
            X_synth, y_synth = fold_generator.generate(n_samples=n_synth)
            
            if self.task_type == 'classification':
                y_synth = y_synth.astype(int)
            else:
                y_synth = y_synth.astype(float)
            
            # Train M_train models
            trained_m_train_models = []
            for config in architectures_M_train:
                model = self.model_selector.train_model(config, X_train_calib, y_train_calib)
                trained_m_train_models.append(model)            

            # Fit BPR calibrator
            bpr_calibrator.fit(
                calibration_models=trained_m_train_models,
                X_synth=X_synth,
                y_synth=y_synth,
                X_real_val=X_test_calib,
                y_real_val=y_test_calib
            )

            fold_weights = bpr_calibrator.compute_weights_for_samples(y_synth)

            # Train M_test models
            trained_m_test_models = []
            for config in architectures_M_test:
                model = self.model_selector.train_model(config, X_train_fold_proc, y_train_fold_proc)
                trained_m_test_models.append(model)

            # Compute losses
            fold_real_losses = []
            fold_synth_losses = []
            fold_calib_losses = []
            fold_model_names = []

            fold_per_sample_real = []
            fold_per_sample_synth = []
            fold_per_sample_calib = []

            for config, model in zip(architectures_M_test, trained_m_test_models):
                real_eval = self.model_selector.evaluate_model(model, X_test_proc, y_test_proc)
                synth_eval = self.model_selector.evaluate_model(model, X_synth, y_synth)
                calib_loss = bpr_calibrator.evaluate_calibrated_loss(model, X_synth, y_synth)

                fold_real_losses.append(real_eval['loss'])
                fold_synth_losses.append(synth_eval['loss'])
                fold_calib_losses.append(calib_loss)
                fold_model_names.append(config.name)

                per_sample_real = bpr_calibrator._compute_sample_losses(model, X_test_proc, y_test_proc)
                per_sample_synth = bpr_calibrator._compute_sample_losses(model, X_synth, y_synth)
                per_sample_calib = per_sample_synth * fold_weights

                fold_per_sample_real.append(per_sample_real)
                fold_per_sample_synth.append(per_sample_synth)
                fold_per_sample_calib.append(per_sample_calib)

            fold_real_losses = np.array(fold_real_losses)
            fold_synth_losses = np.array(fold_synth_losses)
            fold_calib_losses = np.array(fold_calib_losses)

            # Compute correlations
            uncalib_spearman, uncalib_pvalue = self.ci_estimator.compute_kendall(
                fold_real_losses, fold_synth_losses
            )
            calib_spearman, calib_pvalue = self.ci_estimator.compute_kendall(
                fold_real_losses, fold_calib_losses
            )

            all_uncalibrated_spearmans.append(uncalib_spearman)
            all_calibrated_spearmans.append(calib_spearman)

            # Rank preservation analysis
            rank_analysis_uncalib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_synth_losses
            )
            rank_analysis_calib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_calib_losses
            )

            # Store fold results
            fold_result = {
                'fold': fold_idx + 1,
                'n_train_calib': len(X_train_calib),
                'n_test_calib': len(X_test_calib),
                'n_test': len(X_test_proc),
                'n_synth': n_synth,
                'm_train_architectures': m_train_names,
                'm_test_architectures': m_test_names,
                'real_losses': fold_real_losses,
                'synth_losses': fold_synth_losses,
                'calibrated_synth_losses': fold_calib_losses,
                'uncalibrated_spearman': uncalib_spearman,
                'uncalibrated_pvalue': uncalib_pvalue,
                'calibrated_spearman': calib_spearman,
                'calibrated_pvalue': calib_pvalue,
                'rank_analysis_uncalibrated': rank_analysis_uncalib,
                'rank_analysis_calibrated': rank_analysis_calib,
                'model_names': fold_model_names,
                'per_sample_real_losses': fold_per_sample_real,
                'per_sample_synth_losses': fold_per_sample_synth,
                'per_sample_calibrated_losses': fold_per_sample_calib,
                'weights': bpr_calibrator.compute_weights_for_samples(y_synth)
            }
            fold_results.append(fold_result)

            if self.verbose:
                print(f"\n   Uncalibrated ρ: {uncalib_spearman:.3f}")
                print(f"   Calibrated ρ:   {calib_spearman:.3f} ({calib_spearman - uncalib_spearman:+.3f})")

        # ============================================================
        # [6] AGGREGATE RESULTS
        # ============================================================
        if self.verbose:
            print(f"\n[6/6] Computing aggregate statistics...")

        uncalib_stats = self.ci_estimator.aggregate_ci_from_samples(all_uncalibrated_spearmans)
        calib_stats = self.ci_estimator.aggregate_ci_from_samples(all_calibrated_spearmans)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"AGGREGATE RESULTS ({n_folds} folds)")
            print(f"{'='*80}")
            print(f"Uncalibrated: {uncalib_stats['mean']:.3f} ± {uncalib_stats['std']:.3f}")
            print(f"  95% CI: [{uncalib_stats['ci_lower']:.3f}, {uncalib_stats['ci_upper']:.3f}]")
            print(f"Calibrated:   {calib_stats['mean']:.3f} ± {calib_stats['std']:.3f}")
            print(f"  95% CI: [{calib_stats['ci_lower']:.3f}, {calib_stats['ci_upper']:.3f}]")

        # Cache data
        self.xreal = X_test_proc
        self.yreal = y_test_proc
        self.xsynth = X_synth
        self.ysynth = y_synth
        self._synth_data_cached = (X_synth, y_synth)

        if self._synthesizer is None and fold_generator is not None:
            # Store the underlying synthesizer for SDK methods (CTGAN, TVAE, GaussianCopula)
            if hasattr(fold_generator, 'synthesizer') and fold_generator.synthesizer is not None:
                self._synthesizer = fold_generator.synthesizer
            # For TabPFGen, store the generator itself
            elif hasattr(fold_generator, '_tabpfgen') and fold_generator._tabpfgen is not None:
                self._synthesizer = fold_generator
            # For TabDDPM, store the plugin
            elif hasattr(fold_generator, '_tabddpm_plugin') and fold_generator._tabddpm_plugin is not None:
                self._synthesizer = fold_generator
            else:
                # Fallback: store the whole generator
                self._synthesizer = fold_generator
            self._synth_generator = fold_generator

        # SHAP analysis
        shap_analyzer = None
        if analyze_shap and self._synth_data_cached is not None:
            calibration_weights = bpr_calibrator.compute_weights_for_samples(y_synth)

            if calibration_weights is not None and len(calibration_weights) > 0:

                y_real_values = y_test_proc.values if hasattr(y_test_proc, 'values') else np.array(y_test_proc)

                shap_analyzer = self.analyze_calibration_weights_with_shap(
                    X_synth=X_synth,
                    y_synth=y_synth,
                    calibration_weights=calibration_weights,
                    X_real=X_test_proc,
                    y_real=y_real_values,
                    plot_types=shap_plot_types,
                    max_display=shap_max_display,
                    verbose=self.verbose
                )

        # Store results
        self.results = {
            'dataset': self.dataset_name,
            'synth_method': self.synth_method,
            'task_type': self.task_type,
            'n_folds': n_folds,
            'M_calibration': M_calibration,
            'n_evaluation_models': n_total_models - M_calibration,
            'fold_results': fold_results,
            'iteration_results': fold_results,  # Backward compatibility
            'uncalibrated_stats': uncalib_stats,
            'calibrated_stats': calib_stats,
            'uncalibrated_spearmans': all_uncalibrated_spearmans,
            'calibrated_spearmans': all_calibrated_spearmans,
            'shap_analyzer': shap_analyzer
        }

        self._last_bpr_calibrator = bpr_calibrator

        if self.verbose:
            print(f"\n{'='*80}")
            print("EXPERIMENT COMPLETE!")
            print(f"{'='*80}")

        return self.results
    
    def run_kfold_loss_sort_calibration_experiment(self,
                                          n_folds: int = 5,
                                          M_calibration: int = 10,
                                          synth_size_multiplier: float = 1.0,
                                          calib_test_ratio: float = 0.2,
                                          tune_synthetic: bool = False,
                                          n_tune_trials: int = 40,
                                          analyze_shap: bool = True,
                                          shap_plot_types: List[str] = ["dot"],
                                          shap_max_display: int = 15,
                                          setting: int = 1,
                                          var: bool = False) -> Dict:
        """
        Main experiment pipeline with K-fold cross-validation.
        
        Pipeline for each fold:

        1. Split data: D_train (fold train), D_test (fold test)
        2. Further split D_train -> D_train_calib, D_test_calib
        3. Fit preprocessing on D_train_calib, apply to D_test_calib and D_test
        4. Train generative model on D_train_calib (fixed hyperparams)
        5. Split all architectures -> M_train (calibration), M_test (evaluation)
        6. Train M_train and M_test models on D_train_calib
        7. Generate D_synth_test (size = len(D_test))
        8. Compute losses for M_train on D_test_calib, train calibrator
        9. For M_test: compute losses on D_test, D_synth (uncalibrated), D_synth (calibrated)
        10. Compute Spearman correlation for uncalibrated and calibrated rankings

        Args:
            n_folds: Number of cross-validation folds
            M_calibration: Number of models for calibration
            synth_size_multiplier: Multiplier for synthetic data size
            calib_test_ratio: Ratio of train data for calibration test
            tune_synthetic: Whether to tune GAN hyperparameters
            n_tune_trials: Number of Optuna trials for tuning
            analyze_shap: Whether to perform SHAP analysis
            shap_plot_types: SHAP plot types
            shap_max_display: Max features for SHAP plots
            
        Returns:
            Dictionary with comprehensive experiment results
        """
        # ============================================================
        # [1] LOAD RAW DATA
        # ============================================================
        if self.verbose:
            print("\n" + "="*80)
            print("K-FOLD CALIBRATION EXPERIMENT")
            print("="*80)
            print(f"\n[1/6] Loading dataset: {self.dataset_name}...")
        
        df = self.data_loader.load_uci_dataset(self.dataset_name)
        
        target_col = None
        for col in ['income', 'target', 'class']:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]
        
        X_full = df.drop(columns=[target_col]).copy()
        y_full = df[target_col].copy()
        
        if self.verbose:
            print(f"   Samples: {len(X_full)}, Features: {X_full.shape[1]}")
        
        # ============================================================
        # [2] OPTIONAL: TUNE GAN ON REFERENCE SPLIT
        # ============================================================
        if tune_synthetic and self._synthesizer is None:
            if self.verbose:
                print(f"\n[2/6] Tuning GAN hyperparameters...")
            
            if self.task_type == 'classification':
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE, stratify=y_full
                )
            else:
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )
            
            X_ref_train_proc, _, y_ref_train_proc, _, _ = self.data_loader.prepare_data(
                X_ref_train, X_ref_test, y_ref_train, y_ref_test, task_type=self.task_type
            )

            if self.task_type == 'classification':
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, 
                    random_state=CV_RANDOM_STATE, stratify=y_ref_train_proc
                )
            else:
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )
            
            ref_generator = SyntheticDataGenerator(method=self.synth_method, task_type=self.task_type)
            ref_generator.fit(
                X_train=X_ref_train_proc,
                y_train=y_ref_train_proc,
                X_val=X_ref_val_proc,
                y_val=y_ref_val_proc,
                tune_hyperparams=True,
                n_trials=n_tune_trials,
                quality_metric='swd',
                verbose=self.verbose
            )
            
            self._best_hyperparams = ref_generator.best_hyperparams
            self._synthesizer = ref_generator.synthesizer
            self._synth_generator = ref_generator
            
            if self.verbose:
                print(f"   Best hyperparameters: {self._best_hyperparams}")
        else:
            if self.verbose:
                print(f"\n[2/6] Skipping GAN tuning")
        
        # ============================================================
        # [3] GET ALL MODEL ARCHITECTURES
        # ============================================================
        if self.verbose:
            print(f"\n[3/6] Setting up model architectures...")
        
        all_architectures = self.model_selector.get_model_architectures()
        n_total_models = len(all_architectures)
        
        if M_calibration >= n_total_models:
            raise ValueError(f"M_calibration ({M_calibration}) must be < total architectures ({n_total_models})")
        
        if self.verbose:
            print(f"   Total: {n_total_models}, M_train: {M_calibration}, M_test: {n_total_models - M_calibration}")
        
        # ============================================================
        # [4] SETUP K-FOLD CROSS-VALIDATION
        # ============================================================
        if self.verbose:
            print(f"\n[4/6] Setting up {n_folds}-fold cross-validation...")
        
        if self.task_type == 'classification':
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full, y_full)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full)
        
        fold_results = []
        all_uncalibrated_spearmans = []
        all_calibrated_spearmans = []
        
        # ============================================================
        # [5] RUN EXPERIMENT FOR EACH FOLD
        # ============================================================
        if self.verbose:
            print(f"\n[5/6] Running {n_folds}-fold experiment...")
        
        for fold_idx, (train_index, test_index) in enumerate(fold_iterator):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"FOLD {fold_idx + 1}/{n_folds}")
                print(f"{'='*70}")
            
            # Split data
            X_train_fold = X_full.iloc[train_index].reset_index(drop=True)
            y_train_fold = y_full.iloc[train_index].reset_index(drop=True)
            X_test_fold = X_full.iloc[test_index].reset_index(drop=True)
            y_test_fold = y_full.iloc[test_index].reset_index(drop=True)

            X_train_fold_proc, X_test_proc, y_train_fold_proc, y_test_proc, _ = self.data_loader.prepare_data(
                X_train_fold, X_test_fold, y_train_fold, y_test_fold, task_type=self.task_type
            )

            if self.verbose:
                print(f"   D_train size: {len(X_train_fold_proc)}")
                print(f"   D_test size: {len(X_test_proc)}")
            
            # Split for calibration
            if self.task_type == 'classification':
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx,
                    stratify=y_train_fold_proc
                )
            else:
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx
                )

            # Train ALL architectures on calib split; compute real losses to determine M_train / M_test split
            real_losses = {}
            trained_all_models  = {}
            
            for config in all_architectures:
                model = self.model_selector.train_model(config, X_train_calib, y_train_calib)
                
                if var==True:
                    real_losses[config.name] = self.calibrator._compute_real_var(model, X_test_calib, y_test_calib)
                else:
                    real_losses[config.name] = self.calibrator._compute_real_loss(model, X_test_calib, y_test_calib)

                trained_all_models [config.name] = model
            
            real_losses = dict(sorted(real_losses.items(), key=lambda x: x[1]))

            print(real_losses)

            head_num = M_calibration // 2
            tail_num = M_calibration - head_num

            if setting == 1:
                m_train_names = list(real_losses.keys())[:head_num] + list(real_losses.keys())[-tail_num:]
                m_test_names = list(real_losses.keys())[M_calibration:-M_calibration]
            elif setting == 2:
                m_train_names = list(real_losses.keys())[:M_calibration]
                m_test_names = list(real_losses.keys())[M_calibration:-M_calibration]
            elif setting == 3:
                m_train_names = list(real_losses.keys())[-M_calibration:]
                m_test_names = list(real_losses.keys())[M_calibration:-M_calibration]


            architectures_M_test = [c for c in all_architectures if c.name in set(m_test_names)]
            
            if self.verbose:
                print(f"   M_train models: {m_train_names[:3]}... ({len(m_train_names)} total)")
                print(f"   M_test models: {m_test_names[:3]}... ({len(m_test_names)} total)")
            
            # Train generative model
            if self.verbose:
                print(f"   Training {self.synth_method}...")
            
            fold_generator = self._train_generative_model_for_fold(
                X_train_fold_proc, y_train_fold_proc,
                use_cached_hyperparams=(self._best_hyperparams is not None)
            )
            
            # Generate synthetic data
            n_synth = int(len(X_test_proc) * synth_size_multiplier)
            X_synth, y_synth = fold_generator.generate(n_samples=n_synth)
            
            if self.task_type == 'classification':
                y_synth = y_synth.astype(int)
            else:
                y_synth = y_synth.astype(float)
            
            # Fit calibrator
            self.calibrator.fit(
                calibration_models=[trained_all_models[name] for name in m_train_names],
                X_synth=X_synth,
                y_synth=y_synth,
                X_real_val=X_test_calib,
                y_real_val=y_test_calib
            )
            
            # Train M_test models
            trained_m_test_models = [self.model_selector.train_model(config, X_train_fold_proc, y_train_fold_proc)
                for config in architectures_M_test
            ]
            
            # Compute losses
            fold_real_losses = []
            fold_synth_losses = []
            fold_calib_losses = []
            fold_model_names = []

            fold_per_sample_real = []
            fold_per_sample_synth = []
            fold_per_sample_calib = []
            
            for name, model in zip(m_test_names, trained_m_test_models):
                real_eval = self.model_selector.evaluate_model(model, X_test_proc, y_test_proc)
                synth_eval = self.model_selector.evaluate_model(model, X_synth, y_synth)
                calib_loss = self.calibrator.evaluate_calibrated_loss(model, X_synth, y_synth)
                
                fold_real_losses.append(real_eval['loss'])
                fold_synth_losses.append(synth_eval['loss'])
                fold_calib_losses.append(calib_loss)
                fold_model_names.append(name)

                per_sample_real = self.calibrator._compute_sample_losses(model, X_test_proc, y_test_proc)
                per_sample_synth = self.calibrator._compute_sample_losses(model, X_synth, y_synth)
                per_sample_calib = per_sample_synth * self.calibrator.weights

                fold_per_sample_real.append(per_sample_real)
                fold_per_sample_synth.append(per_sample_synth)
                fold_per_sample_calib.append(per_sample_calib)
            
            fold_real_losses = np.array(fold_real_losses)
            fold_synth_losses = np.array(fold_synth_losses)
            fold_calib_losses = np.array(fold_calib_losses)
            
            # Compute correlations
            uncalib_spearman, uncalib_pvalue = self.ci_estimator.compute_spearman(
                fold_real_losses, fold_synth_losses
            )
            calib_spearman, calib_pvalue = self.ci_estimator.compute_spearman(
                fold_real_losses, fold_calib_losses
            )
            
            all_uncalibrated_spearmans.append(uncalib_spearman)
            all_calibrated_spearmans.append(calib_spearman)

            # Rank preservation analysis
            rank_analysis_uncalib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_synth_losses
            )
            rank_analysis_calib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_calib_losses
            )
            
            # Store fold results
            fold_result = {
                'fold': fold_idx + 1,
                'n_train_calib': len(X_train_calib),
                'n_test_calib': len(X_test_calib),
                'n_test': len(X_test_proc),
                'n_synth': n_synth,
                'm_train_architectures': m_train_names,
                'm_test_architectures': m_test_names,
                'real_losses': fold_real_losses,
                'synth_losses': fold_synth_losses,
                'calibrated_synth_losses': fold_calib_losses,
                'uncalibrated_spearman': uncalib_spearman,
                'uncalibrated_pvalue': uncalib_pvalue,
                'calibrated_spearman': calib_spearman,
                'calibrated_pvalue': calib_pvalue,
                'rank_analysis_uncalibrated': rank_analysis_uncalib,
                'rank_analysis_calibrated': rank_analysis_calib,
                'model_names': fold_model_names,
                'per_sample_real_losses': fold_per_sample_real,
                'per_sample_synth_losses': fold_per_sample_synth,
                'per_sample_calibrated_losses': fold_per_sample_calib,
                'weights': self.calibrator.weights.copy()
            }
            fold_results.append(fold_result)
            
            if self.verbose:
                print(f"\n   Uncalibrated ρ: {uncalib_spearman:.3f}")
                print(f"   Calibrated ρ:   {calib_spearman:.3f} ({calib_spearman - uncalib_spearman:+.3f})")
        
        # ============================================================
        # [6] AGGREGATE RESULTS
        # ============================================================
        if self.verbose:
            print(f"\n[6/6] Computing aggregate statistics...")
        
        uncalib_stats = self.ci_estimator.aggregate_ci_from_samples(all_uncalibrated_spearmans)
        calib_stats = self.ci_estimator.aggregate_ci_from_samples(all_calibrated_spearmans)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"AGGREGATE RESULTS ({n_folds} folds)")
            print(f"{'='*80}")
            print(f"Uncalibrated: {uncalib_stats['mean']:.3f} ± {uncalib_stats['std']:.3f}")
            print(f"  95% CI: [{uncalib_stats['ci_lower']:.3f}, {uncalib_stats['ci_upper']:.3f}]")
            print(f"Calibrated:   {calib_stats['mean']:.3f} ± {calib_stats['std']:.3f}")
            print(f"  95% CI: [{calib_stats['ci_lower']:.3f}, {calib_stats['ci_upper']:.3f}]")
        
        # Cache data
        self.xreal = X_test_proc
        self.yreal = y_test_proc
        self.xsynth = X_synth
        self.ysynth = y_synth
        self._synth_data_cached = (X_synth, y_synth)

        if self._synthesizer is None and fold_generator is not None:
            # Store the underlying synthesizer for SDK methods (CTGAN, TVAE, GaussianCopula)
            if hasattr(fold_generator, 'synthesizer') and fold_generator.synthesizer is not None:
                self._synthesizer = fold_generator.synthesizer
            # For TabPFGen, store the generator itself
            elif hasattr(fold_generator, '_tabpfgen') and fold_generator._tabpfgen is not None:
                self._synthesizer = fold_generator
            # For TabDDPM, store the plugin
            elif hasattr(fold_generator, '_tabddpm_plugin') and fold_generator._tabddpm_plugin is not None:
                self._synthesizer = fold_generator
            else:
                # Fallback: store the whole generator
                self._synthesizer = fold_generator
            self._synth_generator = fold_generator
        
        # SHAP analysis
        shap_analyzer = None
        if analyze_shap and self._synth_data_cached is not None:
            calibration_weights = self.calibrator.weights
            
            if calibration_weights is not None and len(calibration_weights) > 0:

                y_real_values = y_test_proc.values if hasattr(y_test_proc, 'values') else np.array(y_test_proc)
                
                shap_analyzer = self.analyze_calibration_weights_with_shap(
                    X_synth=X_synth,
                    y_synth=y_synth,
                    calibration_weights=calibration_weights,
                    X_real=X_test_proc,
                    y_real=y_real_values,
                    plot_types=shap_plot_types,
                    max_display=shap_max_display,
                    verbose=self.verbose
                )

        # Store results
        self.results = {
            'dataset': self.dataset_name,
            'synth_method': self.synth_method,
            'task_type': self.task_type,
            'n_folds': n_folds,
            'M_calibration': M_calibration,
            'n_evaluation_models': n_total_models - M_calibration,
            'fold_results': fold_results,
            'iteration_results': fold_results,  # Backward compatibility
            'uncalibrated_stats': uncalib_stats,
            'calibrated_stats': calib_stats,
            'uncalibrated_spearmans': all_uncalibrated_spearmans,
            'calibrated_spearmans': all_calibrated_spearmans,
            'shap_analyzer': shap_analyzer
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("EXPERIMENT COMPLETE!")
            print(f"{'='*80}")
        
        return self.results
    
    def _get_model_names(self, method_name) -> set:
        """Return set of linear model names for calibration."""
        if self.task_type == 'classification':
            if method_name == 'linear':
                return {
                    'LogReg_L2_LBFGS', 'LogReg_L1_SAGA', 'LogReg_ElasticNet',
                    'LogReg_L2_LBFGS_CV', 'LogReg_None', 'SGD_Log_Loss',
                    'Calibrated_LinearSVC', 'Calibrated_SGD_Hinge', 'Calibrated_LinearSVC_Isotonic',
                    'SVC_Linear', 'LDA_SVD'
                }
            elif method_name == 'ensemble':
                return {
                    'RandomForest_Gini', 'RandomForest_Entropy',
                    'RandomForest_ShallowTrees', 'RandomForest_DeepTrees',
                    'RandomForest_Bootstrap', 'GradientBoosting',
                    'GradientBoosting_Deep', 'HistGradientBoosting',
                    'ExtraTrees', 'ExtraTrees_Entropy',
                    'Bagging_Tree', 'Bagging_LogReg',
                    'AdaBoost_Tree', 'AdaBoost_LogReg', 'AdaBoost_LowLR'
                }
            elif method_name == 'svm':
                return {
                    'SVC_RBF', 'SVC_Poly', 'SVC_Linear', 
                    'SVC_Sigmoid', 'NuSVC'
                }
            elif method_name == 'knn':
                return {
                    'KNN_k3_Uniform', 'KNN_k7_Distance', 'KNN_k15_Uniform',
                    'KNN_L1_k5_Uniform', 'KNN_L1_k10_Distance', 'KNN_Cosine_k10_Distance'
                }
            elif method_name == 'naive bayes':
                return {
                    'GaussianNB_Default', 'GaussianNB_Smooth1e-8', 'BernoulliNB_Alpha1_Bin0',
                    'BernoulliNB_Alpha0p1_Bin05', 'MultinomialNB_Alpha1', 'MultinomialNB_Alpha0p1',
                    'ComplementNB_Alpha1_NoNorm', 'ComplementNB_Alpha1_Norm'
                }
            elif method_name == 'nn':
                return {
                    'MLP_ReLU_Adam_Mid', 'MLP_ReLU_Adam_Deep', 'MLP_ReLU_Adam_Wide',
                    'MLP_ReLU_Adam_Small', 'MLP_Tanh_SGD_Mid', 'MLP_Tanh_SGD_Deep',
                    'MLP_Logistic_Adam', 'MLP_ReLU_LBFGS_Small'
                }
            elif method_name == 'tree':
                return {
                    'DecisionTree_Gini_Full', 'DecisionTree_Gini_Depth10', 'DecisionTree_Entropy_Depth15',
                    'DecisionTree_Gini_RandomSplit', 'ExtraTree_Gini_Deep', 'ExtraTree_Entropy_Shallower',
                    'ExtraTree_Gini_Full_MaxFeatSqrt', 'ExtraTree_Gini_Depth3'
                }

        else:  # regression
            if method_name == 'linear':
                return {
                    'LinearRegression', 'Ridge_Alpha1', 'Ridge_Alpha10', 'Ridge_Alpha01',
                    'Lasso_Alpha1', 'Lasso_Alpha01', 'ElasticNet_L1_05', 'ElasticNet_L1_02',
                    'SGD_L2', 'SGD_L1', 'SGD_Huber', 'SVR_Linear'
                }
            elif method_name == 'svm':
                return {
                    'SVR_RBF', 'SVR_Linear', 'SVR_Poly',
                    'NuSVR', 'SVR_RBF_C1_e0p1', 'SVR_RBF_C10_e0p01',
                    'SVR_RBF_C0p1_e0p2', 'LinSVR_C1', 'LinSVR_C0p1_L2Loss',
                    'SVR_Poly_deg2', 'SVR_Poly_deg3_C10', 'NuSVR_RBF_nu0p5'
                }
            elif method_name == 'knn':
                return {
                    'KNN_k3_Uniform', 'KNN_k7_Distance', 'KNN_k15_Uniform',
                    'KNN_L1_k5_Uniform', 'KNN_L1_k10_Distance', 'KNN_Cosine_k10_Distance'
                }
            elif method_name == 'tree':
                return {
                    'DTree_SqErr_Full', 'DTree_SqErr_Shallow', 'DTree_SqErr_Depth15_MaxFeatSqrt',
                    'DTree_AbsErr_Depth15', 'ETree_SqErr_Full', 'ETree_SqErr_Depth10',
                    'ETree_AbsErr_Depth15_MaxFeatSqrt', 'ETree_SqErr_Depth3'
                }
            elif method_name == 'nn':
                return {
                    'MLP_ReLU_Adam_Mid', 'MLP_ReLU_Adam_Deep', 'MLP_ReLU_Adam_Wide',
                    'MLP_ReLU_Adam_Small', 'MLP_Tanh_SGD_Mid', 'MLP_Tanh_SGD_Deep',
                    'MLP_Logistic_Adam', 'MLP_ReLU_LBFGS_Small'
                }

    def run_method_calibration_experiment(self,
                                          n_folds: int = 5,
                                          M: int = 10,
                                          method_name: str = 'linear',
                                          synth_size_multiplier: float = 1.0,
                                          calib_test_ratio: float = 0.2,
                                          tune_synthetic: bool = False,
                                          n_tune_trials: int = 40,
                                          analyze_shap: bool = True,
                                          shap_plot_types: List[str] = ["dot"],
                                          shap_max_display: int = 15) -> Dict:
        """
        Linear-model calibration experiment with K-fold cross-validation.
        
        Uses only linear models for calibration and all non-linear models for evaluation.
        
        Pipeline for each fold:

        1. Split data: D_train (fold train), D_test (fold test)
        2. Further split D_train -> D_train_calib, D_test_calib
        3. Fit preprocessing on D_train_calib, apply to D_test_calib and D_test
        4. Train generative model on D_train_calib (fixed hyperparams)
        5. Split architectures: linear models -> M_train (calibration), non-linear -> M_test (evaluation)
        6. Train M_train and M_test models on D_train_calib
        7. Generate D_synth_test (size = len(D_test))
        8. Compute losses for M_train on D_test_calib, train calibrator
        9. For M_test: compute losses on D_test, D_synth (uncalibrated), D_synth (calibrated)
        10. Compute Spearman correlation for uncalibrated and calibrated rankings

        Args:
            n_folds: Number of cross-validation folds
            synth_size_multiplier: Multiplier for synthetic data size
            calib_test_ratio: Ratio of train data for calibration test
            tune_synthetic: Whether to tune GAN hyperparameters
            n_tune_trials: Number of Optuna trials for tuning
            analyze_shap: Whether to perform SHAP analysis
            shap_plot_types: SHAP plot types
            shap_max_display: Max features for SHAP plots
            
        Returns:
            Dictionary with comprehensive experiment results
        """
        # ============================================================
        # [1] LOAD RAW DATA
        # ============================================================
        if self.verbose:
            print("\n" + "="*80)
            print("LINEAR-MODEL CALIBRATION EXPERIMENT")
            print("="*80)
            print(f"\n[1/6] Loading dataset: {self.dataset_name}...")
        
        df = self.data_loader.load_uci_dataset(self.dataset_name)
        
        target_col = None
        for col in ['income', 'target', 'class']:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]
        
        X_full = df.drop(columns=[target_col]).copy()
        y_full = df[target_col].copy()
        
        if self.verbose:
            print(f"   Samples: {len(X_full)}, Features: {X_full.shape[1]}")
        
        # ============================================================
        # [2] OPTIONAL: TUNE GAN ON REFERENCE SPLIT
        # ============================================================
        if tune_synthetic and self._synthesizer is None:
            if self.verbose:
                print(f"\n[2/6] Tuning GAN hyperparameters...")
            
            if self.task_type == 'classification':
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE, stratify=y_full
                )
            else:
                X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
                    X_full, y_full, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )
            
            X_ref_train_proc, _, y_ref_train_proc, _, _ = self.data_loader.prepare_data(
                X_ref_train, X_ref_test, y_ref_train, y_ref_test, task_type=self.task_type
            )

            if self.task_type == 'classification':
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, 
                    random_state=CV_RANDOM_STATE, stratify=y_ref_train_proc
                )
            else:
                X_ref_train_proc, X_ref_val_proc, y_ref_train_proc, y_ref_val_proc = train_test_split(
                    X_ref_train_proc, y_ref_train_proc, test_size=calib_test_ratio, random_state=CV_RANDOM_STATE
                )
            
            ref_generator = SyntheticDataGenerator(method=self.synth_method, task_type=self.task_type)
            ref_generator.fit(
                X_train=X_ref_train_proc,
                y_train=y_ref_train_proc,
                X_val=X_ref_val_proc,
                y_val=y_ref_val_proc,
                tune_hyperparams=True,
                n_trials=n_tune_trials,
                quality_metric='swd',
                verbose=self.verbose
            )
            
            self._best_hyperparams = ref_generator.best_hyperparams
            self._synthesizer = ref_generator.synthesizer
            self._synth_generator = ref_generator
            
            if self.verbose:
                print(f"   Best hyperparameters: {self._best_hyperparams}")
        else:
            if self.verbose:
                print(f"\n[2/6] Skipping GAN tuning")
        
        # ============================================================
        # [3] GET ALL MODEL ARCHITECTURES
        # ============================================================
        if self.verbose:
            print(f"\n[3/6] Setting up model architectures...")
        
        all_architectures = self.model_selector.get_model_architectures()
        n_total_models = len(all_architectures)
        
        # Separate linear models (for calibration) from non-linear (for evaluation)
        linear_model_names = self._get_model_names(method_name)
        architectures_M_train = [a for a in all_architectures if a.name in linear_model_names]
        architectures_M_test = [a for a in all_architectures if a.name not in linear_model_names]
        
        M_calibration = len(architectures_M_train)
        
        if M_calibration == 0:
            raise ValueError("No linear models found for calibration!")
        if len(architectures_M_test) == 0:
            raise ValueError("No non-linear models found for evaluation!")
        
        if self.verbose:
            print(f"   Total: {n_total_models}, Linear (M_train): {M_calibration}, Non-linear (M_test): {len(architectures_M_test)}")
        
        # ============================================================
        # [4] SETUP K-FOLD CROSS-VALIDATION
        # ============================================================
        if self.verbose:
            print(f"\n[4/6] Setting up {n_folds}-fold cross-validation...")
        
        if self.task_type == 'classification':
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full, y_full)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=CV_RANDOM_STATE)
            fold_iterator = kfold.split(X_full)
        
        fold_results = []
        all_uncalibrated_spearmans = []
        all_calibrated_spearmans = []
        
        # ============================================================
        # [5] RUN EXPERIMENT FOR EACH FOLD
        # ============================================================
        if self.verbose:
            print(f"\n[5/6] Running {n_folds}-fold experiment...")
        
        for fold_idx, (train_index, test_index) in enumerate(fold_iterator):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"FOLD {fold_idx + 1}/{n_folds}")
                print(f"{'='*70}")
            
            # Split data
            X_train_fold = X_full.iloc[train_index].reset_index(drop=True)
            y_train_fold = y_full.iloc[train_index].reset_index(drop=True)
            X_test_fold = X_full.iloc[test_index].reset_index(drop=True)
            y_test_fold = y_full.iloc[test_index].reset_index(drop=True)

            X_train_fold_proc, X_test_proc, y_train_fold_proc, y_test_proc, _ = self.data_loader.prepare_data(
                X_train_fold, X_test_fold, y_train_fold, y_test_fold, task_type=self.task_type
            )

            if self.verbose:
                print(f"   D_train size: {len(X_train_fold_proc)}")
                print(f"   D_test size: {len(X_test_proc)}")
            
            # Split for calibration
            if self.task_type == 'classification':
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx,
                    stratify=y_train_fold_proc
                )
            else:
                X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                    X_train_fold_proc, y_train_fold_proc,
                    test_size=calib_test_ratio,
                    random_state=CV_RANDOM_STATE + fold_idx
                )
            
            # Use pre-split linear (M_train) and non-linear (M_test) architectures

            rng = np.random.RandomState(CV_RANDOM_STATE + fold_idx)

            shuffled_architectures = architectures_M_train.copy()
            rng.shuffle(shuffled_architectures)
            
            architectures_M_train = shuffled_architectures[:M]

            m_train_names = [a.name for a in architectures_M_train]
            m_test_names = [a.name for a in architectures_M_test]
            
            if self.verbose:
                print(f"   M_train (linear): {m_train_names[:3]}... ({len(m_train_names)} total)")
                print(f"   M_test (non-linear): {m_test_names[:3]}... ({len(m_test_names)} total)")
            
            # Train generative model
            if self.verbose:
                print(f"   Training {self.synth_method}...")
            
            fold_generator = self._train_generative_model_for_fold(
                X_train_fold_proc, y_train_fold_proc,
                use_cached_hyperparams=(self._best_hyperparams is not None)
            )
            
            # Generate synthetic data
            n_synth = int(len(X_test_proc) * synth_size_multiplier)
            X_synth, y_synth = fold_generator.generate(n_samples=n_synth)
            
            if self.task_type == 'classification':
                y_synth = y_synth.astype(int)
            else:
                y_synth = y_synth.astype(float)
            
            # Train M_train models
            trained_m_train_models = []
            for config in architectures_M_train:
                model = self.model_selector.train_model(config, X_train_calib, y_train_calib)
                trained_m_train_models.append(model)
            
            # Fit calibrator
            self.calibrator.fit(
                calibration_models=trained_m_train_models,
                X_synth=X_synth,
                y_synth=y_synth,
                X_real_val=X_test_calib,
                y_real_val=y_test_calib
            )
            
            # Train M_test models
            trained_m_test_models = []
            for config in architectures_M_test:
                model = self.model_selector.train_model(config, X_train_fold_proc, y_train_fold_proc)
                trained_m_test_models.append(model)
            
            # Compute losses
            fold_real_losses = []
            fold_synth_losses = []
            fold_calib_losses = []
            fold_model_names = []

            fold_per_sample_real = []
            fold_per_sample_synth = []
            fold_per_sample_calib = []
            
            for config, model in zip(architectures_M_test, trained_m_test_models):
                real_eval = self.model_selector.evaluate_model(model, X_test_proc, y_test_proc)
                synth_eval = self.model_selector.evaluate_model(model, X_synth, y_synth)
                calib_loss = self.calibrator.evaluate_calibrated_loss(model, X_synth, y_synth)
                
                fold_real_losses.append(real_eval['loss'])
                fold_synth_losses.append(synth_eval['loss'])
                fold_calib_losses.append(calib_loss)
                fold_model_names.append(config.name)

                per_sample_real = self.calibrator._compute_sample_losses(model, X_test_proc, y_test_proc)
                per_sample_synth = self.calibrator._compute_sample_losses(model, X_synth, y_synth)
                per_sample_calib = per_sample_synth * self.calibrator.weights

                fold_per_sample_real.append(per_sample_real)
                fold_per_sample_synth.append(per_sample_synth)
                fold_per_sample_calib.append(per_sample_calib)
            
            fold_real_losses = np.array(fold_real_losses)
            fold_synth_losses = np.array(fold_synth_losses)
            fold_calib_losses = np.array(fold_calib_losses)
            
            # Compute correlations
            uncalib_spearman, uncalib_pvalue = self.ci_estimator.compute_spearman(
                fold_real_losses, fold_synth_losses
            )
            calib_spearman, calib_pvalue = self.ci_estimator.compute_spearman(
                fold_real_losses, fold_calib_losses
            )
            
            all_uncalibrated_spearmans.append(uncalib_spearman)
            all_calibrated_spearmans.append(calib_spearman)

                        # Rank preservation analysis
            rank_analysis_uncalib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_synth_losses
            )
            rank_analysis_calib = self.metrics.compute_rank_preservation_with_guarantees(
                fold_real_losses, fold_calib_losses
            )
            
            # Store fold results
            fold_result = {
                'fold': fold_idx + 1,
                'n_train_calib': len(X_train_calib),
                'n_test_calib': len(X_test_calib),
                'n_test': len(X_test_proc),
                'n_synth': n_synth,
                'm_train_architectures': m_train_names,
                'm_test_architectures': m_test_names,
                'real_losses': fold_real_losses,
                'synth_losses': fold_synth_losses,
                'calibrated_synth_losses': fold_calib_losses,
                'uncalibrated_spearman': uncalib_spearman,
                'uncalibrated_pvalue': uncalib_pvalue,
                'calibrated_spearman': calib_spearman,
                'calibrated_pvalue': calib_pvalue,
                'rank_analysis_uncalibrated': rank_analysis_uncalib,
                'rank_analysis_calibrated': rank_analysis_calib,
                'model_names': fold_model_names,
                'per_sample_real_losses': fold_per_sample_real,
                'per_sample_synth_losses': fold_per_sample_synth,
                'per_sample_calibrated_losses': fold_per_sample_calib,
                'weights': self.calibrator.weights.copy()
            }
            fold_results.append(fold_result)
            
            if self.verbose:
                print(f"\n   Uncalibrated ρ: {uncalib_spearman:.3f}")
                print(f"   Calibrated ρ:   {calib_spearman:.3f} ({calib_spearman - uncalib_spearman:+.3f})")
        
        # ============================================================
        # [6] AGGREGATE RESULTS
        # ============================================================
        if self.verbose:
            print(f"\n[6/6] Computing aggregate statistics...")
        
        uncalib_stats = self.ci_estimator.aggregate_ci_from_samples(all_uncalibrated_spearmans)
        calib_stats = self.ci_estimator.aggregate_ci_from_samples(all_calibrated_spearmans)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"AGGREGATE RESULTS ({n_folds} folds)")
            print(f"{'='*80}")
            print(f"Uncalibrated: {uncalib_stats['mean']:.3f} ± {uncalib_stats['std']:.3f}")
            print(f"  95% CI: [{uncalib_stats['ci_lower']:.3f}, {uncalib_stats['ci_upper']:.3f}]")
            print(f"Calibrated:   {calib_stats['mean']:.3f} ± {calib_stats['std']:.3f}")
            print(f"  95% CI: [{calib_stats['ci_lower']:.3f}, {calib_stats['ci_upper']:.3f}]")
        
        # Cache data
        self.xreal = X_test_proc
        self.yreal = y_test_proc
        self.xsynth = X_synth
        self.ysynth = y_synth
        self._synth_data_cached = (X_synth, y_synth)

        if self._synthesizer is None and fold_generator is not None:
            # Store the underlying synthesizer for SDK methods (CTGAN, TVAE, GaussianCopula)
            if hasattr(fold_generator, 'synthesizer') and fold_generator.synthesizer is not None:
                self._synthesizer = fold_generator.synthesizer
            # For TabPFGen, store the generator itself
            elif hasattr(fold_generator, '_tabpfgen') and fold_generator._tabpfgen is not None:
                self._synthesizer = fold_generator
            # For TabDDPM, store the plugin
            elif hasattr(fold_generator, '_tabddpm_plugin') and fold_generator._tabddpm_plugin is not None:
                self._synthesizer = fold_generator
            else:
                # Fallback: store the whole generator
                self._synthesizer = fold_generator
            self._synth_generator = fold_generator
        
        # SHAP analysis
        shap_analyzer = None
        if analyze_shap and self._synth_data_cached is not None:
            calibration_weights = self.calibrator.weights
            
            if calibration_weights is not None and len(calibration_weights) > 0:

                y_real_values = y_test_proc.values if hasattr(y_test_proc, 'values') else np.array(y_test_proc)
                
                shap_analyzer = self.analyze_calibration_weights_with_shap(
                    X_synth=X_synth,
                    y_synth=y_synth,
                    calibration_weights=calibration_weights,
                    X_real=X_test_proc,
                    y_real=y_real_values,
                    plot_types=shap_plot_types,
                    max_display=shap_max_display,
                    verbose=self.verbose
                )

        # Store results
        self.results = {
            'dataset': self.dataset_name,
            'synth_method': self.synth_method,
            'task_type': self.task_type,
            'n_folds': n_folds,
            'M_calibration': M_calibration,
            'n_evaluation_models': n_total_models - M_calibration,
            'fold_results': fold_results,
            'iteration_results': fold_results,  # Backward compatibility
            'uncalibrated_stats': uncalib_stats,
            'calibrated_stats': calib_stats,
            'uncalibrated_spearmans': all_uncalibrated_spearmans,
            'calibrated_spearmans': all_calibrated_spearmans,
            'shap_analyzer': shap_analyzer
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("EXPERIMENT COMPLETE!")
            print(f"{'='*80}")
        
        return self.results

    def visualize_correlation_results(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Visualize correlation results with per-fold scatter plots.
        
        Creates:
        1. Correlation comparison bar chart (uncalibrated vs calibrated)
        2. Individual scatter plots for each fold (real vs synthetic losses)
        """
        if not self.results:
            print("No results to visualize. Run run_kfold_calibration_experiment first.")
            return
        
        n_iterations = self.results['n_folds'] if 'n_folds' in self.results else self.results.get('n_shuffle_iterations', 0)
        iteration_results = self.results.get('fold_results', self.results.get('iteration_results', []))
        
        # Figure 1: Correlation comparison
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
        
        # Ax 1: Bar chart comparison with error bars
        ax = axes1[0]
        methods = ['Uncalibrated', 'Calibrated']
        uncalib_stats = self.results['uncalibrated_stats']
        calib_stats = self.results['calibrated_stats']
        means = [uncalib_stats['mean'], calib_stats['mean']]
        stds = [uncalib_stats['std'], calib_stats['std']]
        
        bars = ax.bar(methods, means, yerr=stds, 
                     color=['coral', 'lightgreen'], alpha=0.7, 
                     edgecolor='black', linewidth=2, capsize=10)
        ax.set_ylabel('Spearman Correlation', fontsize=12)
        ax.set_title(f'Spearman Correlation Comparison\n(Mean ± Std over {n_iterations} folds)', 
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Show mean ± std and 95% CI
        for i, (mean, std) in enumerate(zip(means, stds)):
            ci = [uncalib_stats, calib_stats][i]
            ax.text(i, mean + std + 0.05, f'{mean:.3f}±{std:.3f}\nCI:[{ci["ci_lower"]:.3f},{ci["ci_upper"]:.3f}]',
                   ha='center', fontsize=8, fontweight='bold')
        
        # Ax 2: Distribution of Spearman values across folds
        ax = axes1[1]
        positions = [1, 2]
        data = [
            self.results['uncalibrated_spearmans'],
            self.results['calibrated_spearmans']
        ]
        
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor('coral')
        bp['boxes'][1].set_facecolor('lightgreen')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Uncalibrated', 'Calibrated'])
        ax.set_ylabel('Spearman Correlation', fontsize=12)
        ax.set_title(f'Distribution of Spearman Correlation\nacross {n_iterations} folds', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure(fig1, 'correlation_comparison')
        plt.show()
        
        # Figure 2: Per-fold scatter plots
        n_cols = min(5, n_iterations)
        n_rows = (n_iterations + n_cols - 1) // n_cols
        
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), constrained_layout=False)
        if n_iterations == 1:
            axes2 = np.array([[axes2]])
        elif n_rows == 1:
            axes2 = axes2.reshape(1, -1)
        
        for iter_idx, iter_result in enumerate(iteration_results):
            row = iter_idx // n_cols
            col = iter_idx % n_cols
            ax = axes2[row, col]
            
            real_losses = iter_result['real_losses']
            synth_losses = iter_result['synth_losses']
            calibrated_losses = iter_result['calibrated_synth_losses']
            
            # Plot uncalibrated
            ax.scatter(real_losses, synth_losses, 
                      s=60, alpha=0.6, edgecolors='black', 
                      label=f"Uncalib (ρ={iter_result['uncalibrated_spearman']:.2f})",
                      marker='o', c='coral')
            
            # Plot calibrated
            ax.scatter(real_losses, calibrated_losses,
                      s=60, alpha=0.6, edgecolors='black',
                      label=f"Calib (ρ={iter_result['calibrated_spearman']:.2f})",
                      marker='^', c='lightgreen')
            
            # Perfect correlation line
            all_vals = np.concatenate([real_losses, synth_losses, calibrated_losses])
            min_val, max_val = all_vals.min(), all_vals.max()
            margin = (max_val - min_val) * 0.1
            ax.plot([min_val - margin, max_val + margin], 
                   [min_val - margin, max_val + margin], 
                   'k--', alpha=0.5, linewidth=1, label='Perfect')
            
            ax.set_xlabel('Real Losses', fontsize=8)
            ax.set_ylabel('Synthetic Losses', fontsize=8)
            fold_num = iter_result.get('fold', iter_idx + 1)
            ax.set_title(f'Fold {fold_num}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_iterations, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes2[row, col].axis('off')
        
        plt.subplots_adjust(
            left=0.06, right=0.98,
            bottom=0.06, top=0.90,
            wspace=0.25,
            hspace=0.35
        )
        
        plt.suptitle('Real Losses vs Synthetic Losses (per fold)', 
                    fontsize=14, fontweight='bold', y=1.02)
        self._save_figure(fig2, 'per_fold_scatter')
        plt.show()

    def compare_vectors(self,
                        iteration: int = 1,
                        model_idx: int = 0,
                        top_n: int = 10):
        """
        Compare real vs synthetic sample losses for a specific model.
        
        For each real sample, finds the nearest synthetic sample (by feature distance)
        and compares their losses. Useful for understanding how calibration affects
        the relationship between real and synthetic samples.
        
        Args:
            iteration: Fold number (1-indexed)
            model_idx: Index of the evaluation model to analyze
            top_n: Number of comparisons to print (sorted by distance)
        """
        if not self.results:
            print("No results to visualize. Run run_kfold_calibration_experiment first.")
            return
        
        iteration_results = self.results.get('fold_results', self.results.get('iteration_results', []))
        n_iterations = len(iteration_results)
        
        if iteration < 1 or iteration > n_iterations:
            print(f"Invalid iteration {iteration}. Must be between 1 and {n_iterations}.")
            return
        
        iter_result = iteration_results[iteration - 1]
        
        # Check if per-sample data exists
        if 'per_sample_real_losses' not in iter_result:
            print("Per-sample losses not available. Re-run experiment with updated code.")
            return
        
        model_names = iter_result['model_names']
        if model_idx >= len(model_names):
            print(f"Invalid model_idx {model_idx}. Must be < {len(model_names)}.")
            return
        
        # Get per-sample losses for the specified model
        real_losses = iter_result['per_sample_real_losses'][model_idx]
        synth_losses = iter_result['per_sample_synth_losses'][model_idx]
        calibrated_losses = iter_result['per_sample_calibrated_losses'][model_idx]
        
        # Get weights from calibrator
        weights = iter_result['weights']
        
        # Get cached data
        if self._synth_data_cached is None:
            print("Synthetic data not cached. Cannot compare feature distances.")
            return
        
        X_synth_df, y_synth = self._synth_data_cached
        
        print(f"\n{'='*100}")
        print(f"SAMPLE COMPARISON: {model_names[model_idx]} (Fold {iteration})")
        print(f"{'='*100}")
        print(f"Real test samples: {len(real_losses)}")
        print(f"Synthetic samples: {len(synth_losses)}")
        print(f"Calibrated samples with non-zero weight: {np.sum(weights > 1e-8) if weights is not None else 'N/A'}")
        
        # Nearest-neighbor comparison: find closest synthetic sample for each real sample
        if self.xreal is not None and self._synth_data_cached is not None:
            X_real = self.xreal
            
            # Select only numeric columns for distance calculation
            numeric_cols = X_real.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                print("\nNo numeric columns available for distance calculation.")
            else:
                X_real_numeric = X_real[numeric_cols].values.astype(float)
                X_synth_numeric = X_synth_df[numeric_cols].values.astype(float)
                
                # Normalize features (z-score) for fair distance comparison
                real_mean = np.mean(X_real_numeric, axis=0)
                real_std = np.std(X_real_numeric, axis=0) + 1e-8
                X_real_norm = X_real_numeric - real_mean
                X_synth_norm = X_synth_numeric
                
                print(f"\n{'='*100}")
                print(f"NEAREST NEIGHBOR COMPARISON (using {len(numeric_cols)} numeric features)")
                print(f"{'='*100}")
                print(f"{'Real Idx':<10} {'Nearest Synth':<15} {'Distance':<12} {'Real Loss':<12} {'Synth Loss':<12} {'Calib Loss':<12} {'Loss Diff':<12} {'Weight':<12}")
                print("-"*100)
                
                # For each real sample, find nearest synthetic sample
                comparisons = []
                for r_idx in range(len(X_real_norm)):
                    real_sample = X_real_norm[r_idx]
                    # Compute distances to all synthetic samples
                    dists = np.linalg.norm(X_synth_norm - real_sample, axis=1)
                    nearest_idx = np.argmin(dists)
                    min_dist = dists[nearest_idx]
                    
                    r_loss = real_losses[r_idx] if r_idx < len(real_losses) else np.nan
                    s_loss = synth_losses[nearest_idx]
                    c_loss = s_loss * weights[nearest_idx]
                    weight = weights[nearest_idx]
                    loss_diff = s_loss - r_loss
                    
                    comparisons.append((r_idx, nearest_idx, min_dist, r_loss, s_loss, c_loss, loss_diff, weight))
                
                # Sort by distance and print top_n closest pairs
                comparisons.sort(key=lambda x: x[2])
                for comp in comparisons[:top_n]:
                    r_idx, s_idx, dist, r_loss, s_loss, c_loss, loss_diff, weight = comp
                    print(f"{r_idx:<10} {s_idx:<15} {dist:<12.4f} {r_loss:<12.4f} {s_loss:<12.4f} {c_loss:<12.4f} {loss_diff:<+12.4f} {weight:<12.6f}")

    def visualize_loss_distributions(self, iteration: int = None, bins: int = 15, 
                                     figsize: Tuple[int, int] = None):
        """Visualize distributions of losses across evaluation models."""
        if not self.results:
            print("No results. Run experiment first.")
            return
        
        fold_results = self.results['fold_results']
        
        if iteration is not None:
            if iteration < 1 or iteration > len(fold_results):
                print(f"Invalid iteration {iteration}.")
                return
            iterations_to_show = [fold_results[iteration - 1]]
        else:
            iterations_to_show = fold_results
        
        n_show = len(iterations_to_show)
        if figsize is None:
            figsize = (15, 4 * n_show)
        
        fig, axes = plt.subplots(n_show, 3, figsize=figsize)
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, fold_result in enumerate(iterations_to_show):
            real = fold_result['real_losses']
            synth = fold_result['synth_losses']
            calib = fold_result['calibrated_synth_losses']
            
            all_losses = np.concatenate([real, synth, calib])
            x_range = (all_losses.min() - 0.1*(all_losses.max()-all_losses.min()),
                       all_losses.max() + 0.1*(all_losses.max()-all_losses.min()))
            
            for col, (data, color, title) in enumerate([
                (real, 'steelblue', 'Real'),
                (synth, 'coral', 'Uncalibrated'),
                (calib, 'lightgreen', 'Calibrated')
            ]):
                ax = axes[row_idx, col]
                ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7, range=x_range)
                ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2)
                ax.set_title(f'Fold {fold_result["fold"]}: {title}')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Loss Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, 'loss_distributions')
        plt.show()

    def analyze_calibration_weights_with_shap(self,
                                             X_synth: pd.DataFrame,
                                             y_synth: pd.Series,
                                             calibration_weights: np.ndarray,
                                             X_real: pd.DataFrame,
                                             y_real: np.ndarray,
                                             plot_types: List[str] = ["dot"],
                                             max_display: int = 15,
                                             verbose: bool = True) -> Optional[SHAPWeightsAnalyzer]:
        """
        Perform dual SHAP analysis for calibration interpretability.
        """
        if verbose:
            print(f"\n{'='*80}")
            print("ANALYZING WITH SHAP (Dual Analysis)")
            print(f"{'='*80}")
        
        # Create analyzer with both weight and y_real data
        analyzer = SHAPWeightsAnalyzer(
            X_synth=X_synth,
            y_synth=y_synth, 
            calibration_weights=calibration_weights,
            X_real=X_real,
            y_real=y_real
        )
        analyzer.fit_surrogate_model(verbose=verbose)
        analyzer.explain(verbose=verbose)
        
        # Print summary for weights
        analyzer.summary_report(target="weights")
        
        # Print summary for y_real if available
        if X_real is not None and y_real is not None:
            analyzer.summary_report(target="y_real")
        
        # Plot summaries for weights
        for plot_type in plot_types:
            if plot_type in ["dot", "bar", "violin"]:
                fig = analyzer.plot_summary(
                    plot_type=plot_type,
                    max_display=max_display,
                    show=False,
                    target="weights"
                )
                self._save_figure(fig, f"shap_weights_{plot_type}")
                plt.show()
        
        # Plot summaries for y_real if available
        if X_real is not None and y_real is not None:
            for plot_type in plot_types:
                if plot_type in ["dot", "bar", "violin"]:
                    fig = analyzer.plot_summary(
                        plot_type=plot_type,
                        max_display=max_display,
                        show=False,
                        target="y_real"
                    )
                    self._save_figure(fig, f"shap_y_real_{plot_type}")
                    plt.show() 
        
        return analyzer
    
    def visualize_per_model_sample_losses(self,
                                           iteration: int = 1,
                                           models_per_row: int = 4,
                                           bins: int = 30,
                                           figsize_per_model: Tuple[float, float] = (4, 3)):
        """
        Visualize per-sample loss distributions for each evaluation model.
        
        For each model, shows histogram of:
        - Real test losses (loss for each sample in test set)
        - Uncalibrated synthetic losses (loss for each sample in synthetic set)
        - Calibrated synthetic losses (weighted loss for each sample)
        
        Args:
            iteration: Fold number to visualize (1-indexed)
            models_per_row: Number of model plots per row
            bins: Number of histogram bins
            figsize_per_model: Figure size for each model subplot
        """
        if not self.results:
            print("No results to visualize. Run run_kfold_calibration_experiment first.")
            return
        
        iteration_results = self.results.get('fold_results', self.results.get('iteration_results', []))
        n_iterations = len(iteration_results)
        
        if iteration < 1 or iteration > n_iterations:
            print(f"Invalid iteration {iteration}. Must be between 1 and {n_iterations}.")
            return
        
        iter_result = iteration_results[iteration - 1]
        
        # Check if per-sample data exists
        if 'per_sample_real_losses' not in iter_result:
            print("Per-sample losses not available. Re-run experiment with updated code.")
            return
        
        model_names = iter_result['model_names']
        per_sample_real = iter_result['per_sample_real_losses']
        per_sample_synth = iter_result['per_sample_synth_losses']
        calibrated_losses = iter_result['calibrated_synth_losses']
        
        n_models = len(model_names)
        n_cols = min(models_per_row, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig_width = figsize_per_model[0] * n_cols
        fig_height = figsize_per_model[1] * n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_models == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, model_name in enumerate(model_names):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            real_losses = per_sample_real[idx]
            synth_losses = per_sample_synth[idx]
            calib_loss = calibrated_losses[idx]
            
            # Determine x-range from all data
            all_losses = np.concatenate([real_losses, synth_losses])
            
            x_min, x_max = np.percentile(all_losses, [1, 99])
            margin = (x_max - x_min) * 0.1
            x_range = (max(0, x_min - margin), x_max + margin)
            
            # Plot overlaid histograms
            ax.hist(real_losses, bins=bins, color='steelblue', alpha=0.5, 
                   range=x_range, density=True, label=f'Real (μ={np.mean(real_losses):.3f})')
            ax.hist(synth_losses, bins=bins, color='coral', alpha=0.5,
                   range=x_range, density=True, label=f'Synth (μ={np.mean(synth_losses):.3f})')
                

            ax.axvline(np.mean(real_losses), color='steelblue', linestyle=':')
            ax.axvline(np.mean(synth_losses), color='coral', linestyle=':')
            ax.axvline(calib_loss, color='lightgreen', linestyle=':', label=f'Calib (μ={calib_loss:.3f})')
            
            ax.set_xlabel('Loss', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.set_title(f'{model_name[:20]}', fontsize=9, fontweight='bold')
            ax.legend(fontsize=6, loc='upper right')
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide empty subplots
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        fold_num = iter_result.get('fold', iteration)
        plt.suptitle(f'Per-Sample Loss Distributions by Model (Fold {fold_num}, {n_models} models)\n'
                    f'Real: {len(per_sample_real[0])} test samples | Synth: {len(per_sample_synth[0])} synthetic samples',
                    fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()

        self._save_figure(fig, 'per_model_sample_losses')
        plt.show()
        
        # Print summary statistics table
        print("\n" + "="*110)
        print(f"PER-SAMPLE LOSS STATISTICS BY MODEL (Fold {fold_num})")
        print("="*110)
        print(f"{'Model':<25} {'Real μ':<10} {'Real σ':<10} {'Synth μ':<10} {'Synth σ':<10} {'Bias':<10}")
        print("-"*110)
        
        for idx, model_name in enumerate(model_names):
            real_mean = np.mean(per_sample_real[idx])
            real_std = np.std(per_sample_real[idx])
            synth_mean = np.mean(per_sample_synth[idx])
            synth_std = np.std(per_sample_synth[idx])
            
            bias = synth_mean - real_mean
            
            print(f"{model_name[:24]:<25} {real_mean:<10.4f} {real_std:<10.4f} {synth_mean:<10.4f} {synth_std:<10.4f} {bias:<+10.4f}")
        
        print("="*110)
    
    def print_summary_table(self):
        """Print summary of correlation estimation results."""
        if not self.results:
            print("No results to print. Run run_kfold_calibration_experiment first.")
            return
        
        print("\n" + "="*80)
        print("SUMMARY OF K-FOLD CALIBRATION EXPERIMENT RESULTS")
        print("="*80)
        
        n_folds = self.results.get('n_folds', self.results.get('n_shuffle_iterations', 0))
        uncalib_stats = self.results['uncalibrated_stats']
        calib_stats = self.results['calibrated_stats']
        
        summary_data = {
            'Metric': [
                f'Uncalibrated Spearman (mean over {n_folds} folds)',
                'Uncalibrated Spearman (std)',
                'Uncalibrated 95% CI (t-distribution)',
                'Uncalibrated Standard Error',
                f'Calibrated Spearman (mean over {n_folds} folds)',
                'Calibrated Spearman (std)',
                'Calibrated 95% CI (t-distribution)',
                'Calibrated Standard Error',
                'Spearman Improvement (mean)',
                'Min Uncalibrated Spearman',
                'Max Uncalibrated Spearman',
                'Min Calibrated Spearman',
                'Max Calibrated Spearman'
            ],
            'Value': [
                f"{uncalib_stats['mean']:.3f}",
                f"{uncalib_stats['std']:.3f}",
                f"[{uncalib_stats['ci_lower']:.3f}, {uncalib_stats['ci_upper']:.3f}]",
                f"{uncalib_stats['se']:.4f}",
                f"{calib_stats['mean']:.3f}",
                f"{calib_stats['std']:.3f}",
                f"[{calib_stats['ci_lower']:.3f}, {calib_stats['ci_upper']:.3f}]",
                f"{calib_stats['se']:.4f}",
                f"{calib_stats['mean'] - uncalib_stats['mean']:.3f}",
                f"{min(self.results['uncalibrated_spearmans']):.3f}",
                f"{max(self.results['uncalibrated_spearmans']):.3f}",
                f"{min(self.results['calibrated_spearmans']):.3f}",
                f"{max(self.results['calibrated_spearmans']):.3f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("PER-FOLD RESULTS")
        print("="*80)
        
        fold_data = []
        fold_results = self.results.get('fold_results', self.results.get('iteration_results', []))
        for fold_result in fold_results:
            fold_num = fold_result.get('fold', fold_result.get('iteration', 0))
            fold_data.append({
                'Fold': fold_num,
                'Uncalib ρ': f"{fold_result['uncalibrated_spearman']:.3f}",
                'Uncalib p': f"{fold_result['uncalibrated_pvalue']:.4f}",
                'Calib ρ': f"{fold_result['calibrated_spearman']:.3f}",
                'Calib p': f"{fold_result['calibrated_pvalue']:.4f}",
                'Improvement': f"{fold_result['calibrated_spearman'] - fold_result['uncalibrated_spearman']:.3f}"
            })
        
        fold_df = pd.DataFrame(fold_data)
        print(fold_df.to_string(index=False))
        print()
    
    def plot_weight_histograms(self, figsize=(15, 5), bins=50):
        """
        Plot weight distribution and diagnostics for sample-level calibration.
        
        Args:
            runner: ExperimentRunner with fitted calibrator
            figsize: Figure size
            bins: Number of histogram bins
        
        Note: If runner.save_figures is True, figures will be saved to runner.figures_dir
        """
        if not self.calibrator.fitted:
            print("Calibrator is not fitted yet. Run experiment first.")
            return

        weights = self.calibrator.weights
        
        if weights is None or len(weights) == 0:
            print("No weights found.")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Full weight distribution
        ax = axes[0]
        ax.hist(weights, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(weights), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(weights):.5f}')
        ax.axvline(1/len(weights), color='orange', linestyle=':', linewidth=2, 
                label=f'Uniform: {1/len(weights):.5f}')
        ax.set_title(f'Sample Weight Distribution\n(n={len(weights)})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Weight', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        
        # 2. Non-zero weights only (log scale)
        ax = axes[1]
        nonzero_weights = weights[weights > 1e-8]
        ax.hist(nonzero_weights, bins=min(bins, len(nonzero_weights)), 
                color='coral', edgecolor='black', alpha=0.7)
        ax.set_title(f'Non-zero Weights\n(n={len(nonzero_weights)}/{len(weights)})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Weight', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Cumulative weight distribution
        ax = axes[2]
        sorted_weights = np.sort(weights)[::-1]  # Descending
        cumsum = np.cumsum(sorted_weights)
        ax.plot(np.arange(len(cumsum)), cumsum, linewidth=2, color='green')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% cumulative')
        ax.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90% cumulative')
        
        # Find where 50% and 90% is reached
        idx_50 = np.searchsorted(cumsum, 0.5)
        idx_90 = np.searchsorted(cumsum, 0.9)
        ax.axvline(idx_50, color='red', linestyle=':', alpha=0.5)
        ax.axvline(idx_90, color='orange', linestyle=':', alpha=0.5)
        
        ax.set_title(f'Cumulative Weight\n(50% at {idx_50}, 90% at {idx_90} samples)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Rank (desc by weight)', fontsize=11)
        ax.set_ylabel('Cumulative Weight', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.suptitle('Sample-Level Calibration Weights', fontsize=14, fontweight='bold', y=1.02)
        self._save_figure(fig, 'weight_histograms')
        plt.show()
        
        # Summary statistics
        print("\n" + "="*70)
        print("WEIGHT STATISTICS (Sample-Level Calibration)")
        print("="*70)
        print(f"  Total samples:      {len(weights)}")
        print(f"  Non-zero weights:   {np.sum(weights > 1e-8)}")
        print(f"  Sum of weights:     {np.sum(weights):.6f}")
        print(f"  Mean weight:        {np.mean(weights):.6f}")
        print(f"  Std weight:         {np.std(weights):.6f}")
        print(f"  Min weight:         {np.min(weights):.6f}")
        print(f"  Max weight:         {np.max(weights):.6f}")
        print(f"  Median weight:      {np.median(weights):.6f}")
        print(f"  Effective samples:  {1 / np.sum(weights**2):.1f} (1/sum(w²))")
        print(f"  Gini coefficient:   {(np.sum(np.abs(weights[:, None] - weights[None, :]))) / (2 * len(weights) * np.sum(weights)):.4f}")
        print("="*70)

