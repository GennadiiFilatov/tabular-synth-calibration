import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

import shap
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

from synth_validation.utils import RANDOM_SEED

class SHAPWeightsAnalyzer:
    """
    SHAP analyzer for dual analysis:
    1. Dependency between calibration weights and X_synth features (+ y_synth)
    2. Dependency between y_real and X_real features
    
    Uses CatBoostRegressorRegressor surrogate models for both analyses and provides
    combined visualization including side-by-side waterfall plots.
    """

    def __init__(self, X_synth: pd.DataFrame, y_synth, calibration_weights: np.ndarray, 
                 X_real: pd.DataFrame = None, y_real: np.ndarray = None, figures_dir: str = None):
        """
        Initializes the dual SHAP analyzer.

        Args:
            X_synth (pd.DataFrame): Synthetic features for weight analysis.
            y_synth: Synthetic target values (numpy array, pandas Series, or DataFrame).
            calibration_weights (np.ndarray): Weights obtained from the calibration model.
            X_real (pd.DataFrame): Real features for y_real analysis (optional).
            y_real (np.ndarray): Real target values for feature dependency analysis (optional).
        """
        # Validate weight analysis inputs
        if not isinstance(X_synth, pd.DataFrame):
            raise TypeError("X_synth should be in format pandas DataFrame.")
        
        # Convert y_synth to numpy array (accept Series, DataFrame, or ndarray)
        if isinstance(y_synth, (pd.Series, pd.DataFrame)):
            y_synth_arr = y_synth.values
        elif isinstance(y_synth, np.ndarray):
            y_synth_arr = y_synth
        else:
            y_synth_arr = np.array(y_synth)
        
        # Ensure y_synth is 2D for concatenation
        if y_synth_arr.ndim == 1:
            y_synth_arr = y_synth_arr.reshape(-1, 1)
        
        if not isinstance(calibration_weights, np.ndarray):
            calibration_weights = np.array(calibration_weights)
        if len(X_synth) != len(calibration_weights):
            raise ValueError("X_synth and calibration_weights have to contain the same number of elements.")
        
        self.figures_dir = figures_dir
            
        self.X_synth = X_synth
        
        # Create concatenated array and DataFrame with column names
        self.X_conc = np.concatenate([X_synth.values, y_synth_arr], axis=1)
        
        # Create DataFrame with proper column names for SHAP (X features + y_synth)
        conc_columns = list(X_synth.columns) + ['y_synth']
        self.X_conc_df = pd.DataFrame(self.X_conc, columns=conc_columns)

        self.weights = calibration_weights
        
        # Store real data for y_real analysis
        self.X_real = X_real
        self.y_real = np.array(y_real) if y_real is not None else None
        
        # Validate y_real analysis inputs if provided
        if self.X_real is not None and self.y_real is not None:
            if not isinstance(self.X_real, pd.DataFrame):
                raise TypeError("X_real should be in format pandas DataFrame.")
            if len(self.X_real) != len(self.y_real):
                raise ValueError("X_real and y_real have to contain the same number of elements.")
            self._has_y_analysis = True
        else:
            self._has_y_analysis = False
        
        # Surrogate models and SHAP values for weight analysis
        self.surrogate_model_w = None
        self.explainer_w = None
        self.shap_values_w = None
        
        # Surrogate models and SHAP values for y_real analysis
        self.surrogate_model_y = None
        self.explainer_y = None
        self.shap_values_y = None
        
        print("=" * 60)
        print("SHAP Dual Analyzer Initialized")
        print("=" * 60)
        print(f"\n[Weight Analysis]")
        print(f"  Synthetic samples: {len(self.X_synth)}")
        print(f"  Features (incl. y_synth): {list(self.X_conc_df.columns)}")
        print(f"  Weight stats - Mean: {self.weights.mean():.4f}, Std: {self.weights.std():.4f}")
        
        if self._has_y_analysis:
            print(f"\n[Y_real Analysis]")
            print(f"  Real samples: {len(self.X_real)}")
            print(f"  Features: {list(self.X_real.columns)}")
            print(f"  Y_real stats - Mean: {self.y_real.mean():.4f}, Std: {self.y_real.std():.4f}")
        else:
            print(f"\n[Y_real Analysis] Not configured (X_real/y_real not provided)")

    def fit_surrogate_model(self, verbose: bool = True, validation_split: float = 0.2, **cat_params):
        """
        Trains surrogate models (XGBRegressor) for both analyses:
        1. X_synth + y_synth -> weights
        2. X_real -> y_real (if configured)
        
        Uses validation split for correct R² estimation.

        Args:
            verbose: Print logs
            validation_split: Fraction of data to use for validation (default: 0.2)
            **cat_params: Additional CatBoost parameters
        """
        default_params = {
            'loss_function': 'RMSE',
            'iterations': 300, 
            'learning_rate': 0.05,
            'depth': 4,                 
            'subsample': 0.8,
            'rsm': 0.8,                  
            'l2_leaf_reg': 1.0,          
            'random_seed': RANDOM_SEED,
            'thread_count': -1,
            'verbose': False,             
        }

        default_params.update(cat_params)
        
        # Train weight surrogate model with validation split
        if verbose:
            print("\n[1/2] Training surrogate model for WEIGHTS (CatBoostRegressor)...")
        
        # Split data for weight model
        X_conc_train, X_conc_val, weights_train, weights_val = train_test_split(
            self.X_conc, self.weights, 
            test_size=validation_split, 
            random_state=RANDOM_SEED
        )
        
        self.surrogate_model_w = CatBoostRegressor(**default_params)
        self.surrogate_model_w.fit(X_conc_train, weights_train)
        
        # Evaluate on validation set for correct R² estimation
        preds_w_val = self.surrogate_model_w.predict(X_conc_val)
        mae_w_val = np.mean(np.abs(weights_val - preds_w_val))
        r2_w_val = 1 - (np.sum((weights_val - preds_w_val)**2) / np.sum((weights_val - weights_val.mean())**2))
        
        # Also compute train metrics for comparison
        preds_w_train = self.surrogate_model_w.predict(X_conc_train)
        r2_w_train = 1 - (np.sum((weights_train - preds_w_train)**2) / np.sum((weights_train - weights_train.mean())**2))
        
        print(f"  Surrogate model (weights) trained")
        print(f"  Train R²: {r2_w_train:.4f}")
        print(f"  Validation R²: {r2_w_val:.4f}, MAE: {mae_w_val:.4f}")
        
        # Refit on full data for SHAP analysis (better explanations with more data)
        self.surrogate_model_w = CatBoostRegressor(**default_params)
        self.surrogate_model_w.fit(self.X_conc, self.weights)
        if verbose:
            print(f"  Model refitted on full data for SHAP analysis")
        
        # Train y_real surrogate model if configured
        if self._has_y_analysis:
            if verbose:
                print("\n[2/2] Training surrogate model for Y_REAL (CatBoostRegressor)...")
            
            # Split data for y_real model
            X_real_train, X_real_val, y_real_train, y_real_val = train_test_split(
                self.X_real, self.y_real,
                test_size=validation_split,
                random_state=RANDOM_SEED
            )
            
            self.surrogate_model_y = CatBoostRegressor(**default_params)
            self.surrogate_model_y.fit(X_real_train, y_real_train)
            
            # Evaluate on validation set for correct R² estimation
            preds_y_val = self.surrogate_model_y.predict(X_real_val)
            mae_y_val = np.mean(np.abs(y_real_val - preds_y_val))
            r2_y_val = 1 - (np.sum((y_real_val - preds_y_val)**2) / np.sum((y_real_val - y_real_val.mean())**2))
            
            # Also compute train metrics for comparison
            preds_y_train = self.surrogate_model_y.predict(X_real_train)
            r2_y_train = 1 - (np.sum((y_real_train - preds_y_train)**2) / np.sum((y_real_train - y_real_train.mean())**2))
            
            print(f"  Surrogate model (y_real) trained")
            print(f"  Train R²: {r2_y_train:.4f}")
            print(f"  Validation R²: {r2_y_val:.4f}, MAE: {mae_y_val:.4f}")
            
            # Refit on full data for SHAP analysis
            self.surrogate_model_y = CatBoostRegressor(**default_params)
            self.surrogate_model_y.fit(self.X_real, self.y_real)
            if verbose:
                print(f"  Model refitted on full data for SHAP analysis")
        else:
            if verbose:
                print("\n[2/2] Skipping y_real surrogate model (not configured)")
        
        return self

    def explain(self, verbose: bool = True):
        """
        Computes SHAP values for both analyses using TreeExplainer.
        """
        if self.surrogate_model_w is None:
            raise RuntimeError("Train surrogate model first with .fit_surrogate_model()")
            
        if verbose:
            print("\n[1/2] Computing SHAP values for WEIGHTS...")
        
        self.explainer_w = shap.TreeExplainer(self.surrogate_model_w)
        # Use X_conc_df (includes y_synth) - matches what the model was trained on
        self.shap_values_w = self.explainer_w(self.X_conc_df)
        
        if verbose:
            print(f"  SHAP values (weights) computed")
            print(f"  Base value: {self.explainer_w.expected_value:.4f}")
        
        # Compute y_real SHAP values if configured
        if self._has_y_analysis and self.surrogate_model_y is not None:
            if verbose:
                print("\n[2/2] Computing SHAP values for Y_REAL...")
            
            self.explainer_y = shap.TreeExplainer(self.surrogate_model_y)
            self.shap_values_y = self.explainer_y(self.X_real)
            
            if verbose:
                print(f"  SHAP values (y_real) computed")
                print(f"  Base value: {self.explainer_y.expected_value:.4f}")
        else:
            if verbose:
                print("\n[2/2] Skipping y_real SHAP (not configured)")
        
        return self

    def plot_summary(self, plot_type: str = "dot", max_display: int = 20, 
                     figsize: Tuple[int, int] = (12, 8), show: bool = True,
                     target: str = "weights"):
        """
        Plot SHAP summary for weights or y_real analysis.
        
        Args:
            plot_type: "dot", "bar", or "violin"
            max_display: Maximum features to display
            figsize: Figure size
            show: Show plot
            target: "weights" or "y_real"
            
        Returns:
            matplotlib.figure.Figure: The generated SHAP summary plot figure
        """
        if target == "weights":
            if self.shap_values_w is None:
                raise RuntimeError("Compute SHAP values first with .explain()")
            shap_values = self.shap_values_w
            X_data = self.X_conc_df  # Use X_conc_df for weights
            title_suffix = "Calibration Weights"
        elif target == "y_real":
            if not self._has_y_analysis or self.shap_values_y is None:
                raise RuntimeError("Y_real SHAP not available. Provide X_real and y_real.")
            shap_values = self.shap_values_y
            X_data = self.X_real
            title_suffix = "Y_real (Target)"
        else:
            raise ValueError(f"Unknown target: {target}. Use 'weights' or 'y_real'.")
        
        print(f"\nPlotting SHAP Summary ({plot_type}) for {target}...")
        
        plt.figure(figsize=figsize)
        
        shap.summary_plot(
            shap_values, 
            X_data, 
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        plt.title(
            f"Feature Impact on {title_suffix}\n",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        plt.xlabel("SHAP value (impact on model output)", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()

        if show:
            plt.show()
        
        return plt.gcf()

    def plot_dependence(self, feature: str, interaction_feature: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6), show: bool = True,
                       target: str = "weights"):
        """
        Plot SHAP dependence for a specific feature.
        
        Args:
            feature: Feature name to analyze
            interaction_feature: Feature to color by (or "auto")
            figsize: Figure size
            show: Show plot
            target: "weights" or "y_real"
        """
        if target == "weights":
            if self.shap_values_w is None:
                raise RuntimeError("Compute SHAP values first with .explain()")
            shap_values = self.shap_values_w
            X_data = self.X_conc_df  # Use X_conc_df for weights
            title_suffix = "Calibration Weights"
        elif target == "y_real":
            if not self._has_y_analysis or self.shap_values_y is None:
                raise RuntimeError("Y_real SHAP not available.")
            shap_values = self.shap_values_y
            X_data = self.X_real
            title_suffix = "Y_real"
        else:
            raise ValueError(f"Unknown target: {target}")
        
        if feature not in X_data.columns:
            raise ValueError(f"Feature '{feature}' not found.")
        
        if interaction_feature and interaction_feature not in X_data.columns:
            raise ValueError(f"Interaction feature '{interaction_feature}' not found.")
        
        print(f"\nPlotting dependence for '{feature}' ({target})...")
        
        plt.figure(figsize=figsize)
        
        interaction_idx = interaction_feature if interaction_feature else "auto"
        
        shap.dependence_plot(
            ind=feature,
            shap_values=shap_values,
            features=X_data,
            interaction_index=interaction_idx,
            show=False
        )
        
        plt.title(
            f"SHAP Dependence: '{feature}' Impact on {title_suffix}"
            + (f"\n(colored by '{interaction_feature}')" if interaction_feature else ""),
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return plt.gcf()

    def plot_waterfall(self, sample_idx: int = 0, figsize: Tuple[int, int] = (12, 8),
                      show: bool = True, target: str = "weights"):
        """
        Plot single waterfall for weights or y_real.
        
        Args:
            sample_idx: Sample index to explain
            figsize: Figure size
            show: Show plot
            target: "weights" or "y_real"
        """
        if target == "weights":
            if self.shap_values_w is None:
                raise RuntimeError("Compute SHAP values first with .explain()")
            shap_values = self.shap_values_w
            X_data = self.X_conc_df  # Use X_conc_df for weights
            model = self.surrogate_model_w
            title_suffix = "Calibration Weight"
        elif target == "y_real":
            if not self._has_y_analysis or self.shap_values_y is None:
                raise RuntimeError("Y_real SHAP not available.")
            shap_values = self.shap_values_y
            X_data = self.X_real
            model = self.surrogate_model_y
            title_suffix = "Y_real"
        else:
            raise ValueError(f"Unknown target: {target}")
        
        if sample_idx >= len(X_data):
            raise ValueError(f"Sample index {sample_idx} out of range (max: {len(X_data)-1}).")
        
        print(f"\n[SHAP] Plotting waterfall for sample {sample_idx} ({target})...")
        
        plt.figure(figsize=figsize)
        
        shap.waterfall_plot(shap_values[sample_idx], show=False)
        
        pred_val = model.predict(X_data.iloc[sample_idx:sample_idx+1])[0]
        plt.title(
            f"SHAP Waterfall: Feature Contributions for Sample {sample_idx}\n"
            f"(Predicted {title_suffix}: {pred_val:.4f})",
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return plt.gcf()

    def plot_combined_waterfall(self, sample_idx_w: int = 0, sample_idx_y: int = 0,
                                figsize: Tuple[int, int] = (20, 8), show: bool = True,
                                max_display: int = 10):
        """
        Plot side-by-side waterfall plots for both weights and y_real analysis.
        
        Args:
            sample_idx_w: Sample index for weights waterfall (from X_synth)
            sample_idx_y: Sample index for y_real waterfall (from X_real)
            figsize: Figure size (width, height)
            show: Show plot
            max_display: Maximum features to display per waterfall
        
        Returns:
            matplotlib Figure
        """
        if self.shap_values_w is None:
            raise RuntimeError("Compute SHAP values first with .explain()")
        
        if not self._has_y_analysis or self.shap_values_y is None:
            raise RuntimeError("Y_real SHAP not available. Provide X_real and y_real.")
        
        if sample_idx_w >= len(self.X_conc_df):
            raise ValueError(f"sample_idx_w {sample_idx_w} out of range.")
        if sample_idx_y >= len(self.X_real):
            raise ValueError(f"sample_idx_y {sample_idx_y} out of range.")
        
        print(f"\n[SHAP] Plotting combined waterfall (weights sample {sample_idx_w}, y_real sample {sample_idx_y})...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Weights waterfall
        plt.sca(axes[0])
        shap.waterfall_plot(self.shap_values_w[sample_idx_w], max_display=max_display, show=False)
        pred_w = self.surrogate_model_w.predict(self.X_conc_df.iloc[sample_idx_w:sample_idx_w+1])[0]
        axes[0].set_title(
            f"Calibration Weights\nSample {sample_idx_w} (pred: {pred_w:.4f})",
            fontsize=12, fontweight='bold'
        )
        
        # Right: Y_real waterfall
        plt.sca(axes[1])
        shap.waterfall_plot(self.shap_values_y[sample_idx_y], max_display=max_display, show=False)
        pred_y = self.surrogate_model_y.predict(self.X_real.iloc[sample_idx_y:sample_idx_y+1])[0]
        axes[1].set_title(
            f"Y_real (Target)\nSample {sample_idx_y} (pred: {pred_y:.4f})",
            fontsize=12, fontweight='bold'
        )
        
        plt.suptitle(
            "SHAP Waterfall: Feature Contributions Comparison\n"
            "(Left: X_synth + y_synth → Weights | Right: X_real → Y_real)",
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig

    def get_feature_importance(self, target: str = "weights") -> pd.DataFrame:
        """
        Returns a feature importance table based on mean absolute SHAP values.

        Args:
            target: "weights" or "y_real"
            
        Returns:
            pd.DataFrame with columns: Feature, Mean |SHAP|, Mean SHAP, Impact
        """
        if target == "weights":
            if self.shap_values_w is None:
                raise RuntimeError("Compute SHAP values first with .explain()")
            shap_values = self.shap_values_w
            X_data = self.X_conc_df  # Use X_conc_df for weights
            impact_label = ("increases weight", "decreases weight")
        elif target == "y_real":
            if not self._has_y_analysis or self.shap_values_y is None:
                raise RuntimeError("Y_real SHAP not available.")
            shap_values = self.shap_values_y
            X_data = self.X_real
            impact_label = ("increases y", "decreases y")
        else:
            raise ValueError(f"Unknown target: {target}")
        
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        mean_shap = shap_values.values.mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': X_data.columns,
            'Mean |SHAP|': mean_abs_shap,
            'Mean SHAP': mean_shap,
            'Impact': [impact_label[0] if x > 0 else impact_label[1] for x in mean_shap]
        }).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
        
        return importance_df

    def summary_report(self, target: str = "weights"):
        """
        Print summary report for weights or y_real SHAP analysis.
        
        Args:
            target: "weights" or "y_real"
        """
        if target == "weights":
            if self.shap_values_w is None:
                print("No SHAP analysis performed yet. Call explain() first.")
                return
            title = "WEIGHTS"
        elif target == "y_real":
            if not self._has_y_analysis or self.shap_values_y is None:
                print("Y_real SHAP not available.")
                return
            title = "Y_REAL"
        else:
            raise ValueError(f"Unknown target: {target}")
        
        print(f"\n{'='*60}")
        print(f"SHAP ANALYSIS SUMMARY: {title}")
        print(f"{'='*60}")
        
        importance_df = self.get_feature_importance(target=target)
        print("\nFeature Importance (by Mean |SHAP|):")
        print(importance_df.to_string(index=False))
        
        print("\nKey Insights:")
        top_3 = importance_df.head(3)
        print(f"\nTop 3 most impactful features for {target}:")
        for idx, row in top_3.iterrows():
            print(f"  {idx+1}. {row['Feature']}: {row['Mean |SHAP|']:.4f} (avg: {row['Mean SHAP']:.4f})")
            print(f"     -> {row['Impact']}")
        print()

    def full_summary_report(self):
        """
        Print summary statistics for the configured analysis targets.
        """
        self.summary_report(target="weights")
        
        if self._has_y_analysis and self.shap_values_y is not None:
            self.summary_report(target="y_real")