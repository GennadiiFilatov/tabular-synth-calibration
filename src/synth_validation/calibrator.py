import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import minimize, Bounds


class SyntheticDataCalibrator:
    """
    Calibration of synthetic samples using sample-level losses.
    Supports both classification and regression tasks.
    
    Solves the constrained optimization problem:
        w = argmin ||l_r - L^T @ w||^2 + λ||w||^2
        
    Subject to:
        w >= 0  (non-negativity constraint)
    
    Where:
        - L[i, m] is the loss of model m on synthetic sample i (shape: n_synth x M)
        - l_r[m] is the average loss of model m on real validation data (shape: M,)
        - w[i] is the weight for synthetic sample i (shape: n_synth,)
    """

    def __init__(self, lambda_reg: float = 0.1, verbose: bool = True, 
                 loss_type: str = 'log_loss', use_sample_wise_loss: bool = True):
        """
        Args:
            lambda_reg: Regularization strength (λ)
            verbose: Print logs
            loss_type: Type of loss to use:
                - 'log_loss': Log-loss for classification (default)
                - 'accuracy': 0/1 loss for classification
                - 'mse': Mean Squared Error for regression
                - 'mae': Mean Absolute Error for regression
            use_sample_wise_loss: If True, compute per-sample losses
        """
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.loss_type = loss_type
        self.use_sample_wise_loss = use_sample_wise_loss
        
        self.weights: np.ndarray = None
        self.fitted = False
        
        self.optimization_result = None
        self.loss_matrix: np.ndarray = None
        self.real_losses: np.ndarray = None
        self.final_loss: float = 0.0

    def _compute_sample_losses(self, model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute per-sample losses for a model."""
        n_samples = len(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        if self.loss_type == 'log_loss':
            if not hasattr(model, "predict_proba"):
                preds = model.predict(X)
                return (preds != y_arr).astype(np.float64)

            try:
                proba = model.predict_proba(X)
            except (ValueError, RuntimeError):
                preds = model.predict(X)
                return (preds != y_arr).astype(np.float64)

            if np.any(np.isnan(proba)):
                preds = model.predict(X)
                return (preds != y_arr).astype(np.float64)

            classes = model.classes_
            eps = 1e-15
            proba = np.clip(proba, eps, 1 - eps)

            sample_losses = np.full(n_samples, -np.log(eps), dtype=np.float64)

            for i in range(n_samples):
                true_label = y_arr[i]
                class_idx = np.where(classes == true_label)[0]
                if len(class_idx) == 0:
                    continue
                sample_losses[i] = -np.log(proba[i, class_idx[0]])

            return sample_losses

        elif self.loss_type == 'accuracy':
            preds = model.predict(X)
            return (preds != y_arr).astype(np.float64)

        elif self.loss_type == 'mse':
            preds = model.predict(X)
            return ((preds - y_arr) ** 2).astype(np.float64)

        elif self.loss_type == 'mae':
            preds = model.predict(X)
            return np.abs(preds - y_arr).astype(np.float64)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _compute_real_loss(self, model, X_real: pd.DataFrame, y_real: pd.Series) -> float:
        """Compute average loss on real validation data."""
        sample_losses = self._compute_sample_losses(model, X_real, y_real)
        return np.mean(sample_losses)

    def fit(self, 
            calibration_models: List[Any],
            X_synth: pd.DataFrame, y_synth: pd.Series,
            X_real_val: pd.DataFrame, y_real_val: pd.Series) -> None:
        """
        Fit calibration weights on entire synthetic sample set.
        
        Solves:
            w = argmin ||l_r - L^T @ w||^2 + λ||w||^2
            s.t. w >= 0
        """
        N = len(X_synth)
        M = len(calibration_models)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"CALIBRATING SYNTHETIC SAMPLES")
            print(f"{'='*70}")
            print(f"  λ: {self.lambda_reg}, Loss: {self.loss_type}")
            print(f"  Calibration Models (M): {M}, Synthetic samples (N): {N}")
        
        L = np.zeros((N, M), dtype=np.float64)
        l_r = np.zeros(M, dtype=np.float64)
        
        for m, model in enumerate(calibration_models):
            if self.verbose:
                print(f"  Model {m+1}/{M}...", end=" ")

            L[:, m] = self._compute_sample_losses(model, X_synth, y_synth)
            l_r[m] = self._compute_real_loss(model, X_real_val, y_real_val)

            if self.verbose:
                print(f"synth_loss_mean={np.mean(L[:, m]):.4f}, real_loss={l_r[m]:.4f}")
        
        self.loss_matrix = L
        self.real_losses = l_r
        
        w_opt, opt_result = self._solve_constrained_optimization(L, l_r)
        
        self.weights = w_opt
        self.optimization_result = opt_result
        self.fitted = True
        
        residual = l_r - L.T @ w_opt
        self.final_loss = np.sum(residual ** 2) + self.lambda_reg * np.sum(w_opt ** 2)
        
        if self.verbose:
            print(f"  Final objective: {self.final_loss:.6f}")
            print(f"  Non-zero weights: {np.sum(w_opt > 1e-6)}/{N}")
            print(f"  Weight sum: {np.sum(w_opt):.6f}")
            print(f"  Non-zero weights: {np.sum(w_opt > 1e-6)}/{N}")
            print(f"  Max weight: {np.max(w_opt):.6f}")

    def _solve_constrained_optimization(self, L: np.ndarray, l_r: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Solve the constrained optimization problem."""
        N, M = L.shape
        
        def objective(w):
            residual = l_r - L.T @ w
            return np.sum(residual ** 2) + self.lambda_reg * np.sum(w ** 2)
        
        def gradient(w):
            return 2 * (L @ L.T @ w - L @ l_r + self.lambda_reg * w)
        
        def callback(xk):
            loss = objective(xk)
            loss_history.append(loss)
        
        w0 = np.ones(N) / N

        loss_history = []

        result = minimize(
            objective, w0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=Bounds(lb=0.0, ub=1.0),
            callback=callback,
            options={
                'maxiter': 15000,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'disp': False
            }
        )

        plt.semilogy(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.grid(True)
        plt.show()
        
        return result.x, result

    def evaluate_calibrated_loss(self, model: Any, X_synth: pd.DataFrame, 
                                  y_synth: pd.Series) -> float:
        """Evaluate weighted (calibrated) loss for a model on synthetic data."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        sample_losses = self._compute_sample_losses(model, X_synth, y_synth)
        return sample_losses.T @ self.weights