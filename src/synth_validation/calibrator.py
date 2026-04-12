import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import minimize, Bounds, BFGS, LinearConstraint


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

    def __init__(self, lambda_reg: float = 0.1, verbose: bool = True, task_type: str = 'classification', 
                 loss_type: str = 'log_loss', use_sample_wise_loss: bool = True, cl_type: str = 'straight',
                 regression_n_bins: int = 10):
        """
        Args:
            lambda_reg: Regularization strength (λ)
            verbose: Print logs
            task_type: 'classification' or 'regression'
            loss_type: Type of loss to use:
                - 'log_loss': Log-loss for classification (default)
                - 'accuracy': 0/1 loss for classification
                - 'mse': Mean Squared Error for regression
                - 'mae': Mean Absolute Error for regression
            use_sample_wise_loss: If True, compute per-sample losses
            cl_type: 'straight' (global) or 'per_class' (group-wise)
            regression_n_bins: Number of target bins when task_type='regression' and cl_type='per_class'
        """
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.task_type = task_type
        self.loss_type = loss_type
        self.use_sample_wise_loss = use_sample_wise_loss
        self.regression_n_bins = max(2, int(regression_n_bins))
        self.group_bin_edges: Optional[np.ndarray] = None
        
        if cl_type != 'per_class':
            self.weights: np.ndarray = None
        else:
            self.weights: Dict[Any, np.ndarray] = {}
            self.sample_indices: Dict[Any, np.ndarray] = {}

        self.fitted = False
        self.cl_type = cl_type
        
        self.optimization_result = None
        self.loss_matrix: np.ndarray = None
        self.real_losses: np.ndarray = None
        self.final_loss: float = 0.0

    def _to_numpy_1d(self, y: Any) -> np.ndarray:
        """Convert targets to a flat numpy array."""
        if hasattr(y, 'values'):
            return np.asarray(y.values).reshape(-1)
        return np.asarray(y).reshape(-1)

    def _slice_target(self, y: Any, indices: np.ndarray) -> np.ndarray:
        """Slice target values by integer indices."""
        if hasattr(y, 'iloc'):
            return np.asarray(y.iloc[indices].values)
        y_arr = np.asarray(y)
        return y_arr[indices]

    def _assign_regression_bins(self, y: Any, bin_edges: np.ndarray) -> np.ndarray:
        """Assign continuous targets to integer bin ids."""
        y_arr = self._to_numpy_1d(y).astype(np.float64, copy=False)
        internal_edges = bin_edges[1:-1]
        return np.digitize(y_arr, internal_edges, right=True)

    def _prepare_per_class_groups(self, y_synth: Any, y_real: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare shared group labels for per-class/per-bin calibration."""
        y_synth_arr = self._to_numpy_1d(y_synth)
        y_real_arr = self._to_numpy_1d(y_real)

        if self.task_type == 'regression':
            combined_targets = np.concatenate([
                y_synth_arr.astype(np.float64, copy=False),
                y_real_arr.astype(np.float64, copy=False)
            ])

            n_unique = len(np.unique(combined_targets))
            if n_unique <= 1:
                bin_edges = np.array([-np.inf, np.inf], dtype=np.float64)
            else:
                n_bins = min(self.regression_n_bins, n_unique)
                quantiles = np.linspace(0.0, 1.0, n_bins + 1)
                raw_edges = np.quantile(combined_targets, quantiles)
                unique_edges = np.unique(raw_edges)

                if len(unique_edges) < 2:
                    bin_edges = np.array([-np.inf, np.inf], dtype=np.float64)
                else:
                    bin_edges = unique_edges.astype(np.float64)
                    bin_edges[0] = -np.inf
                    bin_edges[-1] = np.inf

            self.group_bin_edges = bin_edges
            synth_groups = self._assign_regression_bins(y_synth_arr, bin_edges)
            real_groups = self._assign_regression_bins(y_real_arr, bin_edges)
        else:
            self.group_bin_edges = None
            synth_groups = y_synth_arr
            real_groups = y_real_arr

        shared_groups = np.intersect1d(np.unique(synth_groups), np.unique(real_groups))
        return synth_groups, real_groups, shared_groups

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
    
    def _compute_real_var(self, model, X_real: pd.DataFrame, y_real: pd.Series) -> float:
        """Compute variance on real validation data."""
        sample_losses = self._compute_sample_losses(model, X_real, y_real)
        return np.var(sample_losses)

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
        if self.cl_type != 'per_class':
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
        else:
            self.weights = {}
            self.sample_indices = {}

            synth_groups, real_groups, groups = self._prepare_per_class_groups(y_synth, y_real_val)

            if self.task_type == 'classification':
                group_name = 'Classes'
                header_name = 'CLASS'
            else:
                group_name = 'Target bins'
                header_name = 'BIN'

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"CALIBRATING SYNTHETIC SAMPLES PER {header_name}")
                print(f"{'='*70}")
                print(f"  Regularization: {self.lambda_reg}")
                print(f"  Calibration models (M): {len(calibration_models)}")
                print(f"  Real training samples: {len(X_real_val)}")
                print(f"  Synthetic samples: {len(X_synth)}")
                print(f"  {group_name}: {len(groups)}")
                if self.task_type == 'regression' and self.group_bin_edges is not None:
                    print(f"  Requested bins: {self.regression_n_bins}, effective bins: {len(self.group_bin_edges) - 1}")
                print(f"{'='*70}\n")
            for group_label in groups:
                synth_indices = np.where(synth_groups == group_label)[0]
                real_indices = np.where(real_groups == group_label)[0]

                X_synth_c = X_synth.iloc[synth_indices]
                X_real_c = X_real_val.iloc[real_indices]
                y_synth_c = self._slice_target(y_synth, synth_indices)
                y_real_c = self._slice_target(y_real_val, real_indices)

                N_c = len(X_synth_c)
                M = len(calibration_models)

                if N_c == 0 or len(y_real_c) == 0:
                    if self.verbose:
                        singular_group_name = 'Class' if self.task_type == 'classification' else 'Bin'
                        print(f"  {singular_group_name} {group_label}: SKIPPED (empty)")
                    continue

                L = np.zeros((N_c, M))
                l_r = np.zeros(M)

                for m, model in enumerate(calibration_models):
                    if self.verbose:
                        print(f"  Model {m+1}/{M}...", end=" ")

                    L[:, m] = self._compute_sample_losses(model, X_synth_c, y_synth_c)
                    l_r[m] = self._compute_real_loss(model, X_real_c, y_real_c)

                    if self.verbose:
                        print(f"synth_loss_mean={np.mean(L[:, m]):.4f}, real_loss={l_r[m]:.4f}")
                
                w_opt, opt_result = self._solve_constrained_optimization(L, l_r)
                
                self.weights[group_label] = w_opt
                self.sample_indices[group_label] = synth_indices
                self.optimization_result = opt_result

                residual = l_r - L.T @ w_opt
                self.final_loss = np.sum(residual ** 2) + self.lambda_reg * np.sum(w_opt ** 2)
                    
                if self.verbose:
                    print(f"  Final objective: {self.final_loss:.6f}")
                    print(f"  Non-zero weights: {np.sum(w_opt > 1e-6)}/{N_c}")
                    print(f"  Weight sum: {np.sum(w_opt):.6f}")
                    print(f"  Non-zero weights: {np.sum(w_opt > 1e-6)}/{N_c}")
                    print(f"  Max weight: {np.max(w_opt):.6f}")

            self.fitted = True



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
        
        if self.cl_type != 'per_class':
            sample_losses = self._compute_sample_losses(model, X_synth, y_synth)
            return sample_losses.T @ self.weights
        else:
            sample_losses = self._compute_sample_losses(model, X_synth, y_synth)

            if self.task_type == 'regression':
                if self.group_bin_edges is None:
                    raise ValueError("Regression bins are not available. Call fit() first.")
                synth_groups = self._assign_regression_bins(y_synth, self.group_bin_edges)
            else:
                synth_groups = self._to_numpy_1d(y_synth)

            total_error_weighted = 0.0
            total_samples = 0
            
            for group_label, w_c in self.weights.items():
                mask = (synth_groups == group_label)
                n_samples = int(np.sum(mask))
                
                if n_samples == 0:
                    continue
                l_r = sample_losses[mask]

                if len(l_r) != len(w_c):
                    raise ValueError(f"MISMATCH for group {group_label}: {len(l_r)} != {len(w_c)}")
                
                calibrated_group_error_rate = np.dot(l_r, w_c)
                total_error_weighted += calibrated_group_error_rate * n_samples
                total_samples += n_samples
            
            if total_samples == 0:
                return 0.0

            return total_error_weighted / total_samples
        
    def compute_weights_for_samples(self, y_synth: pd.Series) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        if self.cl_type != 'per_class':
            if self.weights is None:
                raise ValueError("Weights are not available.")
            return self.weights.copy()

        weights = np.zeros(len(y_synth), dtype=np.float64)
        
        for class_label, w_c in self.weights.items():
            indices = self.sample_indices[class_label]
            
            if len(indices) != len(w_c):
                raise ValueError(f"MISMATCH: {len(indices)} != {len(w_c)}")
            
            weights[indices] = w_c
        
        return weights



class SyntheticBPRCalibrator:
    """Calibrate synthetic sample weights using a BPR-style objective.

    This calibrator learns a box-constrained weight vector over synthetic
    samples so that model ranking induced by weighted synthetic losses matches
    the ranking observed on real validation losses.
    """

    def __init__(self, eps: float = 0.0, beta: float = 1.0, lambda_reg: float = 0.5, mu: float = 0.0, 
                 verbose: bool = False, task_type: str = 'classification', loss_type: str = 'log_loss'):
        """Initialize the BPR calibrator.

        Args:
            eps: Preference threshold for building pairwise model comparisons.
            beta: Sigmoid temperature for BPR margins.
            lambda_reg: L2 regularization strength on sample weights.
            mu: KL-regularization strength toward a uniform prior.
            verbose: Whether to print progress and diagnostics.
            task_type: Task type, either 'classification' or 'regression'.
            loss_type: Loss to compute per sample.
        """
        self.eps = eps
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.mu = mu

        self.task_type = task_type
        self.loss_type = loss_type

        self.verbose = verbose

        self.weights: np.ndarray = None
        self.fitted = False

        self.pref_set: np.ndarray = None
        self.diff_matrix: np.ndarray = None
        self.loss_matrix: np.ndarray = None
        self.real_losses: np.ndarray = None
        
        self.optimization_result = None
        self.final_loss: float = 0.0

    def _to_numpy_1d(self, y: Any) -> np.ndarray:
        """Convert targets to a flat numpy array.

        Args:
            y: Target values as numpy array, pandas Series, or similar.

        Returns:
            Flattened target array.
        """
        if hasattr(y, 'values'):
            return np.asarray(y.values).reshape(-1)
        return np.asarray(y).reshape(-1)

    def _compute_sample_losses(self, model: Any, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute per-sample losses for a model.

        Args:
            model: Trained sklearn-compatible model.
            X: Feature matrix.
            y: Ground-truth targets.

        Returns:
            Per-sample loss vector with shape (n_samples,).
        """
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

    def _compute_real_loss(self, model: Any, X_real: pd.DataFrame, y_real: pd.Series) -> float:
        """Compute average loss on real validation data.

        Args:
            model: Trained sklearn-compatible model.
            X_real: Real validation features.
            y_real: Real validation targets.

        Returns:
            Mean loss on real validation data.
        """
        sample_losses = self._compute_sample_losses(model, X_real, y_real)
        return np.mean(sample_losses)

    def _simplex_project(self, v: np.ndarray) -> np.ndarray:
        """Project vector onto the probability simplex.

        Args:
            v: Input vector.

        Returns:
            Euclidean projection of ``v`` onto ``{w: sum(w)=1, w>=0}``.
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
        rho = int(rho_candidates[-1]) if len(rho_candidates) > 0 else 0
        theta = (cssv[rho] - 1.0) / float(rho + 1)
        return np.maximum(v - theta, 0.0)

    def _build_pref_set(self, r: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
        """Build pairwise model preferences from real-data losses.

        Args:
            r: Real-data mean losses per model, shape (M,).
            eps: Preference threshold.

        Returns:
            Tuple containing:
                - pref_pairs: Integer index pairs (a, b), shape (|P|, 2)
                - d: Preference strengths, shape (|P|,)
        """
        m_models = len(r)
        pref_pairs_list: List[Tuple[int, int]] = []
        d_list: List[float] = []

        for a in range(m_models):
            for b in range(m_models):
                if a == b:
                    continue
                if r[a] > r[b] + eps:
                    pref_pairs_list.append((a, b))
                    #d_list.append(float(r[a] - r[b]))
                    d_list.append(float(r[a] - r[b]))

        if len(pref_pairs_list) == 0:
            if self.verbose:
                print("  WARNING: Empty preference set after eps filtering; falling back to all ordered pairs with uniform weights.")
            for a in range(m_models):
                for b in range(m_models):
                    if a != b:
                        pref_pairs_list.append((a, b))
            d = np.ones(len(pref_pairs_list), dtype=np.float64)
        else:
            d = np.asarray(d_list, dtype=np.float64)

        if len(pref_pairs_list) > 0:
            pref_pairs = np.asarray(pref_pairs_list, dtype=np.int64)
        else:
            pref_pairs = np.empty((0, 2), dtype=np.int64)
            d = np.empty((0,), dtype=np.float64)

        self.pref_set = pref_pairs
        return pref_pairs, d

    def _build_diff_matrix(self, L: np.ndarray, pref_pairs: np.ndarray) -> np.ndarray:
        """Build BPR difference matrix.

        Args:
            L: Synthetic loss matrix with shape (Ns, M).
            pref_pairs: Preference pairs with shape (|P|, 2).

        Returns:
            Difference matrix D with shape (|P|, Ns), where
            D[(a,b), i] = L[i, b] - L[i, a].
        """
        if pref_pairs.size == 0:
            D = np.zeros((0, L.shape[0]), dtype=np.float64)
        else:
            a_idx = pref_pairs[:, 0]
            b_idx = pref_pairs[:, 1]
            D = (L[:, b_idx] - L[:, a_idx]).T.astype(np.float64, copy=False)

        self.diff_matrix = D
        return D

    def _compute_bpr_objective(self, w: np.ndarray, D: np.ndarray, d: np.ndarray) -> float:
        """Compute BPR objective value for current weights.

        Args:
            w: Current sample weights, shape (Ns,).
            D: Difference matrix, shape (|P|, Ns).
            d: Preference weights, shape (|P|,).

        Returns:
            Scalar objective value.
        """
        n_samples = len(w)
        u = 1.0 / n_samples

        margins = D @ w
        logits = np.clip(self.beta * margins, -500.0, 500.0)
        sigmoid_vals = 1.0 / (1.0 + np.exp(-logits))

        bpr_term = -float(d @ np.log(sigmoid_vals + 1e-15))
        l2_term = self.lambda_reg * float(np.sum(w ** 2))
        w_clip = np.maximum(w, 1e-15)
        kl_term = self.mu * float(np.sum(w * (np.log(w_clip) - np.log(u))))

        return bpr_term + l2_term + kl_term

    def _solve_bpr_optimization(self, L: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Solve BPR optimization using L-BFGS-B with box constraints.

        Args:
            L: Synthetic loss matrix, shape (Ns, M).
            r: Real-data model losses, shape (M,).

        Returns:
            Tuple containing:
                - w_opt: Optimized weights, shape (Ns,)
                - loss_history: Objective values over optimization
        """
        n_samples = L.shape[0]
        if n_samples <= 0:
            raise ValueError("Synthetic dataset must contain at least one sample.")

        pref_pairs, d = self._build_pref_set(r, self.eps)
        D = self._build_diff_matrix(L, pref_pairs)

        w0 = np.ones(n_samples, dtype=np.float64) / n_samples
        u = 1.0 / n_samples
        log_u = np.log(u)

        def objective(w: np.ndarray) -> float:
            return self._compute_bpr_objective(w, D, d)

        def gradient(w: np.ndarray) -> np.ndarray:
            margins = D @ w
            neg_logits = np.clip(-self.beta * margins, -500.0, 500.0)
            sigma_neg = 1.0 / (1.0 + np.exp(-neg_logits))

            w_clip = np.maximum(w, 1e-15)
            grad = (
                -self.beta * (D.T @ (d * sigma_neg))
                + 2.0 * self.lambda_reg * w
                + self.mu * (np.log(w_clip) - log_u + 1.0)
            )
            return grad

        loss_history: List[float] = [objective(w0)]

        def callback(wk: np.ndarray, state: Any) -> None:
            loss_history.append(objective(wk))

        result = minimize(
            objective, w0,
            method='trust-constr',
            jac=gradient,
            hess=BFGS(),
            bounds=Bounds(lb=1e-8, ub=1.0, keep_feasible=True),
            constraints=LinearConstraint(A=np.ones((1, n_samples)), 
                                         lb=1.0, ub=1.0, keep_feasible=True),
            callback=callback,
            options={
                'maxiter': 1000,
                'gtol': 1e-8,
                'xtol': 1e-10,
                'initial_constr_penalty': 1.0,
                'initial_barrier_parameter': 0.01,  # чем меньше, тем дальше от границы
                'initial_barrier_tolerance': 0.01,
                'verbose': 0
            }
        )

        final_loss = objective(result.x)
        if len(loss_history) == 0 or not np.isclose(loss_history[-1], final_loss):
            loss_history.append(final_loss)

        self.optimization_result = result

        return result.x, loss_history

    def fit(self,
            calibration_models: List[Any],
            X_synth: pd.DataFrame, y_synth: pd.Series,
            X_real_val: pd.DataFrame, y_real_val: pd.Series) -> None:
        """Fit BPR calibration weights on synthetic samples.

        Args:
            calibration_models: Fitted calibration models.
            X_synth: Synthetic feature matrix.
            y_synth: Synthetic targets.
            X_real_val: Real validation feature matrix.
            y_real_val: Real validation targets.
        """
        n_samples = len(X_synth)
        n_models = len(calibration_models)
        if n_samples <= 0:
            raise ValueError("Synthetic dataset must contain at least one sample.")
        if n_models <= 0:
            raise ValueError("At least one calibration model is required.")

        if self.verbose:
            print(f"\n{'='*70}")
            print("CALIBRATING SYNTHETIC SAMPLES (BPR)")
            print(f"{'='*70}")
            print(f"  Calibration Models (M): {n_models}, Synthetic samples (Ns): {n_samples}")
            print(f"  eps: {self.eps}, beta: {self.beta}, lambda: {self.lambda_reg}, mu: {self.mu}")

        L = np.zeros((n_samples, n_models), dtype=np.float64)
        r = np.zeros(n_models, dtype=np.float64)

        for m, model in enumerate(calibration_models):
            if self.verbose:
                print(f"  Model {m+1}/{n_models}...", end=" ")

            L[:, m] = self._compute_sample_losses(model, X_synth, y_synth)
            r[m] = self._compute_real_loss(model, X_real_val, y_real_val)

            if self.verbose:
                print(f"synth_loss_mean={np.mean(L[:, m]):.4f}, real_loss={r[m]:.4f}")

        w_opt, loss_history = self._solve_bpr_optimization(L, r)

        self.weights = w_opt
        self.loss_matrix = L
        self.real_losses = r
        self.fitted = True
        self.final_loss = float(loss_history[-1]) if len(loss_history) > 0 else 0.0

        if len(loss_history) > 0:
            plt.semilogy(loss_history)
            plt.xlabel('Iteration')
            plt.ylabel('Loss (log scale)')
            plt.grid(True)
            plt.show()

        if self.verbose:
            print(f"  Final objective: {self.final_loss:.6f}")
            print(f"  Non-zero weights: {np.sum(self.weights > 1e-8)}/{n_samples}")
            print(f"  Weight sum: {np.sum(self.weights):.6f}")
            print(f"  Max weight: {np.max(self.weights):.6f}")

    def evaluate_calibrated_loss(self, model: Any, X_synth: pd.DataFrame,
                                 y_synth: pd.Series) -> float:
        """Evaluate weighted synthetic loss for a model.

        Args:
            model: Trained model to evaluate.
            X_synth: Synthetic feature matrix.
            y_synth: Synthetic targets.

        Returns:
            Weighted synthetic loss scalar.
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        sample_losses = self._compute_sample_losses(model, X_synth, y_synth)
        return float(sample_losses @ self.weights)

    def compute_weights_for_samples(self, y_synth: Optional[pd.Series] = None) -> np.ndarray:
        """Return full sample weight vector.

        Args:
            y_synth: Unused. Present only for API compatibility.

        Returns:
            Copy of calibrated sample weights.
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        if self.weights is None:
            raise ValueError("Weights are not available.")
        return self.weights.copy()