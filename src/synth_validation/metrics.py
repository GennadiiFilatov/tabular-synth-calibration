"""
Evaluation Metrics Module.

Implements the metric set required by the semester research plan
(Scientific_Research.pdf, Section 3 "Metrics" and WP3 "Deployment-time
diagnostics"):

    Reg@1 / NormReg@1   - selection regret and its uniform-baseline-normalized
                          form (Section 3.1: w* = argmin ...; NormReg@1 < 1
                          means calibration helps).
    Hit@k, Kendall tau  - top-k accuracy and full-ranking concordance on Heval.
    HarmRate            - fraction of trials where calibration is worse than
                          the uniform baseline (Reg@1(w*) > Reg@1(u)).
    ExcessHarm / Gain   - one-sided magnitudes of harm/gain vs. uniform.
    LOO-residual        - leave-one-out cross-fitted residual on Hcal
                          (deployment-time diagnostic, no access to Heval).
    ESS, w_max, H(w)    - effective sample size, max weight, weight entropy
                          (deployment-time diagnostics of the weight vector).
    kappa(Lsyn_cal)     - condition number of the calibration-model loss
                          matrix (deployment-time diagnostic of ill-posedness).
    rho_cal             - Spearman correlation between synthetic and uniform
                          real losses on Hcal (secondary diagnostic).

The module also provides the theoretical rank-preservation bound and NDCG@k.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Any, Callable, Dict, List, Optional, Tuple


class EvaluationMetrics:
    """
    Evaluation metrics for synthetic-data-based model ranking/selection,
    for synthetic-data model selection and deployment diagnostics.
    """

    def __init__(self):
        self.results = {}

    # ------------------------------------------------------------------
    # Core rank-agreement metrics
    # ------------------------------------------------------------------

    @staticmethod
    def spearman_correlation(losses_synth_eval: np.ndarray,
                              losses_real_eval: np.ndarray,
                              alternative: str = 'two-sided') -> Dict[str, float]:
        """
        Spearman rank correlation for the model evaluation set (Heval).

        The real losses must come from the outer evaluation data split, not
        the calibration holdout. Calibration-only correlation is exposed by
        ``calibration_spearman_correlation`` and is reported as ``rho_cal``.
        """
        losses_synth_eval = np.asarray(losses_synth_eval, dtype=float)
        losses_real_eval = np.asarray(losses_real_eval, dtype=float)
        if len(losses_synth_eval) != len(losses_real_eval):
            raise ValueError("Synthetic and real evaluation losses must have the same length.")
        if len(losses_synth_eval) < 3:
            return {'correlation': np.nan, 'p_value': np.nan, 'n_samples': len(losses_synth_eval)}

        corr, p_value = stats.spearmanr(
            losses_synth_eval, losses_real_eval, alternative=alternative
        )
        return {'correlation': corr, 'p_value': p_value, 'n_samples': len(losses_synth_eval)}

    @staticmethod
    def calibration_spearman_correlation(losses_synth_cal: np.ndarray,
                                          losses_real_cal: np.ndarray,
                                          alternative: str = 'two-sided') -> Dict[str, float]:
        """Spearman correlation for the calibration-only diagnostic ``rho_cal``."""
        return EvaluationMetrics.spearman_correlation(
            losses_synth_cal, losses_real_cal, alternative=alternative
        )

    @staticmethod
    def kendall_tau(losses_a: np.ndarray, losses_b: np.ndarray) -> Dict[str, float]:
        """Kendall's tau rank concordance (Section 3 metric table)."""
        losses_a = np.asarray(losses_a, dtype=float)
        losses_b = np.asarray(losses_b, dtype=float)
        if len(losses_a) < 2:
            return {'tau': np.nan, 'p_value': np.nan}
        tau, p_value = stats.kendalltau(losses_a, losses_b)
        return {'tau': tau, 'p_value': p_value}

    @staticmethod
    def _clamp_k(k: int, n_models: int) -> int:
        if n_models <= 0:
            return 0
        return max(1, min(int(k), n_models))

    @staticmethod
    def hit_at_k(real_losses: np.ndarray,
                 pred_losses: np.ndarray,
                 k: int) -> float:
        """
        Hit@k: overlap between the top-k models under real loss and the
        top-k models under predicted (synthetic/calibrated) loss.
        Section 3 metric table: "Hit@k, Kendall tau - top-k accuracy and
        ranking on Heval".
        """
        real_losses = np.asarray(real_losses, dtype=float)
        pred_losses = np.asarray(pred_losses, dtype=float)
        n_models = len(real_losses)
        if n_models == 0:
            return np.nan

        k = EvaluationMetrics._clamp_k(k, n_models)
        if k == 0:
            return np.nan

        topk_real = set(np.argsort(real_losses)[:k])
        topk_pred = set(np.argsort(pred_losses)[:k])
        return len(topk_real & topk_pred) / k

    @staticmethod
    def ndcg_at_k(real_losses: np.ndarray,
                  pred_losses: np.ndarray,
                  k: int) -> float:
        """NDCG@k with relevance derived from negative real losses, scaled to [0, 1]."""
        real_losses = np.asarray(real_losses, dtype=float)
        pred_losses = np.asarray(pred_losses, dtype=float)
        n_models = len(real_losses)
        if n_models == 0:
            return np.nan

        k = EvaluationMetrics._clamp_k(k, n_models)
        if k == 0:
            return np.nan

        relevance = -real_losses
        min_rel = np.min(relevance)
        if min_rel < 0:
            relevance = relevance - min_rel
        max_rel = np.max(relevance)
        if max_rel > 0:
            relevance = relevance / max_rel

        pred_order = np.argsort(pred_losses)[:k]
        ideal_order = np.argsort(real_losses)[:k]
        discounts = 1.0 / np.log2(np.arange(2, k + 2))

        dcg = np.sum(relevance[pred_order] * discounts)
        idcg = np.sum(relevance[ideal_order] * discounts)

        return dcg / idcg if idcg > 0 else np.nan

    @staticmethod
    def topk_metrics(real_losses: np.ndarray,
                      pred_losses: np.ndarray,
                      k: int) -> Dict[str, float]:
        """Bundle of Spearman@k, NDCG@k, Hit@k for a fixed cutoff k."""
        real_losses = np.asarray(real_losses, dtype=float)
        pred_losses = np.asarray(pred_losses, dtype=float)
        n_models = len(real_losses)
        if n_models == 0:
            return {'spearman': np.nan, 'ndcg': np.nan, 'hit_rate': np.nan, 'k': 0}

        k = EvaluationMetrics._clamp_k(k, n_models)
        topk_idx = np.argsort(real_losses)[:k]

        if len(topk_idx) < 2:
            spearman = np.nan
        else:
            spearman, _ = stats.spearmanr(real_losses[topk_idx], pred_losses[topk_idx])

        return {
            'spearman': float(spearman),
            'ndcg': float(EvaluationMetrics.ndcg_at_k(real_losses, pred_losses, k)),
            'hit_rate': float(EvaluationMetrics.hit_at_k(real_losses, pred_losses, k)),
            'k': int(k)
        }

    @staticmethod
    def rank_preservation(losses_synth: np.ndarray,
                           losses_test: np.ndarray) -> Dict[str, float]:
        """
        Non-top-k rank-agreement diagnostics: Kendall's tau, mean absolute
        rank difference, and best-model agreement. Top-k overlap content
        was removed here as a duplicate of hit_at_k/topk_metrics.
        """
        losses_synth = np.asarray(losses_synth, dtype=float)
        losses_test = np.asarray(losses_test, dtype=float)

        ranks_synth = stats.rankdata(losses_synth, method='ordinal')
        ranks_test = stats.rankdata(losses_test, method='ordinal')

        tau_result = EvaluationMetrics.kendall_tau(losses_synth, losses_test)

        best_synth = np.argmin(losses_synth)
        best_test = np.argmin(losses_test)

        return {
            'kendall_tau': tau_result['tau'],
            'kendall_tau_p_value': tau_result['p_value'],
            'mean_abs_rank_diff': np.mean(np.abs(ranks_synth - ranks_test)),
            'best_model_match': int(best_synth == best_test),
            'best_synth_rank_in_test': int(ranks_test[best_synth]),
        }

    # ------------------------------------------------------------------
    # Selection regret: Reg@1, NormReg@1, HarmRate, ExcessHarm/Gain
    # (Section 3.1 of the plan)
    # ------------------------------------------------------------------

    @staticmethod
    def reg_at_1(real_losses_eval: np.ndarray,
                 pred_losses_eval: np.ndarray) -> float:
        """
        Reg@1(w) = Rreal(h*_w) - Rreal(h*), where h*_w = argmin_Heval pred_losses
        and h* = argmin_Heval real_losses.

        `pred_losses_eval` should be the risk estimate under weighting w
        (e.g. weighted/calibrated synthetic loss) for each model in Heval;
        `real_losses_eval` the cross-fitted real loss for the same models.
        """
        real_losses_eval = np.asarray(real_losses_eval, dtype=float)
        pred_losses_eval = np.asarray(pred_losses_eval, dtype=float)
        if len(real_losses_eval) == 0:
            return np.nan

        selected_idx = int(np.argmin(pred_losses_eval))
        optimal_idx = int(np.argmin(real_losses_eval))
        return float(real_losses_eval[selected_idx] - real_losses_eval[optimal_idx])

    @staticmethod
    def norm_reg_at_1(real_losses_eval: np.ndarray,
                       calibrated_losses_eval: np.ndarray,
                       uniform_losses_eval: np.ndarray,
                       eps: float = 1e-12) -> float:
        """
        NormReg@1(w*) = Reg@1(w*) / Reg@1(u).

        NormReg@1 < 1 means the calibrated weighting w* outperforms the
        uniform baseline u for top-1 model selection on Heval.
        """
        reg_w = EvaluationMetrics.reg_at_1(real_losses_eval, calibrated_losses_eval)
        reg_u = EvaluationMetrics.reg_at_1(real_losses_eval, uniform_losses_eval)
        if reg_u <= eps:
            return np.nan if reg_w <= eps else np.inf
        return reg_w / reg_u

    @staticmethod
    def robust_normreg_summary(norm_reg_at_1_values: List[float],
                                reg_u_values: Optional[List[float]] = None,
                                degenerate_eps: float = 1e-12) -> Dict[str, Any]:
        """
        Returns:
            mean_finite: mean over folds excluding inf/nan (may still be biased if
                many folds are excluded -- check n_inf/n_total).
            median_all: median treating inf as missing (robust to a single
                degenerate fold, unlike the mean).
            n_total, n_inf, n_nan: diagnostic counts.
            degenerate_uniform_frac: fraction of folds where Reg@1(u) <= eps,
                i.e. where NormReg@1 is structurally unstable by construction
                (requires reg_u_values to be passed).
        """
        values = np.asarray(norm_reg_at_1_values, dtype=float)
        finite_values = values[np.isfinite(values)]
        n_total = len(values)
        n_inf = int(np.sum(np.isinf(values)))
        n_nan = int(np.sum(np.isnan(values)))

        values_for_median = np.where(np.isinf(values), np.nan, values)
        median_all = float(np.nanmedian(values_for_median)) if n_total > 0 else np.nan

        degenerate_frac = np.nan
        if reg_u_values is not None:
            reg_u_arr = np.asarray(reg_u_values, dtype=float)
            if len(reg_u_arr) > 0:
                degenerate_frac = float(np.mean(reg_u_arr <= degenerate_eps))

        return {
            'mean_finite': float(np.mean(finite_values)) if len(finite_values) > 0 else np.nan,
            'median_all': median_all,
            'n_total': n_total,
            'n_inf': n_inf,
            'n_nan': n_nan,
            'degenerate_uniform_frac': degenerate_frac,
        }

    @staticmethod
    def harm_rate(norm_reg_at_1_values: List[float]) -> float:
        """
        HarmRate = fraction of trials/datasets/folds where
        Reg@1(w*) > Reg@1(u), i.e. NormReg@1 > 1.
        """
        values = np.asarray(norm_reg_at_1_values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return np.nan
        return float(np.mean(values > 1.0))

    @staticmethod
    def excess_harm_gain(reg_w: float, reg_u: float) -> Dict[str, float]:
        """
        ExcessHarm = max(0, Reg@1(w*) - Reg@1(u))
        Gain       = max(0, Reg@1(u) - Reg@1(w*))
        (Section 3 metric table: "max(0, ±(Reg@1(w*) - Reg@1(u)))").
        """
        diff = reg_w - reg_u
        return {
            'excess_harm': float(max(0.0, diff)),
            'gain': float(max(0.0, -diff)),
        }

    # ------------------------------------------------------------------
    # WP3 deployment-time diagnostics (computed without access to Heval)
    # ------------------------------------------------------------------

    @staticmethod
    def effective_sample_size(weights: np.ndarray) -> float:
        """ESS = 1 / sum(w_i^2), for weights normalized to sum to 1."""
        weights = np.asarray(weights, dtype=float)
        total = weights.sum()
        if total <= 0:
            return np.nan
        w_norm = weights / total
        denom = np.sum(w_norm ** 2)
        return float(1.0 / denom) if denom > 0 else np.nan

    @staticmethod
    def weight_entropy(weights: np.ndarray, eps: float = 1e-15) -> float:
        """Shannon entropy H(w) of the (normalized) calibration weight vector."""
        weights = np.asarray(weights, dtype=float)
        total = weights.sum()
        if total <= 0:
            return np.nan
        w_norm = weights / total
        w_norm = np.clip(w_norm, eps, None)
        return float(-np.sum(w_norm * np.log(w_norm)))

    @staticmethod
    def weight_max(weights: np.ndarray) -> float:
        """w_max: largest normalized weight (concentration diagnostic)."""
        weights = np.asarray(weights, dtype=float)
        total = weights.sum()
        if total <= 0:
            return np.nan
        return float(np.max(weights / total))

    @staticmethod
    def condition_number(loss_matrix_cal: np.ndarray) -> float:
        """
        kappa(Lsyn_cal): condition number of the calibration-model synthetic
        loss matrix (models x samples). High kappa flags ill-posedness of
        the weight-fitting problem.
        """
        loss_matrix_cal = np.asarray(loss_matrix_cal, dtype=float)
        if loss_matrix_cal.ndim != 2 or min(loss_matrix_cal.shape) == 0:
            return np.nan
        try:
            return float(np.linalg.cond(loss_matrix_cal))
        except np.linalg.LinAlgError:
            return np.nan

    @staticmethod
    def loo_residual(calibration_models_losses_real: np.ndarray,
                      calibration_models_losses_synth: np.ndarray,
                      fit_weights_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      predict_risk_fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
        """
        Leave-one-out cross-fitted residual on Hcal (WP3 deployment-time
        diagnostic, no access to Heval):

        For each model k in Hcal, refit weights on Hcal \\ {k} and measure
        the residual between the predicted risk for k and its known
        cross-fitted real loss. Returns the mean absolute LOO residual.

        Args:
            calibration_models_losses_real: shape (M,) real losses on Hcal
            calibration_models_losses_synth: shape (M, ns) synthetic
                per-sample losses on Hcal
            fit_weights_fn: (losses_real_subset, losses_synth_subset) -> weights
            predict_risk_fn: (weights, losses_synth_for_model) -> predicted risk
        """
        real = np.asarray(calibration_models_losses_real, dtype=float)
        synth = np.asarray(calibration_models_losses_synth, dtype=float)
        m = len(real)
        if m < 2:
            return np.nan

        residuals = []
        for k in range(m):
            mask = np.ones(m, dtype=bool)
            mask[k] = False
            w_loo = fit_weights_fn(real[mask], synth[mask])
            pred_k = predict_risk_fn(w_loo, synth[k])
            residuals.append(abs(pred_k - real[k]))

        return float(np.mean(residuals))

    @classmethod
    def deployment_diagnostics(cls,
                                weights: np.ndarray,
                                loss_matrix_cal: np.ndarray,
                                synth_losses_cal: np.ndarray,
                                real_losses_cal: np.ndarray,
                                task_type: str,
                                n_real: int,
                                loo_residual_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Bundle of all WP3 deployment-time diagnostics computable without
        access to Heval: LOO-residual, ESS, w_max, H(w), kappa(Lsyn_cal),
        rho_cal, M, ns, n, task_type.
        """
        rho_cal = cls.calibration_spearman_correlation(
            synth_losses_cal, real_losses_cal
        )['correlation']
        return {
            'loo_residual': loo_residual_value,
            'ess': cls.effective_sample_size(weights),
            'w_max': cls.weight_max(weights),
            'weight_entropy': cls.weight_entropy(weights),
            'condition_number': cls.condition_number(loss_matrix_cal),
            'rho_cal': rho_cal,
            'M': len(real_losses_cal),
            'n_synth': loss_matrix_cal.shape[1] if loss_matrix_cal.ndim == 2 else np.nan,
            'n_real': n_real,
            'task_type': task_type,
        }

    # ------------------------------------------------------------------
    # Theoretical rank-preservation bound (Corollary 3.3, paper [1])
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rank_preservation_with_guarantees(real_errors: np.ndarray,
                                                   synth_errors: np.ndarray) -> Dict[str, float]:
        """
        Rank preservation with theoretical guarantees based on total
        variation distance (Corollary 3.3). Retained as-is: not redundant
        with the plan's Reg@1/NormReg@1 (this is a pairwise, TV-bound-based
        analysis rather than a top-1 selection-regret analysis).
        """
        n_models = len(real_errors)
        real_errors = np.asarray(real_errors, dtype=float)
        synth_errors = np.asarray(synth_errors, dtype=float)

        delta_tv = np.mean(np.abs(synth_errors - real_errors))

        total_pairs = 0
        guaranteed_preserved = 0
        empirical_preserved = 0
        violations = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                total_pairs += 1

                delta_real = real_errors[j] - real_errors[i]
                delta_synth = synth_errors[j] - synth_errors[i]

                has_guarantee = abs(delta_synth) >= delta_tv
                signs_match = (np.sign(delta_real) == np.sign(delta_synth) or
                               np.sign(delta_real) == 0 or
                               np.sign(delta_synth) == 0)

                if signs_match:
                    if has_guarantee:
                        guaranteed_preserved += 1
                    else:
                        empirical_preserved += 1
                else:
                    violations += 1

        preservation_rate = (guaranteed_preserved + empirical_preserved) / total_pairs if total_pairs > 0 else 0.0

        return {
            'preservation_rate': preservation_rate,
            'guaranteed_rate': guaranteed_preserved / total_pairs if total_pairs > 0 else 0.0,
            'violation_rate': violations / total_pairs if total_pairs > 0 else 0.0,
            'total_variation': delta_tv,
            'total_pairs': total_pairs
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def plot_rank_comparison(losses_synth: np.ndarray,
                              losses_test: np.ndarray,
                              model_names: Optional[List[str]] = None,
                              calibrated_losses: Optional[np.ndarray] = None,
                              figsize: Tuple[int, int] = (14, 10),
                              title: str = "Model Ranking Comparison"):
        """Visualize rank comparison between synthetic and test losses."""
        n_models = len(losses_synth)

        if model_names is None:
            model_names = [f'Model_{i}' for i in range(n_models)]

        ranks_synth = stats.rankdata(losses_synth, method='ordinal')
        ranks_test = stats.rankdata(losses_test, method='ordinal')

        n_plots = 3 if calibrated_losses is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        ax1 = axes[0]
        ax1.scatter(losses_synth, losses_test, alpha=0.7, edgecolors='k', s=50)
        ax1.plot([min(losses_synth), max(losses_synth)],
                  [min(losses_synth), max(losses_synth)], 'r--', label='y=x')
        corr = EvaluationMetrics.spearman_correlation(losses_synth, losses_test)
        ax1.set_xlabel('Synthetic Loss', fontsize=12)
        ax1.set_ylabel('Test Loss', fontsize=12)
        ax1.set_title(f'Loss Comparison\nSpearman rho = {corr["correlation"]:.3f}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        for i in range(n_models):
            color = 'green' if abs(ranks_synth[i] - ranks_test[i]) <= n_models * 0.1 else 'red'
            ax2.plot([0, 1], [ranks_synth[i], ranks_test[i]],
                      color=color, alpha=0.5, linewidth=1)
        ax2.scatter([0] * n_models, ranks_synth, alpha=0.7, label='Synth Rank', s=30)
        ax2.scatter([1] * n_models, ranks_test, alpha=0.7, label='Test Rank', s=30)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Synthetic', 'Test'])
        ax2.set_ylabel('Rank (lower = better)', fontsize=12)
        ax2.set_title('Rank Transitions', fontsize=14)
        ax2.legend()
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)

        if calibrated_losses is not None:
            ax3 = axes[2]
            ax3.scatter(calibrated_losses, losses_test, alpha=0.7,
                        edgecolors='k', s=50, color='green')
            ax3.plot([min(calibrated_losses), max(calibrated_losses)],
                      [min(calibrated_losses), max(calibrated_losses)], 'r--', label='y=x')
            corr_cal = EvaluationMetrics.spearman_correlation(calibrated_losses, losses_test)
            ax3.set_xlabel('Calibrated Loss', fontsize=12)
            ax3.set_ylabel('Test Loss', fontsize=12)
            ax3.set_title(f'After Calibration\nSpearman rho = {corr_cal["correlation"]:.3f}', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_topk_overlap(losses_synth: np.ndarray,
                           losses_test: np.ndarray,
                           max_k: int = 20,
                           calibrated_losses: Optional[np.ndarray] = None,
                           figsize: Tuple[int, int] = (10, 6)):
        """
        Plot top-k overlap (%) as a function of k. Delegates the overlap
        computation to hit_at_k instead of recomputing it inline.
        """
        max_k = min(max_k, len(losses_synth))
        ks = range(1, max_k + 1)

        overlaps_raw = [EvaluationMetrics.hit_at_k(losses_test, losses_synth, k) * 100 for k in ks]
        overlaps_calibrated = None
        if calibrated_losses is not None:
            overlaps_calibrated = [EvaluationMetrics.hit_at_k(losses_test, calibrated_losses, k) * 100 for k in ks]

        plt.figure(figsize=figsize)
        plt.plot(ks, overlaps_raw, 'b-o', label='Raw Synthetic', linewidth=2, markersize=4)

        if overlaps_calibrated is not None:
            plt.plot(ks, overlaps_calibrated, 'g-s', label='Calibrated', linewidth=2, markersize=4)

        plt.axhline(y=50, color='r', linestyle='--', label='Random (50%)', alpha=0.7)

        plt.xlabel('k (number of top models)', fontsize=12)
        plt.ylabel('Top-k Overlap (%)', fontsize=12)
        plt.title('Top-k Model Selection Overlap', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.show()
