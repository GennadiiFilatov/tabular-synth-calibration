"""
Evaluation Metrics Module.

Comprehensive metrics for synthetic data validation including:
- Spearman correlation
- Rank preservation  
- Model selection accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for synthetic data validation.
    """
    
    def __init__(self):
        self.results = {}
    
    @staticmethod
    def spearman_correlation(losses_synth: np.ndarray,
                             losses_test: np.ndarray,
                             alternative: str = 'two-sided') -> Dict[str, float]:
        """
        Compute Spearman rank correlation between synthetic and test losses.

        Args:
            losses_synth: Model losses on synthetic data
            losses_test: Model losses on real test data  
            alternative: 'two-sided', 'greater', or 'less'

        Returns:
            Dictionary with correlation coefficient and p-value
        """
        if len(losses_synth) < 3:
            return {'correlation': np.nan, 'p_value': np.nan, 'n_samples': len(losses_synth)}
            
        corr, p_value = stats.spearmanr(losses_synth, losses_test, alternative=alternative)
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'n_samples': len(losses_synth)
        }

    @staticmethod
    def rank_preservation(losses_synth: np.ndarray,
                          losses_test: np.ndarray,
                          top_k: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Compute rank preservation metrics.

        Args:
            losses_synth: Model losses on synthetic data
            losses_test: Model losses on real test data
            top_k: List of k values for top-k overlap (default: [1, 3, 5, 10])

        Returns:
            Dictionary with rank preservation metrics
        """
        if top_k is None:
            top_k = [1, 3, 5, 10]

        n_models = len(losses_synth)

        # Ranks (lower is better for loss)
        ranks_synth = stats.rankdata(losses_synth, method='ordinal')
        ranks_test = stats.rankdata(losses_test, method='ordinal')

        # Top-k overlap metrics
        results = {}
        for k in top_k:
            if k > n_models:
                continue

            top_k_synth = set(np.argsort(losses_synth)[:k])
            top_k_test = set(np.argsort(losses_test)[:k])

            overlap = len(top_k_synth & top_k_test)
            results[f'top_{k}_overlap'] = overlap
            results[f'top_{k}_overlap_pct'] = overlap / k * 100

        # Kendall's tau
        tau, tau_p = stats.kendalltau(losses_synth, losses_test)
        results['kendall_tau'] = tau
        results['kendall_tau_p_value'] = tau_p

        # Mean absolute rank difference
        results['mean_abs_rank_diff'] = np.mean(np.abs(ranks_synth - ranks_test))

        # Best model agreement
        best_synth = np.argmin(losses_synth)
        best_test = np.argmin(losses_test)
        results['best_model_match'] = int(best_synth == best_test)
        results['best_synth_rank_in_test'] = int(ranks_test[best_synth])

        return results

    @staticmethod
    def compute_rank_preservation_with_guarantees(real_errors: np.ndarray,
                                                  synth_errors: np.ndarray) -> Dict[str, float]:
        """
        Compute rank preservation using Corollary 3.3 with detailed analysis.
        
        Analyzes pairwise rank preservation with theoretical guarantees based on
        total variation distance.
        
        Args:
            real_errors: Model errors/losses on real test data
            synth_errors: Model errors/losses on synthetic data
            
        Returns:
            Dictionary with preservation metrics:
                - preservation_rate: Overall preservation rate
                - guaranteed_rate: Rate with theoretical guarantee
                - violation_rate: Rate of rank violations
                - total_variation: Mean absolute difference (δ_TV)
                - total_pairs: Number of model pairs analyzed
        """
        n_models = len(real_errors)
        real_errors = np.array(real_errors)
        synth_errors = np.array(synth_errors)
        
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

    @staticmethod
    def compute_selection_accuracy(losses_synth: np.ndarray, 
                                   losses_test: np.ndarray,
                                   selection_threshold: int = 10) -> Dict[str, float]:
        """
        Compute model selection accuracy metrics.
        
        Args:
            losses_synth: Losses on synthetic data
            losses_test: Losses on real test data
            selection_threshold: Number of top models to consider
            
        Returns:
            Dictionary with selection accuracy metrics
        """
        n_models = len(losses_synth)
        k = min(selection_threshold, n_models)
        
        # Sort by synthetic loss
        synth_order = np.argsort(losses_synth)
        test_order = np.argsort(losses_test)
        
        # Top-k selection accuracy
        top_k_synth = set(synth_order[:k])
        top_k_test = set(test_order[:k])
        
        # Precision: how many of our selections are actually good
        precision = len(top_k_synth & top_k_test) / k
        
        # Recall: how many of the actual good models did we select
        recall = len(top_k_synth & top_k_test) / k
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Best model: did we select the actual best?
        best_model_selected = int(synth_order[0] == test_order[0])
        
        # Regret: how much worse is our selection vs optimal
        synth_best_idx = synth_order[0]
        test_best_idx = test_order[0]
        
        synth_selected_test_loss = losses_test[synth_best_idx]
        optimal_test_loss = losses_test[test_best_idx]
        
        regret = synth_selected_test_loss - optimal_test_loss
        relative_regret = regret / optimal_test_loss if optimal_test_loss != 0 else np.inf
        
        return {
            'precision_at_k': precision,
            'recall_at_k': recall,
            'f1_at_k': f1,
            'best_model_selected': best_model_selected,
            'regret': regret,
            'relative_regret': relative_regret,
            'selection_threshold': k
        }
    
    @staticmethod
    def plot_rank_comparison(losses_synth: np.ndarray,
                             losses_test: np.ndarray,
                             model_names: Optional[List[str]] = None,
                             calibrated_losses: Optional[np.ndarray] = None,
                             figsize: Tuple[int, int] = (14, 10),
                             title: str = "Model Ranking Comparison"):
        """
        Visualize rank comparison between synthetic and test losses.
        
        Args:
            losses_synth: Losses on synthetic data
            losses_test: Losses on real test data
            model_names: Optional list of model names
            calibrated_losses: Optional calibrated losses
            figsize: Figure size
            title: Plot title
        """
        n_models = len(losses_synth)
        
        if model_names is None:
            model_names = [f'Model_{i}' for i in range(n_models)]
        
        # Compute ranks
        ranks_synth = stats.rankdata(losses_synth, method='ordinal')
        ranks_test = stats.rankdata(losses_test, method='ordinal')
        
        if calibrated_losses is not None:
            ranks_calibrated = stats.rankdata(calibrated_losses, method='ordinal')
        
        # Create figure
        n_plots = 3 if calibrated_losses is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        # Plot 1: Scatter plot of losses
        ax1 = axes[0]
        ax1.scatter(losses_synth, losses_test, alpha=0.7, edgecolors='k', s=50)
        ax1.plot([min(losses_synth), max(losses_synth)],
                 [min(losses_synth), max(losses_synth)], 'r--', label='y=x')
        corr = EvaluationMetrics.spearman_correlation(losses_synth, losses_test)
        ax1.set_xlabel('Synthetic Loss', fontsize=12)
        ax1.set_ylabel('Test Loss', fontsize=12)
        ax1.set_title(f'Loss Comparison\nSpearman ρ = {corr["correlation"]:.3f}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rank comparison
        ax2 = axes[1]
        for i in range(n_models):
            color = 'green' if abs(ranks_synth[i] - ranks_test[i]) <= n_models * 0.1 else 'red'
            ax2.plot([0, 1], [ranks_synth[i], ranks_test[i]], 
                     color=color, alpha=0.5, linewidth=1)
        ax2.scatter([0]*n_models, ranks_synth, alpha=0.7, label='Synth Rank', s=30)
        ax2.scatter([1]*n_models, ranks_test, alpha=0.7, label='Test Rank', s=30)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Synthetic', 'Test'])
        ax2.set_ylabel('Rank (lower = better)', fontsize=12)
        ax2.set_title('Rank Transitions', fontsize=14)
        ax2.legend()
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Calibrated comparison (if available)
        if calibrated_losses is not None:
            ax3 = axes[2]
            ax3.scatter(calibrated_losses, losses_test, alpha=0.7, 
                       edgecolors='k', s=50, color='green')
            ax3.plot([min(calibrated_losses), max(calibrated_losses)],
                     [min(calibrated_losses), max(calibrated_losses)], 'r--', label='y=x')
            corr_cal = EvaluationMetrics.spearman_correlation(calibrated_losses, losses_test)
            ax3.set_xlabel('Calibrated Loss', fontsize=12)
            ax3.set_ylabel('Test Loss', fontsize=12)
            ax3.set_title(f'After Calibration\nSpearman ρ = {corr_cal["correlation"]:.3f}', fontsize=14)
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
        Plot top-k overlap as a function of k.
        
        Args:
            losses_synth: Losses on synthetic data
            losses_test: Losses on real test data
            max_k: Maximum k to plot
            calibrated_losses: Optional calibrated losses
            figsize: Figure size
        """
        max_k = min(max_k, len(losses_synth))
        ks = range(1, max_k + 1)
        
        overlaps_raw = []
        overlaps_calibrated = []
        
        for k in ks:
            top_k_synth = set(np.argsort(losses_synth)[:k])
            top_k_test = set(np.argsort(losses_test)[:k])
            overlaps_raw.append(len(top_k_synth & top_k_test) / k * 100)
            
            if calibrated_losses is not None:
                top_k_cal = set(np.argsort(calibrated_losses)[:k])
                overlaps_calibrated.append(len(top_k_cal & top_k_test) / k * 100)
        
        plt.figure(figsize=figsize)
        plt.plot(ks, overlaps_raw, 'b-o', label='Raw Synthetic', linewidth=2, markersize=4)
        
        if calibrated_losses is not None:
            plt.plot(ks, overlaps_calibrated, 'g-s', label='Calibrated', linewidth=2, markersize=4)
        
        # Random baseline
        plt.axhline(y=50, color='r', linestyle='--', label='Random (50%)', alpha=0.7)
        
        plt.xlabel('k (number of top models)', fontsize=12)
        plt.ylabel('Top-k Overlap (%)', fontsize=12)
        plt.title('Top-k Model Selection Overlap', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.show()
