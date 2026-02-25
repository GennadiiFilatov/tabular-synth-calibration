"""
Theoretical Framework Implementation.

Implementation of Lemma 3.1, Theorem 3.2, and Corollary 3.3
from the synthetic data validation paper.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class TheoreticalFramework:
    """
    Implementation of Lemma 3.1, Theorem 3.2, and Corollary 3.3
    from the paper.
    """

    @staticmethod
    def compute_disagreement_regions(h1_preds: np.ndarray, 
                                     h2_preds: np.ndarray, 
                                     true_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lemma 3.1: Compute Ω1 and Ω2 regions where hypotheses disagree.

        Ω1 = {x | h1(x) ≠ f(x) ∧ h2(x) = f(x)}
        Ω2 = {x | h2(x) ≠ f(x) ∧ h1(x) = f(x)}

        Args:
            h1_preds: Predictions from hypothesis h1
            h2_preds: Predictions from hypothesis h2
            true_labels: True labels f(x)

        Returns:
            Tuple of (Ω1 indices, Ω2 indices)
        """
        h1_correct = (h1_preds == true_labels)
        h2_correct = (h2_preds == true_labels)

        # Ω1: h1 wrong, h2 correct
        omega1 = (~h1_correct) & h2_correct

        # Ω2: h2 wrong, h1 correct
        omega2 = (~h2_correct) & h1_correct

        return omega1, omega2

    @staticmethod
    def compute_risk_difference(h1_preds: np.ndarray, 
                                h2_preds: np.ndarray, 
                                true_labels: np.ndarray) -> float:
        """
        Lemma 3.1: Compute Δε = ε(h2) - ε(h1)

        The risk difference depends only on disagreement regions:
        Δε = ∫_Ω2 μ(x)dx - ∫_Ω1 μ(x)dx

        For empirical risk: Δε = (|Ω2| - |Ω1|) / N
        """
        omega1, omega2 = TheoreticalFramework.compute_disagreement_regions(
            h1_preds, h2_preds, true_labels
        )

        n_samples = len(true_labels)
        delta_epsilon = (omega2.sum() - omega1.sum()) / n_samples

        return delta_epsilon

    @staticmethod
    def compute_total_variation(h1_preds: np.ndarray, 
                                h2_preds: np.ndarray,
                                true_labels_real: np.ndarray,
                                true_labels_synth: np.ndarray,
                                weights_real: Optional[np.ndarray] = None,
                                weights_synth: Optional[np.ndarray] = None) -> float:
        """
        Theorem 3.2: Compute total variation δ_h1⊕h2(μr, μs)

        This measures the distribution gap in disagreement regions.

        Returns:
            Total variation distance
        """
        # Find disagreement region (h1 ⊕ h2)
        omega1_real, omega2_real = TheoreticalFramework.compute_disagreement_regions(
            h1_preds[:len(true_labels_real)], 
            h2_preds[:len(true_labels_real)], 
            true_labels_real
        )
        disagreement_real = omega1_real | omega2_real

        omega1_synth, omega2_synth = TheoreticalFramework.compute_disagreement_regions(
            h1_preds[len(true_labels_real):], 
            h2_preds[len(true_labels_real):], 
            true_labels_synth
        )
        disagreement_synth = omega1_synth | omega2_synth

        # Compute total variation in disagreement region
        if weights_real is None:
            weights_real = np.ones(len(true_labels_real)) / len(true_labels_real)
        if weights_synth is None:
            weights_synth = np.ones(len(true_labels_synth)) / len(true_labels_synth)

        # Total variation: ∫|μr - μs|dx
        tv = 0.5 * (np.abs(weights_real[disagreement_real].sum() - 
                           weights_synth[disagreement_synth].sum()))

        return tv

    @staticmethod
    def check_rank_preservation(delta_epsilon_synth: float, 
                                total_variation: float) -> bool:
        """
        Corollary 3.3: Check if rank will be preserved.

        If Δεs ≥ δ(μr, μs), then Δεr ≥ 0 (rank preserved)

        Args:
            delta_epsilon_synth: Risk difference on synthetic data
            total_variation: Total variation between distributions

        Returns:
            True if rank preservation is guaranteed
        """
        return abs(delta_epsilon_synth) >= total_variation

    @staticmethod
    def analyze_rank_preservation(models_real_loss: List[float],
                                  models_synth_loss: List[float],
                                  model_names: List[str]) -> pd.DataFrame:
        """
        Complete rank preservation analysis for multiple models.

        Returns:
            DataFrame with detailed analysis
        """
        results = []

        for i in range(len(models_real_loss)):
            for j in range(i + 1, len(models_real_loss)):
                delta_real = models_real_loss[j] - models_real_loss[i]
                delta_synth = models_synth_loss[j] - models_synth_loss[i]

                # Check if rank is preserved
                real_rank = np.sign(delta_real)
                synth_rank = np.sign(delta_synth)
                # Properly handle ties: rank preserved if same sign OR either is a tie (0)
                preserved = (real_rank == synth_rank) or (real_rank == 0) or (synth_rank == 0)

                results.append({
                    'Model 1': model_names[i],
                    'Model 2': model_names[j],
                    'Δε_real': delta_real,
                    'Δε_synth': delta_synth,
                    'Rank Preserved': preserved,
                    'Loss Gap': abs(delta_real - delta_synth)
                })

        return pd.DataFrame(results)
