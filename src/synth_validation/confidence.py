"""
Confidence Interval Estimation Module.

Bootstrap and analytical methods for computing confidence intervals.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict


class ConfidenceIntervalEstimator:
    """
    Compute confidence intervals using bootstrap and analytical methods.
    """

    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000, 
                 random_state: int = 42):
        """
        Initialize confidence interval estimator.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.results = {}

    def bootstrap_ci(self,
                     data: np.ndarray,
                     statistic: Callable = np.mean,
                     method: str = 'percentile') -> Dict[str, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            data: Input data array
            statistic: Function to compute statistic (default: np.mean)
            method: 'percentile', 'bca', or 'basic'

        Returns:
            Dictionary with CI bounds and point estimate
        """
        np.random.seed(self.random_state)
        
        n = len(data)
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        point_estimate = statistic(data)
        
        alpha = 1 - self.confidence_level
        
        if method == 'percentile':
            lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
            upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        
        elif method == 'basic':
            lower = 2 * point_estimate - np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
            upper = 2 * point_estimate - np.percentile(bootstrap_stats, alpha / 2 * 100)
        
        elif method == 'bca':
            # BCa (Bias-Corrected and Accelerated) method
            # Bias correction
            z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))
            
            # Acceleration (jackknife estimate)
            jackknife_stats = []
            for i in range(n):
                jack_sample = np.delete(data, i)
                jackknife_stats.append(statistic(jack_sample))
            jackknife_stats = np.array(jackknife_stats)
            jackknife_mean = np.mean(jackknife_stats)
            
            num = np.sum((jackknife_mean - jackknife_stats) ** 3)
            den = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
            acc = num / den if den != 0 else 0
            
            # Adjusted percentiles
            z_alpha = stats.norm.ppf(alpha / 2)
            z_1_alpha = stats.norm.ppf(1 - alpha / 2)
            
            alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - acc * (z0 + z_alpha)))
            alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - acc * (z0 + z_1_alpha)))
            
            lower = np.percentile(bootstrap_stats, alpha1 * 100)
            upper = np.percentile(bootstrap_stats, alpha2 * 100)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'point_estimate': point_estimate,
            'lower': lower,
            'upper': upper,
            'ci_width': upper - lower,
            'se': np.std(bootstrap_stats),
            'method': method,
            'confidence_level': self.confidence_level
        }

    def analytical_ci_spearman(self, 
                               rho: float, 
                               n: int) -> Dict[str, float]:
        """
        Compute analytical confidence interval for Spearman correlation.
        
        Uses Fisher transformation for correlation confidence intervals.

        Args:
            rho: Spearman correlation coefficient
            n: Sample size

        Returns:
            Dictionary with CI bounds
        """
        # Fisher transformation
        z = 0.5 * np.log((1 + rho) / (1 - rho + 1e-10))
        
        # Standard error
        se_z = 1.0 / np.sqrt(n - 3) if n > 3 else np.inf
        
        # Critical value
        alpha = 1 - self.confidence_level
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        # CI in z-space
        z_lower = z - z_crit * se_z
        z_upper = z + z_crit * se_z
        
        # Transform back
        rho_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        rho_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return {
            'point_estimate': rho,
            'lower': max(-1, rho_lower),
            'upper': min(1, rho_upper),
            'ci_width': rho_upper - rho_lower,
            'se': se_z,
            'method': 'analytical_fisher',
            'confidence_level': self.confidence_level,
            'n': n
        }
    
    def analytical_ci_mean(self, 
                           data: np.ndarray) -> Dict[str, float]:
        """
        Compute analytical confidence interval for mean using t-distribution.

        Args:
            data: Input data array

        Returns:
            Dictionary with CI bounds
        """
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        
        alpha = 1 - self.confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        
        margin = t_crit * se
        
        return {
            'point_estimate': mean,
            'lower': mean - margin,
            'upper': mean + margin,
            'ci_width': 2 * margin,
            'se': se,
            'method': 'analytical_t',
            'confidence_level': self.confidence_level,
            'n': n
        }

    def bootstrap_ci_correlation(self,
                                 x: np.ndarray,
                                 y: np.ndarray,
                                 correlation_type: str = 'spearman') -> Dict[str, float]:
        """
        Compute bootstrap CI for correlation coefficient.

        Args:
            x: First variable
            y: Second variable  
            correlation_type: 'spearman' or 'pearson'

        Returns:
            Dictionary with CI bounds
        """
        np.random.seed(self.random_state)
        
        n = len(x)
        bootstrap_corrs = []
        
        if correlation_type == 'spearman':
            corr_func = lambda a, b: stats.spearmanr(a, b)[0]
        else:
            corr_func = lambda a, b: stats.pearsonr(a, b)[0]
        
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            try:
                bootstrap_corrs.append(corr_func(x_boot, y_boot))
            except:
                continue
        
        bootstrap_corrs = np.array(bootstrap_corrs)
        
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_corrs, alpha / 2 * 100)
        upper = np.percentile(bootstrap_corrs, (1 - alpha / 2) * 100)
        
        point_estimate = corr_func(x, y)
        
        return {
            'point_estimate': point_estimate,
            'lower': lower,
            'upper': upper,
            'ci_width': upper - lower,
            'se': np.std(bootstrap_corrs),
            'method': f'bootstrap_{correlation_type}',
            'confidence_level': self.confidence_level,
            'n_valid_bootstrap': len(bootstrap_corrs)
        }

    def compare_methods(self, 
                        data: np.ndarray,
                        statistic: Callable = np.mean) -> Dict[str, Dict]:
        """
        Compare different CI estimation methods.

        Args:
            data: Input data array
            statistic: Function to compute statistic

        Returns:
            Dictionary comparing different methods
        """
        results = {}
        
        # Bootstrap methods
        for method in ['percentile', 'basic', 'bca']:
            results[f'bootstrap_{method}'] = self.bootstrap_ci(data, statistic, method)
        
        # Analytical (for mean)
        if statistic == np.mean:
            results['analytical_t'] = self.analytical_ci_mean(data)
        
        return results

    def compute_spearman(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Compute Spearman correlation coefficient.
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Tuple of (correlation, p-value)
        """
        if len(x) < 3:
            return np.nan, np.nan
        corr, pval = stats.spearmanr(x, y)
        return corr, pval
    
    def compute_kendall(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Compute Kendall tau-b rank correlation coefficient.

        Args:
            x: First array (e.g. real losses)
            y: Second array (e.g. synthetic losses)

        Returns:
            Tuple of (tau, p-value)
        """
        if len(x) < 3:
            return np.nan, np.nan
        tau, pval = stats.kendalltau(x, y)
        return float(tau), float(pval)

    def aggregate_ci_from_samples(self, samples: List[float]) -> Dict[str, float]:
        """
        Compute CI from fold samples using t-distribution.
        
        Args:
            samples: List of values from each fold
            
        Returns:
            Dictionary with mean, std, se, ci_lower, ci_upper, n
        """
        samples = np.array(samples)
        n = len(samples)
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        se = std / np.sqrt(n)
        
        alpha = 1 - self.confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        
        return {
            'mean': mean,
            'std': std,
            'se': se,
            'ci_lower': mean - t_crit * se,
            'ci_upper': mean + t_crit * se,
            'n': n
        }

    def aggregate_cis_across_folds(self, 
                                   fold_results: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate confidence intervals across multiple cross-validation folds.

        Args:
            fold_results: List of dictionaries with fold-level results

        Returns:
            Aggregated confidence intervals
        """
        aggregated = defaultdict(list)
        
        # Collect all metrics across folds
        for fold_dict in fold_results:
            for key, value in fold_dict.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    aggregated[key].append(value)
        
        # Compute CIs for each metric
        results = {}
        for key, values in aggregated.items():
            if len(values) >= 2:
                values_arr = np.array(values)
                
                # Bootstrap CI
                bootstrap_result = self.bootstrap_ci(values_arr, np.mean, 'percentile')
                
                # Analytical CI
                analytical_result = self.analytical_ci_mean(values_arr)
                
                results[key] = {
                    'mean': np.mean(values_arr),
                    'std': np.std(values_arr),
                    'n_folds': len(values_arr),
                    'bootstrap_ci': (bootstrap_result['lower'], bootstrap_result['upper']),
                    'analytical_ci': (analytical_result['lower'], analytical_result['upper']),
                    'values': values_arr.tolist()
                }
        
        return results
