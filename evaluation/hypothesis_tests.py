"""
evaluation/hypothesis_tests.py
Statistical Hypothesis Testing Suite for Volatility Forecasting:
1. Stationary / Moving Block Bootstrap for QLIKE Confidence Intervals
2. Diebold-Mariano (DM) & Giacomini-White (GW) tests with Newey-West HAC
3. Two One-Sided Tests (TOST) for equivalence testing
4. Holm-Bonferroni step-down correction for multiple testing
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any, List

def newey_west_hac(d: np.ndarray, max_lags: int = None) -> float:
    """
    Computes Newey-West HAC variance estimator for loss differential series d.
    """
    n = len(d)
    if max_lags is None:
        max_lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    max_lags = max(1, min(max_lags, n - 1))
    
    d_centered = d - np.mean(d)
    gamma_0 = np.mean(d_centered ** 2)
    
    var_sum = gamma_0
    for l in range(1, max_lags + 1):
        weight = 1.0 - (l / (max_lags + 1.0))
        gamma_l = np.mean(d_centered[l:] * d_centered[:-l])
        var_sum += 2.0 * weight * gamma_l
        
    return max(1e-12, var_sum / n)

def diebold_mariano_test(loss1: np.ndarray, loss2: np.ndarray, max_lags: int = None) -> Tuple[float, float]:
    """
    Computes Diebold-Mariano test statistic and two-sided p-value.
    d = loss1 - loss2
    H0: E[d] = 0 (equal predictive accuracy)
    H1: E[d] != 0
    Positive stat indicates model 2 has lower loss than model 1.
    """
    d = loss1 - loss2
    n = len(d)
    mean_d = np.mean(d)
    var_d = newey_west_hac(d, max_lags=max_lags)
    stat = mean_d / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat)))
    return float(stat), float(p_value)

def block_bootstrap_ci(losses: np.ndarray, block_size: int = 48, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    """
    Computes Moving Block Bootstrap confidence interval for mean loss.
    Returns: (mean, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(losses)
    num_blocks = int(np.ceil(n / block_size))
    
    boot_means = np.zeros(n_boot)
    max_start = n - block_size
    if max_start <= 0:
        return float(np.mean(losses)), float(np.min(losses)), float(np.max(losses))
        
    for b in range(n_boot):
        start_indices = rng.integers(0, max_start, size=num_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in start_indices])[:n]
        boot_means[b] = np.mean(losses[indices])
        
    ci_lower = np.percentile(boot_means, 100.0 * (alpha / 2.0))
    ci_upper = np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))
    return float(np.mean(losses)), float(ci_lower), float(ci_upper)

def tost_equivalence_test(loss_test: np.ndarray, loss_ref: np.ndarray, margin: float = 0.02) -> Dict[str, Any]:
    """
    Two One-Sided Tests (TOST) for equivalence of predictive accuracy.
    H0_lower: mu_test - mu_ref <= -margin
    H0_upper: mu_test - mu_ref >= margin
    If both are rejected at level alpha, we conclude equivalence within [-margin, margin].
    """
    d = loss_test - loss_ref
    n = len(d)
    mean_diff = np.mean(d)
    se = np.sqrt(newey_west_hac(d))
    
    t_lower = (mean_diff - (-margin)) / se
    p_lower = 1.0 - stats.norm.cdf(t_lower)
    
    t_upper = (mean_diff - margin) / se
    p_upper = stats.norm.cdf(t_upper)
    
    p_tost = max(p_lower, p_upper)
    is_equivalent = bool(p_tost < 0.05)
    
    return {
        "mean_diff": float(mean_diff),
        "margin": float(margin),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost": float(p_tost),
        "is_equivalent": is_equivalent
    }

def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Applies Holm-Bonferroni step-down correction for multiple testing.
    Returns: list of (adjusted_p_value, is_significant)
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    adj_p = np.zeros(m)
    
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adj = (m - rank) * p
        running_max = max(running_max, adj)
        adj_p[idx] = min(1.0, running_max)
        
    results = [(float(adj_p[i]), bool(adj_p[i] < alpha)) for i in range(m)]
    return results

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    l1 = rng.standard_normal(1000) ** 2
    l2 = l1 + 0.1 * rng.standard_normal(1000)
    stat, pval = diebold_mariano_test(l1, l2)
    print(f"DM stat: {stat:.4f}, p-value: {pval:.4f}")
    mean_val, cil, ciu = block_bootstrap_ci(l1)
    print(f"Loss CI: {mean_val:.4f} [{cil:.4f}, {ciu:.4f}]")
    tost_res = tost_equivalence_test(l1, l2, margin=0.05)
    print("TOST res:", tost_res)

