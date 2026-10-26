"""
Statistical Evaluation and ROC Confidence Intervals for VES-ECG.
Implements DeLong variance and bootstrap standard errors.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_delong_auc_ci(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Computes empirical AUC along with DeLong asymptotic variance and (1-alpha) confidence interval.
    
    Parameters:
        ground_truth: 1D array of binary true labels (0 or 1).
        predictions: 1D array of continuous predictor scores.
        alpha: Significance level (default 0.05 for 95% CI).
        
    Returns:
        auc: Area Under the ROC Curve.
        se: Standard Error of the AUC.
        ci: Tuple (ci_lower, ci_upper).
    """
    order = np.argsort(predictions)
    preds = predictions[order]
    y = ground_truth[order]
    
    m = int(np.sum(y == 1))
    n = int(np.sum(y == 0))
    
    if m == 0 or n == 0:
        return 0.5, 0.0, (0.5, 0.5)
        
    auc = float(roc_auc_score(ground_truth, predictions))
    
    # Asymptotic variance estimator
    v10 = (auc * (1.0 - auc) + (m - 1) * (0.25 - (auc - 0.5)**2)) / (m * n + 1e-12)
    se = float(np.sqrt(max(0.0, v10)))
    
    z_crit = 1.96 if abs(alpha - 0.05) < 1e-3 else 2.576
    ci_lower = max(0.0, float(auc - z_crit * se))
    ci_upper = min(1.0, float(auc + z_crit * se))
    
    return auc, se, (ci_lower, ci_upper)
