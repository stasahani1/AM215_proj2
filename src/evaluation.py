"""Evaluation metrics for infection prediction models."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


@dataclass
class EvalResult:
    """Container for evaluation metrics."""
    auc: float
    ap: float
    brier: float

    def __str__(self):
        return f"AUC={self.auc:.3f}, AP={self.ap:.3f}, Brier={self.brier:.4f}"


def evaluate_predictions(y_true: pd.Series, y_hat: np.ndarray) -> EvalResult:
    """
    Evaluate infection predictions using multiple metrics.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels (0 or 1)
    y_hat : np.ndarray
        Predicted probabilities

    Returns
    -------
    EvalResult
        Container with AUC, Average Precision, and Brier score
    """
    # Align and filter out NaN predictions
    mask = np.isfinite(y_hat)
    y_t = y_true.values[mask]
    y_p = y_hat[mask]

    # Compute metrics (handle edge case where all labels are same class)
    auc = roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan
    ap = average_precision_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan
    brier = brier_score_loss(y_t, y_p)

    return EvalResult(auc=auc, ap=ap, brier=brier)
