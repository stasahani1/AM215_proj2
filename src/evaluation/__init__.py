"""
Evaluation Module - Model Performance Assessment.

This module provides comprehensive evaluation tools:

- Backtester: Run models on test chunks and collect results
- ModelEvaluator: Compute metrics across multiple dimensions
  * Price accuracy (MSE, MAE, MAPE, R²)
  * Return accuracy
  * Volatility matching
  * Correlation structure matching
  * Distribution similarity (Wasserstein, JS divergence)
  * State accuracy (for state-based models)

The evaluation framework ensures fair comparison by testing all models
on identical out-of-sample data with multiple Monte Carlo simulations.
"""

from .backtester import Backtester
from .metrics import ModelEvaluator

__all__ = [
    'Backtester',
    'ModelEvaluator'
]
