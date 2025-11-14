"""
Visualization Module - Publication-Ready Plotting Functions.

This module provides visualization tools for model evaluation:

- plot_price_trajectories: Actual vs simulated price paths
- plot_state_heatmap: State evolution over time
- plot_return_distributions: Histogram comparison
- plot_metric_comparison: Bar charts of model performance
- plot_correlation_matrices: Cross-stock correlation structure

All functions generate publication-ready figures with proper styling,
labels, legends, and can save to file.
"""

from .plots import (
    plot_price_trajectories,
    plot_state_heatmap,
    plot_return_distributions,
    plot_metric_comparison,
    plot_correlation_matrices
)

__all__ = [
    'plot_price_trajectories',
    'plot_state_heatmap',
    'plot_return_distributions',
    'plot_metric_comparison',
    'plot_correlation_matrices'
]
