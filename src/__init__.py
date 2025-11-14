"""
SIR Contagion Model for Stock Price Prediction.

This package implements a multi-state regime-switching model with epidemic-inspired
contagion dynamics for predicting stock prices during crisis periods.

Main Components:
- data: Stock price loading and time chunk selection
- models: SIR contagion model and baseline models
- training: HMM training, state inference, contagion parameter learning
- evaluation: Backtesting framework and comprehensive metrics
- visualization: Publication-ready plotting functions

For usage examples, see the scripts/ directory and README.md.
"""

__version__ = '1.0.0'
__author__ = 'AM215 Project Team'