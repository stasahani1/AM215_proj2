"""
Training Module - State Learning and Contagion Parameter Estimation.

This module provides algorithms for training the SIR model:

- GaussianHMM: Baum-Welch algorithm for state parameter learning
- MultiStockStateInference: Coordinate HMM training across multiple stocks
- ContagionLearner: Maximum likelihood estimation of β, γ, α parameters

The training pipeline:
1. Learn state parameters (μ, σ) for each stock using HMM
2. Infer historical state sequences using Viterbi algorithm
3. Estimate contagion parameters from state transitions and network structure
"""

from .hmm_trainer import GaussianHMM, HMMParameters
from .state_inference import MultiStockStateInference
from .contagion_learner import ContagionLearner

__all__ = [
    'GaussianHMM',
    'HMMParameters',
    'MultiStockStateInference',
    'ContagionLearner'
]
