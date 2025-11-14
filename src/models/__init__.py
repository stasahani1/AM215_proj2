"""
Models Module - Price Prediction Models.

This module implements the SIR contagion model and three baseline models:

Main Model:
- SIRContagionModel: Multi-state random walk with epidemic contagion dynamics

Baselines:
- SingleRandomWalkModel: Simple GBM with constant μ, σ
- IndependentHMMModel: Multi-state without contagion
- CorrelatedRandomWalkModel: Correlated returns, no states

All models inherit from BaseModel and implement the same interface for
training (fit) and prediction (simulate).
"""

from .base_model import BaseModel, StateBasedModel, SimulationResult
from .sir_contagion_model import SIRContagionModel
from .baseline_single_rw import SingleRandomWalkModel
from .baseline_independent_hmm import IndependentHMMModel
from .baseline_correlated_rw import CorrelatedRandomWalkModel

__all__ = [
    'BaseModel',
    'StateBasedModel',
    'SimulationResult',
    'SIRContagionModel',
    'SingleRandomWalkModel',
    'IndependentHMMModel',
    'CorrelatedRandomWalkModel'
]
