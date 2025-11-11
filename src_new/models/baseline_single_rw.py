"""
Baseline 1: Single Random Walk (Geometric Brownian Motion)
Each stock has one set of (μ, σ) parameters.
"""

import numpy as np
import pandas as pd
from typing import List
import sys
sys.path.append('..')
from models.base_model import BaseModel, SimulationResult


class SingleRandomWalkModel(BaseModel):
    """
    Simple single-state random walk model.
    Each stock has constant drift and volatility.
    """
    
    def __init__(self, tickers: List[str]):
        super().__init__(name="Single Random Walk")
        self.tickers = tickers
        self.n_stocks = len(tickers)
        
        # Parameters to learn
        self.means = None  # (n_stocks,)
        self.stds = None   # (n_stocks,)
    
    def fit(self, chunks: List, **kwargs):
        """
        Estimate mean and std for each stock from training data.
        
        Args:
            chunks: List of DataChunk objects
        """
        print(f"\nTraining {self.name}...")
        
        # Concatenate all returns
        all_returns = pd.concat([chunk.returns for chunk in chunks], axis=0)
        
        # Compute mean and std for each stock
        self.means = all_returns[self.tickers].mean().values
        self.stds = all_returns[self.tickers].std().values
        
        print(f"  Learned parameters for {self.n_stocks} stocks")
        print(f"  Mean returns: {self.means.mean():.5f} ± {self.means.std():.5f}")
        print(f"  Mean volatility: {self.stds.mean():.5f} ± {self.stds.std():.5f}")
        
        self.is_trained = True
    
    def simulate(self, 
                initial_prices: np.ndarray,
                n_steps: int,
                dates: List[pd.Timestamp] = None,
                **kwargs) -> SimulationResult:
        """
        Simulate prices using simple random walk.
        
        Args:
            initial_prices: Starting prices (n_stocks,)
            n_steps: Number of steps
            dates: Optional dates
            
        Returns:
            SimulationResult
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        prices = np.zeros((n_steps, self.n_stocks))
        returns = np.zeros((n_steps, self.n_stocks))
        
        prices[0] = initial_prices
        
        for t in range(1, n_steps):
            # Generate returns from learned distribution
            returns[t] = np.random.normal(self.means, self.stds)
            prices[t] = prices[t-1] * np.exp(returns[t])
        
        return SimulationResult(
            prices=prices,
            returns=returns,
            states=None,  # No states in this model
            dates=dates
        )
    
    def get_params(self):
        """Get model parameters."""
        params = super().get_params()
        if self.is_trained:
            params['means'] = self.means
            params['stds'] = self.stds
        return params

