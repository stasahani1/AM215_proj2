"""
Baseline Model 3: Correlated Random Walk.

This module implements a random walk model with correlated returns across stocks.
It captures cross-stock co-movement through a multivariate normal distribution
but does not include regime switching.

The model generates correlated returns from:
    r(t) ~ MVN(μ, Σ)
where μ is the mean vector and Σ is the covariance matrix learned from data.
This allows the model to capture correlations during crisis periods without
explicit state dynamics.
"""

import numpy as np
import pandas as pd
from typing import List

from models.base_model import BaseModel, SimulationResult


class CorrelatedRandomWalkModel(BaseModel):
    """
    Random walk with correlated returns across stocks.
    No state switching, just correlated noise.
    """
    
    def __init__(self, tickers: List[str]):
        super().__init__(name="Correlated Random Walk")
        self.tickers = tickers
        self.n_stocks = len(tickers)
        
        # Parameters to learn
        self.means = None  # (n_stocks,)
        self.cov_matrix = None  # (n_stocks, n_stocks)
    
    def fit(self, chunks: List, **kwargs):
        """
        Estimate mean vector and covariance matrix from training data.
        
        Args:
            chunks: List of DataChunk objects
        """
        print(f"\nTraining {self.name}...")
        
        # Concatenate all returns
        all_returns = pd.concat([chunk.returns for chunk in chunks], axis=0)
        returns_data = all_returns[self.tickers]
        
        # Compute mean and covariance
        self.means = returns_data.mean().values
        self.cov_matrix = returns_data.cov().values
        
        # Compute average correlation
        corr_matrix = returns_data.corr().values
        n = len(corr_matrix)
        avg_corr = (corr_matrix.sum() - n) / (n * (n - 1))
        
        print(f"  Learned parameters for {self.n_stocks} stocks")
        print(f"  Mean returns: {self.means.mean():.5f}")
        print(f"  Average correlation: {avg_corr:.3f}")
        print(f"  Average volatility: {np.sqrt(np.diag(self.cov_matrix)).mean():.5f}")
        
        self.is_trained = True
    
    def simulate(self, 
                initial_prices: np.ndarray,
                n_steps: int,
                dates: List[pd.Timestamp] = None,
                **kwargs) -> SimulationResult:
        """
        Simulate with correlated returns.
        
        Args:
            initial_prices: Starting prices
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
        
        # Generate correlated returns
        for t in range(1, n_steps):
            # Draw from multivariate normal
            returns[t] = np.random.multivariate_normal(self.means, self.cov_matrix)
            prices[t] = prices[t-1] * np.exp(returns[t])
        
        return SimulationResult(
            prices=prices,
            returns=returns,
            states=None,
            dates=dates
        )
    
    def get_params(self):
        """Get model parameters."""
        params = super().get_params()
        if self.is_trained:
            params['means'] = self.means
            params['cov_matrix'] = self.cov_matrix
            # Compute average correlation
            corr_matrix = self.cov_matrix / np.outer(
                np.sqrt(np.diag(self.cov_matrix)),
                np.sqrt(np.diag(self.cov_matrix))
            )
            n = len(corr_matrix)
            params['avg_correlation'] = (corr_matrix.sum() - n) / (n * (n - 1))
        return params

