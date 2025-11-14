"""
Base model interface for all price prediction models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Result from a single simulation run."""
    prices: np.ndarray  # Shape: (n_steps, n_stocks)
    returns: np.ndarray  # Shape: (n_steps, n_stocks)
    states: np.ndarray = None  # Shape: (n_steps, n_stocks) - for state-based models
    dates: List[pd.Timestamp] = None


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str):
        """
        Initialize model.
        
        Args:
            name: Human-readable name for the model
        """
        self.name = name
        self.is_trained = False
        self.tickers = None
        
    @abstractmethod
    def fit(self, chunks: List, **kwargs):
        """
        Train model on data chunks.
        
        Args:
            chunks: List of DataChunk objects for training
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def simulate(self, 
                initial_prices: np.ndarray,
                n_steps: int,
                dates: List[pd.Timestamp] = None,
                **kwargs) -> SimulationResult:
        """
        Simulate price trajectories forward.
        
        Args:
            initial_prices: Starting prices for each stock (n_stocks,)
            n_steps: Number of time steps to simulate
            dates: Optional dates for the simulation
            **kwargs: Additional model-specific parameters
            
        Returns:
            SimulationResult with simulated prices and other data
        """
        pass
    
    def simulate_multiple(self,
                         initial_prices: np.ndarray,
                         n_steps: int,
                         n_simulations: int = 1000,
                         dates: List[pd.Timestamp] = None,
                         **kwargs) -> List[SimulationResult]:
        """
        Run multiple Monte Carlo simulations.
        
        Args:
            initial_prices: Starting prices
            n_steps: Number of steps per simulation
            n_simulations: Number of simulations to run
            dates: Optional dates
            **kwargs: Model-specific parameters
            
        Returns:
            List of SimulationResult objects
        """
        results = []
        for i in range(n_simulations):
            if i % 100 == 0 and i > 0:
                print(f"  Simulations: {i}/{n_simulations}", end='\r')
            result = self.simulate(initial_prices, n_steps, dates, **kwargs)
            results.append(result)
        print(f"  Simulations: {n_simulations}/{n_simulations} ✓")
        return results
    
    def predict_mean(self,
                    initial_prices: np.ndarray,
                    n_steps: int,
                    n_simulations: int = 1000,
                    **kwargs) -> np.ndarray:
        """
        Get mean prediction across multiple simulations.
        
        Returns:
            Array of shape (n_steps, n_stocks) with mean prices
        """
        results = self.simulate_multiple(initial_prices, n_steps, n_simulations, **kwargs)
        all_prices = np.array([r.prices for r in results])  # (n_sims, n_steps, n_stocks)
        return all_prices.mean(axis=0)
    
    def get_params(self) -> Dict:
        """Get model parameters as dictionary."""
        return {'name': self.name, 'is_trained': self.is_trained}
    
    def __str__(self):
        return f"{self.name} (trained={self.is_trained})"
    
    def __repr__(self):
        return self.__str__()


class StateBasedModel(BaseModel):
    """Base class for models with hidden states."""
    
    def __init__(self, name: str, n_states: int = 3):
        super().__init__(name)
        self.n_states = n_states
        self.state_names = None
        
    @abstractmethod
    def infer_states(self, prices: pd.DataFrame) -> np.ndarray:
        """
        Infer hidden states from observed prices.
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            Array of shape (n_steps, n_stocks) with state indices
        """
        pass
    
    def get_state_statistics(self, states: np.ndarray) -> Dict:
        """Compute statistics about state occupancy."""
        n_steps, n_stocks = states.shape
        stats = {}
        
        for state in range(self.n_states):
            mask = (states == state)
            stats[f'state_{state}_pct'] = mask.sum() / (n_steps * n_stocks) * 100
            
        return stats

