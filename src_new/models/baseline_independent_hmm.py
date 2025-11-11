"""
Baseline 2: Independent HMM
Each stock has 3 states, but transitions are independent (no contagion).
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import sys
sys.path.append('..')
from models.base_model import StateBasedModel, SimulationResult
from training.state_inference import MultiStockStateInference


class IndependentHMMModel(StateBasedModel):
    """
    Multi-state model where each stock has independent state transitions.
    No contagion effect - states don't depend on neighbors.
    """
    
    def __init__(self, tickers: List[str]):
        super().__init__(name="Independent HMM", n_states=3)
        self.tickers = tickers
        self.n_stocks = len(tickers)
        
        # Components
        self.state_inference = None
        self.transition_matrices = {}  # Independent transition matrix per stock
        
        self.state_names = {0: 'R', 1: 'S', 2: 'I'}
    
    def fit(self, chunks: List, **kwargs):
        """
        Learn state parameters and independent transition probabilities.
        
        Args:
            chunks: List of DataChunk objects
        """
        print(f"\nTraining {self.name}...")
        
        # Step 1: Learn state parameters (same as SIR model)
        self.state_inference = MultiStockStateInference(self.tickers, n_states=3)
        self.state_inference.fit(chunks, max_iter=30)
        
        # Step 2: Learn independent transition matrices
        print("\nLearning independent transition matrices...")
        transition_counts = self.state_inference.get_transition_counts(chunks)
        
        for ticker in self.tickers:
            if ticker in transition_counts:
                counts = transition_counts[ticker]
                # Convert counts to probabilities
                trans_matrix = counts / (counts.sum(axis=1, keepdims=True) + 1e-10)
                self.transition_matrices[ticker] = trans_matrix
            else:
                # Default transition matrix
                self.transition_matrices[ticker] = np.eye(3) * 0.9 + 0.1 / 3
        
        # Print average transition probabilities
        avg_trans = np.mean([self.transition_matrices[t] for t in self.tickers], axis=0)
        print("\nAverage transition matrix:")
        print("       R      S      I")
        for i, from_state in enumerate(['R', 'S', 'I']):
            print(f"  {from_state}: ", end='')
            for j in range(3):
                print(f"{avg_trans[i,j]:.3f}  ", end='')
            print()
        
        self.is_trained = True
    
    def simulate(self, 
                initial_prices: np.ndarray,
                n_steps: int,
                dates: List[pd.Timestamp] = None,
                initial_states: np.ndarray = None,
                **kwargs) -> SimulationResult:
        """
        Simulate with independent state transitions.
        
        Args:
            initial_prices: Starting prices
            n_steps: Number of steps
            dates: Optional dates
            initial_states: Initial states (if None, assume all Susceptible)
            
        Returns:
            SimulationResult
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        prices = np.zeros((n_steps, self.n_stocks))
        returns = np.zeros((n_steps, self.n_stocks))
        states = np.zeros((n_steps, self.n_stocks), dtype=int)
        
        prices[0] = initial_prices
        
        if initial_states is None:
            states[0] = 1  # All start as Susceptible
        else:
            states[0] = initial_states
        
        # Simulate forward
        for t in range(1, n_steps):
            for i, ticker in enumerate(self.tickers):
                current_state = states[t-1, i]
                
                # Get transition probabilities (INDEPENDENT - no neighbor effects)
                trans_probs = self.transition_matrices[ticker][current_state]
                
                # Sample next state
                next_state = np.random.choice([0, 1, 2], p=trans_probs)
                states[t, i] = next_state
                
                # Generate return based on state
                state_params = self.state_inference.state_params[ticker][next_state]
                mu = state_params['mu']
                sigma = state_params['sigma']
                
                ret = np.random.normal(mu, sigma)
                returns[t, i] = ret
                prices[t, i] = prices[t-1, i] * np.exp(ret)
        
        return SimulationResult(
            prices=prices,
            returns=returns,
            states=states,
            dates=dates
        )
    
    def infer_states(self, prices: pd.DataFrame) -> np.ndarray:
        """Infer states from observed prices."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        returns = np.log(prices / prices.shift(1)).dropna()
        states_df = self.state_inference.infer_states(returns)
        states = states_df.values
        
        # Prepend first row with default state (Susceptible = 1) to match prices length
        if len(states) < len(prices):
            first_row = np.ones((1, states.shape[1]), dtype=int)
            states = np.vstack([first_row, states])
        
        return states
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        if self.is_trained:
            params['n_stocks'] = self.n_stocks
            params['avg_transition_matrix'] = np.mean(
                [self.transition_matrices[t] for t in self.tickers], axis=0
            )
        return params

