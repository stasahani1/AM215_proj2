"""
State inference module for learning states across multiple stocks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from .hmm_trainer import GaussianHMM, HMMParameters


class MultiStockStateInference:
    """Infer states for multiple stocks simultaneously."""
    
    def __init__(self, tickers: List[str], n_states: int = 3):
        """
        Initialize multi-stock state inference.
        
        Args:
            tickers: List of stock tickers
            n_states: Number of states per stock
        """
        self.tickers = tickers
        self.n_states = n_states
        self.hmms = {}  # Dictionary of HMM per stock
        self.state_params = {}  # Dictionary of (mu, sigma) per stock per state
        
    def fit(self, chunks: List, max_iter: int = 100, min_iter: int = 10) -> 'MultiStockStateInference':
        """
        Fit HMM for each stock independently on training chunks.
        
        Args:
            chunks: List of DataChunk objects
            max_iter: Maximum iterations for HMM training
            min_iter: Minimum iterations before convergence
            
        Returns:
            self
        """
        print(f"\nFitting HMMs for {len(self.tickers)} stocks...")
        
        # Concatenate all training returns for each stock
        all_returns = {ticker: [] for ticker in self.tickers}
        
        for chunk in chunks:
            for ticker in self.tickers:
                if ticker in chunk.returns.columns:
                    returns = chunk.returns[ticker].values
                    all_returns[ticker].extend(returns)
        
        # Fit HMM for each stock
        for ticker in self.tickers:
            print(f"\n  Training {ticker}...")
            returns = np.array(all_returns[ticker])
            
            hmm = GaussianHMM(n_states=self.n_states, random_state=42)
            hmm.fit(returns, max_iter=max_iter, min_iter=min_iter)
            
            self.hmms[ticker] = hmm
            
            # Store state parameters
            self.state_params[ticker] = {}
            for state in range(self.n_states):
                self.state_params[ticker][state] = {
                    'mu': hmm.params.means[state],
                    'sigma': hmm.params.stds[state]
                }
            
            # Print learned parameters
            labels = hmm.get_state_labels()
            print(f"    Learned states:")
            for state in range(self.n_states):
                label = labels.get(state, f'State_{state}')
                mu = hmm.params.means[state]
                sigma = hmm.params.stds[state]
                print(f"      {label}: μ={mu:.5f}, σ={sigma:.5f}")
        
        return self
    
    def infer_states(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Infer states for all stocks given returns.
        
        Args:
            returns: DataFrame of returns (dates x tickers)
            
        Returns:
            DataFrame of states (dates x tickers), with state indices 0, 1, 2
        """
        states = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=int)
        
        for ticker in returns.columns:
            if ticker in self.hmms:
                ticker_returns = returns[ticker].dropna().values
                if len(ticker_returns) == 0:
                    # If no valid returns, default to Susceptible
                    states[ticker] = 1
                else:
                    ticker_states = self.hmms[ticker].viterbi(ticker_returns)
                    # If viterbi returns fewer states than expected, pad with Susceptible
                    if len(ticker_states) < len(returns):
                        full_states = np.ones(len(returns), dtype=int)
                        full_states[-len(ticker_states):] = ticker_states
                        states[ticker] = full_states
                    else:
                        states[ticker] = ticker_states
            else:
                # Default to state 1 (Susceptible) if no HMM
                states[ticker] = 1
        
        return states
    
    def infer_states_for_chunks(self, chunks: List) -> List[pd.DataFrame]:
        """
        Infer states for each chunk.
        
        Returns:
            List of state DataFrames, one per chunk
        """
        chunk_states = []
        
        for chunk in chunks:
            states = self.infer_states(chunk.returns)
            chunk_states.append(states)
        
        return chunk_states
    
    def get_state_summary(self, chunks: List) -> pd.DataFrame:
        """
        Get summary of state occupancy across all chunks.
        
        Returns:
            DataFrame with state percentages per stock
        """
        chunk_states = self.infer_states_for_chunks(chunks)
        
        # Concatenate all states
        all_states = pd.concat(chunk_states, axis=0)
        
        # Compute percentage in each state
        summary = []
        for ticker in self.tickers:
            if ticker in all_states.columns:
                states = all_states[ticker].values
                total = len(states)
                
                summary.append({
                    'ticker': ticker,
                    'pct_R': (states == 0).sum() / total * 100,
                    'pct_S': (states == 1).sum() / total * 100,
                    'pct_I': (states == 2).sum() / total * 100
                })
        
        return pd.DataFrame(summary)
    
    def get_transition_counts(self, chunks: List) -> Dict[str, np.ndarray]:
        """
        Count transitions between states for each stock.
        
        Returns:
            Dictionary of transition count matrices per ticker
        """
        chunk_states = self.infer_states_for_chunks(chunks)
        all_states = pd.concat(chunk_states, axis=0)
        
        transition_counts = {}
        
        for ticker in self.tickers:
            if ticker not in all_states.columns:
                continue
                
            states = all_states[ticker].values
            counts = np.zeros((self.n_states, self.n_states))
            
            for i in range(len(states) - 1):
                from_state = states[i]
                to_state = states[i+1]
                counts[from_state, to_state] += 1
            
            transition_counts[ticker] = counts
        
        return transition_counts

