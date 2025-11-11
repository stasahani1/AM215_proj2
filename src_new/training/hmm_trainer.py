"""
Hidden Markov Model trainer for discovering states in stock returns.
Uses Gaussian emissions for each state.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class HMMParameters:
    """Parameters for a Gaussian HMM."""
    n_states: int
    means: np.ndarray  # Shape: (n_states,)
    stds: np.ndarray   # Shape: (n_states,)
    transition_matrix: np.ndarray  # Shape: (n_states, n_states)
    initial_probs: np.ndarray  # Shape: (n_states,)
    

class GaussianHMM:
    """
    Gaussian Hidden Markov Model for stock returns.
    Each state has a Gaussian distribution N(μ, σ²).
    """
    
    def __init__(self, n_states: int = 3, random_state: int = 42):
        """
        Initialize HMM.
        
        Args:
            n_states: Number of hidden states
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Parameters to be learned
        self.params = None
        
    def _initialize_parameters(self, returns: np.ndarray) -> HMMParameters:
        """Initialize parameters randomly."""
        # Initialize means by dividing return range into n_states regions
        return_mean = returns.mean()
        return_std = returns.std()
        
        means = np.linspace(
            return_mean - 2*return_std,
            return_mean + 2*return_std,
            self.n_states
        )
        
        # Initialize stds as increasing (low, medium, high volatility)
        stds = np.linspace(return_std * 0.5, return_std * 2.0, self.n_states)
        
        # Initialize transition matrix (slight preference to stay in same state)
        trans = np.ones((self.n_states, self.n_states)) * 0.1
        trans += np.eye(self.n_states) * 0.7
        trans = trans / trans.sum(axis=1, keepdims=True)
        
        # Uniform initial probabilities
        initial = np.ones(self.n_states) / self.n_states
        
        return HMMParameters(
            n_states=self.n_states,
            means=means,
            stds=stds,
            transition_matrix=trans,
            initial_probs=initial
        )
    
    def _emission_probability(self, returns: np.ndarray, state: int) -> np.ndarray:
        """
        Compute P(observation | state).
        
        Args:
            returns: Array of returns
            state: State index
            
        Returns:
            Array of probabilities
        """
        mu = self.params.means[state]
        sigma = self.params.stds[state]
        return norm.pdf(returns, mu, sigma)
    
    def _forward(self, returns: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm.
        
        Returns:
            alpha: Forward probabilities (T, n_states)
            log_likelihood: Log likelihood of sequence
        """
        T = len(returns)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        for s in range(self.n_states):
            alpha[0, s] = (self.params.initial_probs[s] * 
                          self._emission_probability(returns[0:1], s)[0])
        
        # Normalization to prevent underflow
        alpha[0] = alpha[0] / (alpha[0].sum() + 1e-10)
        
        # Recursion
        for t in range(1, T):
            for s in range(self.n_states):
                trans_prob = alpha[t-1] @ self.params.transition_matrix[:, s]
                emission_prob = self._emission_probability(returns[t:t+1], s)[0]
                alpha[t, s] = trans_prob * emission_prob
            
            # Normalize
            alpha[t] = alpha[t] / (alpha[t].sum() + 1e-10)
        
        log_likelihood = np.log(alpha.sum(axis=1) + 1e-10).sum()
        
        return alpha, log_likelihood
    
    def _backward(self, returns: np.ndarray) -> np.ndarray:
        """
        Backward algorithm.
        
        Returns:
            beta: Backward probabilities (T, n_states)
        """
        T = len(returns)
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[T-1] = 1.0
        
        # Recursion
        for t in range(T-2, -1, -1):
            for s in range(self.n_states):
                for s_next in range(self.n_states):
                    emission_prob = self._emission_probability(returns[t+1:t+2], s_next)[0]
                    beta[t, s] += (self.params.transition_matrix[s, s_next] * 
                                  emission_prob * beta[t+1, s_next])
            
            # Normalize
            beta[t] = beta[t] / (beta[t].sum() + 1e-10)
        
        return beta
    
    def _baum_welch_step(self, returns: np.ndarray) -> float:
        """
        Single E-M step of Baum-Welch algorithm.
        
        Returns:
            Log likelihood of sequence
        """
        T = len(returns)
        
        # E-step: Compute forward and backward probabilities
        alpha, log_likelihood = self._forward(returns)
        beta = self._backward(returns)
        
        # Compute gamma (state occupation probabilities)
        gamma = alpha * beta
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)
        
        # Compute xi (transition probabilities)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission_prob = self._emission_probability(returns[t+1:t+2], j)[0]
                    xi[t, i, j] = (alpha[t, i] * 
                                  self.params.transition_matrix[i, j] * 
                                  emission_prob * 
                                  beta[t+1, j])
            xi[t] = xi[t] / (xi[t].sum() + 1e-10)
        
        # M-step: Update parameters
        # Update initial probabilities
        self.params.initial_probs = gamma[0]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = xi[:, i, j].sum()
                denominator = gamma[:-1, i].sum()
                self.params.transition_matrix[i, j] = numerator / (denominator + 1e-10)
        
        # Normalize rows
        self.params.transition_matrix = (self.params.transition_matrix / 
                                        self.params.transition_matrix.sum(axis=1, keepdims=True))
        
        # Update means and stds
        for s in range(self.n_states):
            weights = gamma[:, s]
            weight_sum = weights.sum() + 1e-10
            
            self.params.means[s] = (weights * returns).sum() / weight_sum
            self.params.stds[s] = np.sqrt((weights * (returns - self.params.means[s])**2).sum() / weight_sum)
            self.params.stds[s] = max(self.params.stds[s], 1e-6)  # Prevent collapse
        
        return log_likelihood
    
    def fit(self, returns: np.ndarray, max_iter: int = 100, tol: float = 1e-4) -> 'GaussianHMM':
        """
        Fit HMM to returns using Baum-Welch algorithm.
        
        Args:
            returns: 1D array of returns
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            
        Returns:
            self
        """
        # Initialize parameters
        self.params = self._initialize_parameters(returns)
        
        prev_ll = -np.inf
        for iteration in range(max_iter):
            log_likelihood = self._baum_welch_step(returns)
            
            if iteration % 10 == 0:
                print(f"    Iteration {iteration}: log-likelihood = {log_likelihood:.2f}")
            
            # Check convergence
            if abs(log_likelihood - prev_ll) < tol:
                print(f"    Converged at iteration {iteration}")
                break
                
            prev_ll = log_likelihood
        
        # Sort states by volatility (std)
        order = np.argsort(self.params.stds)
        self.params.means = self.params.means[order]
        self.params.stds = self.params.stds[order]
        self.params.transition_matrix = self.params.transition_matrix[order][:, order]
        self.params.initial_probs = self.params.initial_probs[order]
        
        return self
    
    def viterbi(self, returns: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm to find most likely state sequence.
        
        Args:
            returns: 1D array of returns
            
        Returns:
            Array of state indices (0 to n_states-1)
        """
        T = len(returns)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        for s in range(self.n_states):
            delta[0, s] = (np.log(self.params.initial_probs[s] + 1e-10) + 
                          np.log(self._emission_probability(returns[0:1], s)[0] + 1e-10))
        
        # Recursion
        for t in range(1, T):
            for s in range(self.n_states):
                trans_probs = delta[t-1] + np.log(self.params.transition_matrix[:, s] + 1e-10)
                psi[t, s] = np.argmax(trans_probs)
                delta[t, s] = (trans_probs[psi[t, s]] + 
                              np.log(self._emission_probability(returns[t:t+1], s)[0] + 1e-10))
        
        # Backtracking
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def get_state_labels(self) -> Dict[int, str]:
        """Get interpretable labels for states based on volatility."""
        if self.params is None:
            return {}
        
        # States are sorted by std, so:
        # 0 = lowest volatility (Recovered)
        # 1 = medium volatility (Susceptible)
        # 2 = highest volatility (Infected)
        
        if self.n_states == 3:
            return {0: 'R', 1: 'S', 2: 'I'}
        else:
            return {i: f'State_{i}' for i in range(self.n_states)}


if __name__ == '__main__':
    # Test HMM
    np.random.seed(42)
    
    # Generate synthetic data with 3 regimes
    T = 500
    true_states = np.random.choice([0, 1, 2], size=T, p=[0.3, 0.5, 0.2])
    
    returns = np.zeros(T)
    for t in range(T):
        if true_states[t] == 0:  # Low vol
            returns[t] = np.random.normal(0.001, 0.01)
        elif true_states[t] == 1:  # Medium vol
            returns[t] = np.random.normal(0.0, 0.02)
        else:  # High vol
            returns[t] = np.random.normal(-0.002, 0.04)
    
    # Fit HMM
    print("Fitting HMM...")
    hmm = GaussianHMM(n_states=3)
    hmm.fit(returns, max_iter=50)
    
    # Infer states
    inferred_states = hmm.viterbi(returns)
    
    print("\nLearned parameters:")
    print(f"Means: {hmm.params.means}")
    print(f"Stds:  {hmm.params.stds}")
    print(f"\nState labels: {hmm.get_state_labels()}")
    print(f"Accuracy: {(inferred_states == true_states).mean()*100:.1f}%")

