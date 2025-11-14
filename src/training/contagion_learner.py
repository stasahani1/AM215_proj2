"""
Module for learning contagion dynamics between stocks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


class ContagionLearner:
    """Learn contagion parameters (β, γ, α) from state sequences."""
    
    def __init__(self, tickers: List[str], network: np.ndarray = None):
        """
        Initialize contagion learner.
        
        Args:
            tickers: List of stock tickers
            network: Adjacency matrix (n_stocks x n_stocks) defining connections
                    If None, uses fully connected network
        """
        self.tickers = tickers
        self.n_stocks = len(tickers)
        
        if network is None:
            # Fully connected (all stocks can infect all others)
            self.network = np.ones((self.n_stocks, self.n_stocks))
            np.fill_diagonal(self.network, 0)  # Stock doesn't infect itself
        else:
            self.network = network
        
        # Parameters to learn
        self.beta = None   # Contagion rate (S -> I)
        self.gamma = None  # Recovery rate (I -> R)
        self.alpha = None  # Re-susceptibility rate (R -> S)
        
        # Baseline transition rates (without contagion)
        self.baseline_SI = None
        self.baseline_IR = None
        self.baseline_RS = None
    
    def build_network_from_correlation(self, returns: pd.DataFrame, threshold: float = 0.6) -> np.ndarray:
        """
        Build network from correlation structure.
        
        Args:
            returns: DataFrame of returns
            threshold: Correlation threshold for connection
            
        Returns:
            Adjacency matrix
        """
        corr = returns[self.tickers].corr()
        network = (corr.values > threshold).astype(float)
        np.fill_diagonal(network, 0)
        
        print(f"Network connectivity: {network.sum() / (self.n_stocks * (self.n_stocks - 1)):.1%}")
        return network
    
    def _count_transitions(self, states_list: List[pd.DataFrame]) -> Dict:
        """
        Count state transitions and neighbor effects.
        
        Returns:
            Dictionary with transition statistics
        """
        # Track: for each S->I transition, how many neighbors were infected?
        SI_transitions = []  # (stock_idx, num_infected_neighbors, did_infect)
        IR_transitions = []  # (from I to R)
        RS_transitions = []  # (from R to S)
        
        for states_df in states_list:
            states = states_df.values  # (T, n_stocks)
            T = len(states)
            
            for t in range(T - 1):
                for i, ticker in enumerate(self.tickers):
                    if ticker not in states_df.columns:
                        continue
                    
                    current_state = states[t, i]
                    next_state = states[t+1, i]
                    
                    # Get neighbors
                    neighbors = self.network[i] > 0
                    neighbor_states = states[t, neighbors]
                    num_infected = (neighbor_states == 2).sum()  # State 2 = Infected
                    num_neighbors = neighbors.sum()
                    
                    if num_neighbors == 0:
                        continue
                    
                    infected_fraction = num_infected / num_neighbors
                    
                    # Track S -> I transitions
                    if current_state == 1:  # Susceptible
                        did_infect = (next_state == 2)
                        SI_transitions.append({
                            'stock': i,
                            'infected_fraction': infected_fraction,
                            'did_infect': did_infect
                        })
                    
                    # Track I -> R transitions
                    elif current_state == 2:  # Infected
                        did_recover = (next_state == 0)
                        IR_transitions.append({
                            'stock': i,
                            'did_recover': did_recover
                        })
                    
                    # Track R -> S transitions
                    elif current_state == 0:  # Recovered
                        became_susceptible = (next_state == 1)
                        RS_transitions.append({
                            'stock': i,
                            'became_susceptible': became_susceptible
                        })
        
        return {
            'SI': pd.DataFrame(SI_transitions),
            'IR': pd.DataFrame(IR_transitions),
            'RS': pd.DataFrame(RS_transitions)
        }
    
    def fit(self, states_list: List[pd.DataFrame]) -> 'ContagionLearner':
        """
        Learn contagion parameters from state sequences.
        
        Args:
            states_list: List of state DataFrames from training chunks
            
        Returns:
            self
        """
        print("\nLearning contagion parameters...")
        
        # Count transitions
        transitions = self._count_transitions(states_list)
        
        # Learn S -> I dynamics (contagion effect)
        if len(transitions['SI']) > 0:
            SI_data = transitions['SI']
            
            # Baseline infection rate (when no neighbors infected)
            baseline_mask = SI_data['infected_fraction'] == 0
            if baseline_mask.sum() > 0:
                self.baseline_SI = SI_data[baseline_mask]['did_infect'].mean()
            else:
                self.baseline_SI = 0.05  # Default
            
            # Learn β using logistic regression
            X = SI_data[['infected_fraction']].values
            y = SI_data['did_infect'].values
            
            if y.sum() > 0 and (~y).sum() > 0:  # Need both classes
                lr = LogisticRegression(fit_intercept=True)
                lr.fit(X, y)
                self.beta = lr.coef_[0, 0]
                
                # Compute effective contagion rate
                print(f"  Baseline S->I rate: {self.baseline_SI:.4f}")
                print(f"  Contagion coefficient β: {self.beta:.4f}")
                print(f"  S->I transitions analyzed: {len(SI_data)}")
            else:
                self.beta = 0.0
                print(f"  Insufficient variation in S->I transitions")
        else:
            self.baseline_SI = 0.05
            self.beta = 0.0
            print("  No S->I transitions found")
        
        # Learn I -> R dynamics (recovery rate)
        if len(transitions['IR']) > 0:
            IR_data = transitions['IR']
            self.gamma = IR_data['did_recover'].mean()
            print(f"  Recovery rate γ: {self.gamma:.4f}")
            print(f"  I->R transitions analyzed: {len(IR_data)}")
        else:
            self.gamma = 0.1
            print("  No I->R transitions found, using default γ=0.1")
        
        # Learn R -> S dynamics (re-susceptibility rate)
        if len(transitions['RS']) > 0:
            RS_data = transitions['RS']
            self.alpha = RS_data['became_susceptible'].mean()
            print(f"  Re-susceptibility rate α: {self.alpha:.4f}")
            print(f"  R->S transitions analyzed: {len(RS_data)}")
        else:
            self.alpha = 0.05
            print("  No R->S transitions found, using default α=0.05")
        
        return self
    
    def compute_infection_probability(self, 
                                     current_state: int,
                                     neighbor_states: np.ndarray,
                                     dt: float = 1.0) -> Dict[int, float]:
        """
        Compute transition probabilities given current state and neighbors.
        
        Args:
            current_state: Current state (0=R, 1=S, 2=I)
            neighbor_states: Array of neighbor states
            dt: Time step (default 1 day)
            
        Returns:
            Dictionary with probabilities for each next state
        """
        if len(neighbor_states) > 0:
            infected_fraction = (neighbor_states == 2).sum() / len(neighbor_states)
        else:
            infected_fraction = 0
        
        if current_state == 1:  # Susceptible
            # P(S->I) depends on infected neighbors
            p_SI = self.baseline_SI + self.beta * infected_fraction
            p_SI = min(max(p_SI, 0.0), 1.0)  # Clip to [0, 1]
            p_SI = 1 - np.exp(-p_SI * dt)
            
            return {
                0: 0.0,
                1: 1 - p_SI,
                2: p_SI
            }
        
        elif current_state == 2:  # Infected
            # Recovery is independent of neighbors
            p_IR = 1 - np.exp(-self.gamma * dt)
            
            return {
                0: p_IR,
                1: 0.0,
                2: 1 - p_IR
            }
        
        elif current_state == 0:  # Recovered
            # Re-susceptibility
            p_RS = 1 - np.exp(-self.alpha * dt)
            
            return {
                0: 1 - p_RS,
                1: p_RS,
                2: 0.0
            }
        
        return {0: 0.0, 1: 0.0, 2: 0.0}
    
    def get_params(self) -> Dict:
        """Get learned parameters."""
        return {
            'beta': self.beta,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'baseline_SI': self.baseline_SI,
            'network_density': self.network.sum() / (self.n_stocks * (self.n_stocks - 1))
        }


if __name__ == '__main__':
    # Test contagion learner
    np.random.seed(42)
    
    tickers = ['A', 'B', 'C', 'D', 'E']
    
    # Create synthetic state sequences with contagion
    T = 200
    n_stocks = len(tickers)
    states = np.ones((T, n_stocks), dtype=int)  # Start all susceptible
    
    # Simulate contagion
    for t in range(1, T):
        states[t] = states[t-1].copy()
        
        for i in range(n_stocks):
            # Count infected neighbors
            infected_count = (states[t-1] == 2).sum()
            
            if states[t-1, i] == 1:  # Susceptible
                # Higher infection prob with more infected
                p_infect = 0.05 + 0.3 * (infected_count / n_stocks)
                if np.random.rand() < p_infect:
                    states[t, i] = 2
            
            elif states[t-1, i] == 2:  # Infected
                if np.random.rand() < 0.2:
                    states[t, i] = 0
            
            elif states[t-1, i] == 0:  # Recovered
                if np.random.rand() < 0.1:
                    states[t, i] = 1
    
    # Convert to DataFrame
    states_df = pd.DataFrame(states, columns=tickers)
    
    # Learn parameters
    learner = ContagionLearner(tickers)
    learner.fit([states_df])
    
    print(f"\nLearned parameters:")
    print(learner.get_params())

