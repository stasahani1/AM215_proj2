"""
SIR Contagion Model for multi-stock price dynamics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import sys
sys.path.append('..')
from models.base_model import StateBasedModel, SimulationResult
from training.state_inference import MultiStockStateInference
from training.contagion_learner import ContagionLearner


class SIRContagionModel(StateBasedModel):
    """
    Multi-state random walk with SIR contagion dynamics.
    
    Each stock has 3 states (S, I, R) with different random walk parameters.
    State transitions depend on neighbors' states (contagion effect).
    """
    
    def __init__(self, tickers: List[str], network: np.ndarray = None):
        """
        Initialize SIR contagion model.
        
        Args:
            tickers: List of stock tickers
            network: Adjacency matrix for contagion network
        """
        super().__init__(name="SIR Contagion Model", n_states=3)
        self.tickers = tickers
        self.n_stocks = len(tickers)
        self.network = network
        
        # Components
        self.state_inference = None
        self.contagion_learner = None
        
        # State names
        self.state_names = {0: 'R', 1: 'S', 2: 'I'}
        
    def fit(self, chunks: List, correlation_threshold: float = 0.6, **kwargs):
        """
        Train the model on data chunks.
        
        Args:
            chunks: List of DataChunk objects
            correlation_threshold: Threshold for building correlation network
        """
        print(f"\n{'='*60}")
        print("Training SIR Contagion Model")
        print(f"{'='*60}")
        
        # Step 1: Learn state parameters for each stock
        self.state_inference = MultiStockStateInference(self.tickers, n_states=3)
        self.state_inference.fit(chunks, max_iter=30)
        
        # Step 2: Build network if not provided
        if self.network is None:
            print(f"\nBuilding correlation network (threshold={correlation_threshold})...")
            # Concatenate all returns
            all_returns = pd.concat([chunk.returns for chunk in chunks], axis=0)
            self.network = self.contagion_learner = ContagionLearner(
                self.tickers
            ).build_network_from_correlation(all_returns, correlation_threshold)
        
        # Step 3: Infer historical states
        print("\nInferring historical states...")
        states_list = self.state_inference.infer_states_for_chunks(chunks)
        
        # Print state summary
        state_summary = self.state_inference.get_state_summary(chunks)
        print("\nState occupancy:")
        print(state_summary.to_string(index=False))
        
        # Step 4: Learn contagion parameters
        self.contagion_learner = ContagionLearner(self.tickers, self.network)
        self.contagion_learner.fit(states_list)
        
        self.is_trained = True
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}\n")
    
    def simulate(self, 
                initial_prices: np.ndarray,
                n_steps: int,
                dates: List[pd.Timestamp] = None,
                initial_states: np.ndarray = None,
                **kwargs) -> SimulationResult:
        """
        Simulate price trajectories with SIR contagion dynamics.
        
        Args:
            initial_prices: Starting prices (n_stocks,)
            n_steps: Number of steps to simulate
            dates: Optional dates for simulation
            initial_states: Initial states (n_stocks,), if None assumes all Susceptible
            
        Returns:
            SimulationResult with prices, returns, and states
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before simulation")
        
        # Initialize
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
                
                # Get neighbor states
                neighbors_mask = self.network[i] > 0
                neighbor_states = states[t-1, neighbors_mask]
                
                # Compute transition probabilities (with contagion)
                trans_probs = self.contagion_learner.compute_infection_probability(
                    current_state, neighbor_states
                )
                
                # Sample next state
                next_state = np.random.choice(
                    list(trans_probs.keys()),
                    p=list(trans_probs.values())
                )
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
            params['contagion_params'] = self.contagion_learner.get_params()
            params['n_stocks'] = self.n_stocks
            params['network_density'] = self.network.sum() / (self.n_stocks * (self.n_stocks - 1))
        
        return params
    
    def print_summary(self):
        """Print model summary."""
        print(f"\n{'='*60}")
        print(f"SIR Contagion Model Summary")
        print(f"{'='*60}")
        print(f"Stocks: {self.n_stocks}")
        print(f"Trained: {self.is_trained}")
        
        if self.is_trained:
            print(f"\nContagion Parameters:")
            params = self.contagion_learner.get_params()
            print(f"  β (contagion rate):      {params['beta']:.4f}")
            print(f"  γ (recovery rate):       {params['gamma']:.4f}")
            print(f"  α (re-susceptibility):   {params['alpha']:.4f}")
            print(f"  Network density:         {params['network_density']:.2%}")
            
            print(f"\nState Parameters (sample for first stock):")
            ticker = self.tickers[0]
            for state in range(3):
                label = self.state_names[state]
                params = self.state_inference.state_params[ticker][state]
                print(f"  State {label}: μ={params['mu']:.5f}, σ={params['sigma']:.5f}")
        
        print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test SIR model
    print("Testing SIR Contagion Model...")
    
    # Create synthetic data
    from data.data_loader import DataLoader, get_default_tickers
    from data.chunk_selector import prepare_dataset
    
    tickers = get_default_tickers()[:5]
    print(f"Loading data for {tickers}...")
    
    loader = DataLoader(tickers, years_back=5)
    prices, vix = loader.load_data()
    
    dataset = prepare_dataset(prices, vix, n_chunks=30, train_ratio=0.7)
    
    # Train model
    model = SIRContagionModel(tickers)
    model.fit(dataset['train_chunks'])
    model.print_summary()
    
    # Test simulation
    print("Testing simulation...")
    test_chunk = dataset['test_chunks'][0]
    initial_prices = test_chunk.prices.iloc[0].values
    
    result = model.simulate(initial_prices, n_steps=10)
    print(f"Simulated {result.prices.shape[0]} steps for {result.prices.shape[1]} stocks")
    print(f"Final prices: {result.prices[-1]}")
    print(f"Final states: {result.states[-1]}")

