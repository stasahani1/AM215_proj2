"""
Evaluation Metrics for Model Performance Comparison.

This module provides comprehensive metrics to evaluate stock price prediction
models across multiple dimensions:

1. **Price Accuracy**: MSE, MAE, MAPE, R² for price predictions
2. **Return Accuracy**: Return prediction metrics
3. **Volatility Matching**: How well models capture volatility patterns
4. **Correlation Matching**: Cross-stock correlation structure accuracy
5. **Distribution Matching**: Wasserstein distance, JS divergence for returns
6. **State Metrics**: State accuracy for state-based models

All metrics are computed by comparing actual historical data against
Monte Carlo simulations from the model.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

from models.base_model import SimulationResult


class ModelEvaluator:
    """Compute various metrics to evaluate model performance."""
    
    def __init__(self):
        pass
    
    def compute_price_accuracy_metrics(self, 
                                       actual_prices: np.ndarray,
                                       simulated_results: List[SimulationResult]) -> Dict:
        """
        Compute price prediction accuracy metrics.
        
        Args:
            actual_prices: Actual observed prices (n_steps, n_stocks)
            simulated_results: List of simulation results
            
        Returns:
            Dictionary of metrics
        """
        # Get mean prediction across simulations
        all_sim_prices = np.array([r.prices for r in simulated_results])  # (n_sims, n_steps, n_stocks)
        mean_prices = all_sim_prices.mean(axis=0)  # (n_steps, n_stocks)
        std_prices = all_sim_prices.std(axis=0)
        
        # Mean Squared Error
        mse = np.mean((actual_prices - mean_prices) ** 2)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual_prices - mean_prices))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual_prices - mean_prices) / (actual_prices + 1e-10))) * 100
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # R-squared (coefficient of determination)
        ss_res = np.sum((actual_prices - mean_prices) ** 2)
        ss_tot = np.sum((actual_prices - actual_prices.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Log-likelihood (assuming Gaussian)
        log_likelihood = -0.5 * np.sum(
            ((actual_prices - mean_prices) / (std_prices + 1e-10)) ** 2 + 
            np.log(2 * np.pi * (std_prices ** 2 + 1e-10))
        )
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'r2': r2,
            'log_likelihood': log_likelihood
        }
    
    def compute_return_accuracy_metrics(self,
                                       actual_returns: np.ndarray,
                                       simulated_results: List[SimulationResult]) -> Dict:
        """
        Compute return prediction accuracy.
        
        Args:
            actual_returns: Actual returns (n_steps-1, n_stocks)
            simulated_results: List of simulation results
            
        Returns:
            Dictionary of metrics
        """
        # Skip first row of simulated returns (which is typically zeros)
        all_sim_returns = np.array([r.returns[1:] for r in simulated_results])
        mean_returns = all_sim_returns.mean(axis=0)
        
        # Ensure shapes match
        min_len = min(len(actual_returns), len(mean_returns))
        actual_returns = actual_returns[:min_len]
        mean_returns = mean_returns[:min_len]
        
        # Return MSE
        return_mse = np.mean((actual_returns - mean_returns) ** 2)
        
        # Return correlation
        actual_flat = actual_returns.flatten()
        simulated_flat = mean_returns.flatten()
        return_corr = np.corrcoef(actual_flat, simulated_flat)[0, 1]
        
        return {
            'return_mse': return_mse,
            'return_correlation': return_corr
        }
    
    def compute_volatility_metrics(self,
                                   actual_prices: np.ndarray,
                                   simulated_results: List[SimulationResult]) -> Dict:
        """
        Evaluate how well model captures volatility.
        
        Args:
            actual_prices: Actual prices
            simulated_results: List of simulation results
            
        Returns:
            Dictionary of volatility metrics
        """
        # Compute actual volatility (std of returns)
        actual_returns = np.diff(np.log(actual_prices), axis=0)
        actual_vol = actual_returns.std(axis=0)  # Per stock
        
        # Compute simulated volatility distribution
        sim_vols = []
        for result in simulated_results:
            sim_returns = np.diff(np.log(result.prices), axis=0)
            sim_vol = sim_returns.std(axis=0)
            sim_vols.append(sim_vol)
        
        sim_vols = np.array(sim_vols)  # (n_sims, n_stocks)
        mean_sim_vol = sim_vols.mean(axis=0)
        
        # Volatility error
        vol_mae = np.mean(np.abs(actual_vol - mean_sim_vol))
        vol_mape = np.mean(np.abs((actual_vol - mean_sim_vol) / (actual_vol + 1e-10))) * 100
        
        # Check if model reproduces volatility clustering
        # (high volatility followed by high volatility)
        actual_vol_ts = actual_returns.std(axis=1)  # Time series of volatility
        actual_vol_autocorr = np.corrcoef(actual_vol_ts[:-1], actual_vol_ts[1:])[0, 1]
        
        sim_vol_autocorrs = []
        for result in simulated_results:
            sim_returns = np.diff(np.log(result.prices), axis=0)
            sim_vol_ts = sim_returns.std(axis=1)
            if len(sim_vol_ts) > 1:
                autocorr = np.corrcoef(sim_vol_ts[:-1], sim_vol_ts[1:])[0, 1]
                if not np.isnan(autocorr):
                    sim_vol_autocorrs.append(autocorr)
        
        mean_sim_vol_autocorr = np.mean(sim_vol_autocorrs) if sim_vol_autocorrs else 0
        
        return {
            'volatility_mae': vol_mae,
            'volatility_mape': vol_mape,
            'actual_vol_autocorr': actual_vol_autocorr,
            'simulated_vol_autocorr': mean_sim_vol_autocorr,
            'vol_autocorr_error': abs(actual_vol_autocorr - mean_sim_vol_autocorr)
        }
    
    def compute_correlation_metrics(self,
                                    actual_prices: np.ndarray,
                                    simulated_results: List[SimulationResult]) -> Dict:
        """
        Evaluate how well model captures cross-stock correlations.
        
        Args:
            actual_prices: Actual prices
            simulated_results: List of simulation results
            
        Returns:
            Dictionary of correlation metrics
        """
        # Compute actual correlation matrix
        actual_returns = np.diff(np.log(actual_prices), axis=0)
        actual_corr = np.corrcoef(actual_returns.T)
        
        # Compute simulated correlation matrices
        sim_corrs = []
        for result in simulated_results:
            sim_returns = np.diff(np.log(result.prices), axis=0)
            sim_corr = np.corrcoef(sim_returns.T)
            sim_corrs.append(sim_corr)
        
        mean_sim_corr = np.mean(sim_corrs, axis=0)
        
        # Frobenius norm of difference
        corr_frobenius_error = np.linalg.norm(actual_corr - mean_sim_corr, 'fro')
        
        # Mean absolute error in correlations
        # (only off-diagonal elements)
        n_stocks = actual_corr.shape[0]
        mask = ~np.eye(n_stocks, dtype=bool)
        corr_mae = np.mean(np.abs(actual_corr[mask] - mean_sim_corr[mask]))
        
        # Average correlation
        actual_avg_corr = actual_corr[mask].mean()
        sim_avg_corr = mean_sim_corr[mask].mean()
        
        return {
            'correlation_frobenius_error': corr_frobenius_error,
            'correlation_mae': corr_mae,
            'actual_avg_correlation': actual_avg_corr,
            'simulated_avg_correlation': sim_avg_corr,
            'avg_correlation_error': abs(actual_avg_corr - sim_avg_corr)
        }
    
    def compute_distribution_metrics(self,
                                    actual_returns: np.ndarray,
                                    simulated_results: List[SimulationResult]) -> Dict:
        """
        Compare distributions of actual vs simulated returns.
        
        Args:
            actual_returns: Actual returns
            simulated_results: List of simulation results
            
        Returns:
            Dictionary of distribution metrics
        """
        actual_flat = actual_returns.flatten()
        
        # Collect all simulated returns (skip first row which is typically zeros)
        all_sim_returns = []
        for result in simulated_results:
            all_sim_returns.extend(result.returns[1:].flatten())
        all_sim_returns = np.array(all_sim_returns)
        
        # Wasserstein distance
        w_dist = wasserstein_distance(actual_flat, all_sim_returns)
        
        # Jensen-Shannon divergence (requires binning)
        # Create histogram bins
        bins = np.linspace(
            min(actual_flat.min(), all_sim_returns.min()),
            max(actual_flat.max(), all_sim_returns.max()),
            50
        )
        
        actual_hist, _ = np.histogram(actual_flat, bins=bins, density=True)
        sim_hist, _ = np.histogram(all_sim_returns, bins=bins, density=True)
        
        # Normalize to probabilities
        actual_hist = actual_hist / (actual_hist.sum() + 1e-10)
        sim_hist = sim_hist / (sim_hist.sum() + 1e-10)
        
        js_div = jensenshannon(actual_hist + 1e-10, sim_hist + 1e-10)
        
        # Compare moments
        actual_mean = actual_flat.mean()
        actual_std = actual_flat.std()
        actual_skew = ((actual_flat - actual_mean) ** 3).mean() / (actual_std ** 3)
        actual_kurt = ((actual_flat - actual_mean) ** 4).mean() / (actual_std ** 4)
        
        sim_mean = all_sim_returns.mean()
        sim_std = all_sim_returns.std()
        sim_skew = ((all_sim_returns - sim_mean) ** 3).mean() / (sim_std ** 3)
        sim_kurt = ((all_sim_returns - sim_mean) ** 4).mean() / (sim_std ** 4)
        
        return {
            'wasserstein_distance': w_dist,
            'js_divergence': js_div,
            'mean_error': abs(actual_mean - sim_mean),
            'std_error': abs(actual_std - sim_std),
            'skew_error': abs(actual_skew - sim_skew),
            'kurtosis_error': abs(actual_kurt - sim_kurt)
        }
    
    def compute_state_metrics(self,
                             actual_states: np.ndarray,
                             simulated_results: List[SimulationResult]) -> Dict:
        """
        Evaluate state inference accuracy (for state-based models).
        
        Args:
            actual_states: Inferred actual states
            simulated_results: List of simulation results
            
        Returns:
            Dictionary of state metrics
        """
        if simulated_results[0].states is None:
            return {}
        
        # Get mean simulated states
        all_sim_states = np.array([r.states for r in simulated_results])
        
        # State accuracy (most common state)
        mode_states = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=all_sim_states
        )
        
        state_accuracy = (mode_states == actual_states).mean() * 100
        
        # State distribution
        actual_state_dist = np.array([
            (actual_states == 0).mean(),
            (actual_states == 1).mean(),
            (actual_states == 2).mean()
        ])
        
        sim_state_dist = np.array([
            (all_sim_states == 0).mean(),
            (all_sim_states == 1).mean(),
            (all_sim_states == 2).mean()
        ])
        
        state_dist_error = np.abs(actual_state_dist - sim_state_dist).mean()
        
        return {
            'state_accuracy': state_accuracy,
            'state_distribution_error': state_dist_error,
            'actual_infected_pct': actual_state_dist[2] * 100,
            'simulated_infected_pct': sim_state_dist[2] * 100
        }
    
    def evaluate_all(self,
                    actual_prices: np.ndarray,
                    simulated_results: List[SimulationResult],
                    actual_states: np.ndarray = None) -> Dict:
        """
        Compute all metrics.
        
        Args:
            actual_prices: Actual prices
            simulated_results: List of simulation results
            actual_states: Optional actual states for state-based models
            
        Returns:
            Dictionary with all metrics
        """
        actual_returns = np.diff(np.log(actual_prices), axis=0)
        
        metrics = {}
        
        # Price accuracy
        metrics.update(self.compute_price_accuracy_metrics(actual_prices, simulated_results))
        
        # Return accuracy
        metrics.update(self.compute_return_accuracy_metrics(actual_returns, simulated_results))
        
        # Volatility
        metrics.update(self.compute_volatility_metrics(actual_prices, simulated_results))
        
        # Correlation
        if actual_prices.shape[1] > 1:  # Only if multiple stocks
            metrics.update(self.compute_correlation_metrics(actual_prices, simulated_results))
        
        # Distribution
        metrics.update(self.compute_distribution_metrics(actual_returns, simulated_results))
        
        # States (if applicable)
        if actual_states is not None:
            metrics.update(self.compute_state_metrics(actual_states, simulated_results))
        
        return metrics


def print_metrics_comparison(results_dict: Dict[str, Dict], top_n: int = 10):
    """
    Print comparison of metrics across models.
    
    Args:
        results_dict: Dictionary mapping model_name -> metrics_dict
        top_n: Number of top metrics to display
    """
    # Get all metric names
    all_metrics = set()
    for metrics in results_dict.values():
        all_metrics.update(metrics.keys())
    
    # Sort metrics alphabetically
    all_metrics = sorted(all_metrics)[:top_n]
    
    print(f"\n{'='*80}")
    print("Model Comparison (Lower is Better for Error Metrics)")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'Metric':<30}", end='')
    for model_name in results_dict.keys():
        print(f"{model_name:<20}", end='')
    print()
    print("-" * 80)
    
    for metric in all_metrics:
        print(f"{metric:<30}", end='')
        for model_name in results_dict.keys():
            value = results_dict[model_name].get(metric, np.nan)
            if isinstance(value, (int, float)):
                print(f"{value:>19.4f}", end=' ')
            else:
                print(f"{str(value):>19}", end=' ')
        print()
    
    print(f"{'='*80}\n")

