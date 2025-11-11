"""
Backtesting framework for evaluating models on test chunks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import sys
sys.path.append('..')
from models.base_model import BaseModel
from evaluation.metrics import ModelEvaluator


class Backtester:
    """Run backtest evaluation on test chunks."""
    
    def __init__(self, models: List[BaseModel], n_simulations: int = 1000):
        """
        Initialize backtester.
        
        Args:
            models: List of models to evaluate
            n_simulations: Number of Monte Carlo simulations per test chunk
        """
        self.models = models
        self.n_simulations = n_simulations
        self.evaluator = ModelEvaluator()
        
        self.results = {}  # Store results per model
    
    def run_backtest(self, test_chunks: List, verbose: bool = True) -> Dict:
        """
        Run backtest on test chunks.
        
        Args:
            test_chunks: List of DataChunk objects to test on
            verbose: Print progress
            
        Returns:
            Dictionary with results per model
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Backtest")
            print(f"{'='*60}")
            print(f"Test chunks: {len(test_chunks)}")
            print(f"Simulations per chunk: {self.n_simulations}")
            print(f"Models: {len(self.models)}")
        
        # Initialize results storage
        for model in self.models:
            self.results[model.name] = {
                'chunk_metrics': [],
                'aggregate_metrics': {},
                'chunk_ids': []
            }
        
        # Iterate over test chunks
        for chunk_idx, chunk in enumerate(test_chunks):
            if verbose:
                print(f"\n{'-'*60}")
                print(f"Test Chunk {chunk_idx + 1}/{len(test_chunks)}")
                print(f"  Period: {chunk.start_date.date()} to {chunk.end_date.date()}")
                print(f"  Mean VIX: {chunk.vix_mean:.2f}")
            
            # Get actual prices
            actual_prices = chunk.prices.values
            initial_prices = actual_prices[0]
            n_steps = len(actual_prices)
            
            # Evaluate each model
            for model in self.models:
                if verbose:
                    print(f"\n  Evaluating {model.name}...")
                
                try:
                    # Run simulations
                    sim_results = model.simulate_multiple(
                        initial_prices=initial_prices,
                        n_steps=n_steps,
                        n_simulations=self.n_simulations,
                        dates=chunk.prices.index.tolist()
                    )
                    
                    # Infer actual states if model is state-based
                    actual_states = None
                    if hasattr(model, 'infer_states') and hasattr(model, 'is_trained') and model.is_trained:
                        try:
                            actual_states = model.infer_states(chunk.prices)
                        except:
                            actual_states = None
                    
                    # Compute metrics
                    metrics = self.evaluator.evaluate_all(
                        actual_prices=actual_prices,
                        simulated_results=sim_results,
                        actual_states=actual_states
                    )
                    
                    # Store results
                    self.results[model.name]['chunk_metrics'].append(metrics)
                    self.results[model.name]['chunk_ids'].append(chunk.chunk_id)
                    
                    if verbose:
                        print(f"    MSE: {metrics.get('mse', np.nan):.6f}")
                        print(f"    MAPE: {metrics.get('mape', np.nan):.2f}%")
                        print(f"    R²: {metrics.get('r2', np.nan):.4f}")
                
                except Exception as e:
                    if verbose:
                        print(f"    Error: {e}")
                    # Store NaN metrics
                    self.results[model.name]['chunk_metrics'].append({})
        
        # Aggregate results across all chunks
        if verbose:
            print(f"\n{'-'*60}")
            print("Aggregating results...")
        
        for model in self.models:
            chunk_metrics = self.results[model.name]['chunk_metrics']
            
            # Compute mean metrics across chunks
            all_metric_names = set()
            for metrics in chunk_metrics:
                all_metric_names.update(metrics.keys())
            
            aggregate = {}
            for metric_name in all_metric_names:
                values = [m.get(metric_name, np.nan) for m in chunk_metrics]
                values = [v for v in values if not np.isnan(v)]
                
                if values:
                    aggregate[f'{metric_name}_mean'] = np.mean(values)
                    aggregate[f'{metric_name}_std'] = np.std(values)
                    aggregate[f'{metric_name}_median'] = np.median(values)
            
            self.results[model.name]['aggregate_metrics'] = aggregate
        
        if verbose:
            print(f"{'='*60}\n")
        
        return self.results
    
    def get_comparison_table(self, metrics_to_compare: List[str] = None) -> pd.DataFrame:
        """
        Get comparison table of model performance.
        
        Args:
            metrics_to_compare: List of metric names to compare
                               If None, uses default key metrics
                               
        Returns:
            DataFrame with models as rows, metrics as columns
        """
        if metrics_to_compare is None:
            metrics_to_compare = [
                'mse_mean', 'mape_mean', 'r2_mean',
                'volatility_mae_mean', 'correlation_mae_mean',
                'wasserstein_distance_mean'
            ]
        
        comparison = []
        for model in self.models:
            row = {'model': model.name}
            aggregate = self.results[model.name]['aggregate_metrics']
            
            for metric in metrics_to_compare:
                row[metric] = aggregate.get(metric, np.nan)
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.set_index('model')
        
        return df
    
    def get_best_model(self, metric: str = 'mse_mean', lower_is_better: bool = True) -> str:
        """
        Find best performing model based on a metric.
        
        Args:
            metric: Metric name to compare
            lower_is_better: If True, lower values are better
            
        Returns:
            Name of best model
        """
        values = {}
        for model in self.models:
            value = self.results[model.name]['aggregate_metrics'].get(metric, np.nan)
            if not np.isnan(value):
                values[model.name] = value
        
        if not values:
            return None
        
        if lower_is_better:
            best_model = min(values, key=values.get)
        else:
            best_model = max(values, key=values.get)
        
        return best_model
    
    def print_summary(self):
        """Print summary of backtest results."""
        print(f"\n{'='*80}")
        print("Backtest Summary")
        print(f"{'='*80}\n")
        
        comparison = self.get_comparison_table()
        print(comparison.to_string())
        
        print(f"\n{'='*80}")
        print("Best Models by Metric:")
        print(f"{'='*80}")
        
        key_metrics = [
            ('mse_mean', True, 'Price MSE'),
            ('mape_mean', True, 'Price MAPE'),
            ('r2_mean', False, 'R-squared'),
            ('volatility_mae_mean', True, 'Volatility Error'),
            ('correlation_mae_mean', True, 'Correlation Error')
        ]
        
        for metric, lower_is_better, label in key_metrics:
            best = self.get_best_model(metric, lower_is_better)
            if best:
                value = self.results[best]['aggregate_metrics'].get(metric, np.nan)
                print(f"  {label:<25}: {best:<25} ({value:.4f})")
        
        print(f"{'='*80}\n")
    
    def export_results(self, filename: str = 'backtest_results.csv'):
        """Export results to CSV."""
        all_rows = []
        
        for model in self.models:
            chunk_metrics = self.results[model.name]['chunk_metrics']
            chunk_ids = self.results[model.name]['chunk_ids']
            
            for chunk_id, metrics in zip(chunk_ids, chunk_metrics):
                row = {
                    'model': model.name,
                    'chunk_id': chunk_id,
                    **metrics
                }
                all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

