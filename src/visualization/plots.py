"""
Visualization Tools for Model Evaluation.

This module provides plotting functions to visualize model performance
and results. It includes:

1. **Price Trajectories**: Actual vs simulated price paths
2. **State Heatmaps**: Evolution of hidden states over time
3. **Return Distributions**: Histogram comparison of actual vs simulated
4. **Metric Comparison**: Bar charts comparing model performance
5. **Correlation Matrices**: Cross-stock correlation structure comparison

All plots are publication-ready with proper styling, labels, and legends.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

from models.base_model import SimulationResult

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_price_trajectories(actual_prices: pd.DataFrame,
                            simulated_results: Dict[str, List[SimulationResult]],
                            stock_idx: int = 0,
                            n_trajectories: int = 100,
                            save_path: str = None):
    """
    Plot actual vs simulated price trajectories.
    
    Args:
        actual_prices: DataFrame of actual prices
        simulated_results: Dict mapping model_name -> list of SimulationResults
        stock_idx: Index of stock to plot
        n_trajectories: Number of trajectories to show per model
        save_path: Path to save figure
    """
    n_models = len(simulated_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    stock_name = actual_prices.columns[stock_idx]
    actual = actual_prices[stock_name].values
    dates = actual_prices.index
    
    for ax, (model_name, results) in zip(axes, simulated_results.items()):
        # Plot sample trajectories
        for i in range(min(n_trajectories, len(results))):
            ax.plot(dates, results[i].prices[:, stock_idx], 
                   alpha=0.05, color='blue', linewidth=0.5)
        
        # Plot mean trajectory
        mean_prices = np.mean([r.prices[:, stock_idx] for r in results], axis=0)
        ax.plot(dates, mean_prices, color='blue', linewidth=2, 
               label='Mean Simulated', alpha=0.8)
        
        # Plot actual
        ax.plot(dates, actual, color='red', linewidth=2, 
               label='Actual', linestyle='--')
        
        ax.set_title(f'{model_name}\n{stock_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_state_heatmap(states: np.ndarray,
                       tickers: List[str],
                       dates: pd.DatetimeIndex,
                       title: str = "State Evolution",
                       save_path: str = None):
    """
    Plot heatmap of states over time.
    
    Args:
        states: Array of states (n_steps, n_stocks)
        tickers: List of stock tickers
        dates: DatetimeIndex
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, max(6, len(tickers) * 0.5)))
    
    # Create heatmap using matplotlib
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['green', 'yellow', 'red'])
    
    im = ax.imshow(states.T, aspect='auto', cmap=cmap, vmin=0, vmax=2)
    
    # Set ticks and labels
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers)
    
    # Set x-axis labels (dates) - show every Nth date
    n_dates = len(dates)
    step = max(1, n_dates // 10)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in range(0, n_dates, step)], rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('State (R=0, S=1, I=2)')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_return_distributions(actual_returns: np.ndarray,
                              simulated_results: Dict[str, List[SimulationResult]],
                              save_path: str = None):
    """
    Plot distribution of returns (actual vs simulated).
    
    Args:
        actual_returns: Actual returns
        simulated_results: Dict of simulated results per model
        save_path: Path to save figure
    """
    n_models = len(simulated_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    actual_flat = actual_returns.flatten()
    
    for ax, (model_name, results) in zip(axes, simulated_results.items()):
        # Collect simulated returns
        sim_returns = np.concatenate([r.returns.flatten() for r in results])
        
        # Plot distributions
        ax.hist(actual_flat, bins=50, alpha=0.5, density=True, 
               label='Actual', color='red')
        ax.hist(sim_returns, bins=50, alpha=0.5, density=True, 
               label='Simulated', color='blue')
        
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_metric_comparison(backtest_results: Dict,
                           metrics: List[str] = None,
                           save_path: str = None):
    """
    Plot bar chart comparing models across metrics.
    
    Args:
        backtest_results: Results from Backtester
        metrics: List of metrics to plot
        save_path: Path to save figure
    """
    if metrics is None:
        metrics = ['mse_mean', 'mape_mean', 'volatility_mae_mean', 
                  'correlation_mae_mean', 'wasserstein_distance_mean']
    
    # Prepare data
    model_names = list(backtest_results.keys())
    n_metrics = len(metrics)
    n_models = len(model_names)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = []
        for model_name in model_names:
            value = backtest_results[model_name]['aggregate_metrics'].get(metric, np.nan)
            values.append(value)
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))
        bars = ax.bar(range(n_models), values, color=colors)
        
        # Highlight best model
        if not all(np.isnan(values)):
            best_idx = np.nanargmin(values)
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(3)
        
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_correlation_matrices(actual_returns: pd.DataFrame,
                              simulated_results: Dict[str, List[SimulationResult]],
                              tickers: List[str],
                              save_path: str = None):
    """
    Plot correlation matrices (actual vs simulated).
    
    Args:
        actual_returns: DataFrame of actual returns
        simulated_results: Dict of simulated results
        tickers: List of stock tickers
        save_path: Path to save figure
    """
    n_models = len(simulated_results)
    fig, axes = plt.subplots(1, n_models+1, figsize=(5*(n_models+1), 5))
    
    # Actual correlation
    actual_corr = actual_returns.corr().values
    im = axes[0].imshow(actual_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_xticks(range(len(tickers)))
    axes[0].set_yticks(range(len(tickers)))
    axes[0].set_xticklabels(tickers, rotation=45, ha='right')
    axes[0].set_yticklabels(tickers)
    axes[0].set_title('Actual Correlation')
    
    # Add text annotations
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = axes[0].text(j, i, f'{actual_corr[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=axes[0])
    
    # Simulated correlations
    for ax, (model_name, results) in zip(axes[1:], simulated_results.items()):
        # Average correlation across simulations
        all_corrs = []
        for result in results:
            sim_df = pd.DataFrame(result.returns, columns=tickers)
            sim_corr = sim_df.corr().values
            all_corrs.append(sim_corr)
        
        mean_corr = np.mean(all_corrs, axis=0)
        
        im = ax.imshow(mean_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha='right')
        ax.set_yticklabels(tickers)
        ax.set_title(f'{model_name}')
        
        # Add text annotations
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                text = ax.text(j, i, f'{mean_corr[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_chunk_performance(backtest_results: Dict,
                          chunk_ids: List[int],
                          metric: str = 'mse',
                          save_path: str = None):
    """
    Plot performance across test chunks.
    
    Args:
        backtest_results: Results from Backtester
        chunk_ids: List of chunk IDs
        metric: Metric to plot
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for model_name, results in backtest_results.items():
        chunk_metrics = results['chunk_metrics']
        values = [m.get(metric, np.nan) for m in chunk_metrics]
        
        ax.plot(chunk_ids, values, marker='o', label=model_name, linewidth=2)
    
    ax.set_xlabel('Chunk ID')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Across Test Chunks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def create_summary_report(backtest_results: Dict,
                         test_chunk: any,
                         models: List,
                         output_dir: str = 'results'):
    """
    Create comprehensive summary report with multiple plots.
    
    Args:
        backtest_results: Results from backtester
        test_chunk: A test chunk for visualization
        models: List of models
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating summary report in {output_dir}/...")
    
    # Run simulations on one test chunk for visualization
    sim_results = {}
    actual_prices = test_chunk.prices
    initial_prices = actual_prices.iloc[0].values
    
    for model in models:
        print(f"  Simulating {model.name}...")
        results = model.simulate_multiple(initial_prices, len(actual_prices), n_simulations=200)
        sim_results[model.name] = results
    
    # Generate plots
    print("\n  Creating plots...")
    
    # 1. Price trajectories
    plot_price_trajectories(
        actual_prices, sim_results, stock_idx=0,
        save_path=f'{output_dir}/price_trajectories.png'
    )
    
    # 2. Return distributions
    actual_returns = np.log(actual_prices / actual_prices.shift(1)).dropna().values
    plot_return_distributions(
        actual_returns, sim_results,
        save_path=f'{output_dir}/return_distributions.png'
    )
    
    # 3. Metric comparison
    plot_metric_comparison(
        backtest_results,
        save_path=f'{output_dir}/metric_comparison.png'
    )
    
    # 4. Correlation matrices (if multiple stocks)
    if len(actual_prices.columns) > 1:
        plot_correlation_matrices(
            np.log(actual_prices / actual_prices.shift(1)).dropna(),
            sim_results,
            actual_prices.columns.tolist(),
            save_path=f'{output_dir}/correlation_matrices.png'
        )
    
    print(f"\n  Report generated in {output_dir}/")

