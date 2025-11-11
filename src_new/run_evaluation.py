"""
Main script to run complete evaluation of SIR contagion model vs baselines.

Usage:
    python run_evaluation.py
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add paths
sys.path.append(os.path.dirname(__file__))

# Import modules
from data.data_loader import DataLoader, get_default_tickers
from data.chunk_selector import prepare_dataset
from models.sir_contagion_model import SIRContagionModel
from models.baseline_single_rw import SingleRandomWalkModel
from models.baseline_independent_hmm import IndependentHMMModel
from models.baseline_correlated_rw import CorrelatedRandomWalkModel
from evaluation.backtester import Backtester
from visualization.plots import create_summary_report


def main():
    """Run complete evaluation pipeline."""
    
    print("\n" + "="*80)
    print("SIR CONTAGION MODEL EVALUATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # ==================== CONFIGURATION ====================
    
    # Data configuration
    TICKERS = get_default_tickers()[:10]  # Use 10 stocks for faster computation
    YEARS_BACK = 15  # Use 15 years of data
    
    # Chunking configuration
    CHUNK_DAYS = 10  # 2 weeks
    N_CHUNKS = 60  # Select top 60 high-volatility chunks
    TRAIN_RATIO = 0.6  # 60% for training, 40% for testing
    
    # Evaluation configuration
    N_SIMULATIONS = 500  # Monte Carlo simulations per test chunk
    
    # Network configuration
    CORRELATION_THRESHOLD = 0.5  # For building contagion network
    
    print("Configuration:")
    print(f"  Tickers: {TICKERS}")
    print(f"  Years of data: {YEARS_BACK}")
    print(f"  Chunk size: {CHUNK_DAYS} days")
    print(f"  High-volatility chunks: {N_CHUNKS}")
    print(f"  Train/test split: {TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}")
    print(f"  Simulations per test: {N_SIMULATIONS}")
    print(f"  Correlation threshold: {CORRELATION_THRESHOLD}")
    
    # ==================== STEP 1: LOAD DATA ====================
    
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    
    loader = DataLoader(TICKERS, years_back=YEARS_BACK)
    prices, vix = loader.load_data()
    
    print("\nData Summary:")
    print(loader.get_summary_stats().to_string())
    
    # ==================== STEP 2: CREATE CHUNKS ====================
    
    print("\n" + "="*80)
    print("STEP 2: Creating and Selecting High-Volatility Chunks")
    print("="*80)
    
    dataset = prepare_dataset(
        prices=prices,
        vix=vix,
        chunk_days=CHUNK_DAYS,
        n_chunks=N_CHUNKS,
        train_ratio=TRAIN_RATIO,
        volatility_metric='vix'
    )
    
    train_chunks = dataset['train_chunks']
    test_chunks = dataset['test_chunks']
    
    print(f"\nDataset prepared:")
    print(f"  Training chunks: {len(train_chunks)}")
    print(f"  Testing chunks: {len(test_chunks)}")
    
    # ==================== STEP 3: TRAIN MODELS ====================
    
    print("\n" + "="*80)
    print("STEP 3: Training Models")
    print("="*80)
    
    # Initialize models
    models = []
    
    # 1. SIR Contagion Model (our model)
    print("\n" + "-"*60)
    print("Training Model 1/4: SIR Contagion Model")
    print("-"*60)
    sir_model = SIRContagionModel(TICKERS)
    sir_model.fit(train_chunks, correlation_threshold=CORRELATION_THRESHOLD)
    sir_model.print_summary()
    models.append(sir_model)
    
    # 2. Independent HMM (Baseline 2 - same states, no contagion)
    print("\n" + "-"*60)
    print("Training Model 2/4: Independent HMM")
    print("-"*60)
    hmm_model = IndependentHMMModel(TICKERS)
    hmm_model.fit(train_chunks)
    models.append(hmm_model)
    
    # 3. Single Random Walk (Baseline 1 - simplest)
    print("\n" + "-"*60)
    print("Training Model 3/4: Single Random Walk")
    print("-"*60)
    single_rw_model = SingleRandomWalkModel(TICKERS)
    single_rw_model.fit(train_chunks)
    models.append(single_rw_model)
    
    # 4. Correlated Random Walk (Baseline 3 - captures correlation)
    print("\n" + "-"*60)
    print("Training Model 4/4: Correlated Random Walk")
    print("-"*60)
    corr_rw_model = CorrelatedRandomWalkModel(TICKERS)
    corr_rw_model.fit(train_chunks)
    models.append(corr_rw_model)
    
    print("\n" + "="*80)
    print("All models trained successfully!")
    print("="*80)
    
    # ==================== STEP 4: RUN BACKTEST ====================
    
    print("\n" + "="*80)
    print("STEP 4: Running Backtest on Test Chunks")
    print("="*80)
    
    backtester = Backtester(models, n_simulations=N_SIMULATIONS)
    results = backtester.run_backtest(test_chunks, verbose=True)
    
    # ==================== STEP 5: ANALYZE RESULTS ====================
    
    print("\n" + "="*80)
    print("STEP 5: Results Analysis")
    print("="*80)
    
    backtester.print_summary()
    
    # Export detailed results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    backtester.export_results(f'{output_dir}/backtest_results.csv')
    
    # ==================== STEP 6: VISUALIZATIONS ====================
    
    print("\n" + "="*80)
    print("STEP 6: Creating Visualizations")
    print("="*80)
    
    # Use first test chunk for visualization
    test_chunk_for_viz = test_chunks[0]
    
    create_summary_report(
        backtest_results=results,
        test_chunk=test_chunk_for_viz,
        models=models,
        output_dir=output_dir
    )
    
    # ==================== STEP 7: FINAL SUMMARY ====================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    # Get key metrics
    comparison = backtester.get_comparison_table()
    
    print("\nKey Performance Metrics:")
    print(comparison[['mse_mean', 'mape_mean', 'r2_mean']].to_string())
    
    # Determine winner
    print("\n" + "-"*80)
    print("Model Rankings (Lower is Better):")
    print("-"*80)
    
    for metric in ['mse_mean', 'mape_mean']:
        values = comparison[metric].sort_values()
        print(f"\n{metric.replace('_', ' ').title()}:")
        for i, (model, value) in enumerate(values.items(), 1):
            print(f"  {i}. {model:<30} {value:.6f}")
    
    # Check if SIR model wins
    sir_mse = results['SIR Contagion Model']['aggregate_metrics']['mse_mean']
    best_baseline_mse = min([
        results['Independent HMM']['aggregate_metrics']['mse_mean'],
        results['Single Random Walk']['aggregate_metrics']['mse_mean'],
        results['Correlated Random Walk']['aggregate_metrics']['mse_mean']
    ])
    
    print("\n" + "="*80)
    if sir_mse < best_baseline_mse:
        improvement = (best_baseline_mse - sir_mse) / best_baseline_mse * 100
        print(f"✓ SUCCESS: SIR Contagion Model outperforms all baselines!")
        print(f"  Improvement over best baseline: {improvement:.2f}%")
    else:
        print(f"✗ SIR model did not outperform baselines")
        print(f"  This may indicate contagion effects are weak in selected periods")
        print(f"  or parameters need tuning.")
    print("="*80)
    
    # ==================== COMPLETION ====================
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - backtest_results.csv")
    print(f"  - price_trajectories.png")
    print(f"  - return_distributions.png")
    print(f"  - metric_comparison.png")
    print(f"  - correlation_matrices.png")
    print("="*80 + "\n")
    
    return results


if __name__ == '__main__':
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

