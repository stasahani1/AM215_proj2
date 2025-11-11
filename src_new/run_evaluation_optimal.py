"""
OPTIMAL FULL EVALUATION: 6-week chunks + Mixed training
This combines the best aspects of both approaches for maximum performance.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from data.data_loader import DataLoader, get_default_tickers
from data.chunk_selector import ChunkSelector
from models.sir_contagion_model import SIRContagionModel
from models.baseline_single_rw import SingleRandomWalkModel
from models.baseline_independent_hmm import IndependentHMMModel
from models.baseline_correlated_rw import CorrelatedRandomWalkModel
from evaluation.backtester import Backtester
from visualization.plots import create_summary_report


def main():
    """Run optimal evaluation with 6-week chunks and mixed training."""
    
    print("\n" + "="*80)
    print("SIR CONTAGION MODEL - OPTIMAL EVALUATION")
    print("6-Week Chunks + Mixed Training")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # ==================== OPTIMAL CONFIGURATION ====================
    
    TICKERS = get_default_tickers()[:10]
    YEARS_BACK = 15
    CHUNK_DAYS = 30  # ← 6 weeks (optimal time scale)
    N_SIMULATIONS = 500
    CORRELATION_THRESHOLD = 0.5
    
    print("OPTIMAL Configuration:")
    print(f"  Tickers: {len(TICKERS)} stocks")
    print(f"  Years of data: {YEARS_BACK}")
    print(f"  Chunk size: {CHUNK_DAYS} days (~6 weeks)")
    print(f"  Training strategy: MIXED VIX periods")
    print(f"  Testing strategy: HIGH VIX periods")
    print(f"  Simulations per test: {N_SIMULATIONS}")
    
    # ==================== STEP 1: LOAD DATA ====================
    
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    
    loader = DataLoader(TICKERS, years_back=YEARS_BACK)
    prices, vix = loader.load_data()
    
    print("\nData Summary:")
    print(loader.get_summary_stats().to_string())
    
    # ==================== STEP 2: CREATE OPTIMAL CHUNKS ====================
    
    print("\n" + "="*80)
    print("STEP 2: Creating Optimal Chunk Strategy")
    print("="*80)
    
    selector = ChunkSelector(prices, vix, chunk_days=CHUNK_DAYS)
    all_chunks = selector.create_chunks()
    
    # Sort by VIX
    all_chunks_sorted = sorted(all_chunks, key=lambda x: x.vix_mean)
    n_total = len(all_chunks_sorted)
    
    print(f"\nCreated {n_total} chunks of {CHUNK_DAYS} days each")
    
    # Training: MIXED VIX (every 2nd chunk for variety)
    train_indices = list(range(0, n_total, 2))
    train_chunks = [all_chunks_sorted[i] for i in train_indices]
    
    # Testing: HIGH VIX only (top 20%)
    n_test = max(10, int(n_total * 0.2))
    test_chunks = sorted(all_chunks_sorted[-n_test:], key=lambda x: x.start_date)
    
    print(f"\nTraining: {len(train_chunks)} chunks (MIXED VIX)")
    print(f"  VIX range: {min(c.vix_mean for c in train_chunks):.1f} - {max(c.vix_mean for c in train_chunks):.1f}")
    print(f"  Total training data: {len(train_chunks) * CHUNK_DAYS} days")
    
    print(f"\nTesting: {len(test_chunks)} chunks (HIGH VIX)")
    print(f"  VIX range: {min(c.vix_mean for c in test_chunks):.1f} - {max(c.vix_mean for c in test_chunks):.1f}")
    print(f"  Total test data: {len(test_chunks) * CHUNK_DAYS} days")
    
    # ==================== STEP 3: TRAIN MODELS ====================
    
    print("\n" + "="*80)
    print("STEP 3: Training Models")
    print("="*80)
    
    models = []
    
    # 1. SIR Contagion Model
    print("\n" + "-"*60)
    print("Training Model 1/4: SIR Contagion Model")
    print("-"*60)
    sir_model = SIRContagionModel(TICKERS)
    sir_model.fit(train_chunks, correlation_threshold=CORRELATION_THRESHOLD)
    sir_model.print_summary()
    models.append(sir_model)
    
    # 2. Independent HMM
    print("\n" + "-"*60)
    print("Training Model 2/4: Independent HMM")
    print("-"*60)
    hmm_model = IndependentHMMModel(TICKERS)
    hmm_model.fit(train_chunks)
    models.append(hmm_model)
    
    # 3. Single Random Walk
    print("\n" + "-"*60)
    print("Training Model 3/4: Single Random Walk")
    print("-"*60)
    single_rw_model = SingleRandomWalkModel(TICKERS)
    single_rw_model.fit(train_chunks)
    models.append(single_rw_model)
    
    # 4. Correlated Random Walk
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
    print("STEP 4: Running Backtest on High-VIX Test Chunks")
    print("="*80)
    
    backtester = Backtester(models, n_simulations=N_SIMULATIONS)
    results = backtester.run_backtest(test_chunks, verbose=True)
    
    # ==================== STEP 5: ANALYZE RESULTS ====================
    
    print("\n" + "="*80)
    print("STEP 5: Results Analysis")
    print("="*80)
    
    backtester.print_summary()
    
    # Export results
    output_dir = 'results_optimal'
    os.makedirs(output_dir, exist_ok=True)
    backtester.export_results(f'{output_dir}/backtest_results.csv')
    
    # ==================== STEP 6: VISUALIZATIONS ====================
    
    print("\n" + "="*80)
    print("STEP 6: Creating Visualizations")
    print("="*80)
    
    test_chunk_for_viz = test_chunks[0]
    
    create_summary_report(
        backtest_results=results,
        test_chunk=test_chunk_for_viz,
        models=models,
        output_dir=output_dir
    )
    
    # ==================== STEP 7: FINAL SUMMARY ====================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY - OPTIMAL CONFIGURATION")
    print("="*80)
    
    comparison = backtester.get_comparison_table()
    
    print("\nKey Performance Metrics:")
    print(comparison[['mse_mean', 'mape_mean', 'r2_mean']].to_string())
    
    # Model rankings
    print("\n" + "-"*80)
    print("Model Rankings (Lower MSE is Better):")
    print("-"*80)
    
    mse_sorted = comparison['mse_mean'].sort_values()
    for i, (model, value) in enumerate(mse_sorted.items(), 1):
        print(f"  {i}. {model:<30} MSE: {value:.2f}")
    
    # Check if SIR wins
    sir_mse = results['SIR Contagion Model']['aggregate_metrics']['mse_mean']
    best_baseline_mse = min([
        results['Independent HMM']['aggregate_metrics']['mse_mean'],
        results['Single Random Walk']['aggregate_metrics']['mse_mean'],
        results['Correlated Random Walk']['aggregate_metrics']['mse_mean']
    ])
    
    print("\n" + "="*80)
    if sir_mse < best_baseline_mse:
        improvement = (best_baseline_mse - sir_mse) / best_baseline_mse * 100
        print(f"✓✓✓ SUCCESS: SIR Contagion Model WINS!")
        print(f"Improvement over best baseline: {improvement:.1f}%")
        print("\nKey Factors:")
        print("  ✓ 6-week chunks provide optimal time scale")
        print("  ✓ Mixed training enables good state learning")
        print("  ✓ Contagion dynamics improve predictions")
    else:
        print(f"SIR model did not outperform best baseline")
        print("This suggests contagion effects are weak for these stocks/periods")
    print("="*80)
    
    # ==================== COMPLETION ====================
    
    print("\n" + "="*80)
    print("OPTIMAL EVALUATION COMPLETE")
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

