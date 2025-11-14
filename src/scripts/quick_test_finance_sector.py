"""
Quick test with FINANCE SECTOR ONLY stocks.

Hypothesis: Financial contagion (2008 crisis) should show strong β in finance sector.
"""

import sys
import time
import numpy as np
import pandas as pd

# Imports
from data.data_loader import DataLoader
from data.chunk_selector import ChunkSelector
from models.sir_contagion_model import SIRContagionModel
from models.baseline_single_rw import SingleRandomWalkModel
from models.baseline_independent_hmm import IndependentHMMModel
from models.baseline_correlated_rw import CorrelatedRandomWalkModel
from evaluation.backtester import Backtester

# Configuration
FINANCE_STOCKS = [
    'JPM',    # JPMorgan Chase
    'BAC',    # Bank of America
    'WFC',    # Wells Fargo
    'GS',     # Goldman Sachs
    'MS',     # Morgan Stanley
    'C',      # Citigroup
    'BLK',    # BlackRock
    'SCHW',   # Charles Schwab
    'AXP',    # American Express
    'USB'     # US Bancorp
]

YEARS_BACK = 15  # Include 2008 crisis!
CHUNK_DAYS = 30  # 6 weeks
N_CHUNKS = 30
TRAIN_RATIO = 0.6
N_SIMULATIONS = 200

print("="*80)
print("FINANCE SECTOR ONLY TEST")
print("="*80)
print(f"\nHypothesis: SIR contagion should be STRONGEST in finance")
print(f"(Financial crisis 2008 is the canonical example of contagion)")
print(f"\nConfiguration:")
print(f"  Stocks: {len(FINANCE_STOCKS)} financial stocks")
print(f"  Period: Last {YEARS_BACK} years (includes 2008 crisis)")
print(f"  Chunk size: {CHUNK_DAYS} days (~6 weeks)")
print(f"  Total chunks: {N_CHUNKS}")
print(f"  Train/test split: {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}")
print(f"  Simulations per test: {N_SIMULATIONS}")
print("="*80)

# Step 1: Load data
print("\n" + "="*80)
print("STEP 1: Loading Data for Finance Stocks")
print("="*80)
loader = DataLoader(FINANCE_STOCKS, years_back=YEARS_BACK)
prices, vix = loader.load_data()

if prices.empty:
    print("ERROR: No data loaded")
    sys.exit(1)

print(f"\nLoaded {len(prices)} days of data for {len(prices.columns)} stocks")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Step 2: Create chunks (mixed training: all periods, test: high VIX)
print("\n" + "="*80)
print("STEP 2: Creating Time Chunks")
print("="*80)
print(f"Splitting into {CHUNK_DAYS}-day chunks...")

chunk_selector = ChunkSelector(prices, vix, chunk_days=CHUNK_DAYS)
chunks = chunk_selector.create_chunks()

if len(chunks) < N_CHUNKS:
    print(f"Warning: Only {len(chunks)} chunks available, using all")
    N_CHUNKS = len(chunks)

# Sort by VIX and create mixed training set
chunks_sorted = sorted(chunks, key=lambda c: c.vix_mean, reverse=True)

# Training: Mix of all VIX levels
n_train = int(N_CHUNKS * TRAIN_RATIO)
n_test = N_CHUNKS - n_train

all_chunks_by_time = sorted(chunks, key=lambda c: c.start_date)
step = max(1, len(all_chunks_by_time) // n_train)
train_chunks = all_chunks_by_time[::step][:n_train]

# Testing: High VIX only
test_chunks = chunks_sorted[:n_test]

print(f"\nTraining chunks: {len(train_chunks)} (mixed VIX levels)")
train_vix_range = [c.vix_mean for c in train_chunks]
print(f"  VIX range: {min(train_vix_range):.1f} - {max(train_vix_range):.1f}")

print(f"\nTest chunks: {len(test_chunks)} (high VIX only)")
test_vix_range = [c.vix_mean for c in test_chunks]
print(f"  VIX range: {min(test_vix_range):.1f} - {max(test_vix_range):.1f}")

# Step 3: Initialize models
print("\n" + "="*80)
print("STEP 3: Initializing Models")
print("="*80)

models = {
    'SIR Contagion Model': SIRContagionModel(FINANCE_STOCKS),
    'Independent HMM': IndependentHMMModel(FINANCE_STOCKS),
    'Single Random Walk': SingleRandomWalkModel(FINANCE_STOCKS),
    'Correlated Random Walk': CorrelatedRandomWalkModel(FINANCE_STOCKS)
}

print(f"Initialized {len(models)} models")

# Step 4: Train models
print("\n" + "="*80)
print("STEP 4: Training Models")
print("="*80)

start_time = time.time()

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    model.fit(train_chunks, correlation_threshold=0.6)

train_time = time.time() - start_time
print(f"\nTotal training time: {train_time:.1f}s")

# Print SIR contagion parameters
print("\n" + "="*80)
print("FINANCE SECTOR CONTAGION PARAMETERS")
print("="*80)
sir_model = models['SIR Contagion Model']
if sir_model.is_trained:
    params = sir_model.get_params()
    print(f"\nContagion Parameters:")
    print(f"  β (contagion rate):      {params['contagion_params']['beta']:.4f}")
    print(f"  γ (recovery rate):       {params['contagion_params']['gamma']:.4f}")
    print(f"  α (re-susceptibility):   {params['contagion_params']['alpha']:.4f}")
    print(f"  Network density:         {params['network_density']:.2%}")
    print(f"\nExpectation: β should be VERY HIGH for finance (2008 crisis contagion)")

# Step 5: Backtest
print("\n" + "="*80)
print("STEP 5: Backtesting on High-VIX Periods")
print("="*80)

backtester = Backtester(list(models.values()), n_simulations=N_SIMULATIONS)
results = backtester.run_backtest(test_chunks)

# Step 6: Analyze results
print("\n" + "="*80)
print("STEP 6: Results Analysis")
print("="*80)

# Print summary using backtester's built-in method
backtester.print_summary()

# Get comparison table
summary = backtester.get_comparison_table()

print("\n" + "="*80)
print("Backtest Summary - FINANCE SECTOR ONLY")
print("="*80)
print(summary[['mse_mean', 'mape_mean', 'r2_mean']].to_string())

# Rankings
print("\n" + "="*80)
print("Model Rankings (Lower MSE is Better):")
print("="*80)
ranked = summary.sort_values('mse_mean')
for idx, (model_name, row) in enumerate(ranked.iterrows(), 1):
    print(f"  {idx}. {model_name:30s} MSE: {row['mse_mean']:.2f}")

# Check if SIR won
sir_mse = summary.loc['SIR Contagion Model', 'mse_mean']
best_baseline_mse = summary.drop('SIR Contagion Model')['mse_mean'].min()
best_baseline_name = summary.drop('SIR Contagion Model')['mse_mean'].idxmin()

print("\n" + "="*80)
print("FINANCE SECTOR HYPOTHESIS TEST")
print("="*80)
print(f"SIR Contagion Model MSE:     {sir_mse:.2f}")
print(f"Best Baseline ({best_baseline_name}): {best_baseline_mse:.2f}")

improvement = (best_baseline_mse - sir_mse) / best_baseline_mse * 100
if sir_mse < best_baseline_mse:
    print(f"\n✅ SIR OUTPERFORMS by {improvement:.1f}%")
    print(f"   Financial contagion validated!")
else:
    print(f"\n❌ SIR underperforms by {-improvement:.1f}%")
    print(f"   Surprising - even in finance, contagion effects weak")

print("="*80)
print(f"Total runtime: {time.time() - start_time:.1f}s")
print("="*80)

