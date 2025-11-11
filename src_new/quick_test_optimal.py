"""
OPTIMAL TEST: 6-week chunks + Mixed training (Best of both worlds)
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from data.data_loader import DataLoader
from data.chunk_selector import ChunkSelector
from models.sir_contagion_model import SIRContagionModel
from models.baseline_single_rw import SingleRandomWalkModel
from models.baseline_independent_hmm import IndependentHMMModel
from evaluation.backtester import Backtester

print("\n" + "="*80)
print("OPTIMAL TEST - 6-Week Chunks + Mixed Training")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

TICKERS = ['AAPL', 'MSFT', 'JPM']
YEARS_BACK = 5
CHUNK_DAYS = 30  # 6 weeks (good time scale)
N_SIMULATIONS = 50

print(f"Configuration: {len(TICKERS)} stocks, {YEARS_BACK} years")
print(f"  Chunk size: {CHUNK_DAYS} days (~6 weeks)")
print("  Training: MIXED VIX periods (calm + crisis)")
print("  Testing: HIGH VIX periods\n")

# Step 1: Load data
print("Step 1: Loading data...")
loader = DataLoader(TICKERS, years_back=YEARS_BACK)
prices, vix = loader.load_data()
print(f"✓ Loaded {len(prices)} days\n")

# Step 2: Create 6-week chunks with MIXED training
print("Step 2: Creating 6-week chunks with mixed training set...")
selector = ChunkSelector(prices, vix, chunk_days=CHUNK_DAYS)
all_chunks = selector.create_chunks()

# Sort by VIX
all_chunks_sorted = sorted(all_chunks, key=lambda x: x.vix_mean)
n_total = len(all_chunks_sorted)

print(f"Created {n_total} chunks of {CHUNK_DAYS} days each\n")

# Training: Select MIXED VIX levels (every 2nd chunk to get variety)
# This ensures we see low, medium, and high VIX
train_indices = list(range(0, n_total, 2))  # Every other chunk
train_chunks = [all_chunks_sorted[i] for i in train_indices]

# Testing: Top HIGH-VIX chunks (where contagion matters)
test_chunks = sorted(all_chunks_sorted[-8:], key=lambda x: x.start_date)

print(f"Training set: {len(train_chunks)} chunks")
print(f"  VIX range: {min(c.vix_mean for c in train_chunks):.1f} - {max(c.vix_mean for c in train_chunks):.1f}")
print(f"  Total training days: {len(train_chunks) * CHUNK_DAYS}")
print(f"\nTest set: {len(test_chunks)} HIGH-VIX chunks")
print(f"  VIX range: {min(c.vix_mean for c in test_chunks):.1f} - {max(c.vix_mean for c in test_chunks):.1f}")
print(f"  Total test days: {len(test_chunks) * CHUNK_DAYS}\n")

# Step 3: Train models
print("Step 3: Training models...")

print("  Training SIR Contagion Model...")
sir_model = SIRContagionModel(TICKERS)
sir_model.fit(train_chunks)

print("\n  Training Single Random Walk...")
single_rw = SingleRandomWalkModel(TICKERS)
single_rw.fit(train_chunks)

print("\n  Training Independent HMM...")
ind_hmm = IndependentHMMModel(TICKERS)
ind_hmm.fit(train_chunks)

models = [sir_model, single_rw, ind_hmm]
print("\n✓ All models trained\n")

# Step 4: Backtest on high-VIX periods
print("Step 4: Testing on HIGH-VIX periods...")
backtester = Backtester(models, n_simulations=N_SIMULATIONS)
results = backtester.run_backtest(test_chunks[:3], verbose=True)  # Test on 3 chunks

# Step 5: Results
print("\n" + "="*80)
print("OPTIMAL RESULTS - 6-Week Chunks + Mixed Training")
print("="*80)
backtester.print_summary()

# Compare to previous results
print("\n" + "="*80)
print("COMPARISON TO PREVIOUS APPROACHES")
print("="*80)
print("\nPrevious Results:")
print("  6-Week (High-VIX only):")
print("    - SIR MSE: 172.93 (WINS)")
print("    - But β not learned")
print("\n  Mixed Periods (20-day):")
print("    - SIR MSE: 139.33 (LOSES)")
print("    - But β learned (0.64)")
print("\n  Current (6-Week + Mixed):")
sir_mse = results['SIR Contagion Model']['aggregate_metrics'].get('mse_mean', 'N/A')
print(f"    - SIR MSE: {sir_mse:.2f}")
print("    - Check if β learned AND wins!")

print("\n" + "="*80)
print("✓✓✓ OPTIMAL TEST COMPLETE ✓✓✓")
print("="*80)
print("This combination should give:")
print("  1. Better state learning (mixed VIX)")
print("  2. Better predictions (6-week time scale)")
print("  3. Learned contagion (β) that actually helps!")
print("="*80 + "\n")

