"""
Quick test training on MIXED periods (calm + crisis) for better state learning.
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
print("QUICK TEST - Train on Mixed Periods (Calm + Crisis)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

TICKERS = ['AAPL', 'MSFT', 'JPM']
YEARS_BACK = 5
N_SIMULATIONS = 50

print(f"Configuration: {len(TICKERS)} stocks, {YEARS_BACK} years")
print("Strategy: Train on ALL periods, test on HIGH-VIX periods\n")

# Step 1: Load data
print("Step 1: Loading data...")
loader = DataLoader(TICKERS, years_back=YEARS_BACK)
prices, vix = loader.load_data()
print(f"✓ Loaded {len(prices)} days\n")

# Step 2: Create chunks differently
print("Step 2: Creating mixed training set...")
selector = ChunkSelector(prices, vix, chunk_days=20)  # Longer chunks
all_chunks = selector.create_chunks()

# Select chunks with MIXED volatility for training
all_chunks_sorted = sorted(all_chunks, key=lambda x: x.vix_mean)
n_total = len(all_chunks_sorted)

# Training: Mix of low, medium, high VIX (to learn all regimes)
train_indices = list(range(0, n_total, 3))  # Every 3rd chunk (mixed VIX)
train_chunks = [all_chunks_sorted[i] for i in train_indices[:20]]

# Testing: Only HIGH VIX periods (where contagion matters)
test_chunks = sorted(all_chunks_sorted[-10:], key=lambda x: x.start_date)

print(f"✓ Training: {len(train_chunks)} mixed-VIX chunks")
print(f"  VIX range: {min(c.vix_mean for c in train_chunks):.1f} - {max(c.vix_mean for c in train_chunks):.1f}")
print(f"✓ Testing: {len(test_chunks)} high-VIX chunks")
print(f"  VIX range: {min(c.vix_mean for c in test_chunks):.1f} - {max(c.vix_mean for c in test_chunks):.1f}\n")

# Step 3: Train models
print("Step 3: Training models on MIXED periods...")

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

# Step 4: Test on HIGH-VIX periods
print("Step 4: Testing on HIGH-VIX periods (where contagion matters)...")
backtester = Backtester(models, n_simulations=N_SIMULATIONS)
results = backtester.run_backtest(test_chunks[:2], verbose=True)

# Step 5: Results
print("\n" + "="*80)
print("RESULTS - Training on Mixed, Testing on Crisis")
print("="*80)
backtester.print_summary()

print("\n" + "="*80)
print("✓✓✓ TEST COMPLETE ✓✓✓")
print("="*80)
print("Training on mixed periods should help learn distinct regimes!")
print("="*80 + "\n")

