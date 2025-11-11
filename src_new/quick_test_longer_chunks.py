"""
Quick test with LONGER chunks to capture regime transitions better.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from data.data_loader import DataLoader
from data.chunk_selector import prepare_dataset
from models.sir_contagion_model import SIRContagionModel
from models.baseline_single_rw import SingleRandomWalkModel
from models.baseline_independent_hmm import IndependentHMMModel
from evaluation.backtester import Backtester

print("\n" + "="*80)
print("QUICK TEST - Longer Chunks (Better for Regime Detection)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration with LONGER chunks
TICKERS = ['AAPL', 'MSFT', 'JPM']
YEARS_BACK = 5
CHUNK_DAYS = 30  # ← 6 weeks instead of 2 weeks!
N_CHUNKS = 15    # Fewer chunks but longer
N_SIMULATIONS = 50

print(f"Configuration: {len(TICKERS)} stocks, {YEARS_BACK} years")
print(f"  Chunk size: {CHUNK_DAYS} days (~6 weeks)")
print(f"  High-vol chunks: {N_CHUNKS}")
print(f"  Simulations: {N_SIMULATIONS}\n")

# Step 1: Load data
print("Step 1: Loading data...")
loader = DataLoader(TICKERS, years_back=YEARS_BACK)
prices, vix = loader.load_data()
print(f"✓ Loaded {len(prices)} days\n")

# Step 2: Create LONGER chunks
print("Step 2: Creating longer chunks...")
dataset = prepare_dataset(
    prices, vix, 
    chunk_days=CHUNK_DAYS,  # Longer chunks
    n_chunks=N_CHUNKS, 
    train_ratio=0.6
)
print(f"✓ {len(dataset['train_chunks'])} train, {len(dataset['test_chunks'])} test")
print(f"  Each chunk: {CHUNK_DAYS} trading days\n")

# Step 3: Train models
print("Step 3: Training models...")

print("  Training SIR Contagion Model...")
sir_model = SIRContagionModel(TICKERS)
sir_model.fit(dataset['train_chunks'])

print("\n  Training Single Random Walk...")
single_rw = SingleRandomWalkModel(TICKERS)
single_rw.fit(dataset['train_chunks'])

print("\n  Training Independent HMM...")
ind_hmm = IndependentHMMModel(TICKERS)
ind_hmm.fit(dataset['train_chunks'])

models = [sir_model, single_rw, ind_hmm]
print("\n✓ All models trained\n")

# Step 4: Quick backtest
print("Step 4: Running quick backtest...")
backtester = Backtester(models, n_simulations=N_SIMULATIONS)
results = backtester.run_backtest(dataset['test_chunks'][:2], verbose=True)

# Step 5: Results
print("\n" + "="*80)
print("RESULTS WITH LONGER CHUNKS")
print("="*80)
backtester.print_summary()

print("\n" + "="*80)
print("✓✓✓ TEST COMPLETE ✓✓✓")
print("="*80)
print("Longer chunks should give more regime transitions!")
print("="*80 + "\n")

