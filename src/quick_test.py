"""
Quick test script to verify the framework works before running full evaluation.
This tests with minimal data (3 stocks, 5 years, 20 chunks, 50 simulations).
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add paths
sys.path.append(os.path.dirname(__file__))

# Import modules
from data.data_loader import DataLoader
from data.chunk_selector import prepare_dataset
from models.sir_contagion_model import SIRContagionModel
from models.baseline_single_rw import SingleRandomWalkModel
from models.baseline_independent_hmm import IndependentHMMModel
from evaluation.backtester import Backtester

print("\n" + "="*80)
print("QUICK TEST - SIR Contagion Model Framework")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Minimal configuration for quick testing
TICKERS = ['AAPL', 'MSFT', 'JPM']
YEARS_BACK = 5
N_CHUNKS = 20
N_SIMULATIONS = 50

print(f"Configuration: {len(TICKERS)} stocks, {YEARS_BACK} years, {N_CHUNKS} chunks, {N_SIMULATIONS} sims\n")

# Step 1: Load data
print("Step 1: Loading data...")
loader = DataLoader(TICKERS, years_back=YEARS_BACK)
prices, vix = loader.load_data()
print(f"✓ Loaded {len(prices)} days\n")

# Step 2: Create chunks
print("Step 2: Creating chunks...")
dataset = prepare_dataset(prices, vix, chunk_days=10, n_chunks=N_CHUNKS, train_ratio=0.6)
print(f"✓ {len(dataset['train_chunks'])} train, {len(dataset['test_chunks'])} test\n")

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

# Step 4: Quick backtest on first 2 test chunks
print("Step 4: Running quick backtest...")
backtester = Backtester(models, n_simulations=N_SIMULATIONS)
results = backtester.run_backtest(dataset['test_chunks'][:2], verbose=True)

# Step 5: Results
print("\n" + "="*80)
print("QUICK TEST RESULTS")
print("="*80)
backtester.print_summary()

print("\n" + "="*80)
print("✓✓✓ QUICK TEST COMPLETE ✓✓✓")
print("="*80)
print("The framework is working correctly!")
print("\nNote: This quick test only shows numerical results.")
print("To generate graphs and full visualizations, run:")
print("  python run_evaluation.py")
print("="*80 + "\n")

