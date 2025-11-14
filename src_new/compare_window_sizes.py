"""
Comprehensive comparison of different time window sizes:
- 2 weeks (10 days)
- 6 weeks (30 days)
- 12 weeks (60 days)

Tests each window size with proper statistical analysis including error bars.
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB'
]

WINDOW_SIZES = [
    ('2-week', 10),
    ('6-week', 30),
    ('12-week', 60)
]

YEARS_BACK = 15
N_CHUNKS = 30
TRAIN_RATIO = 0.6
N_SIMULATIONS = 200

print("="*80)
print("WINDOW SIZE COMPARISON STUDY")
print("="*80)
print(f"\nTesting window sizes: {[w[0] for w in WINDOW_SIZES]}")
print(f"Stocks: {len(FINANCE_STOCKS)} financial institutions")
print(f"Period: {YEARS_BACK} years (2010-2025)")
print(f"Simulations per test: {N_SIMULATIONS}")
print("="*80)

# Load data once
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)
loader = DataLoader(FINANCE_STOCKS, years_back=YEARS_BACK)
prices, vix = loader.load_data()

if prices.empty:
    print("ERROR: No data loaded")
    sys.exit(1)

print(f"\nLoaded {len(prices)} days of data for {len(prices.columns)} stocks")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Store results for each window size
all_results = {}

# Test each window size
for window_name, chunk_days in WINDOW_SIZES:
    print("\n" + "="*80)
    print(f"TESTING WINDOW SIZE: {window_name} ({chunk_days} days)")
    print("="*80)

    start_time = time.time()

    # Create chunks for this window size
    print(f"\nCreating {chunk_days}-day chunks...")
    chunk_selector = ChunkSelector(prices, vix, chunk_days=chunk_days)
    chunks = chunk_selector.create_chunks()

    # Sort and split
    chunks_sorted = sorted(chunks, key=lambda c: c.vix_mean, reverse=True)
    n_train = int(N_CHUNKS * TRAIN_RATIO)
    n_test = N_CHUNKS - n_train

    # Training: Mix of all VIX levels
    all_chunks_by_time = sorted(chunks, key=lambda c: c.start_date)
    step = max(1, len(all_chunks_by_time) // n_train)
    train_chunks = all_chunks_by_time[::step][:n_train]

    # Testing: High VIX only
    test_chunks = chunks_sorted[:n_test]

    print(f"Training chunks: {len(train_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")
    train_vix = [c.vix_mean for c in train_chunks]
    test_vix = [c.vix_mean for c in test_chunks]
    print(f"  Train VIX: {min(train_vix):.1f} - {max(train_vix):.1f}")
    print(f"  Test VIX:  {min(test_vix):.1f} - {max(test_vix):.1f}")

    # Initialize models
    print(f"\nInitializing models for {window_name}...")
    models = {
        'SIR Contagion Model': SIRContagionModel(FINANCE_STOCKS),
        'Independent HMM': IndependentHMMModel(FINANCE_STOCKS),
        'Single Random Walk': SingleRandomWalkModel(FINANCE_STOCKS),
        'Correlated Random Walk': CorrelatedRandomWalkModel(FINANCE_STOCKS)
    }

    # Train models
    print(f"\nTraining models for {window_name}...")
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(train_chunks, correlation_threshold=0.6)

    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.1f}s")

    # Print SIR parameters
    sir_model = models['SIR Contagion Model']
    if sir_model.is_trained:
        params = sir_model.get_params()
        print(f"\nSIR Contagion Parameters ({window_name}):")
        print(f"  β (contagion rate):    {params['contagion_params']['beta']:.4f}")
        print(f"  γ (recovery rate):     {params['contagion_params']['gamma']:.4f}")
        print(f"  α (re-susceptibility): {params['contagion_params']['alpha']:.4f}")

    # Backtest
    print(f"\nBacktesting {window_name}...")
    backtester = Backtester(list(models.values()), n_simulations=N_SIMULATIONS)
    results = backtester.run_backtest(test_chunks, verbose=False)

    # Store results
    all_results[window_name] = {
        'chunk_days': chunk_days,
        'backtester': backtester,
        'results': results,
        'train_time': train_time,
        'n_train': len(train_chunks),
        'n_test': len(test_chunks)
    }

    total_time = time.time() - start_time
    print(f"\n{window_name} complete! Total time: {total_time:.1f}s")
    print("="*80)

# Analyze and compare results
print("\n" + "="*80)
print("COMPARATIVE ANALYSIS")
print("="*80)

# Extract metrics for comparison
comparison_data = []

for window_name, window_results in all_results.items():
    backtester = window_results['backtester']

    for model in backtester.models:
        model_results = backtester.results[model.name]
        chunk_metrics = model_results['chunk_metrics']

        # Extract MSE for each chunk
        mse_values = [m.get('mse', np.nan) for m in chunk_metrics]
        mse_values = [v for v in mse_values if not np.isnan(v)]

        if mse_values:
            comparison_data.append({
                'window': window_name,
                'model': model.name,
                'mse_mean': np.mean(mse_values),
                'mse_std': np.std(mse_values),
                'mse_sem': np.std(mse_values) / np.sqrt(len(mse_values)),  # Standard error
                'n_chunks': len(mse_values)
            })

comparison_df = pd.DataFrame(comparison_data)

# Print results table
print("\n" + "="*80)
print("RESULTS BY WINDOW SIZE")
print("="*80)

for window_name in [w[0] for w in WINDOW_SIZES]:
    print(f"\n{window_name.upper()}:")
    window_data = comparison_df[comparison_df['window'] == window_name]
    window_data_sorted = window_data.sort_values('mse_mean')

    print(f"{'Model':<30} {'MSE Mean':<12} {'MSE Std':<12} {'Rank':<6}")
    print("-" * 80)
    for idx, (_, row) in enumerate(window_data_sorted.iterrows(), 1):
        print(f"{row['model']:<30} {row['mse_mean']:<12.2f} {row['mse_std']:<12.2f} {idx:<6}")

# Statistical significance test
print("\n" + "="*80)
print("STATISTICAL ANALYSIS: SIR vs Best Baseline")
print("="*80)

for window_name, window_results in all_results.items():
    print(f"\n{window_name.upper()}:")

    backtester = window_results['backtester']

    # Get SIR MSE values
    sir_metrics = backtester.results['SIR Contagion Model']['chunk_metrics']
    sir_mse = [m.get('mse', np.nan) for m in sir_metrics]
    sir_mse = [v for v in sir_mse if not np.isnan(v)]

    # Get best baseline
    baseline_names = ['Independent HMM', 'Single Random Walk', 'Correlated Random Walk']
    best_baseline_mse = float('inf')
    best_baseline_name = None
    best_baseline_values = None

    for baseline_name in baseline_names:
        if baseline_name in backtester.results:
            baseline_metrics = backtester.results[baseline_name]['chunk_metrics']
            baseline_mse = [m.get('mse', np.nan) for m in baseline_metrics]
            baseline_mse = [v for v in baseline_mse if not np.isnan(v)]

            if baseline_mse and np.mean(baseline_mse) < best_baseline_mse:
                best_baseline_mse = np.mean(baseline_mse)
                best_baseline_name = baseline_name
                best_baseline_values = baseline_mse

    if sir_mse and best_baseline_values:
        # Paired t-test (each chunk is a paired observation)
        t_stat, p_value = stats.ttest_rel(sir_mse, best_baseline_values)

        sir_mean = np.mean(sir_mse)
        baseline_mean = np.mean(best_baseline_values)
        improvement = (baseline_mean - sir_mean) / baseline_mean * 100

        print(f"  SIR Contagion:        {sir_mean:.2f} ± {np.std(sir_mse):.2f}")
        print(f"  Best Baseline ({best_baseline_name}): {baseline_mean:.2f} ± {np.std(best_baseline_values):.2f}")
        print(f"  Improvement:          {improvement:+.1f}%")
        print(f"  t-statistic:          {t_stat:.3f}")
        print(f"  p-value:              {p_value:.4f}")

        if p_value < 0.05:
            if improvement > 0:
                print(f"  ✅ SIR significantly BETTER (p < 0.05)")
            else:
                print(f"  ❌ SIR significantly WORSE (p < 0.05)")
        else:
            print(f"  ⚠️  No significant difference (p >= 0.05)")

# Create visualization
print("\n" + "="*80)
print("GENERATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: MSE by window size with error bars
ax1 = axes[0]
models_to_plot = ['SIR Contagion Model', 'Independent HMM', 'Correlated Random Walk', 'Single Random Walk']
colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']

x_positions = np.arange(len(WINDOW_SIZES))
width = 0.2

for idx, model_name in enumerate(models_to_plot):
    model_data = comparison_df[comparison_df['model'] == model_name]

    means = []
    sems = []
    for window_name in [w[0] for w in WINDOW_SIZES]:
        row = model_data[model_data['window'] == window_name]
        if not row.empty:
            means.append(row['mse_mean'].values[0])
            sems.append(row['mse_sem'].values[0])
        else:
            means.append(0)
            sems.append(0)

    offset = (idx - 1.5) * width
    ax1.bar(x_positions + offset, means, width, yerr=sems,
            label=model_name, color=colors[idx], alpha=0.8, capsize=5)

ax1.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance by Window Size', fontsize=14, fontweight='bold')
ax1.set_xticks(x_positions)
ax1.set_xticklabels([w[0] for w in WINDOW_SIZES])
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: SIR improvement over best baseline
ax2 = axes[1]

improvements = []
improvement_errors = []
window_labels = []

for window_name, window_results in all_results.items():
    backtester = window_results['backtester']

    # Get SIR MSE
    sir_metrics = backtester.results['SIR Contagion Model']['chunk_metrics']
    sir_mse = [m.get('mse', np.nan) for m in sir_metrics]
    sir_mse = [v for v in sir_mse if not np.isnan(v)]

    # Get best baseline MSE
    baseline_names = ['Independent HMM', 'Single Random Walk', 'Correlated Random Walk']
    best_baseline_mse = float('inf')
    best_baseline_values = None

    for baseline_name in baseline_names:
        if baseline_name in backtester.results:
            baseline_metrics = backtester.results[baseline_name]['chunk_metrics']
            baseline_mse = [m.get('mse', np.nan) for m in baseline_metrics]
            baseline_mse = [v for v in baseline_mse if not np.isnan(v)]

            if baseline_mse and np.mean(baseline_mse) < best_baseline_mse:
                best_baseline_mse = np.mean(baseline_mse)
                best_baseline_values = baseline_mse

    if sir_mse and best_baseline_values:
        # Calculate improvement for each chunk
        chunk_improvements = [(b - s) / b * 100 for s, b in zip(sir_mse, best_baseline_values)]

        improvements.append(np.mean(chunk_improvements))
        improvement_errors.append(np.std(chunk_improvements) / np.sqrt(len(chunk_improvements)))
        window_labels.append(window_name)

bars = ax2.bar(window_labels, improvements, yerr=improvement_errors,
               color=['#2ecc71' if i > 0 else '#e74c3c' for i in improvements],
               alpha=0.8, capsize=5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Window Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('Improvement over Best Baseline (%)', fontsize=12, fontweight='bold')
ax2.set_title('SIR Contagion Model Performance Gain',
              fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val, err in zip(bars, improvements, improvement_errors):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%\n±{err:.1f}%',
             ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.savefig('window_size_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization: window_size_comparison.png")

# Export detailed results
print("\nExporting detailed results...")
comparison_df.to_csv('window_size_comparison_results.csv', index=False)
print("✅ Saved results: window_size_comparison_results.csv")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

best_window = None
best_improvement = -float('inf')

for window_name in [w[0] for w in WINDOW_SIZES]:
    window_data = comparison_df[comparison_df['window'] == window_name]
    sir_data = window_data[window_data['model'] == 'SIR Contagion Model']
    baseline_data = window_data[window_data['model'] != 'SIR Contagion Model']

    if not sir_data.empty and not baseline_data.empty:
        sir_mse = sir_data['mse_mean'].values[0]
        best_baseline_mse = baseline_data['mse_mean'].min()
        improvement = (best_baseline_mse - sir_mse) / best_baseline_mse * 100

        print(f"\n{window_name}:")
        print(f"  SIR MSE:               {sir_mse:.2f}")
        print(f"  Best Baseline MSE:     {best_baseline_mse:.2f}")
        print(f"  Improvement:           {improvement:+.1f}%")

        if improvement > best_improvement:
            best_improvement = improvement
            best_window = window_name

print(f"\n{'='*80}")
print(f"BEST WINDOW SIZE: {best_window}")
print(f"Improvement: {best_improvement:+.1f}%")
print(f"{'='*80}\n")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nFiles generated:")
print("  1. window_size_comparison.png - Visualization")
print("  2. window_size_comparison_results.csv - Detailed results")
print("="*80)
