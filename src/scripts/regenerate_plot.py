"""Regenerate the window size comparison plot with updated styling."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the results
comparison_df = pd.read_csv('window_size_comparison_results.csv')

WINDOW_SIZES = [
    ('2-week', 10),
    ('6-week', 30),
    ('12-week', 60)
]

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

# Calculate improvements
improvements = []
improvement_errors = []
window_labels = []

for window_name in [w[0] for w in WINDOW_SIZES]:
    window_data = comparison_df[comparison_df['window'] == window_name]

    # Get SIR MSE
    sir_data = window_data[window_data['model'] == 'SIR Contagion Model']
    if sir_data.empty:
        continue
    sir_mse = sir_data['mse_mean'].values[0]
    sir_sem = sir_data['mse_sem'].values[0]

    # Get best baseline MSE
    baseline_data = window_data[window_data['model'] != 'SIR Contagion Model']
    best_baseline_mse = baseline_data['mse_mean'].min()

    # Calculate improvement
    improvement = (best_baseline_mse - sir_mse) / best_baseline_mse * 100

    # For error, use propagation of uncertainty (simplified)
    improvement_error = (sir_sem / best_baseline_mse) * 100

    improvements.append(improvement)
    improvement_errors.append(improvement_error)
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
print("✅ Regenerated plot: window_size_comparison.png")
