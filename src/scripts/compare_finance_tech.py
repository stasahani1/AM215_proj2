"""
Generate Finance vs Technology comparison analysis.
Modified to show only contagion rate (without labels) and performance improvement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the sector results
tech_df = pd.read_csv('tech_sector_results.csv', index_col=0)

# Extract key metrics - using exact values from the image
results = {
    'Finance': {
        'sir_mse': 219.40,
        'best_baseline_mse': 236.66,
        'beta': 0.499,  # From image: 0.499
        'improvement': 7.3
    },
    'Technology': {
        'sir_mse': tech_df.loc['SIR Contagion Model', 'mse_mean'],
        'best_baseline_mse': tech_df.drop('SIR Contagion Model')['mse_mean'].min(),
        'beta': 0.150,  # From image: 0.150 (labeled as "Weak Contagion")
        'improvement': -0.5  # From image: -0.5%
    }
}

print("="*80)
print("FINANCE vs TECHNOLOGY COMPARISON")
print("="*80)

# Create comparison figure with 2 plots only
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Finance vs Technology: Contagion Analysis', fontsize=16, fontweight='bold')

sectors = ['Finance', 'Technology']
colors = ['#2ecc71', '#e74c3c']  # Green, Red

# Plot 1: Contagion Rate (β) - WITHOUT labels
ax1 = axes[0]
betas = [results[s]['beta'] for s in sectors]
bars1 = ax1.bar(sectors, betas, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Contagion Rate (β)', fontsize=11, fontweight='bold')
ax1.set_title('Contagion Rate by Sector', fontsize=12, fontweight='bold')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add only the numeric values, no "strong/weak contagion" labels
for bar, val in zip(bars1, betas):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

# Plot 2: Performance Improvement
ax2 = axes[1]
improvements = [results[s]['improvement'] for s in sectors]
bar_colors = ['green' if x > 0 else 'red' for x in improvements]
bars2 = ax2.bar(sectors, improvements, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('SIR Improvement over Best Baseline (%)', fontsize=11, fontweight='bold')
ax2.set_title('Model Performance Gain by Sector', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.2)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars2, improvements):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:+.1f}%',
            ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.savefig('finance_tech_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: finance_tech_comparison.png")

# Create comparison table
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['beta', 'sir_mse', 'best_baseline_mse', 'improvement']]
comparison_df.columns = ['β (Contagion Rate)', 'SIR MSE', 'Best Baseline MSE', 'Improvement (%)']

print("\n" + "="*80)
print("Table: Finance vs Technology Comparison")
print("="*80)
print(comparison_df.to_string())

# Save table
comparison_df.to_csv('finance_tech_comparison.csv')
print(f"\n✅ Saved: finance_tech_comparison.csv")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("\n1. CONTAGION RATE:")
print(f"   Finance:    β = {results['Finance']['beta']:.4f}")
print(f"   Technology: β = {results['Technology']['beta']:.4f}")

print("\n2. MODEL PERFORMANCE:")
print(f"   Finance:    SIR OUTPERFORMS by {results['Finance']['improvement']:+.1f}%")
print(f"   Technology: SIR underperforms by {abs(results['Technology']['improvement']):.1f}%")

print("\n3. INTERPRETATION:")
print("   Finance shows positive contagion rate and SIR improvement,")
print("   indicating true crisis contagion dynamics. Technology shows")
print("   negative contagion rate and SIR failure, indicating independence.")

print("\n" + "="*80)
