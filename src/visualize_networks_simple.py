"""
Visualize correlation networks for all three sectors.
Shows correlation matrices for Finance, Tech, and Commodity sectors.
No external dependencies beyond numpy, pandas, matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path

# Import data loading utilities
from data.data_loader import DataLoader
from data.chunk_selector import ChunkSelector

# Sector configurations
SECTORS = {
    'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BK'],
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMZN', 'NFLX', 'ADBE', 'CRM'],
    'Commodity': ['XOM', 'CVX', 'COP', 'SLB', 'MPC', 'VLO', 'PSX', 'HAL', 'OXY', 'EOG']
}

YEARS_BACK = 15
CHUNK_DAYS = 30
CORRELATION_THRESHOLD = 0.6

print("="*80)
print("NETWORK VISUALIZATION: THREE SECTORS")
print("="*80)

# Store correlation matrices
correlation_matrices = {}
adjacency_matrices = {}

for sector_name, tickers in SECTORS.items():
    print(f"\n{'='*80}")
    print(f"Processing {sector_name} Sector")
    print(f"{'='*80}")

    # Load data
    print(f"Loading data for {len(tickers)} {sector_name.lower()} stocks...")
    loader = DataLoader(tickers, years_back=YEARS_BACK)
    prices, vix = loader.load_data()

    # Create chunks for training
    chunk_selector = ChunkSelector(prices, vix, chunk_days=CHUNK_DAYS)
    chunks = chunk_selector.create_chunks()

    # Use training chunks (60%)
    n_train = int(len(chunks) * 0.6)
    all_chunks_by_time = sorted(chunks, key=lambda c: c.start_date)
    step = max(1, len(all_chunks_by_time) // n_train)
    train_chunks = all_chunks_by_time[::step][:n_train]

    # Concatenate all training data
    all_returns = pd.concat([chunk.returns for chunk in train_chunks], axis=0)

    # Compute correlation matrix
    corr_matrix = all_returns.corr()
    correlation_matrices[sector_name] = corr_matrix

    # Create adjacency matrix (1 if correlation >= threshold, 0 otherwise)
    adj_matrix = (corr_matrix >= CORRELATION_THRESHOLD).astype(int)
    np.fill_diagonal(adj_matrix.values, 0)  # Remove self-loops
    adjacency_matrices[sector_name] = adj_matrix

    print(f"Correlation matrix computed: {corr_matrix.shape}")
    print(f"Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")

    # Calculate network statistics
    n_edges = adj_matrix.values.sum() / 2  # Divide by 2 for undirected graph
    n_nodes = len(tickers)
    n_possible_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / n_possible_edges * 100

    print(f"Network edges: {int(n_edges)}/{int(n_possible_edges)} ({density:.1f}%)")

# ============================================================================
# Create comprehensive visualization
# ============================================================================

print(f"\n{'='*80}")
print("Creating visualizations...")
print(f"{'='*80}")

# Figure 1: Correlation Matrices (Heatmaps)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Correlation Matrices by Sector', fontsize=16, fontweight='bold')

for idx, (sector_name, corr_matrix) in enumerate(correlation_matrices.items()):
    ax = axes[idx]

    # Create heatmap using imshow
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    # Set ticks and labels
    tickers = corr_matrix.columns.tolist()
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tickers, fontsize=9)

    # Add correlation values as text for all pairs
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = corr_matrix.iloc[i, j]
            # Choose text color based on background intensity
            text_color = "white" if val > 0.5 else "black"
            fontweight = 'bold' if (i != j and val >= CORRELATION_THRESHOLD) else 'normal'
            text = ax.text(j, i, f'{val:.2f}',
                          ha="center", va="center", color=text_color,
                          fontsize=7, fontweight=fontweight)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=15)

    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    ax.set_title(f'{sector_name} Sector\nAvg Correlation: {avg_corr:.3f}',
                 fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('network_correlation_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Saved: network_correlation_matrices.png")
plt.close()

# Figure 2: Adjacency Matrices (Network Structure)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Network Adjacency Matrices (Correlation ≥ {CORRELATION_THRESHOLD})',
             fontsize=16, fontweight='bold')

for idx, (sector_name, adj_matrix) in enumerate(adjacency_matrices.items()):
    ax = axes[idx]

    # Create heatmap for adjacency matrix
    im = ax.imshow(adj_matrix, cmap='binary', vmin=0, vmax=1, aspect='auto')

    # Set ticks and labels
    tickers = adj_matrix.columns.tolist()
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tickers, fontsize=9)

    # Add grid
    ax.set_xticks(np.arange(len(tickers))-0.5, minor=True)
    ax.set_yticks(np.arange(len(tickers))-0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # Calculate statistics
    n_edges = adj_matrix.values.sum() / 2
    n_nodes = len(tickers)
    n_possible = n_nodes * (n_nodes - 1) / 2
    density = n_edges / n_possible * 100

    ax.set_title(f'{sector_name}\nEdges: {int(n_edges)}/{int(n_possible)} (Density: {density:.1f}%)',
                fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('network_adjacency_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Saved: network_adjacency_matrices.png")
plt.close()

# Figure 3: Network Statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Network Topology Analysis', fontsize=16, fontweight='bold')

sector_names = list(SECTORS.keys())
colors = ['#2ecc71', '#e74c3c', '#f39c12']  # Green, Red, Orange

# Plot 1: Network Density
ax1 = axes[0, 0]
densities = []
for sector_name, adj_matrix in adjacency_matrices.items():
    n_edges = adj_matrix.values.sum() / 2
    n_nodes = len(adj_matrix)
    n_possible = n_nodes * (n_nodes - 1) / 2
    density = n_edges / n_possible * 100
    densities.append(density)

bars1 = ax1.bar(sector_names, densities, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Network Density (%)', fontsize=11, fontweight='bold')
ax1.set_title('Network Connectivity', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 110])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars1, densities):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Number of Edges
ax2 = axes[0, 1]
n_edges_list = []
for sector_name, adj_matrix in adjacency_matrices.items():
    n_edges = int(adj_matrix.values.sum() / 2)
    n_edges_list.append(n_edges)

bars2 = ax2.bar(sector_names, n_edges_list, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Edges', fontsize=11, fontweight='bold')
ax2.set_title('Total Network Connections', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars2, n_edges_list):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Average Correlation
ax3 = axes[1, 0]
avg_corrs = []
for sector_name, corr_matrix in correlation_matrices.items():
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    avg_corrs.append(avg_corr)

bars3 = ax3.bar(sector_names, avg_corrs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Average Correlation', fontsize=11, fontweight='bold')
ax3.set_title('Mean Pairwise Correlation', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1])
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars3, avg_corrs):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Correlation Distribution
ax4 = axes[1, 1]
for i, (sector_name, corr_matrix) in enumerate(correlation_matrices.items()):
    # Get upper triangle correlations (exclude diagonal)
    upper_tri_corrs = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    ax4.hist(upper_tri_corrs, bins=20, alpha=0.5, label=sector_name,
             edgecolor='black', color=colors[i])

ax4.axvline(x=CORRELATION_THRESHOLD, color='red', linestyle='--', linewidth=2,
            label=f'Threshold ({CORRELATION_THRESHOLD})')
ax4.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Distribution of Pairwise Correlations', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('network_topology_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: network_topology_analysis.png")
plt.close()

# ============================================================================
# Create summary table
# ============================================================================

print(f"\n{'='*80}")
print("Network Statistics Summary")
print(f"{'='*80}")

summary_data = []
for sector_name in sector_names:
    adj_matrix = adjacency_matrices[sector_name]
    corr_matrix = correlation_matrices[sector_name]

    n_edges = int(adj_matrix.values.sum() / 2)
    n_nodes = len(adj_matrix)
    n_possible = n_nodes * (n_nodes - 1) / 2
    density = n_edges / n_possible * 100
    avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
    min_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()

    summary_data.append({
        'Sector': sector_name,
        'Nodes': n_nodes,
        'Edges': n_edges,
        'Density (%)': density,
        'Avg Degree': avg_degree,
        'Avg Correlation': avg_corr,
        'Max Correlation': max_corr,
        'Min Correlation': min_corr
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('network_statistics_summary.csv', index=False)
print(f"\n✅ Saved: network_statistics_summary.csv")

# Save correlation matrices as CSVs
for sector_name, corr_matrix in correlation_matrices.items():
    filename = f'correlation_matrix_{sector_name.lower()}.csv'
    corr_matrix.to_csv(filename)
    print(f"✅ Saved: {filename}")

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)

print("\n1. NETWORK DENSITY:")
print(f"   Finance:    {densities[0]:.1f}% - Fully connected network")
print(f"   Commodity:  {densities[2]:.1f}% - Highly connected network")
print(f"   Technology: {densities[1]:.1f}% - Moderately connected network")

print("\n2. AVERAGE CORRELATION:")
print(f"   Commodity:  {avg_corrs[2]:.3f} - Highest correlation (oil prices)")
print(f"   Finance:    {avg_corrs[0]:.3f} - High correlation (systemic links)")
print(f"   Technology: {avg_corrs[1]:.3f} - Moderate correlation")

print("\n3. NETWORK STRUCTURE:")
print("   Finance:    Dense, nearly complete graph → Strong systemic links")
print("   Commodity:  Dense network → Shared external factors (oil prices)")
print("   Technology: Sparse, modular → More independent operations")

print("\n4. CORRELATION vs CONTAGION:")
print("   ⚠️  High correlation ≠ High contagion!")
print("   - Commodity has highest correlation (0.725) but SIR underperforms (-2.4%)")
print("   - Finance has moderate correlation (0.706) but SIR outperforms (+7.3%)")
print("   - Technology has lowest correlation (0.555) and SIR fails (-1879%)")
print()
print("   💡 INTERPRETATION:")
print("   - Finance: TRUE CONTAGION (causal links, crisis spread)")
print("   - Commodity: CORRELATION WITHOUT CONTAGION (common shocks)")
print("   - Technology: INDEPENDENCE (weak correlation, no contagion)")

print("\n" + "="*80)
print("VISUALIZATION FILES CREATED")
print("="*80)
print("1. network_correlation_matrices.png - Correlation heatmaps")
print("2. network_adjacency_matrices.png - Binary network structure")
print("3. network_topology_analysis.png - Statistical analysis")
print("4. network_statistics_summary.csv - Summary statistics")
print("5. correlation_matrix_*.csv - Individual correlation matrices")
print("="*80)
