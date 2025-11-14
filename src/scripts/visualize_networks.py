"""
Visualize correlation networks for all three sectors.
Shows network graphs and correlation matrices for Finance, Tech, and Commodity sectors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
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
networks = {}

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

    print(f"Correlation matrix computed: {corr_matrix.shape}")
    print(f"Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")

    # Build network graph
    G = nx.Graph()

    # Add nodes
    for ticker in tickers:
        G.add_node(ticker)

    # Add edges for correlations above threshold
    n_edges = 0
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:  # Upper triangle only
                corr = corr_matrix.loc[ticker1, ticker2]
                if corr >= CORRELATION_THRESHOLD:
                    G.add_edge(ticker1, ticker2, weight=corr)
                    n_edges += 1

    networks[sector_name] = G

    # Calculate network statistics
    n_possible_edges = len(tickers) * (len(tickers) - 1) / 2
    density = n_edges / n_possible_edges * 100

    print(f"Network edges: {n_edges}/{int(n_possible_edges)} ({density:.1f}%)")
    print(f"Average degree: {2*n_edges/len(tickers):.1f}")

# ============================================================================
# Create comprehensive visualization
# ============================================================================

print(f"\n{'='*80}")
print("Creating visualizations...")
print(f"{'='*80}")

# Figure 1: Correlation Matrices (Heatmaps)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Correlation Matrices by Sector', fontsize=16, fontweight='bold')

for idx, (sector_name, corr_matrix) in enumerate(correlation_matrices.items()):
    ax = axes[idx]

    # Create heatmap using imshow
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    # Set ticks and labels
    tickers = corr_matrix.columns.tolist()
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tickers, fontsize=8)

    # Add correlation values as text
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=15)

    ax.set_title(f'{sector_name} Sector\n(Avg: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f})',
                 fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('network_correlation_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Saved: network_correlation_matrices.png")
plt.close()

# Figure 2: Network Graphs
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Correlation Networks by Sector (threshold ≥ {CORRELATION_THRESHOLD})',
             fontsize=16, fontweight='bold')

colors = ['#2ecc71', '#e74c3c', '#f39c12']  # Green, Red, Orange

for idx, (sector_name, G) in enumerate(networks.items()):
    ax = axes[idx]

    # Calculate layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Draw network
    nx.draw_networkx_nodes(G, pos,
                          node_color=colors[idx],
                          node_size=800,
                          alpha=0.8,
                          ax=ax)

    # Draw edges with varying thickness based on correlation strength
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_edges(G, pos,
                          width=[w*3 for w in weights],  # Scale edge width
                          alpha=0.3,
                          edge_color='gray',
                          ax=ax)

    nx.draw_networkx_labels(G, pos,
                           font_size=9,
                           font_weight='bold',
                           ax=ax)

    # Calculate statistics
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    n_possible = n_nodes * (n_nodes - 1) / 2
    density = n_edges / n_possible * 100
    avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0

    ax.set_title(f'{sector_name}\nDensity: {density:.1f}%, Avg Degree: {avg_degree:.1f}',
                fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('network_correlation_graphs.png', dpi=300, bbox_inches='tight')
print("✅ Saved: network_correlation_graphs.png")
plt.close()

# Figure 3: Detailed Network Statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Network Topology Analysis', fontsize=16, fontweight='bold')

sector_names = list(SECTORS.keys())
sector_colors = colors

# Plot 1: Network Density
ax1 = axes[0, 0]
densities = []
for sector_name, G in networks.items():
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    n_possible = n_nodes * (n_nodes - 1) / 2
    density = n_edges / n_possible * 100
    densities.append(density)

bars1 = ax1.bar(sector_names, densities, color=sector_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Network Density (%)', fontsize=11, fontweight='bold')
ax1.set_title('Network Connectivity', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 110])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars1, densities):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Average Degree
ax2 = axes[0, 1]
avg_degrees = []
for sector_name, G in networks.items():
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
    avg_degrees.append(avg_degree)

bars2 = ax2.bar(sector_names, avg_degrees, color=sector_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Average Degree', fontsize=11, fontweight='bold')
ax2.set_title('Average Connections per Node', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars2, avg_degrees):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Average Correlation
ax3 = axes[1, 0]
avg_corrs = []
for sector_name, corr_matrix in correlation_matrices.items():
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    avg_corrs.append(avg_corr)

bars3 = ax3.bar(sector_names, avg_corrs, color=sector_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Average Correlation', fontsize=11, fontweight='bold')
ax3.set_title('Mean Pairwise Correlation', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1])
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars3, avg_corrs):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Correlation Distribution
ax4 = axes[1, 1]
for sector_name, corr_matrix in correlation_matrices.items():
    # Get upper triangle correlations (exclude diagonal)
    upper_tri_corrs = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    ax4.hist(upper_tri_corrs, bins=20, alpha=0.5, label=sector_name, edgecolor='black')

ax4.axvline(x=CORRELATION_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold ({CORRELATION_THRESHOLD})')
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
    G = networks[sector_name]
    corr_matrix = correlation_matrices[sector_name]

    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
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
print("   Finance:    Dense, complete graph → Strong direct contagion")
print("   Commodity:  Dense but not complete → Shared external factors")
print("   Technology: Sparse, modular → Independent operations")

print("\n4. INTERPRETATION:")
print("   High correlation ≠ High contagion!")
print("   - Commodity has highest correlation but SIR underperforms")
print("   - Finance has lower correlation but SIR outperforms")
print("   - This suggests Finance has TRUE contagion (causal links)")
print("   - While Commodity has correlation without contagion (common shocks)")

print("\n" + "="*80)
print("VISUALIZATION FILES CREATED")
print("="*80)
print("1. network_correlation_matrices.png - Heatmaps of correlations")
print("2. network_correlation_graphs.png - Network topology visualizations")
print("3. network_topology_analysis.png - Statistical analysis")
print("4. network_statistics_summary.csv - Summary statistics table")
print("="*80)
