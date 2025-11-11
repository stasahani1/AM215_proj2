"""
Compare SIR contagion across Tech vs Finance sectors with visualizations.

This script runs both sector tests and creates comparison plots.
"""

import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import re
import time

print("="*80)
print("SECTOR COMPARISON WITH VISUALIZATIONS")
print("="*80)
print("\nThis will:")
print("  1. Run Tech sector test (10 tech stocks)")
print("  2. Run Finance sector test (10 finance stocks)")
print("  3. Compare contagion parameters (β)")
print("  4. Compare model performance (MSE, MAPE, R²)")
print("  5. Generate comparison plots")
print("\nEstimated time: ~15-20 minutes")
print("="*80)

# Create results directory
results_dir = Path("results_sector_comparison")
results_dir.mkdir(exist_ok=True)

# Store results
results = {'Tech': {}, 'Finance': {}}

# Run Tech Sector Test
print("\n" + "="*80)
print("RUNNING TECH SECTOR TEST")
print("="*80)
start_time = time.time()

try:
    result = subprocess.run(
        [sys.executable, "quick_test_tech_sector.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=900  # 15 min timeout
    )
    
    output = result.stdout
    print(output)  # Print output so user can see progress
    
    # Extract metrics from output
    for line in output.split('\n'):
        if 'β (contagion rate):' in line and 'Contagion Parameters' in output[max(0, output.find(line)-200):output.find(line)]:
            try:
                results['Tech']['beta'] = float(re.search(r'([-\d.]+)', line.split(':')[1]).group(1))
            except:
                pass
        if 'γ (recovery rate):' in line:
            try:
                results['Tech']['gamma'] = float(re.search(r'([-\d.]+)', line.split(':')[1]).group(1))
            except:
                pass
        if 'α (re-susceptibility):' in line:
            try:
                results['Tech']['alpha'] = float(re.search(r'([-\d.]+)', line.split(':')[1]).group(1))
            except:
                pass
        if 'Network density:' in line and ':' in line:
            try:
                results['Tech']['network_density'] = float(re.search(r'([\d.]+)%', line).group(1))
            except:
                pass
        if 'SIR Contagion Model MSE:' in line:
            try:
                results['Tech']['sir_mse'] = float(line.split(':')[-1].strip())
            except:
                pass
        if 'Best Baseline' in line and ':' in line:
            # Format: "Best Baseline (Model Name): 123.45"
            try:
                # Extract the number after the last colon
                value_str = line.split(':')[-1].strip()
                results['Tech']['baseline_mse'] = float(value_str)
                # Extract baseline name
                baseline_match = re.search(r'Best Baseline \(([^)]+)\)', line)
                if baseline_match:
                    results['Tech']['best_baseline'] = baseline_match.group(1)
            except:
                pass
    
    tech_time = time.time() - start_time
    results['Tech']['runtime'] = tech_time
    print(f"\n✓ Tech sector test complete ({tech_time:.1f}s)")
    
except Exception as e:
    print(f"\n✗ Tech sector test failed: {e}")
    results['Tech'] = None

# Run Finance Sector Test
print("\n" + "="*80)
print("RUNNING FINANCE SECTOR TEST")
print("="*80)
start_time = time.time()

try:
    result = subprocess.run(
        [sys.executable, "quick_test_finance_sector.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=900
    )
    
    output = result.stdout
    print(output)
    
    # Extract metrics
    for line in output.split('\n'):
        if 'β (contagion rate):' in line and 'Contagion Parameters' in output[max(0, output.find(line)-200):output.find(line)]:
            try:
                results['Finance']['beta'] = float(re.search(r'([-\d.]+)', line.split(':')[1]).group(1))
            except:
                pass
        if 'γ (recovery rate):' in line:
            try:
                results['Finance']['gamma'] = float(re.search(r'([-\d.]+)', line.split(':')[1]).group(1))
            except:
                pass
        if 'α (re-susceptibility):' in line:
            try:
                results['Finance']['alpha'] = float(re.search(r'([-\d.]+)', line.split(':')[1]).group(1))
            except:
                pass
        if 'Network density:' in line and ':' in line:
            try:
                results['Finance']['network_density'] = float(re.search(r'([\d.]+)%', line).group(1))
            except:
                pass
        if 'SIR Contagion Model MSE:' in line:
            try:
                results['Finance']['sir_mse'] = float(line.split(':')[-1].strip())
            except:
                pass
        if 'Best Baseline' in line and ':' in line:
            try:
                # Extract the number after the last colon
                value_str = line.split(':')[-1].strip()
                results['Finance']['baseline_mse'] = float(value_str)
                # Extract baseline name
                baseline_match = re.search(r'Best Baseline \(([^)]+)\)', line)
                if baseline_match:
                    results['Finance']['best_baseline'] = baseline_match.group(1)
            except:
                pass
    
    finance_time = time.time() - start_time
    results['Finance']['runtime'] = finance_time
    print(f"\n✓ Finance sector test complete ({finance_time:.1f}s)")
    
except Exception as e:
    print(f"\n✗ Finance sector test failed: {e}")
    results['Finance'] = None

# Create Visualizations
if results['Tech'] and results['Finance']:
    # Check if we have all required data
    required_keys = ['beta', 'gamma', 'alpha', 'network_density', 'sir_mse', 'baseline_mse']
    tech_complete = all(key in results['Tech'] for key in required_keys)
    finance_complete = all(key in results['Finance'] for key in required_keys)
    
    if not tech_complete:
        print(f"\n✗ Tech results incomplete. Missing: {[k for k in required_keys if k not in results['Tech']]}")
        print("Tech results:", results['Tech'])
    if not finance_complete:
        print(f"\n✗ Finance results incomplete. Missing: {[k for k in required_keys if k not in results['Finance']]}")
        print("Finance results:", results['Finance'])
    
    if not (tech_complete and finance_complete):
        print("\n✗ Cannot create visualizations - incomplete data")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    # Calculate improvements
    results['Tech']['improvement'] = (results['Tech']['baseline_mse'] - results['Tech']['sir_mse']) / results['Tech']['baseline_mse'] * 100
    results['Finance']['improvement'] = (results['Finance']['baseline_mse'] - results['Finance']['sir_mse']) / results['Finance']['baseline_mse'] * 100
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Contagion Parameters Comparison
    ax1 = plt.subplot(2, 3, 1)
    sectors = ['Tech', 'Finance']
    betas = [results['Tech']['beta'], results['Finance']['beta']]
    bars = ax1.bar(sectors, betas, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Contagion Rate (β)', fontsize=12, fontweight='bold')
    ax1.set_title('Contagion Strength by Sector', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, betas):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Model Performance (MSE)
    ax2 = plt.subplot(2, 3, 2)
    x = np.arange(len(sectors))
    width = 0.35
    
    sir_mses = [results['Tech']['sir_mse'], results['Finance']['sir_mse']]
    baseline_mses = [results['Tech']['baseline_mse'], results['Finance']['baseline_mse']]
    
    bars1 = ax2.bar(x - width/2, sir_mses, width, label='SIR Contagion', 
                    color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, baseline_mses, width, label='Best Baseline',
                    color='#95a5a6', alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sectors)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Performance Improvement
    ax3 = plt.subplot(2, 3, 3)
    improvements = [results['Tech']['improvement'], results['Finance']['improvement']]
    colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in improvements]
    bars = ax3.bar(sectors, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_ylabel('Improvement vs Baseline (%)', fontsize=12, fontweight='bold')
    ax3.set_title('SIR Model Performance Gain', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va=va, fontweight='bold')
    
    # 4. All Contagion Parameters
    ax4 = plt.subplot(2, 3, 4)
    params = ['β\n(Contagion)', 'γ\n(Recovery)', 'α\n(Re-susceptible)']
    tech_params = [results['Tech']['beta'], results['Tech']['gamma'], results['Tech']['alpha']]
    finance_params = [results['Finance']['beta'], results['Finance']['gamma'], results['Finance']['alpha']]
    
    x = np.arange(len(params))
    width = 0.35
    
    ax4.bar(x - width/2, tech_params, width, label='Tech', 
            color='#3498db', alpha=0.7, edgecolor='black')
    ax4.bar(x + width/2, finance_params, width, label='Finance',
            color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax4.set_ylabel('Parameter Value', fontsize=12, fontweight='bold')
    ax4.set_title('All SIR Parameters', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(params)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Summary Table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    table_data = [
        ['Metric', 'Tech', 'Finance'],
        ['', '', ''],
        ['Contagion (β)', f"{results['Tech']['beta']:.4f}", f"{results['Finance']['beta']:.4f}"],
        ['Recovery (γ)', f"{results['Tech']['gamma']:.4f}", f"{results['Finance']['gamma']:.4f}"],
        ['Re-suscept (α)', f"{results['Tech']['alpha']:.4f}", f"{results['Finance']['alpha']:.4f}"],
        ['', '', ''],
        ['SIR MSE', f"{results['Tech']['sir_mse']:.2f}", f"{results['Finance']['sir_mse']:.2f}"],
        ['Baseline MSE', f"{results['Tech']['baseline_mse']:.2f}", f"{results['Finance']['baseline_mse']:.2f}"],
        ['Improvement', f"{results['Tech']['improvement']:+.1f}%", f"{results['Finance']['improvement']:+.1f}%"],
        ['', '', ''],
        ['Network Density', f"{results['Tech']['network_density']:.0f}%", f"{results['Finance']['network_density']:.0f}%"],
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight winner
    winner_row = 8  # Improvement row
    if results['Finance']['improvement'] > results['Tech']['improvement']:
        table[(winner_row, 2)].set_facecolor('#2ecc71')
        table[(winner_row, 2)].set_text_props(weight='bold')
    else:
        table[(winner_row, 1)].set_facecolor('#2ecc71')
        table[(winner_row, 1)].set_text_props(weight='bold')
    
    ax5.set_title('Detailed Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # 6. Key Findings
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    findings = []
    
    # Finding 1: β comparison
    if results['Finance']['beta'] > results['Tech']['beta']:
        beta_diff = results['Finance']['beta'] - results['Tech']['beta']
        findings.append(f"✓ Finance shows {beta_diff:.4f} higher\n  contagion rate than Tech")
        findings.append(f"  ({results['Finance']['beta']:.4f} vs {results['Tech']['beta']:.4f})")
    else:
        findings.append(f"✗ Tech shows higher contagion\n  than Finance (unexpected)")
    
    findings.append("")
    
    # Finding 2: Performance
    if results['Finance']['improvement'] > 5:
        findings.append(f"✓ SIR wins in Finance by {results['Finance']['improvement']:.1f}%")
        findings.append(f"  Validates financial contagion!")
    elif results['Finance']['improvement'] > 0:
        findings.append(f"~ SIR slightly better in Finance")
        findings.append(f"  ({results['Finance']['improvement']:.1f}% improvement)")
    else:
        findings.append(f"✗ SIR underperforms in Finance")
    
    findings.append("")
    
    if results['Tech']['improvement'] > 0:
        findings.append(f"Tech: SIR +{results['Tech']['improvement']:.1f}%")
    else:
        findings.append(f"Tech: SIR {results['Tech']['improvement']:.1f}%")
    
    findings.append("")
    findings.append("Conclusion:")
    if results['Finance']['improvement'] > results['Tech']['improvement']:
        findings.append("Sector-specific contagion")
        findings.append("validated! Finance shows")
        findings.append("stronger contagion dynamics.")
    else:
        findings.append("Results suggest contagion")
        findings.append("varies by sector and period.")
    
    findings_text = '\n'.join(findings)
    ax6.text(0.1, 0.95, findings_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax6.set_title('Key Findings', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Tech vs Finance: SIR Contagion Model Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = results_dir / "sector_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    
    plt.show()
    
    # Save results to CSV
    df = pd.DataFrame(results).T
    df.to_csv(results_dir / "sector_comparison_metrics.csv")
    print(f"✓ Saved metrics: {results_dir / 'sector_comparison_metrics.csv'}")
    
    # Print summary
    print("\n" + "="*80)
    print("CROSS-SECTOR COMPARISON SUMMARY")
    print("="*80)
    
    print("\n1. Contagion Rates (β):")
    print("-" * 80)
    print(f"  Tech:    β = {results['Tech']['beta']:.4f}")
    print(f"  Finance: β = {results['Finance']['beta']:.4f}")
    
    if results['Finance']['beta'] > results['Tech']['beta']:
        diff = results['Finance']['beta'] - results['Tech']['beta']
        pct_higher = (diff / results['Tech']['beta']) * 100
        print(f"\n  → Finance shows {pct_higher:.1f}% higher contagion rate")
        print(f"    This matches 2008 financial crisis narrative!")
    
    print("\n2. Model Performance:")
    print("-" * 80)
    print(f"  Tech:    SIR {results['Tech']['improvement']:+.1f}% vs baseline")
    print(f"  Finance: SIR {results['Finance']['improvement']:+.1f}% vs baseline")
    
    if results['Finance']['improvement'] > 5:
        print(f"\n  → SIR model WINS in Finance sector!")
        print(f"    Improvement: {results['Finance']['improvement']:.1f}%")
        print(f"    This validates sector-specific contagion modeling")
    
    print("\n3. Network Structure:")
    print("-" * 80)
    print(f"  Tech:    {results['Tech']['network_density']:.0f}% connected")
    print(f"  Finance: {results['Finance']['network_density']:.0f}% connected")
    
    print("\n" + "="*80)
    print("✓ SECTOR COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}/")
    print("  - sector_comparison.png")
    print("  - sector_comparison_metrics.csv")
    print("="*80)

else:
    print("\n✗ Could not complete comparison - check individual test outputs")

