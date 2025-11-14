"""
Compare SIR contagion parameters across different sectors.

This script runs quick tests on multiple sectors and compares:
1. Contagion rate (β) - should be higher within sectors
2. Model performance (MSE) - SIR should win if contagion matters
"""

import sys
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Run each sector test and capture results
print("="*80)
print("SECTOR COMPARISON STUDY")
print("="*80)
print("\nRunning tests on different sectors to compare contagion dynamics...")
print("This will take ~10-15 minutes\n")

results = {}

# Test 1: Tech Sector
print("\n" + "="*80)
print("TEST 1: TECH SECTOR")
print("="*80)
print("Running tech sector test...")
try:
    result = subprocess.run(
        [sys.executable, "quick_test_tech_sector.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=600
    )
    output = result.stdout
    
    # Extract key metrics
    for line in output.split('\n'):
        if 'β (contagion rate):' in line:
            beta = float(line.split(':')[1].strip())
        if 'SIR Contagion Model MSE:' in line:
            sir_mse = float(line.split(':')[1].strip())
        if 'Best Baseline' in line and 'MSE' in line:
            # Format: "Best Baseline (Model Name): 123.45"
            baseline_mse = float(line.split(':')[-1].strip())
    
    results['Tech'] = {
        'beta': beta,
        'sir_mse': sir_mse,
        'baseline_mse': baseline_mse,
        'improvement': (baseline_mse - sir_mse) / baseline_mse * 100
    }
    print("✓ Tech sector test complete")
except Exception as e:
    print(f"✗ Tech sector test failed: {e}")
    results['Tech'] = None

# Test 2: Finance Sector
print("\n" + "="*80)
print("TEST 2: FINANCE SECTOR")
print("="*80)
print("Running finance sector test...")
try:
    result = subprocess.run(
        [sys.executable, "quick_test_finance_sector.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=600
    )
    output = result.stdout
    
    # Extract key metrics
    for line in output.split('\n'):
        if 'β (contagion rate):' in line:
            beta = float(line.split(':')[1].strip())
        if 'SIR Contagion Model MSE:' in line:
            sir_mse = float(line.split(':')[1].strip())
        if 'Best Baseline' in line and 'MSE' in line:
            baseline_mse = float(line.split(':')[-1].strip())
    
    results['Finance'] = {
        'beta': beta,
        'sir_mse': sir_mse,
        'baseline_mse': baseline_mse,
        'improvement': (baseline_mse - sir_mse) / baseline_mse * 100
    }
    print("✓ Finance sector test complete")
except Exception as e:
    print(f"✗ Finance sector test failed: {e}")
    results['Finance'] = None

# Compare results
print("\n" + "="*80)
print("CROSS-SECTOR COMPARISON")
print("="*80)

if results['Tech'] and results['Finance']:
    comparison = pd.DataFrame(results).T
    comparison.index.name = 'Sector'
    
    print("\n1. Contagion Rate (β):")
    print("-" * 80)
    print(comparison[['beta']].to_string())
    print(f"\nInterpretation:")
    print(f"  Higher β → Stronger contagion within that sector")
    
    if comparison.loc['Finance', 'beta'] > comparison.loc['Tech', 'beta']:
        print(f"  ✓ Finance shows stronger contagion (β={comparison.loc['Finance', 'beta']:.4f})")
        print(f"    This matches 2008 crisis narrative!")
    else:
        print(f"  Tech shows stronger contagion (β={comparison.loc['Tech', 'beta']:.4f})")
    
    print("\n2. Model Performance:")
    print("-" * 80)
    print(comparison[['sir_mse', 'baseline_mse', 'improvement']].to_string())
    print(f"\nInterpretation:")
    print(f"  Positive improvement → SIR beats baselines (contagion matters)")
    print(f"  Negative improvement → Baselines win (contagion weak)")
    
    if comparison['improvement'].max() > 0:
        best_sector = comparison['improvement'].idxmax()
        print(f"\n  ✓ SIR performs best in {best_sector} sector")
        print(f"    Improvement: {comparison.loc[best_sector, 'improvement']:.1f}%")
    else:
        print(f"\n  ✗ SIR underperforms in all sectors tested")
        print(f"    Suggests: Contagion effects weak OR model needs improvement")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Finding 1: Beta comparison
    beta_diff = abs(comparison.loc['Finance', 'beta'] - comparison.loc['Tech', 'beta'])
    if beta_diff > 0.1:
        print(f"\n✓ FINDING 1: Sector-specific contagion detected")
        print(f"  β differs by {beta_diff:.4f} between sectors")
    else:
        print(f"\n✗ FINDING 1: Contagion similar across sectors")
        print(f"  β only differs by {beta_diff:.4f}")
    
    # Finding 2: Performance comparison
    if comparison['improvement'].max() > 5:
        print(f"\n✓ FINDING 2: SIR model shows value in some sectors")
        print(f"  Best improvement: {comparison['improvement'].max():.1f}%")
    else:
        print(f"\n✗ FINDING 2: SIR underperforms or marginally better")
        print(f"  Consider model enhancements (threshold effects, asymmetric states)")
    
    # Save results
    comparison.to_csv('results_sector_comparison.csv')
    print(f"\nResults saved to: results_sector_comparison.csv")

else:
    print("\n✗ Could not complete comparison - check individual test outputs")

print("\n" + "="*80)
print("SECTOR COMPARISON COMPLETE")
print("="*80)

