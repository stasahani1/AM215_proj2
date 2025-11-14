# SIR Contagion Model for Stock Price Prediction

A comprehensive framework for evaluating stock price prediction using an **SIR (Susceptible-Infected-Recovered) contagion model** inspired by epidemic dynamics, adapted for financial markets.

## Overview

This framework implements a **multi-state regime-switching model with network contagion effects**:

- **Three Market States**: Each stock exists in one of three regimes
  - **S (Susceptible)**: Normal volatility, baseline regime
  - **I (Infected)**: High volatility, crisis regime  
  - **R (Recovered)**: Low volatility, stable regime
  
- **State-Dependent Dynamics**: Each state has learned return parameters (μ, σ)

- **Contagion Effects**: State transitions exhibit network effects
  - Infected stocks increase infection probability in correlated neighbors
  - Models systemic risk propagation across financial networks

- **Evaluation**: Trained and tested on historical crisis periods (2008, COVID, etc.)

## Key Features

✓ **Regime-Switching Dynamics**: HMM-based state inference (Hamilton 1989)  
✓ **Financial Contagion**: Network-based transmission of volatility states  
✓ **Sector Analysis**: Tests finance vs tech sector contagion  
✓ **Comprehensive Baselines**: Compares against Independent HMM, Random Walk, Correlated RW  
✓ **Multiple Evaluation Windows**: 6-week and 6-month period analyses  

## Quick Start

### Option 1: Interactive Menu (Recommended)
```bash
cd src
./RUN_ME.sh
```

### Option 2: Direct Execution
```bash
# Quick test (recommended first run)
python src/scripts/quick_test_optimal.py

# Full evaluation with visualizations
python src/scripts/run_evaluation_optimal.py

# Finance sector test (best results!)
python src/scripts/quick_test_finance_sector.py

# Sector comparison
python src/scripts/compare_sectors_with_viz.py
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
AM215_proj2/
├── src/
│   ├── scripts/                          # All runnable scripts
│   │   ├── quick_test_optimal.py         # Quick test with optimal settings ⭐
│   │   ├── run_evaluation_optimal.py     # Full evaluation
│   │   ├── quick_test_finance_sector.py  # Finance sector (best results) 💰
│   │   ├── quick_test_tech_sector.py     # Tech sector analysis
│   │   ├── compare_sectors_with_viz.py   # Sector comparison
│   │   ├── compare_window_sizes.py       # Window size analysis
│   │   ├── generate_latex_tables.py      # LaTeX table generation
│   │   └── visualize_networks.py         # Network visualization
│   │
│   ├── data/                             # Data loading and preprocessing
│   │   ├── data_loader.py                # Fetch stock prices and VIX
│   │   └── chunk_selector.py             # Create time chunks
│   │
│   ├── models/                           # Model implementations
│   │   ├── base_model.py                 # Base class
│   │   ├── sir_contagion_model.py        # SIR contagion (main model)
│   │   ├── baseline_independent_hmm.py   # HMM without contagion
│   │   ├── baseline_single_rw.py         # Simple random walk
│   │   └── baseline_correlated_rw.py     # Correlated random walk
│   │
│   ├── training/                         # Training algorithms
│   │   ├── hmm_trainer.py                # Baum-Welch algorithm
│   │   ├── state_inference.py            # Multi-stock state inference
│   │   └── contagion_learner.py          # Learn β, γ, α parameters
│   │
│   ├── evaluation/                       # Evaluation framework
│   │   ├── backtester.py                 # Backtesting engine
│   │   └── metrics.py                    # Comprehensive metrics
│   │
│   ├── visualization/                    # Plotting tools
│   │   └── plots.py                      # All visualization functions
│   │
│   ├── project_results/                  # Final results for paper
│   │   ├── latex_tables/                 # LaTeX tables
│   │   └── *.png                         # Publication-ready figures
│   │
│   ├── RUN_ME.sh                         # Interactive menu script
│   ├── README.md                         # Detailed documentation
│   ├── GETTING_STARTED.md                # Usage guide
│   ├── OPTIMAL_APPROACH.md               # Methodology
│   ├── SECTOR_ANALYSIS.md                # Sector findings
│   └── FINANCE_SECTOR_SUCCESS.md         # Best results
│
├── requirements.txt                      # Python dependencies
└── venv/                                 # Virtual environment (create this)
```

## Models Compared

### 1. **SIR Contagion Model** (Main Contribution)
- 3 states per stock with HMM-learned (μ, σ)
- **Contagion dynamics**: `P(S→I | neighbors) = baseline + β × (infected_fraction)`
- Learned parameters: 
  - **β** (contagion rate): How infected neighbors increase infection probability
  - **γ** (recovery rate): Transition rate from I → R
  - **α** (re-susceptibility): Transition rate from R → S
- Network structure from historical correlations

### 2. **Independent HMM** (Baseline)
- Same 3 states, same learning procedure
- **No contagion**: transitions independent across stocks
- Tests if network effects matter

### 3. **Single Random Walk** (Baseline)
- One (μ, σ) per stock, no state switching
- Simplest benchmark

### 4. **Correlated Random Walk** (Baseline)
- Captures cross-stock correlation
- No state switching
- Tests if correlation alone suffices

## Key Results

### Finance Sector (2008 Crisis + COVID) ⭐
✅ **SIR model shows strongest performance**
- β = 1.14-1.99 (strong contagion)
- 7-15% improvement over baselines
- High network density (77-100%)
- Best during financial crises

### Tech Sector
- β = 0.86-0.88 (moderate contagion)
- Mixed results vs baselines
- Lower network connectivity
- Less pronounced contagion effects

### Key Insight
**Financial contagion is sector-specific**: Finance sector exhibits strong crisis-driven contagion (supporting "too-big-to-fail" and systemic risk theories), while tech sector shows more independent dynamics.

## Evaluation Metrics

### Price Accuracy
- **MSE**: Mean squared error
- **MAPE**: Mean absolute percentage error
- **R²**: Coefficient of determination

### Volatility Matching
- **Volatility MAE**: Captures volatility accuracy
- **Autocorrelation**: Tests volatility clustering

### Correlation Structure
- **Correlation MAE**: Cross-stock correlation error
- **Frobenius Norm**: Overall structure distance

### Distribution Matching
- **Wasserstein Distance**: Return distribution similarity
- **Jensen-Shannon Divergence**: Distributional difference

### State Metrics (SIR-specific)
- **State accuracy**: Correct state prediction rate
- **State occupancy**: Time spent in each regime
- **Transition patterns**: S→I→R dynamics

## Output Files

After running, results are saved to `src/`:
- `results_optimal/` - Optimal configuration results
- `results_sector_comparison/` - Sector analysis
- `project_results/` - Final publication-ready outputs

**Generated files**:
- `backtest_results.csv` - Detailed per-chunk metrics
- `price_trajectories.png` - Actual vs predicted prices
- `return_distributions.png` - Distribution comparison
- `metric_comparison.png` - Model performance bars
- `correlation_matrices.png` - Correlation structure
- `state_heatmap.png` - State evolution over time (SIR only)

## Configuration

Edit parameters in any script (e.g., `src/scripts/quick_test_optimal.py`):

```python
# Stock selection
STOCKS = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C']  # Finance stocks

# Time periods
YEARS_BACK = 15          # Historical data window
CHUNK_DAYS = 30          # ~6 weeks (optimal setting)

# Evaluation
N_CHUNKS = 30            # Number of test periods
N_SIMULATIONS = 200      # Monte Carlo simulations

# Network
CORRELATION_THRESHOLD = 0.6  # For contagion network construction
```

## Methodology

### Training Pipeline
1. **State Learning**: Baum-Welch algorithm learns (μ, σ) for each state per stock
2. **State Inference**: Viterbi algorithm infers historical state sequences
3. **Network Construction**: Build correlation network from training data
4. **Contagion Learning**: Maximum likelihood estimation of β, γ, α from historical state transitions

### Evaluation Strategy
- **Mixed-period training**: Train on both high and low VIX periods
- **High-VIX testing**: Evaluate on crisis periods
- **Monte Carlo**: Average over 100-500 simulations per test
- **Out-of-sample**: Strict train/test split, no data leakage

## Dependencies

Install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Key packages**:
- `yfinance` - Stock data download
- `pandas`, `numpy` - Data manipulation
- `scipy` - Statistical functions
- `scikit-learn` - ML utilities
- `matplotlib` - Visualization

## Usage Examples

### Run Quick Test
```bash
cd src
python scripts/quick_test_optimal.py
```

### Generate LaTeX Tables for Paper
```bash
cd src
python scripts/generate_latex_tables.py
```

### Compare Finance vs Tech Sectors
```bash
cd src
python scripts/compare_sectors_with_viz.py
```

### Visualize Correlation Networks
```bash
cd src
python scripts/visualize_networks.py
```

## Documentation

Detailed documentation in `src/`:
- `GETTING_STARTED.md` - Step-by-step usage guide
- `OPTIMAL_APPROACH.md` - Methodology explanation
- `SECTOR_ANALYSIS.md` - Sector-specific findings
- `FINANCE_SECTOR_SUCCESS.md` - Best results summary

## References

1. **Kermack, W. O., & McKendrick, A. G. (1927)**. A contribution to the mathematical theory of epidemics. *Proceedings of the Royal Society of London*.

2. **Hamilton, J. D. (1989)**. A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*.

## Support & Troubleshooting

For issues:
1. Check console output logs for errors
2. Verify data downloads completed successfully (requires internet)
3. Try `quick_test_optimal.py` first before full evaluation
4. Reduce `N_SIMULATIONS` for faster testing
5. Ensure virtual environment is activated

Common fixes:
```bash
# If yfinance fails
pip install --upgrade yfinance

# If matplotlib display issues
export MPLBACKEND=Agg

# If permission denied on RUN_ME.sh
chmod +x src/RUN_ME.sh
```

## Citation

```bibtex
@misc{sir_stock_contagion,
  title={SIR Contagion Model for Stock Price Prediction},
  author={Your Name},
  year={2025},
  note={Multi-state regime-switching with epidemic-inspired financial contagion}
}
```

## License

MIT License - see LICENSE file for details

---

**Get started**: `cd src && ./RUN_ME.sh` or `python src/scripts/quick_test_optimal.py`

**Best results**: Finance sector analysis shows 7-15% improvement over baselines with strong contagion dynamics (β ≈ 1.14-1.99)

