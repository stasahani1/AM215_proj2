# Getting Started with SIR Contagion Model Evaluation

## Quick Start (Easiest Way)

### Option 1: Use the Interactive Script
```bash
cd /Users/praghav/Desktop/am215/AM215_proj2/src_new
./RUN_ME.sh
```

Then choose:
- **Option 1**: Quick test (3 stocks, ~5 minutes) - Recommended for first run
- **Option 2**: Full evaluation (10 stocks, ~30 minutes) - For complete results

### Option 2: Run Directly with Python

**Quick Test (Recommended First):**
```bash
cd /Users/praghav/Desktop/am215/AM215_proj2
source venv/bin/activate
cd src_new
python quick_test.py
```

**Full Evaluation:**
```bash
cd /Users/praghav/Desktop/am215/AM215_proj2
source venv/bin/activate
cd src_new
python run_evaluation.py
```

## What You'll See

### Quick Test Output (~5 minutes)
- Loads 3 stocks (AAPL, MSFT, JPM) with 5 years of data
- Trains 3 models on 20 high-volatility chunks
- Runs 50 Monte Carlo simulations per test chunk
- Shows performance comparison table
- **⚠️ No graphs generated** (only numerical results in terminal)

### Full Evaluation Output (~30 minutes)
- Loads 10 stocks with 15 years of data
- Trains 4 models on 60 high-volatility chunks
- Runs 500 Monte Carlo simulations per test chunk
- Generates comprehensive visualizations in `results/` directory:
  - `price_trajectories.png` - Actual vs simulated prices
  - `return_distributions.png` - Distribution comparison
  - `metric_comparison.png` - Performance metrics
  - `correlation_matrices.png` - Cross-stock correlations
  - `backtest_results.csv` - Detailed numerical results

## Understanding the Results

### Key Metrics (Lower is Better)
- **MSE (Mean Squared Error)**: Average squared difference between actual and predicted prices
- **MAPE (Mean Absolute Percentage Error)**: Percentage error, easier to interpret
- **R² (R-squared)**: Goodness of fit (higher is better, max 1.0)

### What Success Looks Like
If the SIR Contagion Model is successful, you should see:
- ✓ Lower MSE and MAPE compared to all baselines
- ✓ Higher R² value
- ✓ Better volatility and correlation matching
- ✓ Message: "✓ SUCCESS: SIR Contagion Model outperforms all baselines!"

### If SIR Model Doesn't Win
This is still valuable research! It might indicate:
- Contagion effects are weak in the selected time periods
- The correlation network needs different threshold
- Different train/test split might be needed
- Consider trying different stock sets (e.g., all financial stocks)

## Models Being Compared

1. **SIR Contagion Model** (Your Model)
   - 3 states: Susceptible, Infected, Recovered
   - State transitions depend on neighbors' states (contagion)
   - Learned parameters: β (contagion rate), γ (recovery), α (re-susceptibility)

2. **Independent HMM** (Baseline 2)
   - Same 3 states as SIR
   - State transitions are independent (no contagion)
   - Tests whether contagion matters

3. **Single Random Walk** (Baseline 1)
   - Simplest: one (μ, σ) per stock
   - No states, no contagion
   - Basic benchmark

4. **Correlated Random Walk** (Baseline 3)
   - Captures cross-stock correlation
   - No state switching
   - Tests if correlation alone is enough

## Customization

Edit parameters in `run_evaluation.py`:

```python
# Line ~30-40
TICKERS = get_default_tickers()[:10]  # Change number of stocks
YEARS_BACK = 15  # More years = more data
N_CHUNKS = 60  # Number of high-volatility periods
N_SIMULATIONS = 500  # More = slower but more accurate
CORRELATION_THRESHOLD = 0.5  # Network connection threshold
```

## Troubleshooting

### "No module named 'yfinance'"
Make sure you activated the virtual environment:
```bash
source /Users/praghav/Desktop/am215/AM215_proj2/venv/bin/activate
```

### Data download fails
- Check internet connection
- yfinance may have rate limits - wait a few minutes and retry
- Some tickers may be delisted - they'll be skipped automatically

### Out of memory
- Reduce `N_SIMULATIONS` (e.g., from 500 to 100)
- Reduce `N_CHUNKS` (e.g., from 60 to 30)
- Use fewer stocks

### Slow execution
- Normal! Full evaluation takes ~30 minutes
- Run `quick_test.py` instead (~5 minutes)
- Reduce `N_SIMULATIONS` for faster results

## Next Steps After Running

1. **Check Results Directory**: Look at generated plots in `results/`
2. **Analyze CSV**: Open `results/backtest_results.csv` in Excel/Numbers
3. **Experiment**: Try different stock sets, time periods, parameters
4. **Extend**: Add your own models by inheriting from `BaseModel`

## Technical Support

If you encounter issues:
1. Make sure virtual environment is activated
2. Check that data was downloaded successfully (look for "✓" marks)
3. Try the quick test first before full evaluation
4. Check the terminal output for specific error messages

## File Overview

```
src_new/
├── RUN_ME.sh ← START HERE (interactive)
├── quick_test.py ← Or start here (quick test)
├── run_evaluation.py ← Full evaluation
├── README.md ← Technical documentation
├── GETTING_STARTED.md ← This file
│
├── data/ ← Data loading modules
├── models/ ← SIR and baseline models
├── training/ ← HMM and contagion learning
├── evaluation/ ← Metrics and backtesting
└── visualization/ ← Plotting functions
```

## Expected Timeline

| Task | Time | What Happens |
|------|------|--------------|
| Data Loading | 2-3 min | Downloads stock and VIX data |
| Chunking | <1 min | Selects high-volatility periods |
| HMM Training | 3-5 min | Learns states for each stock |
| Contagion Learning | <1 min | Learns β, γ, α parameters |
| Backtesting | 15-20 min | Runs Monte Carlo simulations |
| Visualization | 2-3 min | Generates plots |
| **Total** | **~25-35 min** | **Full evaluation** |
| **Quick Test** | **~5 min** | **Abbreviated version** |

## Citation

```
SIR Contagion Model for Stock Price Dynamics
Multi-State Random Walk with Epidemic-Inspired Contagion
Implementation: Python 3.11, NumPy, Pandas, scikit-learn
```

---

**Ready to begin?** Run `./RUN_ME.sh` or `python quick_test.py`!

