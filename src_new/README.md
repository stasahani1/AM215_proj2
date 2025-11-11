# SIR Contagion Model Evaluation Framework

A comprehensive framework for evaluating stock price prediction using an SIR (Susceptible-Infected-Recovered) contagion model compared to baseline approaches.

## Overview

This framework implements a multi-state random walk model where:
- Each stock exists in one of three states: **S** (Susceptible), **I** (Infected), **R** (Recovered)
- Each state has different return dynamics (μ, σ)
- State transitions include **contagion effects**: infected stocks increase infection probability in connected stocks
- The model is trained and evaluated on high-volatility periods from historical data

## Quick Start

### Run the Complete Evaluation

```bash
cd /Users/praghav/Desktop/am215/AM215_proj2/src_new
python run_evaluation.py
```

This will:
1. Load 15 years of stock data for 10 stocks
2. Identify and select 60 high-volatility 2-week periods
3. Train 4 models (SIR + 3 baselines)
4. Run backtests on test periods
5. Generate comprehensive results and visualizations

### Expected Runtime
- Data loading: ~2-3 minutes
- Model training: ~5-10 minutes
- Backtesting: ~15-20 minutes
- **Total: ~25-35 minutes**

## Models Evaluated

### 1. SIR Contagion Model (Main Model)
- 3 states per stock with learned (μ, σ) parameters
- Contagion dynamics: `P(S→I) = f(infected_neighbors)`
- Learned parameters: β (contagion rate), γ (recovery rate), α (re-susceptibility)

### 2. Independent HMM (Baseline)
- Same 3 states as SIR model
- **No contagion**: transitions are independent across stocks
- Tests if contagion effect matters

### 3. Single Random Walk (Baseline)
- Simplest model: one (μ, σ) per stock
- No state switching
- Basic benchmark

### 4. Correlated Random Walk (Baseline)
- Correlated returns across stocks
- No state switching
- Tests if correlation alone is sufficient

## Directory Structure

```
src_new/
├── run_evaluation.py          # Main script - RUN THIS
├── data/
│   ├── data_loader.py          # Fetch stock prices and VIX
│   └── chunk_selector.py       # Create high-volatility chunks
├── models/
│   ├── base_model.py           # Base class for all models
│   ├── sir_contagion_model.py  # SIR contagion implementation
│   ├── baseline_single_rw.py   # Simple random walk
│   ├── baseline_independent_hmm.py  # HMM without contagion
│   └── baseline_correlated_rw.py    # Correlated returns
├── training/
│   ├── hmm_trainer.py          # Hidden Markov Model training
│   ├── state_inference.py      # Multi-stock state inference
│   └── contagion_learner.py    # Learn β, γ, α parameters
├── evaluation/
│   ├── metrics.py              # Comprehensive evaluation metrics
│   └── backtester.py           # Backtesting framework
└── visualization/
    └── plots.py                # Visualization tools
```

## Output

After running, you'll find in `results/`:

1. **backtest_results.csv** - Detailed metrics for each test chunk
2. **price_trajectories.png** - Actual vs simulated price paths
3. **return_distributions.png** - Distribution comparison
4. **metric_comparison.png** - Model performance comparison
5. **correlation_matrices.png** - Cross-stock correlation analysis

## Key Metrics

### Price Accuracy
- **MSE**: Mean squared error of prices
- **MAPE**: Mean absolute percentage error
- **R²**: Coefficient of determination

### Volatility Matching
- **Volatility MAE**: How well model captures volatility
- **Volatility Autocorrelation**: Volatility clustering

### Correlation Matching
- **Correlation MAE**: Cross-stock correlation accuracy
- **Frobenius Error**: Overall correlation structure

### Distribution Matching
- **Wasserstein Distance**: Return distribution similarity
- **JS Divergence**: Jensen-Shannon divergence

## Customization

Edit configuration in `run_evaluation.py`:

```python
# Data configuration
TICKERS = get_default_tickers()[:10]  # Number of stocks
YEARS_BACK = 15  # Years of historical data

# Chunking configuration
CHUNK_DAYS = 10  # Days per chunk (10 = 2 weeks)
N_CHUNKS = 60  # Number of high-volatility chunks
TRAIN_RATIO = 0.6  # Train/test split

# Evaluation
N_SIMULATIONS = 500  # Monte Carlo simulations
CORRELATION_THRESHOLD = 0.5  # For contagion network
```

## Understanding the Results

### If SIR Model Wins:
✓ Contagion effects are significant in crisis periods  
✓ Multi-state dynamics improve predictions  
✓ Network effects matter for price modeling  

### If SIR Model Doesn't Win:
- May need more data or different stocks
- Contagion effects might be weak in selected periods
- Consider tuning correlation threshold
- Try different train/test splits

## Technical Details

### State Interpretation
- **State R (Recovered)**: Low volatility, stable prices
- **State S (Susceptible)**: Normal volatility, baseline regime
- **State I (Infected)**: High volatility, crisis regime

### Contagion Mechanism
```
P(stock_i: S→I) = baseline_rate + β * (infected_neighbors / total_neighbors)
```

### Training Procedure
1. Learn state parameters (μ, σ) via HMM for each stock
2. Infer historical state sequences via Viterbi algorithm
3. Learn contagion parameters (β, γ, α) from transition patterns
4. Build network from historical correlations

## Dependencies

All dependencies are in the parent directory's `requirements.txt`:
- yfinance (data)
- pandas, numpy (data processing)
- scipy, scikit-learn (algorithms)
- matplotlib, seaborn (visualization)

## Citation

If you use this framework, please cite:
```
SIR Contagion Model for Stock Price Dynamics
Multi-State Random Walk with Epidemic-Inspired Contagion
```

## Support

For questions or issues:
1. Check the output logs for error messages
2. Verify data was downloaded successfully
3. Try reducing N_SIMULATIONS for faster testing
4. Ensure all dependencies are installed

