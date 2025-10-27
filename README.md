# Financial Contagion SIR Model

A network-based epidemic model for analyzing financial market contagion using the SIR (Susceptible-Infected-Recovered) framework.

## Overview

This project models financial crises as an epidemic spreading through a network of correlated stocks. By treating market stress as "infections" that propagate through financial networks, we can:

- Identify systemically important stocks
- Predict contagion risk
- Stress-test portfolios under different market scenarios
- Quantify network effects in financial markets

## Methodology

### 1. Crisis State Labeling (S/I/R)

Each stock on each day is classified as:

- **Susceptible (S)**: Operating normally, vulnerable to crisis
- **Infected (I)**: In crisis (drawdown ≥10% OR volatility in top 5% vs. historical)
- **Recovered (R)**: Recently recovered, in 10-day cooldown period

### 2. Network Construction

- Rolling 90-day correlation networks
- Top-k sparsification (keep 10 strongest connections per stock)
- Row-normalized adjacency matrices (weights represent influence)

### 3. Hazard Modeling

Complementary log-log GLM to predict infection probability:

```
P(infection) = 1 - exp(-exp(β₀ + β₁·exposure + β₂·market + β₃·controls))
```

Where:
- **exposure** = network contagion pressure (Σ W_ij × I_j)
- **market** = z-scored VIX (market-wide stress)
- **controls** = lagged returns, etc.

### 4. Scenario Simulation

Forward-simulate SIR dynamics under different market stress scenarios:
- Normal (no stress)
- Stressed (moderate VIX spike)
- Severe (major market crisis)

## Project Structure

```
mini_proj_2/
├── src/
│   ├── data_utils.py        # Data fetching and basic calculations
│   ├── crisis_labeling.py   # S/I/R state classification
│   ├── network.py            # Network construction
│   ├── dataset.py            # Dataset building for modeling
│   ├── model.py              # GLM fitting and recovery estimation
│   ├── evaluation.py         # Evaluation metrics
│   ├── simulation.py         # SIR forward simulation
│   └── run.py                # Main pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd mini_proj_2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

```bash
cd src
python run.py
```

This will:
1. Fetch stock data from Yahoo Finance (2016-2024)
2. Label crisis states for 27 stocks across sectors
3. Build rolling correlation networks
4. Train a GLM model (train: 2016-2019, test: 2020-2024)
5. Evaluate out-of-sample predictions
6. Estimate recovery rates
7. Run Monte Carlo scenario simulations

### Customize Parameters

Edit `run.py` to:
- Change stock tickers
- Adjust date ranges
- Modify crisis thresholds
- Change network parameters

### Using Individual Modules

```python
from data_utils import fetch_yahoo_prices, compute_returns
from crisis_labeling import label_crisis_states
from network import rolling_corr_network

# Fetch data
prices = fetch_yahoo_prices(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01', '2024-01-01')

# Label states
S, I, R = label_crisis_states(prices, dd_thresh=-0.10)

# Build network
returns = compute_returns(prices)
W_by_date = rolling_corr_network(returns, window=90, topk=10)
```

## Key Parameters

### Crisis Labeling
- `dd_thresh`: Drawdown threshold (default: -0.10 = -10%)
- `vol_quantile`: Volatility percentile threshold (default: 0.95)
- `min_spell`: Minimum infection duration (default: 3 days)
- `cooloff_days`: Recovery period (default: 10 days)

### Network
- `window`: Rolling correlation window (default: 90 days)
- `topk`: Number of connections to keep per node (default: 10)
- `nonnegative`: Clip negative correlations to zero (default: True)

### Model
- Train/test split: 2016-2019 train, 2020+ test
- GLM family: Binomial with complementary log-log link

## Output

The pipeline prints:

1. **Data summary**: Stocks, date range, observations
2. **State statistics**: % Susceptible, Infected, Recovered
3. **Network info**: Number of dates, nodes, edges
4. **Dataset info**: Sample size, features, infection rate
5. **Model coefficients**: β estimates with significance tests
6. **OOS performance**: AUC, Average Precision, Brier Score
7. **Recovery rate**: Estimated mean infection duration
8. **Scenario results**: Peak infections, final recovered under stress scenarios

## Example Output

```
================================================================================
SIR CONTAGION MODEL FOR FINANCIAL MARKETS
================================================================================

[1/7] Fetching data from Yahoo Finance...
   Tickers: 27 stocks
   Period: 2016-01-01 to 2024-12-31
   Loaded 2264 days × 27 assets

[2/7] Labeling crisis states (S/I/R)...
   Total observations: 61,128
   Susceptible (S): 47,892 (78.3%)
   Infected (I): 8,214 (13.4%)
   Recovered (R): 5,022 (8.2%)

...

[6/7] Evaluating out-of-sample predictions...
   Out-of-Sample Performance:
   - AUC: 0.742
   - Average Precision: 0.284
   - Brier Score: 0.0891

...

[7/7] Running scenario simulations...
   Scenario: STRESSED
   - Peak infected (mean): 8.4
   - Peak infected (95th percentile): 12.0
   - Final recovered (mean): 15.2
```

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **yfinance**: Yahoo Finance data fetching
- **statsmodels**: Statistical modeling (GLM)
- **scikit-learn**: Evaluation metrics

## Applications

- **Risk Management**: Identify contagion vulnerabilities
- **Portfolio Construction**: Hedge against network effects
- **Stress Testing**: Assess portfolio resilience to market shocks
- **Regulatory Analysis**: Monitor systemic risk buildup

## Theoretical Background

This model combines:
- **Epidemiology**: SIR compartmental models
- **Network Science**: Weighted, time-varying networks
- **Survival Analysis**: Hazard modeling with cloglog link
- **Financial Economics**: Market microstructure, contagion theory

## References

- Allen, F., & Gale, D. (2000). Financial contagion. *Journal of Political Economy*.
- Cont, R., & Wagalath, L. (2016). Fire sales forensics: measuring endogenous risk. *Mathematical Finance*.
- Battiston, S., et al. (2016). Complexity theory and financial regulation. *Science*.

## License

MIT License

## Author

Created for AM215 - Advanced Scientific Computing

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.
