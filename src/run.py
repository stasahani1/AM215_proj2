"""
Main pipeline for SIR contagion modeling on financial markets.

This script:
1. Fetches stock price data from Yahoo Finance
2. Labels crisis states (S/I/R) based on drawdowns and volatility
3. Builds a rolling correlation network
4. Trains a GLM to predict infection hazard
5. Evaluates predictions out-of-sample
6. Estimates recovery rates
7. Runs scenario simulations

Usage:
    python run.py
"""
import numpy as np
import pandas as pd
from datetime import datetime

# Import our modules
from data_utils import (
    fetch_yahoo_prices,
    fetch_vix,
    compute_returns,
    compute_market_stress,
    _safe_align
)
from crisis_labeling import label_crisis_states
from network import rolling_corr_network
from dataset import build_infection_dataset, InfectionDataset
from model import fit_cloglog_glm, predict_prob, estimate_recovery_hazard
from evaluation import evaluate_predictions
from simulation import run_scenario


def main():
    """Run the full SIR contagion pipeline."""
    print("=" * 80)
    print("SIR CONTAGION MODEL FOR FINANCIAL MARKETS")
    print("=" * 80)

    # ========================================================================
    # 1. FETCH DATA FROM YAHOO FINANCE
    # ========================================================================
    print("\n[1/7] Fetching data from Yahoo Finance...")

    # Select a diverse set of stocks (tech, finance, energy, consumer, etc.)
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN',  # Tech
        'JPM', 'BAC', 'GS', 'WFC',  # Finance
        'XOM', 'CVX',  # Energy
        'WMT', 'PG', 'KO', 'MCD',  # Consumer
        'JNJ', 'UNH', 'PFE',  # Healthcare
        'BA', 'CAT', 'MMM',  # Industrial
        'DIS', 'NFLX',  # Entertainment
        'TSLA', 'GM', 'F',  # Auto
        'SPY', 'QQQ'  # ETFs for market reference
    ]

    # Fetch historical data (adjust dates as needed)
    start_date = "2016-01-01"
    end_date = "2024-12-31"

    print(f"   Tickers: {len(tickers)} stocks")
    print(f"   Period: {start_date} to {end_date}")

    prices = fetch_yahoo_prices(tickers, start_date, end_date)
    vix = fetch_vix(start_date, end_date)

    print(f"   Loaded {len(prices)} days × {len(prices.columns)} assets")
    print(f"   Date range: {prices.index[0]} to {prices.index[-1]}")

    # Align data
    prices, vix = _safe_align(prices, vix)
    vix_z = compute_market_stress(vix, window=252)

    # ========================================================================
    # 2. LABEL CRISIS STATES (S/I/R)
    # ========================================================================
    print("\n[2/7] Labeling crisis states (S/I/R)...")

    S, I, R = label_crisis_states(
        prices,
        dd_window=20,
        vol_window=20,
        vol_hist_window=252,
        dd_thresh=-0.10,
        vol_quantile=0.95,
        min_spell=3,
        cooloff_days=10
    )

    # Summary statistics
    total_obs = S.size
    print(f"   Total observations: {total_obs:,}")
    print(f"   Susceptible (S): {S.sum().sum():,} ({100*S.sum().sum()/total_obs:.1f}%)")
    print(f"   Infected (I): {I.sum().sum():,} ({100*I.sum().sum()/total_obs:.1f}%)")
    print(f"   Recovered (R): {R.sum().sum():,} ({100*R.sum().sum()/total_obs:.1f}%)")

    # ========================================================================
    # 3. BUILD ROLLING CORRELATION NETWORK
    # ========================================================================
    print("\n[3/7] Building rolling correlation network...")

    rets = compute_returns(prices)
    W_by_date = rolling_corr_network(
        rets,
        window=90,
        topk=10,
        nonnegative=True
    )

    print(f"   Built network for {len(W_by_date)} dates")
    print(f"   Network size: {rets.shape[1]} nodes, top-k={10} edges per node")

    # ========================================================================
    # 4. BUILD INFECTION DATASET
    # ========================================================================
    print("\n[4/7] Building infection dataset...")

    ds = build_infection_dataset(
        S, I, rets, W_by_date,
        market_stress=vix_z,
        add_controls=True
    )

    print(f"   Dataset size: {len(ds.X):,} observations")
    print(f"   Features: {list(ds.X.columns)}")
    print(f"   Infection rate (labels): {100*ds.y.mean():.2f}%")

    # ========================================================================
    # 5. TRAIN/TEST SPLIT AND MODEL FITTING
    # ========================================================================
    print("\n[5/7] Training infection hazard model...")

    # Split by date (use 2020 as cutoff for train/test)
    split_date = pd.Timestamp("2020-01-01")

    X_tr = ds.X.loc[ds.X.index.get_level_values(0) < split_date]
    y_tr = ds.y.loc[X_tr.index]
    X_te = ds.X.loc[ds.X.index.get_level_values(0) >= split_date]
    y_te = ds.y.loc[X_te.index]

    print(f"   Training: {len(X_tr):,} obs (before {split_date.date()})")
    print(f"   Testing: {len(X_te):,} obs (from {split_date.date()} onward)")

    # Fit model
    train_ds = InfectionDataset(X=X_tr, y=y_tr)
    model = fit_cloglog_glm(train_ds)

    print("\n   Model Summary:")
    print(model.result.summary())

    # ========================================================================
    # 6. EVALUATE OUT-OF-SAMPLE PREDICTIONS
    # ========================================================================
    print("\n[6/7] Evaluating out-of-sample predictions...")

    yhat_te = predict_prob(model, X_te)
    ev = evaluate_predictions(y_te, yhat_te)

    print(f"\n   Out-of-Sample Performance:")
    print(f"   - AUC: {ev.auc:.3f}")
    print(f"   - Average Precision: {ev.ap:.3f}")
    print(f"   - Brier Score: {ev.brier:.4f}")

    # Estimate recovery rate
    mu_hat = estimate_recovery_hazard(I)
    print(f"\n   Estimated daily recovery probability: {mu_hat:.3f}")
    print(f"   (Implies average infection duration: {1/mu_hat:.1f} days)")

    # ========================================================================
    # 7. SCENARIO SIMULATIONS
    # ========================================================================
    print("\n[7/7] Running scenario simulations...")

    # Use the most recent date with network data
    snap_date = max(W_by_date.keys())
    W_snap = W_by_date[snap_date]
    S0 = S.loc[snap_date].values.astype(int)
    I0 = I.loc[snap_date].values.astype(int)
    R0 = R.loc[snap_date].values.astype(int)

    print(f"\n   Snapshot date: {snap_date.date()}")
    print(f"   Initial state: S={S0.sum()}, I={I0.sum()}, R={R0.sum()}")

    # Get beta from model
    beta_hat = float(model.result.params["exposure"])
    print(f"   Using fitted beta (exposure coef): {beta_hat:.3f}")

    # Run scenarios
    scenarios = ["normal", "stressed", "severe"]
    print("\n   Running Monte Carlo simulations (100 runs each)...")

    for scenario in scenarios:
        result = run_scenario(
            S0, I0, R0, W_snap,
            beta=beta_hat,
            mu=mu_hat,
            market_scenario=scenario,
            steps=30,
            n_simulations=100
        )

        print(f"\n   Scenario: {scenario.upper()}")
        print(f"   - Peak infected (mean): {result['peak_infected_mean']:.1f}")
        print(f"   - Peak infected (95th percentile): {result['peak_infected_95th']:.1f}")
        print(f"   - Final recovered (mean): {result['final_recovered_mean']:.1f}")

    # ========================================================================
    # DONE
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
