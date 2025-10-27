"""
Model validation: Compare SIR simulations against historical data.

This script runs backtests from multiple historical dates to validate
that the model can accurately predict future crisis trajectories.
"""
import numpy as np
import pandas as pd
from datetime import datetime

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
from model import fit_cloglog_glm, estimate_recovery_hazard
from backtesting import (
    backtest_simulation,
    compute_accuracy_metrics,
    plot_backtest_comparison,
    run_multiple_backtests
)


def main():
    """Run model validation against historical data."""
    print("=" * 80)
    print("MODEL VALIDATION: BACKTESTING AGAINST HISTORICAL DATA")
    print("=" * 80)

    # ========================================================================
    # 1. LOAD DATA (same as run.py)
    # ========================================================================
    print("\n[1/5] Loading data...")

    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN',
        'JPM', 'BAC', 'GS', 'WFC',
        'XOM', 'CVX',
        'WMT', 'PG', 'KO', 'MCD',
        'JNJ', 'UNH', 'PFE',
        'BA', 'CAT', 'MMM',
        'DIS', 'NFLX',
        'TSLA', 'GM', 'F',
        'SPY', 'QQQ'
    ]

    start_date = "2016-01-01"
    end_date = "2024-12-31"

    prices = fetch_yahoo_prices(tickers, start_date, end_date)
    vix = fetch_vix(start_date, end_date)
    prices, vix = _safe_align(prices, vix)
    vix_z = compute_market_stress(vix, window=252)

    print(f"   Loaded {len(prices)} days × {len(prices.columns)} assets")

    # ========================================================================
    # 2. LABEL STATES AND BUILD NETWORK
    # ========================================================================
    print("\n[2/5] Labeling states and building network...")

    S, I, R = label_crisis_states(prices)
    rets = compute_returns(prices)
    W_by_date = rolling_corr_network(rets, window=90, topk=10)

    print(f"   Network built for {len(W_by_date)} dates")

    # ========================================================================
    # 3. TRAIN MODEL
    # ========================================================================
    print("\n[3/5] Training model...")

    ds = build_infection_dataset(S, I, rets, W_by_date, market_stress=vix_z)
    split_date = pd.Timestamp("2020-01-01")

    X_tr = ds.X.loc[ds.X.index.get_level_values(0) < split_date]
    y_tr = ds.y.loc[X_tr.index]

    train_ds = InfectionDataset(X=X_tr, y=y_tr)
    model = fit_cloglog_glm(train_ds)

    beta_hat = float(model.result.params["exposure"])
    mu_hat = estimate_recovery_hazard(I)

    print(f"   Model fitted: beta={beta_hat:.3f}, mu={mu_hat:.3f}")

    # ========================================================================
    # 4. RUN BACKTESTS ON KEY HISTORICAL PERIODS
    # ========================================================================
    print("\n[4/5] Running backtests...")

    # Select test dates: Find dates with high infection counts for interesting periods
    # Get dates from test period (2020 onwards) with at least 5 infected stocks
    test_period = S.loc[split_date:]
    infected_counts = test_period.loc[I.loc[split_date:].sum(axis=1) >= 5]

    # Select representative dates (evenly spaced)
    if len(infected_counts) > 0:
        indices = np.linspace(0, len(infected_counts)-1, min(5, len(infected_counts)), dtype=int)
        test_dates = [infected_counts.index[i] for i in indices]
        # Add COVID crash specifically if available
        covid_candidates = S.loc["2020-02-01":"2020-03-15"].index
        covid_dates_in_network = [d for d in covid_candidates if d in W_by_date]
        if covid_dates_in_network:
            test_dates.insert(0, covid_dates_in_network[len(covid_dates_in_network)//2])
    else:
        # Fallback: just sample from test period
        test_period_with_network = [d for d in test_period.index if d in W_by_date]
        indices = np.linspace(0, len(test_period_with_network)-1, 5, dtype=int)
        test_dates = [test_period_with_network[i] for i in indices]

    # Remove duplicates and sort
    test_dates = sorted(list(set(test_dates)))

    print(f"   Testing {len(test_dates)} historical periods...")
    print(f"   Periods: {[d.date() for d in test_dates]}")

    # Run detailed backtest for first test period (most interesting)
    if len(test_dates) > 0:
        covid_date = test_dates[0]
        print(f"\n   Running detailed backtest for COVID period (starting {covid_date.date()})...")

        actual, predicted = backtest_simulation(
            S, I, R, W_by_date, vix_z,
            beta=beta_hat,
            mu=mu_hat,
            start_date=covid_date,
            n_days=60,  # 2 months forward
            n_simulations=200
        )

        metrics = compute_accuracy_metrics(actual, predicted)

        print("\n" + "=" * 80)
        print(f"DETAILED BACKTEST RESULTS - PERIOD STARTING {covid_date.date()}")
        print("=" * 80)
        print(f"\nStarting from: {covid_date.date()}")
        print(f"Initial state: S={actual['S'].iloc[0]}, I={actual['I'].iloc[0]}, R={actual['R'].iloc[0]}")
        print(f"\nAfter 60 days:")
        print(f"Actual: S={actual['S'].iloc[-1]}, I={actual['I'].iloc[-1]}, R={actual['R'].iloc[-1]}")
        print(f"Predicted: S={predicted['S_mean'].iloc[-1]:.1f}, I={predicted['I_mean'].iloc[-1]:.1f}, R={predicted['R_mean'].iloc[-1]:.1f}")

        print("\n" + "-" * 80)
        print("ACCURACY METRICS")
        print("-" * 80)
        print(metrics.to_string(index=False))
        print("-" * 80)

        # Create visualization
        print("\n   Generating plot...")
        plot_backtest_comparison(
            actual, predicted, covid_date,
            save_path="../figures/backtest_covid.png"
        )

    # ========================================================================
    # 5. AGGREGATE RESULTS ACROSS MULTIPLE PERIODS
    # ========================================================================
    print("\n[5/5] Running aggregate validation across multiple periods...")

    all_metrics = run_multiple_backtests(
        S, I, R, W_by_date, vix_z,
        beta=beta_hat,
        mu=mu_hat,
        test_dates=test_dates,
        n_days=30,
        n_simulations=100
    )

    if not all_metrics.empty:
        # Compute average metrics across all test periods
        summary = all_metrics.groupby('State')[['RMSE', 'MAE', 'MAPE (%)', 'Coverage (%)', 'Correlation']].mean()

        print("\n" + "=" * 80)
        print("AGGREGATE VALIDATION RESULTS (ACROSS ALL TEST PERIODS)")
        print("=" * 80)
        print(summary.to_string())
        print("=" * 80)

        # Per-period breakdown
        print("\n" + "-" * 80)
        print("PER-PERIOD BREAKDOWN")
        print("-" * 80)
        for date in test_dates:
            period_metrics = all_metrics[all_metrics['start_date'] == date]
            if not period_metrics.empty:
                print(f"\nPeriod starting {date.date()}:")
                print(period_metrics[['State', 'RMSE', 'MAE', 'Coverage (%)']].to_string(index=False))

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nInterpretation:")
    print("- RMSE/MAE: Lower is better (prediction error)")
    print("- Coverage: Should be ~90% (actual values within prediction intervals)")
    print("- Correlation: Higher is better (1.0 = perfect)")
    print("- MAPE: Mean absolute percentage error")


if __name__ == "__main__":
    main()
