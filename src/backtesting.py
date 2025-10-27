"""Backtesting and model validation against historical data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from simulation import simulate_sir


def backtest_simulation(
    S: pd.DataFrame,
    I: pd.DataFrame,
    R: pd.DataFrame,
    W_by_date: Dict[pd.Timestamp, np.ndarray],
    market_stress: pd.Series,
    beta: float,
    mu: float,
    start_date: pd.Timestamp,
    n_days: int = 30,
    n_simulations: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run forward simulation from a historical date and compare to actual outcomes.

    Parameters
    ----------
    S, I, R : pd.DataFrame
        Actual historical states
    W_by_date : Dict
        Network weights by date
    market_stress : pd.Series
        Market stress time series
    beta : float
        Infection strength parameter
    mu : float
        Recovery probability
    start_date : pd.Timestamp
        Date to start simulation from
    n_days : int
        Number of days to simulate forward
    n_simulations : int
        Number of Monte Carlo runs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (actual_counts, predicted_stats) DataFrames with S/I/R counts over time
    """
    # Get initial states
    if start_date not in S.index or start_date not in W_by_date:
        raise ValueError(f"Start date {start_date} not found in data")

    S0 = S.loc[start_date].values.astype(int)
    I0 = I.loc[start_date].values.astype(int)
    R0 = R.loc[start_date].values.astype(int)
    W = W_by_date[start_date]

    # Get actual forward trajectory
    end_idx = S.index.get_loc(start_date) + n_days
    if end_idx >= len(S):
        end_idx = len(S) - 1
        n_days = end_idx - S.index.get_loc(start_date)

    dates = S.index[S.index.get_loc(start_date):end_idx + 1]
    actual_S = S.loc[dates].sum(axis=1).values
    actual_I = I.loc[dates].sum(axis=1).values
    actual_R = R.loc[dates].sum(axis=1).values

    actual_counts = pd.DataFrame({
        'S': actual_S,
        'I': actual_I,
        'R': actual_R,
        'day': range(len(dates))
    })

    # Build market stress path for simulation
    market_dates = dates[dates.isin(market_stress.index)]
    if len(market_dates) > 0:
        market_path = market_stress.loc[market_dates].values
        # Pad if necessary
        if len(market_path) < n_days:
            market_path = np.r_[market_path, np.zeros(n_days - len(market_path))]
    else:
        market_path = np.zeros(n_days)

    # Run multiple simulations
    all_S, all_I, all_R = [], [], []
    for _ in range(n_simulations):
        S_path, I_path, R_path = simulate_sir(
            S0.copy(), I0.copy(), R0.copy(),
            W, beta, mu, market_path, steps=n_days
        )
        all_S.append(S_path.sum(axis=1))
        all_I.append(I_path.sum(axis=1))
        all_R.append(R_path.sum(axis=1))

    all_S = np.array(all_S)  # (n_simulations, n_days+1)
    all_I = np.array(all_I)
    all_R = np.array(all_R)

    # Compute statistics
    predicted_stats = pd.DataFrame({
        'S_mean': all_S.mean(axis=0)[:len(actual_counts)],
        'S_std': all_S.std(axis=0)[:len(actual_counts)],
        'S_q05': np.percentile(all_S, 5, axis=0)[:len(actual_counts)],
        'S_q95': np.percentile(all_S, 95, axis=0)[:len(actual_counts)],
        'I_mean': all_I.mean(axis=0)[:len(actual_counts)],
        'I_std': all_I.std(axis=0)[:len(actual_counts)],
        'I_q05': np.percentile(all_I, 5, axis=0)[:len(actual_counts)],
        'I_q95': np.percentile(all_I, 95, axis=0)[:len(actual_counts)],
        'R_mean': all_R.mean(axis=0)[:len(actual_counts)],
        'R_std': all_R.std(axis=0)[:len(actual_counts)],
        'R_q05': np.percentile(all_R, 5, axis=0)[:len(actual_counts)],
        'R_q95': np.percentile(all_R, 95, axis=0)[:len(actual_counts)],
        'day': range(len(actual_counts))
    })

    return actual_counts, predicted_stats


def compute_accuracy_metrics(actual: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy metrics comparing actual vs predicted.

    Parameters
    ----------
    actual : pd.DataFrame
        Actual S/I/R counts with columns ['S', 'I', 'R', 'day']
    predicted : pd.DataFrame
        Predicted statistics with columns ['S_mean', 'I_mean', 'R_mean', etc.]

    Returns
    -------
    pd.DataFrame
        Metrics for each state variable
    """
    metrics = []

    for state in ['S', 'I', 'R']:
        actual_vals = actual[state].values
        pred_mean = predicted[f'{state}_mean'].values
        pred_q05 = predicted[f'{state}_q05'].values
        pred_q95 = predicted[f'{state}_q95'].values

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((actual_vals - pred_mean) ** 2))

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actual_vals - pred_mean))

        # MAPE (Mean Absolute Percentage Error) - avoid division by zero
        mape = np.mean(np.abs((actual_vals - pred_mean) / np.maximum(actual_vals, 1))) * 100

        # Coverage: % of actual values within 90% prediction interval
        in_interval = (actual_vals >= pred_q05) & (actual_vals <= pred_q95)
        coverage = in_interval.mean() * 100

        # Correlation
        corr = np.corrcoef(actual_vals, pred_mean)[0, 1]

        metrics.append({
            'State': state,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape,
            'Coverage (%)': coverage,
            'Correlation': corr
        })

    return pd.DataFrame(metrics)


def plot_backtest_comparison(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    start_date: pd.Timestamp,
    save_path: str = None
) -> None:
    """
    Create visualization comparing actual vs predicted trajectories.

    Parameters
    ----------
    actual : pd.DataFrame
        Actual S/I/R counts
    predicted : pd.DataFrame
        Predicted statistics
    start_date : pd.Timestamp
        Starting date of simulation
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    states = ['S', 'I', 'R']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    state_names = ['Susceptible', 'Infected', 'Recovered']

    for ax, state, color, name in zip(axes, states, colors, state_names):
        days = actual['day'].values

        # Plot actual
        ax.plot(days, actual[state], 'o-', color=color, linewidth=2,
                markersize=4, label='Actual', alpha=0.8)

        # Plot predicted mean
        ax.plot(days, predicted[f'{state}_mean'], '--', color='black',
                linewidth=2, label='Predicted Mean', alpha=0.7)

        # Plot 90% prediction interval
        ax.fill_between(days,
                        predicted[f'{state}_q05'],
                        predicted[f'{state}_q95'],
                        color=color, alpha=0.2, label='90% Prediction Interval')

        ax.set_xlabel('Days from Start', fontsize=11)
        ax.set_ylabel(f'Number of {name} Stocks', fontsize=11)
        ax.set_title(f'{name} Stocks: Actual vs Predicted', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Backtest Starting from {start_date.date()}',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved to: {save_path}")

    plt.show()


def run_multiple_backtests(
    S: pd.DataFrame,
    I: pd.DataFrame,
    R: pd.DataFrame,
    W_by_date: Dict[pd.Timestamp, np.ndarray],
    market_stress: pd.Series,
    beta: float,
    mu: float,
    test_dates: list,
    n_days: int = 30,
    n_simulations: int = 100,
) -> pd.DataFrame:
    """
    Run backtests for multiple starting dates and aggregate metrics.

    Parameters
    ----------
    test_dates : list
        List of dates to start backtests from

    Returns
    -------
    pd.DataFrame
        Aggregated metrics across all backtests
    """
    all_metrics = []

    for start_date in test_dates:
        try:
            actual, predicted = backtest_simulation(
                S, I, R, W_by_date, market_stress,
                beta, mu, start_date, n_days, n_simulations
            )
            metrics = compute_accuracy_metrics(actual, predicted)
            metrics['start_date'] = start_date
            all_metrics.append(metrics)
        except Exception as e:
            print(f"   Warning: Failed to backtest from {start_date}: {e}")

    if not all_metrics:
        return pd.DataFrame()

    return pd.concat(all_metrics, ignore_index=True)
