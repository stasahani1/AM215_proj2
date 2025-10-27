"""Crisis state labeling using S/I/R framework."""
import numpy as np
import pandas as pd
from typing import Tuple
from data_utils import compute_returns, rolling_drawdown, rolling_realized_vol


def enforce_min_spell_and_cooloff(
    i_series: pd.Series,
    min_spell: int,
    cooloff_days: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Debounce I spells to min_spell, and produce an R series that stays 1 for `cooloff_days`
    after exiting I. Returns (I_clean, R).

    Parameters
    ----------
    i_series : pd.Series
        Raw infection indicator (0 or 1)
    min_spell : int
        Minimum consecutive days to be considered a valid infection spell
    cooloff_days : int
        Number of days to remain in recovered state after infection

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Cleaned infection series and recovery series
    """
    i = i_series.fillna(0).astype(int).values
    n = len(i)

    # Debounce I (merge short blips into 0)
    start = 0
    while start < n:
        # move to next 1
        while start < n and i[start] == 0:
            start += 1
        if start >= n:
            break
        end = start
        while end < n and i[end] == 1:
            end += 1
        if (end - start) < min_spell:
            i[start:end] = 0
        start = end

    # Compute R
    r = np.zeros_like(i)
    t = 1
    while t < n:
        if i[t-1] == 1 and i[t] == 0:
            # exited infection at t
            r[t: min(n, t + cooloff_days)] = 1
        t += 1

    # If we re-enter I, force R to 0 (I dominates)
    r[i == 1] = 0

    return pd.Series(i, index=i_series.index), pd.Series(r, index=i_series.index)


def label_crisis_states(
    prices: pd.DataFrame,
    dd_window: int = 20,
    vol_window: int = 20,
    vol_hist_window: int = 252,
    dd_thresh: float = -0.10,
    vol_quantile: float = 0.95,
    min_spell: int = 3,
    cooloff_days: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build S/I/R in {0,1} for each ticker/date.
    I = 1 if (drawdown_20d <= dd_thresh) OR (rv20 >= 95th pct of trailing 1y rv20).
    R = 1 during cooloff after leaving I. S = (not I) & (not R).

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with dates as index and tickers as columns
    dd_window : int
        Window for computing drawdown (default: 20 days)
    vol_window : int
        Window for computing realized volatility (default: 20 days)
    vol_hist_window : int
        Historical window for volatility quantile (default: 252 days)
    dd_thresh : float
        Drawdown threshold for infection (default: -0.10 = -10%)
    vol_quantile : float
        Volatility quantile threshold (default: 0.95)
    min_spell : int
        Minimum infection spell length (default: 3 days)
    cooloff_days : int
        Recovery period after infection (default: 10 days)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        S, I, R state indicators (all with same shape as prices)
    """
    prices = prices.sort_index()
    rets = compute_returns(prices)

    I_list, R_list = [], []
    for col in prices.columns:
        dd = rolling_drawdown(prices[col], dd_window)
        rv20 = rolling_realized_vol(rets[col], vol_window)
        # rolling historical threshold of rv20
        rv20_q = rv20.rolling(vol_hist_window, min_periods=vol_hist_window).quantile(vol_quantile)

        infected_raw = ((dd <= dd_thresh) | (rv20 >= rv20_q)).astype(int)
        I_clean, R = enforce_min_spell_and_cooloff(
            infected_raw,
            min_spell=min_spell,
            cooloff_days=cooloff_days
        )
        I_list.append(I_clean.rename(col))
        R_list.append(R.rename(col))

    I = pd.concat(I_list, axis=1).fillna(0).astype(int)
    R = pd.concat(R_list, axis=1).fillna(0).astype(int)
    S = ((1 - I) * (1 - R)).astype(int)

    return S, I, R
