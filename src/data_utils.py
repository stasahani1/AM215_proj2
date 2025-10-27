"""Data utilities for loading and processing financial data."""
import numpy as np
import pandas as pd
from typing import List
import yfinance as yf


def _safe_align(*objs):
    """Align by index intersection in time; maintain sort."""
    idx = objs[0].index
    for o in objs[1:]:
        idx = idx.intersection(o.index)
    idx = idx.sort_values()
    return [o.loc[idx] for o in objs]


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from prices."""
    return prices.sort_index().pct_change()


def rolling_drawdown(prices: pd.Series, window: int) -> pd.Series:
    """Windowed pct drawdown over the last `window` days: (price / rolling_max) - 1."""
    roll_max = prices.rolling(window, min_periods=window).max()
    return (prices / roll_max) - 1.0


def rolling_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Rolling std of daily returns (not annualized; we compare to in-sample quantiles)."""
    return returns.rolling(window, min_periods=window).std()


def fetch_yahoo_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Handle single ticker case
    if len(tickers) == 1:
        prices = data['Adj Close'].to_frame()
        prices.columns = tickers
    else:
        prices = data['Adj Close']

    # Drop any columns with all NaN
    prices = prices.dropna(axis=1, how='all')

    return prices


def fetch_vix(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch VIX index from Yahoo Finance as market stress indicator.

    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'

    Returns
    -------
    pd.Series
        VIX values indexed by date
    """
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    return vix_data['Adj Close']


def compute_market_stress(vix: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute z-scored market stress from VIX.

    Parameters
    ----------
    vix : pd.Series
        Raw VIX values
    window : int
        Rolling window for computing mean/std (default: 252 trading days ≈ 1 year)

    Returns
    -------
    pd.Series
        Z-scored market stress indicator
    """
    vix_mean = vix.rolling(window, min_periods=100).mean()
    vix_std = vix.rolling(window, min_periods=100).std()
    vix_z = (vix - vix_mean) / vix_std
    return vix_z.fillna(0.0)
