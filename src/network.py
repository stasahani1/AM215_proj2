"""Network construction based on rolling correlations."""
import numpy as np
import pandas as pd
from typing import Dict


def rolling_corr_network(
    returns: pd.DataFrame,
    window: int = 90,
    topk: int = 10,
    nonnegative: bool = True,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Build a dict: date -> row-stochastic weight matrix W_t (shape n×n),
    using rolling correlations; keep top-k absolute corr per row; zero others.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns with dates as index and tickers as columns
    window : int
        Rolling window size for correlation calculation (default: 90 days)
    topk : int
        Number of top correlations to keep per row (default: 10)
    nonnegative : bool
        If True, clip negative correlations to zero (default: True)

    Returns
    -------
    Dict[pd.Timestamp, np.ndarray]
        Dictionary mapping dates to row-normalized weight matrices
    """
    returns = returns.sort_index()
    tickers = returns.columns.to_list()
    n = len(tickers)
    W_by_date = {}

    for t in range(window, len(returns)):
        dt = returns.index[t]
        sub = returns.iloc[t - window: t]
        C = sub.corr().values  # n×n

        if nonnegative:
            C = np.clip(C, 0, None)

        # zero self loops
        np.fill_diagonal(C, 0.0)

        # sparsify by top-k per row
        W = np.zeros_like(C)
        for i in range(n):
            row = C[i, :]
            if topk >= 1:
                idx = np.argpartition(row, -topk)[-topk:]
                W[i, idx] = row[idx]

        # row-normalize
        rowsum = W.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        W = W / rowsum

        W_by_date[dt] = W

    return W_by_date
