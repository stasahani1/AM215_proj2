"""Dataset construction for infection hazard modeling."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Dict, Optional
from data_utils import _safe_align


@dataclass
class InfectionDataset:
    """Container for features and labels for infection modeling."""
    X: pd.DataFrame
    y: pd.Series


def build_infection_dataset(
    S: pd.DataFrame,
    I: pd.DataFrame,
    returns: pd.DataFrame,
    W_by_date: Dict[pd.Timestamp, np.ndarray],
    market_stress: Optional[pd.Series] = None,
    add_controls: bool = True
) -> InfectionDataset:
    """
    Build daily panel of S->I transitions:
      y_{i,t} = 1 if stock i becomes infected at t+1 (given S at t)
      X contains:
         exposure_{i,t} = sum_j W_{ij,t} * I_{j,t}
         market_stress_t (z-scored)
         last_ret_{i,t}

    Parameters
    ----------
    S : pd.DataFrame
        Susceptible state indicators
    I : pd.DataFrame
        Infected state indicators
    returns : pd.DataFrame
        Daily returns
    W_by_date : Dict[pd.Timestamp, np.ndarray]
        Network weights by date
    market_stress : Optional[pd.Series]
        Market stress indicator (e.g., z-scored VIX)
    add_controls : bool
        Whether to add control variables like lagged returns

    Returns
    -------
    InfectionDataset
        Dataset with features X and labels y
    """
    S, I, returns = _safe_align(S, I, returns)
    if market_stress is not None:
        (market_stress,) = _safe_align(market_stress)

    tickers = S.columns.to_list()
    n = len(tickers)

    rows = []
    ys = []
    for t_idx in range(1, len(S) - 1):  # we will use W at date t and y at t+1
        dt = S.index[t_idx]
        if dt not in W_by_date:
            continue
        W = W_by_date[dt]  # n×n aligned with tickers order we used to build W
        S_t = S.iloc[t_idx].values.astype(int)
        I_t = I.iloc[t_idx].values.astype(int)
        I_next = I.iloc[t_idx + 1].values.astype(int)

        exposure = W @ I_t  # contagion pressure

        # optional features
        last_ret = returns.iloc[t_idx].values
        m = market_stress.loc[dt] if market_stress is not None else 0.0

        for i, tic in enumerate(tickers):
            if S_t[i] != 1:
                continue  # only build obs for currently susceptible
            xrow = {
                "exposure": float(exposure[i]),
                "market": float(m),
            }
            if add_controls:
                xrow["last_ret"] = float(last_ret[i])
            xrow["ticker"] = tic
            xrow["date"] = dt
            rows.append(xrow)
            ys.append(int(I_next[i]))  # did it turn infected tomorrow?

    X = pd.DataFrame(rows).set_index(["date", "ticker"])
    y = pd.Series(ys, index=X.index, name="y")
    # Add intercept
    X = sm.add_constant(X, has_constant="add")

    return InfectionDataset(X=X, y=y)
