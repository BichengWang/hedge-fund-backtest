"""
Alpha signal generation: momentum and statistical arbitrage (pairs trading).

NO LOOK-AHEAD BIAS: All signals are computed on data available at time t
and must be shifted by 1 day before use in portfolio construction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


# ---------------------------------------------------------------------------
# Momentum Signal
# ---------------------------------------------------------------------------

def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    skip_days: int = 1,
) -> pd.DataFrame:
    """
    Cross-sectional momentum signal, z-scored across assets each day.

    Momentum = cumulative return over [t - lookback - skip_days, t - skip_days]
    (skip_days avoids short-term reversal contamination).

    Parameters
    ----------
    prices   : price DataFrame (rows = dates, cols = assets)
    lookback : momentum window in trading days
    skip_days: days to skip before window (avoids 1-day reversal)

    Returns
    -------
    pd.DataFrame of z-scored momentum signals (same shape as prices).
    Signal is NaN for the first `lookback + skip_days` rows.
    """
    # Raw momentum: return from (t - lookback - skip_days) to (t - skip_days)
    end_px = prices.shift(skip_days)
    start_px = prices.shift(lookback + skip_days)
    raw_mom = np.log(end_px / start_px)

    # Cross-sectional z-score each day
    z = raw_mom.sub(raw_mom.mean(axis=1), axis=0).div(
        raw_mom.std(axis=1).replace(0, np.nan), axis=0
    )
    return z


# ---------------------------------------------------------------------------
# Cointegration / Pairs Trading
# ---------------------------------------------------------------------------

@dataclass
class PairInfo:
    asset_a: str
    asset_b: str
    p_value: float
    hedge_ratio: float   # beta of A ~ beta * B
    half_life: float     # mean-reversion half-life in days
    adf_pvalue: float    # ADF p-value on spread


def _estimate_half_life(spread: pd.Series) -> float:
    """Estimate Ornstein-Uhlenbeck half-life via OLS regression."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    idx = spread_lag.index.intersection(spread_diff.index)
    X = add_constant(spread_lag.loc[idx])
    y = spread_diff.loc[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OLS(y, X).fit()
    lam = model.params.iloc[1]
    if lam >= 0:
        return np.inf  # not mean-reverting
    return -np.log(2) / lam


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
    min_half_life: float = 2.0,
    max_half_life: float = 252.0,
) -> List[PairInfo]:
    """
    Scan all pairs for cointegration using the Engle-Granger test.

    Parameters
    ----------
    prices       : price DataFrame
    significance : p-value threshold for cointegration test
    min_half_life: minimum mean-reversion half-life (days)
    max_half_life: maximum mean-reversion half-life (days)

    Returns
    -------
    List of PairInfo sorted by p-value (best first).
    """
    cols = prices.columns.tolist()
    n = len(cols)
    results: List[PairInfo] = []

    print(f"[signals] Scanning {n*(n-1)//2} pairs for cointegration ...")
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cols[i], cols[j]
            ser_a = prices[a].dropna()
            ser_b = prices[b].dropna()
            common = ser_a.index.intersection(ser_b.index)
            if len(common) < 60:
                continue
            ya, yb = ser_a.loc[common], ser_b.loc[common]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score, pvalue, _ = coint(ya, yb)

            if pvalue > significance:
                continue

            # Estimate hedge ratio via OLS: A = alpha + beta * B
            X = add_constant(yb)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg = OLS(ya, X).fit()
            hedge_ratio = reg.params.iloc[1]
            spread = ya - hedge_ratio * yb

            # ADF test on spread
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_result = adfuller(spread)
            adf_pval = adf_result[1]

            hl = _estimate_half_life(spread)
            if not (min_half_life <= hl <= max_half_life):
                continue

            results.append(
                PairInfo(
                    asset_a=a,
                    asset_b=b,
                    p_value=pvalue,
                    hedge_ratio=hedge_ratio,
                    half_life=hl,
                    adf_pvalue=adf_pval,
                )
            )

    results.sort(key=lambda x: x.p_value)
    print(f"[signals] Found {len(results)} cointegrated pairs (p < {significance})")
    return results


def pairs_signal(
    prices: pd.DataFrame,
    pair: PairInfo,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> pd.Series:
    """
    Generate daily position signal for a cointegrated pair.

    Signal convention:
      +1  = long A / short B  (spread is below -entry_z)
      -1  = short A / long B  (spread is above +entry_z)
       0  = flat

    Uses a rolling z-score with a window = 2 * half_life to avoid look-ahead.

    Parameters
    ----------
    prices  : full price history
    pair    : PairInfo from find_cointegrated_pairs
    entry_z : z-score to enter trade
    exit_z  : z-score to exit trade
    stop_z  : z-score to stop-loss exit

    Returns
    -------
    pd.Series of {-1, 0, +1} signals indexed by date.
    """
    a, b = pair.asset_a, pair.asset_b
    if a not in prices.columns or b not in prices.columns:
        return pd.Series(dtype=float)

    spread = prices[a] - pair.hedge_ratio * prices[b]

    window = max(int(2 * pair.half_life), 20)
    roll_mean = spread.rolling(window, min_periods=window // 2).mean()
    roll_std = spread.rolling(window, min_periods=window // 2).std()
    z_score = (spread - roll_mean) / roll_std.replace(0, np.nan)

    signal = pd.Series(0.0, index=prices.index)
    position = 0.0

    for t in range(1, len(z_score)):
        z = z_score.iloc[t]
        if np.isnan(z):
            signal.iloc[t] = 0.0
            position = 0.0
            continue

        if position == 0:
            if z < -entry_z:
                position = 1.0   # long spread
            elif z > entry_z:
                position = -1.0  # short spread
        elif position == 1:
            if z > -exit_z or z > stop_z:
                position = 0.0
        elif position == -1:
            if z < exit_z or z < -stop_z:
                position = 0.0

        signal.iloc[t] = position

    return signal
