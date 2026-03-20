"""
Portfolio construction: long/short weighting, volatility scaling, constraints.

NO LOOK-AHEAD BIAS: Functions here operate on signals that have already been
shifted by 1 day by the caller. Do not shift again inside these functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def build_long_short_portfolio(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    top_pct: float = 0.2,
    dollar_neutral: bool = True,
) -> pd.DataFrame:
    """
    Build a long/short portfolio from cross-sectional alpha signals.

    Each day:
      - Long the top `top_pct` fraction of assets by signal.
      - Short the bottom `top_pct` fraction.
      - Equal-weight within each leg.
      - Optionally enforce dollar-neutrality (long notional = short notional).

    Parameters
    ----------
    signals      : z-scored signal DataFrame (rows=dates, cols=assets)
    prices       : price DataFrame (used only to align index)
    top_pct      : fraction of assets in each leg (e.g. 0.2 = top/bottom quintile)
    dollar_neutral: if True, scale longs and shorts to equal notional

    Returns
    -------
    pd.DataFrame of portfolio weights (same shape as signals).
    Positive = long, negative = short. Rows sum to ~0 (dollar-neutral).
    """
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    n_assets = signals.shape[1]
    n_leg = max(1, int(np.floor(n_assets * top_pct)))

    for date in signals.index:
        row = signals.loc[date].dropna()
        if len(row) < 2 * n_leg:
            continue

        ranked = row.rank(ascending=False)
        longs = ranked[ranked <= n_leg].index
        shorts = ranked[ranked > len(row) - n_leg].index

        w = pd.Series(0.0, index=signals.columns)
        w[longs] = 1.0 / n_leg
        w[shorts] = -1.0 / n_leg

        if dollar_neutral:
            long_sum = w[w > 0].sum()
            short_sum = abs(w[w < 0].sum())
            if long_sum > 0 and short_sum > 0:
                avg = (long_sum + short_sum) / 2
                w[w > 0] *= avg / long_sum
                w[w < 0] *= avg / short_sum

        weights.loc[date] = w

    return weights


def volatility_scale_positions(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.15,
    vol_window: int = 60,
    min_vol: float = 0.005,
) -> pd.DataFrame:
    """
    Scale portfolio weights so that ex-ante portfolio volatility targets `target_vol`.

    Uses a rolling realized volatility estimate (annualized) of the portfolio
    return series to compute a scalar multiplier each day.

    Parameters
    ----------
    weights    : position weight DataFrame
    returns    : asset return DataFrame (log or simple)
    target_vol : annualized target volatility (e.g. 0.15 = 15%)
    vol_window : rolling window for vol estimation (days)
    min_vol    : floor on portfolio vol to avoid extreme leverage

    Returns
    -------
    Scaled weight DataFrame.
    """
    # Compute portfolio returns using lagged weights (no look-ahead)
    port_returns = (weights.shift(1) * returns).sum(axis=1)

    # Rolling annualized vol
    roll_vol = port_returns.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
    roll_vol = roll_vol.clip(lower=min_vol)

    # Scalar multiplier
    scaler = (target_vol / roll_vol).fillna(1.0).clip(upper=5.0)

    scaled = weights.mul(scaler, axis=0)
    return scaled


def apply_constraints(
    weights: pd.DataFrame,
    max_position: float = 0.05,
    max_gross: float = 2.0,
) -> pd.DataFrame:
    """
    Apply position-level constraints to a weight DataFrame.

    Parameters
    ----------
    weights      : weight DataFrame (positive=long, negative=short)
    max_position : maximum absolute weight per asset (e.g. 0.05 = 5%)
    max_gross    : maximum sum of absolute weights (gross leverage cap)

    Returns
    -------
    Clipped and renormalized weight DataFrame.
    """
    # Clip individual positions
    clipped = weights.clip(lower=-max_position, upper=max_position)

    # Enforce gross leverage cap by scaling down proportionally
    gross = clipped.abs().sum(axis=1)
    excess = gross > max_gross
    if excess.any():
        scale = (max_gross / gross).clip(upper=1.0)
        clipped = clipped.mul(scale, axis=0)

    return clipped
