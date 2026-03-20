"""
Transaction cost estimation and application.

Cost model reference:
  - US equity ETFs: ~5-10 bps round-trip (liquid, narrow spread)
  - US large-cap stocks: ~10-20 bps round-trip
  - US small-cap stocks: ~30-50 bps round-trip
  - Market impact scales with sqrt(participation rate) - not modelled here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# Cost tables in basis points (round-trip)
COST_TABLE = {
    "us_equity_etf": 7,      # liquid ETFs, tight spread
    "us_equity": 15,          # typical large-cap stock
    "us_equity_smallcap": 40, # small-cap / illiquid
    "futures": 3,             # equity index futures
    "fx": 5,                  # major FX pairs
}


def estimate_trade_costs(
    turnover: float,
    asset_class: str = "us_equity",
) -> float:
    """
    Estimate one-way transaction cost given turnover and asset class.

    Cost is expressed as a fraction of portfolio value (not bps).

    Parameters
    ----------
    turnover    : one-way portfolio turnover as fraction (e.g. 0.10 = 10%)
    asset_class : one of the keys in COST_TABLE

    Returns
    -------
    float : cost as fraction of portfolio value
    """
    cost_bps = COST_TABLE.get(asset_class, COST_TABLE["us_equity"])
    cost_fraction = turnover * cost_bps / 10_000
    return cost_fraction


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute daily one-way turnover from weight changes.

    Turnover at day t = 0.5 * sum(|w_t - w_{t-1}|)
    (divide by 2 because each trade has a buyer and seller side)

    Parameters
    ----------
    weights : weight DataFrame

    Returns
    -------
    pd.Series of daily one-way turnover fractions.
    """
    delta = weights.diff().abs().sum(axis=1) / 2
    return delta


def apply_costs(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    asset_class: str = "us_equity_etf",
) -> pd.Series:
    """
    Compute net portfolio returns after transaction costs.

    Steps:
      1. Compute gross portfolio return each day using lagged weights.
      2. Compute turnover (weight changes due to rebalancing).
      3. Subtract cost from the return series.

    NO LOOK-AHEAD BIAS: weights are shifted by 1 before applying to returns.

    Parameters
    ----------
    returns     : asset return DataFrame
    weights     : portfolio weight DataFrame (unshifted)
    asset_class : cost model to use

    Returns
    -------
    pd.Series : net daily portfolio returns
    """
    aligned_weights = weights.shift(1).reindex(returns.index).fillna(0.0)
    aligned_returns = returns.reindex(weights.index).fillna(0.0)

    gross_returns = (aligned_weights * aligned_returns).sum(axis=1)

    turnover = compute_turnover(weights).reindex(gross_returns.index).fillna(0.0)
    daily_costs = turnover.apply(lambda t: estimate_trade_costs(t, asset_class))

    net_returns = gross_returns - daily_costs
    return net_returns
