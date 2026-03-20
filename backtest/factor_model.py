"""
Factor model: alpha/beta decomposition using OLS regression.

Helps answer: Is strategy return due to skill (alpha) or market exposure (beta)?
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def get_market_factor(prices: pd.DataFrame) -> pd.Series:
    """
    Compute the equal-weight market return from a price DataFrame.

    In practice, use SPY or a capitalization-weighted benchmark.
    Here we use the equal-weight average of all assets as a simple proxy.

    Parameters
    ----------
    prices : price DataFrame

    Returns
    -------
    pd.Series of daily log returns representing the market factor.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    market = log_returns.mean(axis=1)
    market.name = "market"
    return market


def calculate_factor_exposures(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    additional_factors: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Regress strategy returns on market (and optional additional factors).

    Returns alpha, beta, t-statistics, p-values, and R-squared.

    Parameters
    ----------
    strategy_returns    : daily strategy return series
    market_returns      : daily market return series
    additional_factors  : optional DataFrame of additional factor returns

    Returns
    -------
    dict with keys: alpha, beta, alpha_tstat, beta_tstat,
                    alpha_pval, beta_pval, r_squared, adj_r_squared,
                    annualized_alpha, information_ratio
    """
    common = strategy_returns.index.intersection(market_returns.index)
    y = strategy_returns.loc[common].dropna()

    if additional_factors is not None:
        factors = pd.concat([market_returns.loc[common], additional_factors.loc[common]], axis=1)
    else:
        factors = market_returns.loc[common].to_frame("market")

    factors = factors.dropna()
    common2 = y.index.intersection(factors.index)
    y = y.loc[common2]
    X = add_constant(factors.loc[common2])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OLS(y, X).fit()

    alpha_daily = model.params["const"]
    beta = model.params.get("market", model.params.iloc[1])
    alpha_tstat = model.tvalues["const"]
    beta_tstat = model.tvalues.get("market", model.tvalues.iloc[1])
    alpha_pval = model.pvalues["const"]
    beta_pval = model.pvalues.get("market", model.pvalues.iloc[1])

    annualized_alpha = alpha_daily * 252
    residuals = model.resid
    ir = residuals.mean() / residuals.std() * np.sqrt(252) if residuals.std() > 0 else np.nan

    return {
        "alpha_daily": round(alpha_daily, 6),
        "alpha_annualized": round(annualized_alpha, 4),
        "beta": round(float(beta), 4),
        "alpha_tstat": round(alpha_tstat, 3),
        "beta_tstat": round(float(beta_tstat), 3),
        "alpha_pval": round(alpha_pval, 4),
        "beta_pval": round(float(beta_pval), 4),
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "information_ratio": round(ir, 3) if not np.isnan(ir) else np.nan,
        "n_obs": len(y),
    }


def decompose_returns(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose strategy returns into alpha + factor contributions.

    Returns a DataFrame with columns:
      - alpha_contribution : daily alpha return (intercept)
      - <factor>_contribution : each factor's contribution (beta * factor_return)
      - residual : unexplained component

    Parameters
    ----------
    strategy_returns : daily strategy return series
    factors          : DataFrame of factor returns (cols = factor names)

    Returns
    -------
    pd.DataFrame of daily return decomposition.
    """
    common = strategy_returns.index.intersection(factors.index)
    y = strategy_returns.loc[common].dropna()
    X_raw = factors.loc[common].dropna()
    common2 = y.index.intersection(X_raw.index)
    y = y.loc[common2]
    X_raw = X_raw.loc[common2]

    X = add_constant(X_raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OLS(y, X).fit()

    decomp = pd.DataFrame(index=common2)
    decomp["alpha_contribution"] = model.params["const"]

    for col in X_raw.columns:
        decomp[f"{col}_contribution"] = model.params[col] * X_raw[col]

    fitted = decomp.sum(axis=1)
    decomp["residual"] = y - fitted
    decomp["total_return"] = y

    return decomp


def print_factor_report(exposures: Dict, strategy_name: str = "Strategy") -> None:
    """Pretty-print factor exposure results."""
    print(f"\n{'='*55}")
    print(f"  Factor Model: {strategy_name}")
    print(f"{'='*55}")
    print(f"  Annualized Alpha : {exposures['alpha_annualized']*100:+.2f}%")
    print(f"  Alpha t-stat     : {exposures['alpha_tstat']:+.2f}  (p={exposures['alpha_pval']:.3f})")
    print(f"  Market Beta      : {exposures['beta']:+.3f}")
    print(f"  Beta t-stat      : {exposures['beta_tstat']:+.2f}  (p={exposures['beta_pval']:.3f})")
    print(f"  R-squared        : {exposures['r_squared']:.4f}")
    print(f"  Info Ratio (res) : {exposures['information_ratio']:.3f}")
    print(f"  N observations   : {exposures['n_obs']}")
    alpha_sig = "**SIGNIFICANT**" if exposures['alpha_pval'] < 0.05 else "not significant"
    print(f"  Alpha is         : {alpha_sig} at 5% level")
    print(f"{'='*55}")
