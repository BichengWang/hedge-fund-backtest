"""
Backtesting engine: run_backtest, walk_forward_backtest, calculate_metrics.

Design principles:
  - NO LOOK-AHEAD BIAS: signals are always lagged 1 day before portfolio construction.
  - Walk-forward windows never overlap training and test sets.
  - All metrics are computed on out-of-sample (test) data in walk-forward mode.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.portfolio import (
    apply_constraints,
    build_long_short_portfolio,
    volatility_scale_positions,
)
from backtest.costs import apply_costs


@dataclass
class WalkForwardResult:
    """Result of a single walk-forward window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_returns: pd.Series
    test_weights: pd.DataFrame
    metrics: Dict


def run_backtest(
    prices: pd.DataFrame,
    signal_func: Callable[[pd.DataFrame], pd.DataFrame],
    start: Optional[str] = None,
    end: Optional[str] = None,
    top_pct: float = 0.2,
    target_vol: float = 0.15,
    max_position: float = 0.10,
    asset_class: str = "us_equity_etf",
    apply_vol_scaling: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run a full backtest over the specified date range.

    Parameters
    ----------
    prices       : price DataFrame
    signal_func  : function(prices) -> signal DataFrame
    start / end  : date range filter (ISO strings)
    top_pct      : long/short portfolio leg size
    target_vol   : annualized target volatility
    max_position : per-asset position cap
    asset_class  : cost model
    apply_vol_scaling : whether to volatility-scale positions

    Returns
    -------
    (net_returns, weights) : pd.Series, pd.DataFrame
    """
    if start:
        prices = prices.loc[start:]
    if end:
        prices = prices.loc[:end]

    from backtest.data import calculate_returns
    returns = calculate_returns(prices, method="log")

    # Generate signals and SHIFT BY 1 DAY to avoid look-ahead bias
    raw_signals = signal_func(prices)
    signals = raw_signals.shift(1)  # <-- critical: use yesterday's signal today

    weights = build_long_short_portfolio(signals, prices, top_pct=top_pct)

    if apply_vol_scaling:
        weights = volatility_scale_positions(weights, returns, target_vol=target_vol)

    weights = apply_constraints(weights, max_position=max_position)

    net_returns = apply_costs(returns, weights, asset_class=asset_class)
    net_returns.name = "strategy"

    return net_returns, weights


def walk_forward_backtest(
    prices: pd.DataFrame,
    signal_func: Callable[[pd.DataFrame], pd.DataFrame],
    train_days: int = 504,    # ~2 years
    test_days: int = 63,      # ~3 months
    step_days: int = 21,      # ~1 month step
    top_pct: float = 0.2,
    target_vol: float = 0.15,
    max_position: float = 0.10,
    asset_class: str = "us_equity_etf",
    apply_vol_scaling: bool = True,
) -> List[WalkForwardResult]:
    """
    Walk-forward validation: train on in-sample data, evaluate on out-of-sample.

    The training window NEVER includes any dates from the test window.
    Walk-forward avoids look-ahead bias by design.

    Parameters
    ----------
    train_days : number of trading days in each training window
    test_days  : number of trading days in each test window
    step_days  : number of days to advance the window each iteration

    Returns
    -------
    List of WalkForwardResult (one per test window)
    """
    from backtest.data import calculate_returns

    dates = prices.index
    n = len(dates)
    results: List[WalkForwardResult] = []

    start_idx = train_days
    window_num = 0

    while start_idx + test_days <= n:
        train_start_idx = max(0, start_idx - train_days)
        train_end_idx = start_idx - 1
        test_start_idx = start_idx
        test_end_idx = min(n - 1, start_idx + test_days - 1)

        train_prices = prices.iloc[train_start_idx : train_end_idx + 1]
        test_prices = prices.iloc[test_start_idx : test_end_idx + 1]

        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]

        window_num += 1
        print(
            f"[engine] WF window {window_num}: "
            f"train [{train_start.date()} -> {train_end.date()}], "
            f"test [{test_start.date()} -> {test_end.date()}]"
        )

        try:
            # Fit signal on training data only
            train_signals = signal_func(train_prices)

            # Generate signals on the test price window using FULL history up to
            # test period (train + test prices), then slice to test period only.
            # This simulates a model retrained on train data applied forward.
            combined_prices = prices.iloc[train_start_idx : test_end_idx + 1]
            combined_signals = signal_func(combined_prices)

            # Shift signals by 1 to avoid look-ahead
            combined_signals_lagged = combined_signals.shift(1)

            # Slice to test period
            test_signals = combined_signals_lagged.loc[test_start:test_end]
            test_prices_slice = combined_prices.loc[test_start:test_end]
            test_returns = calculate_returns(
                combined_prices.loc[train_end:test_end], method="log"
            ).loc[test_start:test_end]

            weights = build_long_short_portfolio(
                test_signals, test_prices_slice, top_pct=top_pct
            )

            if apply_vol_scaling:
                # Use full combined returns for vol estimation
                combined_returns = calculate_returns(combined_prices, method="log")
                combined_weights = build_long_short_portfolio(
                    combined_signals_lagged, combined_prices, top_pct=top_pct
                )
                combined_weights_scaled = volatility_scale_positions(
                    combined_weights, combined_returns, target_vol=target_vol
                )
                weights = combined_weights_scaled.loc[test_start:test_end]

            weights = apply_constraints(weights, max_position=max_position)
            net_returns = apply_costs(test_returns, weights, asset_class=asset_class)

        except Exception as exc:
            warnings.warn(f"Walk-forward window {window_num} failed: {exc}")
            start_idx += step_days
            continue

        metrics = calculate_metrics(net_returns)
        results.append(
            WalkForwardResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                test_returns=net_returns,
                test_weights=weights,
                metrics=metrics,
            )
        )
        start_idx += step_days

    print(f"[engine] Walk-forward complete: {len(results)} windows evaluated")
    return results


def calculate_metrics(returns: pd.Series, risk_free: float = 0.04) -> Dict:
    """
    Calculate comprehensive performance metrics.

    Parameters
    ----------
    returns    : daily net return series
    risk_free  : annual risk-free rate

    Returns
    -------
    dict with keys: Sharpe, Sortino, MaxDD, Calmar, WinRate, Turnover,
                    AnnReturn, AnnVol, TotalReturn, SkewReturn, KurtReturn
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return {}

    daily_rf = risk_free / 252
    excess = returns - daily_rf

    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else np.nan

    downside = returns[returns < daily_rf]
    sortino_denom = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = (ann_return - risk_free) / sortino_denom if sortino_denom and sortino_denom > 0 else np.nan

    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    drawdowns = (cum - roll_max) / roll_max
    max_dd = drawdowns.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (returns > 0).mean()
    total_return = cum.iloc[-1] - 1 if len(cum) > 0 else np.nan

    from scipy.stats import skew, kurtosis
    skewness = skew(returns)
    kurt = kurtosis(returns)

    return {
        "AnnReturn": round(ann_return, 4),
        "AnnVol": round(ann_vol, 4),
        "Sharpe": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "Sortino": round(sortino, 3) if not np.isnan(sortino) else np.nan,
        "MaxDD": round(max_dd, 4),
        "Calmar": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "WinRate": round(win_rate, 4),
        "TotalReturn": round(total_return, 4),
        "Skewness": round(skewness, 3),
        "Kurtosis": round(kurt, 3),
        "NumDays": len(returns),
    }
