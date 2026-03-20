"""
Market regime detection for adaptive strategy filtering.

Regimes:
  bull : price > SMA50 > SMA200 AND 20-day vol < 252-day vol
  bear : price < SMA50 < SMA200
  chop : everything else
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class SimpleRegimeDetector:
    """
    Detects daily market regime using equal-weight average of all tickers.

    Uses SMA crossover (trend) and short-vs-long volatility (vol regime)
    to classify each day as bull, bear, or chop.
    """

    def detect(self, prices: pd.DataFrame) -> pd.Series:
        """
        Detect daily market regime.

        Parameters
        ----------
        prices : price DataFrame (rows=dates, cols=tickers)

        Returns
        -------
        pd.Series of str {"bull", "bear", "chop"} indexed by date.
        NaN / insufficient-history days default to "chop".
        """
        # Equal-weight average across all tickers (normalise first to avoid
        # price-level dominance from high-priced tickers)
        normed = prices.div(prices.iloc[0])
        avg_price = normed.mean(axis=1)

        sma50 = avg_price.rolling(50, min_periods=25).mean()
        sma200 = avg_price.rolling(200, min_periods=100).mean()

        daily_ret = avg_price.pct_change()
        vol_20 = daily_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
        vol_252 = daily_ret.rolling(252, min_periods=126).std() * np.sqrt(252)

        regime = pd.Series("chop", index=prices.index, dtype=object)

        bear_mask = (avg_price < sma50) & (sma50 < sma200)
        bull_mask = (
            (avg_price > sma50)
            & (sma50 > sma200)
            & (vol_20 < vol_252)
        )

        # Apply in order: bear first, then bull (bull overrides chop, never bear)
        regime[bear_mask] = "bear"
        regime[bull_mask] = "bull"

        return regime
