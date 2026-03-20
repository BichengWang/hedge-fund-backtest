"""
Data fetching and cleaning utilities.

SURVIVORSHIP BIAS WARNING: Using current-day tickers only. Stocks/ETFs that were
delisted or merged during the backtest period are not included, which may
overstate historical performance. Use point-in-time constituent data for
production systems.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Optional


def fetch_price_data(
    tickers: List[str],
    start: str,
    end: str,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str
    start   : str, e.g. '2010-01-01'
    end     : str, e.g. '2024-12-31'
    price_col : column to extract from yfinance multi-level download

    Returns
    -------
    pd.DataFrame  columns = tickers, index = DatetimeIndex
    """
    print(f"[data] Fetching {len(tickers)} tickers from {start} to {end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when multiple tickers requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker
        prices = raw[["Close"]]
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    print(f"[data] Downloaded {prices.shape[0]} rows x {prices.shape[1]} columns")
    return prices


def clean_data(
    prices: pd.DataFrame,
    max_missing_pct: float = 0.10,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """
    Clean price data:
      - Drop columns with too many missing values.
      - Forward-fill up to `ffill_limit` consecutive NaNs.
      - Drop any remaining rows that are all-NaN.

    Parameters
    ----------
    prices          : raw price DataFrame
    max_missing_pct : drop ticker if > this fraction of data is NaN
    ffill_limit     : max consecutive NaNs to forward-fill

    Returns
    -------
    Cleaned pd.DataFrame
    """
    n = len(prices)
    missing_frac = prices.isna().sum() / n

    drop_cols = missing_frac[missing_frac > max_missing_pct].index.tolist()
    if drop_cols:
        print(f"[data] Dropping {drop_cols} (>{max_missing_pct*100:.0f}% missing)")
    prices = prices.drop(columns=drop_cols)

    prices = prices.ffill(limit=ffill_limit)
    prices = prices.dropna(how="all")

    remaining_na = prices.isna().sum().sum()
    if remaining_na > 0:
        print(f"[data] {remaining_na} NaN cells remain after cleaning (backfilling first row)")
        prices = prices.bfill(limit=1)

    print(f"[data] Clean data: {prices.shape[0]} rows, {prices.shape[1]} tickers")
    return prices


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    """
    Calculate daily returns.

    Parameters
    ----------
    prices : cleaned price DataFrame
    method : 'log' (default) or 'simple'

    Returns
    -------
    pd.DataFrame of returns, first row is NaN (dropped)
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
    elif method == "simple":
        returns = prices.pct_change().dropna(how="all")
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log' or 'simple'.")
    return returns
