"""
Hedge Fund Backtesting Framework — Demo Script
===============================================

Demonstrates:
  1. Data download (S&P 500 sector ETFs, 2010-2024)
  2. Momentum long/short strategy
  3. Cointegration pairs trading (top 5 pairs)
  4. Walk-forward validation (2yr train, 3mo test)
  5. Realistic transaction costs
  6. Factor model decomposition
  7. Full performance report + equity curve plots

SURVIVORSHIP BIAS WARNING:
  This demo uses current-day ETF tickers. ETFs launched after 2010 (e.g. XLRE
  2015, XLC 2018) will have shorter histories; the backtest adjusts automatically.
  For live strategies, always use point-in-time constituent data.

NO LOOK-AHEAD BIAS:
  Signals are shifted by 1 trading day before portfolio construction throughout.
  Walk-forward training windows never include any data from test periods.

Run with: python main.py
"""

import os
import sys
import warnings
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── project imports ────────────────────────────────────────────────────────
from backtest.data import fetch_price_data, clean_data, calculate_returns
from backtest.signals import momentum_signal, find_cointegrated_pairs, pairs_signal, PairInfo
from backtest.portfolio import (
    build_long_short_portfolio,
    volatility_scale_positions,
    apply_constraints,
)
from backtest.costs import apply_costs, compute_turnover
from backtest.engine import run_backtest, walk_forward_backtest, calculate_metrics
from backtest.factor_model import (
    get_market_factor,
    calculate_factor_exposures,
    decompose_returns,
    print_factor_report,
)
from backtest.report import generate_report

# ── configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

MOMENTUM_LOOKBACK = 60       # days for momentum signal (60-day window)
TOP_PCT = 0.3                # top/bottom 30% for L/S legs (larger with 11 ETFs)
TARGET_VOL = 0.12            # 12% annualised target vol
MAX_POSITION = 0.20          # max 20% per asset
N_PAIRS = 5                  # number of cointegrated pairs to trade
TRAIN_DAYS = 504             # ~2 years
TEST_DAYS = 63               # ~3 months
STEP_DAYS = 21               # ~1 month

# Sector groupings for sector-neutral portfolio construction
# (groups sector ETFs into broader themes so sector-neutral z-scoring has effect)
SECTORS = {
    "XLK": "growth",   "XLY": "growth",  "XLF": "growth",   "XLC": "growth",
    "XLI": "cyclical", "XLB": "cyclical", "XLE": "cyclical",
    "XLV": "defensive", "XLP": "defensive", "XLU": "defensive", "XLRE": "defensive",
}

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "="*60)
    print("  SECTION 1: DATA DOWNLOAD & PREPARATION")
    print("="*60)

    prices = fetch_price_data(TICKERS, START_DATE, END_DATE)
    prices = clean_data(prices)
    returns = calculate_returns(prices, method="log")

    print(f"\nPrice data summary:")
    print(f"  Date range : {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Tickers    : {list(prices.columns)}")
    print(f"  Shape      : {prices.shape}")
    print(f"\nFirst/last closing prices:")
    print(prices.iloc[[0, -1]].round(2).T.to_string())
    return prices, returns


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — MOMENTUM STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

def run_momentum_strategy(prices: pd.DataFrame, returns: pd.DataFrame):
    print("\n" + "="*60)
    print("  SECTION 2: MOMENTUM LONG/SHORT STRATEGY")
    print("="*60)

    def momentum_signal_func(px: pd.DataFrame) -> pd.DataFrame:
        return momentum_signal(px, lookback=MOMENTUM_LOOKBACK, weekly_only=True)

    print(f"  Lookback window : {MOMENTUM_LOOKBACK} days")
    print(f"  L/S leg size    : top/bottom {TOP_PCT*100:.0f}%")
    print(f"  Target vol      : {TARGET_VOL*100:.0f}%")
    print(f"  Weekly signals  : True (rebalance on Mondays only)")
    print(f"  Regime filter   : True (go flat in bear markets)")
    print(f"  Sector neutral  : True (normalise within growth/cyclical/defensive)")
    print(f"  Running backtest ...")

    mom_returns, mom_weights = run_backtest(
        prices,
        signal_func=momentum_signal_func,
        top_pct=TOP_PCT,
        target_vol=TARGET_VOL,
        max_position=MAX_POSITION,
        asset_class="us_equity_etf",
        apply_vol_scaling=True,
        use_regime_filter=True,
        sector_neutral=True,
        sectors=SECTORS,
    )

    metrics = calculate_metrics(mom_returns)
    print_metrics_table(metrics, "Momentum L/S")

    avg_turnover = compute_turnover(mom_weights).mean()
    print(f"  Avg daily turnover: {avg_turnover*100:.2f}%  ({avg_turnover*252*100:.0f}% annualised)")

    generate_report(mom_returns, "momentum", OUTPUT_DIR)

    return mom_returns, mom_weights, metrics


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — PAIRS TRADING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

def run_pairs_strategy(prices: pd.DataFrame, returns: pd.DataFrame):
    print("\n" + "="*60)
    print("  SECTION 3: COINTEGRATION PAIRS TRADING")
    print("="*60)

    # Use first half of data to find pairs (avoids in-sample optimisation bias)
    mid_idx = len(prices) // 2
    discovery_prices = prices.iloc[:mid_idx]
    print(f"  Pair discovery window: {discovery_prices.index[0].date()} to {discovery_prices.index[-1].date()}")

    pairs = find_cointegrated_pairs(
        discovery_prices,
        significance=0.10,
        min_half_life=5.0,
        max_half_life=60.0,
    )

    if not pairs:
        print("  No cointegrated pairs found. Skipping pairs strategy.")
        return None, None, None

    best_pairs = pairs[:N_PAIRS]
    print(f"\n  Top {len(best_pairs)} cointegrated pairs:")
    print(f"  {'Pair':<18} {'p-value':>10} {'hedge_ratio':>12} {'half_life':>12} {'ADF p':>10}")
    print(f"  {'-'*64}")
    for p in best_pairs:
        print(
            f"  {p.asset_a+'/'+p.asset_b:<18} {p.p_value:>10.4f} "
            f"{p.hedge_ratio:>12.3f} {p.half_life:>12.1f} {p.adf_pvalue:>10.4f}"
        )

    # Combine pair signals into an equal-weight pairs portfolio
    pair_signal_series = {}
    for pair in best_pairs:
        sig = pairs_signal(prices, pair, entry_z=2.0, exit_z=0.5, stop_z=3.5)
        pair_signal_series[f"{pair.asset_a}/{pair.asset_b}"] = sig

    signals_df = pd.DataFrame(pair_signal_series)

    # Build synthetic pair returns
    all_pair_returns = []
    for pair in best_pairs:
        key = f"{pair.asset_a}/{pair.asset_b}"
        sig = signals_df[key].shift(1)  # lag signal by 1 day

        ret_a = calculate_returns(prices[[pair.asset_a]], method="log")[pair.asset_a]
        ret_b = calculate_returns(prices[[pair.asset_b]], method="log")[pair.asset_b]

        # Position: +signal in A, -signal * hedge_ratio in B (normalised)
        pair_ret = sig * ret_a - sig * pair.hedge_ratio * ret_b
        pair_ret.name = key
        all_pair_returns.append(pair_ret)

    pair_rets_df = pd.concat(all_pair_returns, axis=1).dropna(how="all")

    # Equal-weight portfolio of pairs
    pairs_portfolio_returns = pair_rets_df.mean(axis=1)
    pairs_portfolio_returns.name = "pairs_strategy"

    # Volatility scale
    roll_vol = pairs_portfolio_returns.rolling(60, min_periods=30).std() * np.sqrt(252)
    roll_vol = roll_vol.clip(lower=0.005)
    scaler = (TARGET_VOL / roll_vol).fillna(1.0).clip(upper=5.0)
    pairs_portfolio_returns_scaled = pairs_portfolio_returns * scaler

    # Apply costs (estimate turnover from signal changes)
    signal_turnover = signals_df.diff().abs().sum(axis=1) / 2 / len(best_pairs)
    from backtest.costs import estimate_trade_costs
    daily_costs = signal_turnover.apply(lambda t: estimate_trade_costs(t, "us_equity_etf"))
    pairs_net = pairs_portfolio_returns_scaled - daily_costs.reindex(pairs_portfolio_returns_scaled.index).fillna(0)

    metrics = calculate_metrics(pairs_net)
    print_metrics_table(metrics, "Pairs Trading")

    generate_report(pairs_net, "pairs", OUTPUT_DIR)

    return pairs_net, signals_df, metrics


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def run_walk_forward(prices: pd.DataFrame):
    print("\n" + "="*60)
    print("  SECTION 4: WALK-FORWARD VALIDATION")
    print("="*60)
    print(f"  Train: {TRAIN_DAYS} days (~2yr)  |  Test: {TEST_DAYS} days (~3mo)  |  Step: {STEP_DAYS} days")

    def momentum_signal_func(px: pd.DataFrame) -> pd.DataFrame:
        return momentum_signal(px, lookback=MOMENTUM_LOOKBACK, weekly_only=True)

    print("\n  [4a] Momentum strategy walk-forward (with regime filter) ...")
    wf_results = walk_forward_backtest(
        prices,
        signal_func=momentum_signal_func,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        top_pct=TOP_PCT,
        target_vol=TARGET_VOL,
        max_position=MAX_POSITION,
        asset_class="us_equity_etf",
        use_regime_filter=True,
        sector_neutral=True,
        sectors=SECTORS,
    )

    if not wf_results:
        print("  No walk-forward results generated.")
        return None, []

    # Concatenate out-of-sample returns
    oos_returns = pd.concat([r.test_returns for r in wf_results]).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_metrics = calculate_metrics(oos_returns)

    print(f"\n  Walk-Forward OOS Results ({len(wf_results)} windows):")
    print_metrics_table(oos_metrics, "Momentum WF OOS")

    # Per-window Sharpe summary
    window_sharpes = [r.metrics.get("Sharpe", np.nan) for r in wf_results]
    valid_sharpes = [s for s in window_sharpes if not np.isnan(s)]
    if valid_sharpes:
        print(f"\n  Per-window Sharpe: mean={np.mean(valid_sharpes):.2f}, "
              f"std={np.std(valid_sharpes):.2f}, "
              f"min={np.min(valid_sharpes):.2f}, "
              f"max={np.max(valid_sharpes):.2f}")
        pct_positive = np.mean([s > 0 for s in valid_sharpes]) * 100
        print(f"  % windows with positive Sharpe: {pct_positive:.0f}%")

    return oos_returns, wf_results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — FACTOR DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

def run_factor_decomposition(
    mom_returns: pd.Series,
    pairs_returns: Optional[pd.Series],
    prices: pd.DataFrame,
):
    print("\n" + "="*60)
    print("  SECTION 5: FACTOR MODEL DECOMPOSITION")
    print("="*60)

    market = get_market_factor(prices)

    print("\n  [5a] Momentum strategy vs. equal-weight market ...")
    mom_exposures = calculate_factor_exposures(mom_returns, market)
    print_factor_report(mom_exposures, "Momentum L/S")

    if pairs_returns is not None:
        print("\n  [5b] Pairs strategy vs. equal-weight market ...")
        pairs_exposures = calculate_factor_exposures(pairs_returns, market)
        print_factor_report(pairs_exposures, "Pairs Trading")
    else:
        pairs_exposures = None

    return mom_exposures, pairs_exposures


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def save_plots(
    prices: pd.DataFrame,
    mom_returns: pd.Series,
    pairs_returns: Optional[pd.Series],
    oos_returns: Optional[pd.Series],
    wf_results: list,
):
    print("\n" + "="*60)
    print("  SECTION 6: GENERATING PLOTS")
    print("="*60)

    # ── Plot 1: Equity Curves ────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle("Hedge Fund Backtest — Equity Curves", fontsize=14, fontweight="bold")

    # Momentum full-period
    ax = axes[0]
    _plot_equity_curve(ax, mom_returns, "Momentum L/S (gross)")
    if oos_returns is not None:
        _plot_equity_curve(ax, oos_returns, "Momentum WF OOS", linestyle="--", alpha=0.8)
    ax.set_title("Momentum Long/Short Strategy")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")

    # Pairs
    ax = axes[1]
    if pairs_returns is not None:
        _plot_equity_curve(ax, pairs_returns, "Pairs Trading", color="green")
    ax.set_title("Cointegration Pairs Trading")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")

    # Market benchmark (equal-weight ETFs)
    ax = axes[2]
    market_ret = calculate_returns(prices, method="log").mean(axis=1)
    _plot_equity_curve(ax, market_ret, "Equal-Weight Benchmark", color="orange")
    if mom_returns is not None:
        _plot_equity_curve(ax, mom_returns, "Momentum L/S", linestyle="--", alpha=0.7)
    ax.set_title("Strategy vs. Benchmark")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")

    plt.tight_layout()
    path = OUTPUT_DIR / "equity_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 2: Drawdown ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Drawdown Analysis", fontsize=14, fontweight="bold")

    _plot_drawdown(axes[0], mom_returns, "Momentum L/S")
    if pairs_returns is not None:
        _plot_drawdown(axes[1], pairs_returns, "Pairs Trading")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "drawdowns.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Rolling Sharpe ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Rolling 1-Year Sharpe Ratio", fontsize=12)

    for ret, label, color in [
        (mom_returns, "Momentum L/S", "steelblue"),
        (pairs_returns, "Pairs Trading", "green"),
    ]:
        if ret is None or len(ret) < 252:
            continue
        roll_sharpe = (
            ret.rolling(252).mean() / ret.rolling(252).std() * np.sqrt(252)
        )
        ax.plot(roll_sharpe.index, roll_sharpe.values, label=label, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    path = OUTPUT_DIR / "rolling_sharpe.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 4: Walk-Forward Per-Window Sharpe ────────────────────────
    if wf_results:
        sharpes = [r.metrics.get("Sharpe", np.nan) for r in wf_results]
        test_starts = [r.test_start for r in wf_results]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_title("Walk-Forward: Per-Window Out-of-Sample Sharpe", fontsize=12)
        colors = ["steelblue" if s > 0 else "salmon" for s in sharpes]
        ax.bar(range(len(sharpes)), sharpes, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(len(sharpes)))
        ax.set_xticklabels([str(d.date()) for d in test_starts], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Sharpe Ratio (OOS)")
        ax.set_xlabel("Test Window Start Date")
        plt.tight_layout()
        path = OUTPUT_DIR / "walkforward_sharpe.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # ── Plot 5: Return Distribution ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Return Distributions", fontsize=12)

    for ax, ret, label in [
        (axes[0], mom_returns, "Momentum L/S"),
        (axes[1], pairs_returns, "Pairs Trading"),
    ]:
        if ret is None:
            ax.set_visible(False)
            continue
        ret_clean = ret.dropna()
        ax.hist(ret_clean * 100, bins=60, color="steelblue", edgecolor="white",
                alpha=0.8, density=True)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(ret_clean.mean() * 100, color="red", linewidth=1.5,
                   linestyle="--", label=f"Mean={ret_clean.mean()*100:.3f}%")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend()

    plt.tight_layout()
    path = OUTPUT_DIR / "return_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    print(f"\n  All plots saved to ./{OUTPUT_DIR}/")


def _plot_equity_curve(ax, returns: pd.Series, label: str, **kwargs):
    cum = (1 + returns.dropna()).cumprod()
    ax.plot(cum.index, cum.values, label=label, **kwargs)


def _plot_drawdown(ax, returns: pd.Series, label: str):
    cum = (1 + returns.dropna()).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max * 100
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="salmon")
    ax.plot(dd.index, dd.values, color="red", linewidth=0.8)
    ax.set_title(f"{label} — Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def print_metrics_table(metrics: dict, label: str):
    print(f"\n  ┌─ {label} ─{'─'*(40-len(label))}┐")
    fields = [
        ("Annual Return", "AnnReturn", "{:.2%}"),
        ("Annual Volatility", "AnnVol", "{:.2%}"),
        ("Sharpe Ratio", "Sharpe", "{:.3f}"),
        ("Sortino Ratio", "Sortino", "{:.3f}"),
        ("Max Drawdown", "MaxDD", "{:.2%}"),
        ("Calmar Ratio", "Calmar", "{:.3f}"),
        ("Win Rate", "WinRate", "{:.2%}"),
        ("Total Return", "TotalReturn", "{:.2%}"),
        ("Skewness", "Skewness", "{:.3f}"),
        ("Kurtosis", "Kurtosis", "{:.3f}"),
        ("Num Trading Days", "NumDays", "{:d}"),
    ]
    for display_name, key, fmt in fields:
        val = metrics.get(key, "N/A")
        if val == "N/A" or (isinstance(val, float) and np.isnan(val)):
            formatted = "N/A"
        else:
            try:
                if key == "NumDays":
                    formatted = fmt.format(int(val))
                else:
                    formatted = fmt.format(val)
            except Exception:
                formatted = str(val)
        print(f"  │  {display_name:<22} {formatted:>12}  │")
    print(f"  └{'─'*42}┘")


def print_full_report(
    mom_metrics: dict,
    pairs_metrics: Optional[dict],
    oos_metrics: Optional[dict],
    mom_exposures: dict,
    pairs_exposures: Optional[dict],
):
    print("\n" + "="*60)
    print("  FULL PERFORMANCE REPORT")
    print("="*60)

    print_metrics_table(mom_metrics, "Momentum L/S (full period, net of costs)")

    if pairs_metrics:
        print_metrics_table(pairs_metrics, "Pairs Trading (full period, net of costs)")

    if oos_metrics:
        print_metrics_table(oos_metrics, "Momentum WF OOS (out-of-sample)")

    print("\n  Factor Model Summary:")
    print(f"  {'Strategy':<20} {'Ann Alpha':>12} {'Alpha t':>10} {'Beta':>8} {'R²':>8}")
    print(f"  {'-'*60}")

    def _fmt_row(name, exp):
        return (
            f"  {name:<20} "
            f"{exp['alpha_annualized']*100:>+11.2f}% "
            f"{exp['alpha_tstat']:>10.2f} "
            f"{exp['beta']:>8.3f} "
            f"{exp['r_squared']:>8.4f}"
        )

    print(_fmt_row("Momentum L/S", mom_exposures))
    if pairs_exposures:
        print(_fmt_row("Pairs Trading", pairs_exposures))

    print("\n" + "="*60)
    print("  NOTES & CAVEATS")
    print("="*60)
    print("  1. SURVIVORSHIP BIAS: Only current ETF tickers used.")
    print("     Shorter histories for XLRE (2015) and XLC (2018).")
    print("  2. TRANSACTION COSTS: ~7 bps/side for liquid ETFs.")
    print("     Real costs may differ (bid-ask, market impact, borrow).")
    print("  3. SIGNAL SHIFT: All signals lagged 1 day before execution.")
    print("  4. WALK-FORWARD: OOS data never used in training window.")
    print("  5. ALPHA SIGNIFICANCE: t-stat > 2 required for 95% confidence.")
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — COMBINED PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════

def run_combined_portfolio(
    mom_returns: pd.Series,
    pairs_returns: Optional[pd.Series],
):
    print("\n" + "="*60)
    print("  SECTION 7: COMBINED PORTFOLIO")
    print("="*60)
    print("  Equal-weight combination of Momentum L/S + Pairs Trading")

    strategies = [(r, n) for r, n in [
        (mom_returns, "Momentum"),
        (pairs_returns, "Pairs"),
    ] if r is not None]

    if not strategies:
        print("  No strategies available for combination.")
        return None

    if len(strategies) == 1:
        combined = strategies[0][0].rename("combined")
        print(f"  Only one strategy available ({strategies[0][1]}); using it as combined.")
    else:
        rets_list = [r.rename(n) for r, n in strategies]
        combined_df = pd.concat(rets_list, axis=1).dropna(how="all")
        combined = combined_df.mean(axis=1)
        combined.name = "combined"
        print(f"  Combining: {[n for _, n in strategies]}")

    metrics = calculate_metrics(combined)
    print_metrics_table(metrics, "Combined Portfolio (equal-weight)")

    # ── Equity curve ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Combined Portfolio Equity Curve", fontsize=12)

    cum_combined = (1 + combined.dropna()).cumprod()
    ax.plot(cum_combined.index, cum_combined.values, color="purple",
            linewidth=2, label="Combined (equal-weight)")

    for ret, name, color in [
        (mom_returns, "Momentum L/S", "steelblue"),
        (pairs_returns, "Pairs Trading", "green"),
    ]:
        if ret is not None:
            cum_s = (1 + ret.dropna()).cumprod()
            ax.plot(cum_s.index, cum_s.values, linestyle="--", alpha=0.55,
                    color=color, label=name)

    ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax.legend(loc="upper left")
    ax.set_ylabel("Cumulative Return (gross of costs)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    path = OUTPUT_DIR / "combined_equity_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#"*60)
    print("  HEDGE FUND BACKTESTING FRAMEWORK")
    print("  Sector ETF Demo | 2010-2024")
    print("#"*60)

    # ── 1. Data ──────────────────────────────────────────────────────────
    prices, returns = load_data()

    # ── 2. Momentum ──────────────────────────────────────────────────────
    mom_returns, mom_weights, mom_metrics = run_momentum_strategy(prices, returns)

    # ── 3. Pairs ─────────────────────────────────────────────────────────
    pairs_returns, pairs_signals_df, pairs_metrics = run_pairs_strategy(prices, returns)

    # ── 4. Walk-Forward ──────────────────────────────────────────────────
    oos_returns, wf_results = run_walk_forward(prices)

    # ── 5. Factor Decomposition ──────────────────────────────────────────
    mom_exposures, pairs_exposures = run_factor_decomposition(
        mom_returns, pairs_returns, prices
    )

    # ── 6. Plots ─────────────────────────────────────────────────────────
    save_plots(prices, mom_returns, pairs_returns, oos_returns, wf_results)

    # ── 7. Combined Portfolio ─────────────────────────────────────────────
    combined_returns = run_combined_portfolio(mom_returns, pairs_returns)

    # ── Full Report ───────────────────────────────────────────────────────
    oos_metrics = calculate_metrics(oos_returns) if oos_returns is not None else {}
    print_full_report(
        mom_metrics,
        pairs_metrics,
        oos_metrics if oos_metrics else None,
        mom_exposures,
        pairs_exposures,
    )

    print("\n  Done. Output files written to ./output/")
    print("  equity_curves.png | drawdowns.png | rolling_sharpe.png")
    print("  walkforward_sharpe.png | return_distributions.png\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
