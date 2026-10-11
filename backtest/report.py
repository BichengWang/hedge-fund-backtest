"""
Strategy performance reporting: monthly heatmap, annual Sharpe, text summary.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_report(
    returns: pd.Series,
    strategy_name: str,
    output_dir: Path,
) -> None:
    """
    Generate a single-page performance report for a strategy.

    Produces:
      1. Monthly returns heatmap (years × months) as PNG
      2. Annual Sharpe bar chart by year
      3. Printed text summary: Sharpe by year, best/worst year, % positive years

    Saves to: output_dir / f"{strategy_name}_report.png"

    Parameters
    ----------
    returns       : daily net return series (pd.Series)
    strategy_name : label used in titles and filename
    output_dir    : directory to save the PNG
    """
    returns = returns.dropna()
    if len(returns) < 20:
        print(f"  [report] Insufficient data for {strategy_name} report (< 20 days).")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── Monthly returns ───────────────────────────────────────────────────
    try:
        monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    except Exception:
        monthly = returns.resample("M").apply(lambda r: (1 + r).prod() - 1)

    mdf = monthly.to_frame("ret")
    mdf["year"] = mdf.index.year
    mdf["month"] = mdf.index.month
    pivot = mdf.pivot_table(index="year", columns="month", values="ret", aggfunc="first")
    pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]
    pct_pivot = pivot * 100  # display as %

    # ── Annual Sharpe + returns ───────────────────────────────────────────
    annual_sharpe: dict[int, float] = {}
    annual_returns: dict[int, float] = {}

    for year, grp in returns.groupby(returns.index.year):
        if len(grp) < 10:
            continue
        annual_returns[int(year)] = float((1 + grp).prod() - 1)
        excess = grp - 0.04 / 252
        std = excess.std()
        annual_sharpe[int(year)] = (
            float(excess.mean() / std * np.sqrt(252)) if std > 0 else np.nan
        )

    years = sorted(annual_sharpe.keys())
    valid_sharpes = [v for v in annual_sharpe.values() if not np.isnan(v)]

    # ── Text summary ──────────────────────────────────────────────────────
    print(f"\n  ── {strategy_name.upper()} ANNUAL REPORT ──")
    print(f"  {'Year':<8} {'Ann Return':>12} {'Sharpe':>10}")
    print(f"  {'-'*32}")
    for y in years:
        ann_ret = annual_returns.get(y, np.nan)
        sharpe = annual_sharpe.get(y, np.nan)
        ret_str = f"{ann_ret:.2%}" if not np.isnan(ann_ret) else "N/A"
        shr_str = f"{sharpe:.3f}" if not np.isnan(sharpe) else "N/A"
        print(f"  {y:<8} {ret_str:>12} {shr_str:>10}")

    if annual_returns:
        best_year = max(annual_returns, key=lambda y: annual_returns[y])
        worst_year = min(annual_returns, key=lambda y: annual_returns[y])
        pct_pos = sum(1 for v in annual_returns.values() if v > 0) / len(annual_returns) * 100
        print(f"\n  Best year  : {best_year} ({annual_returns[best_year]:.2%})")
        print(f"  Worst year : {worst_year} ({annual_returns[worst_year]:.2%})")
        print(f"  % positive : {pct_pos:.0f}% of years")
    if valid_sharpes:
        print(f"  Avg Sharpe : {np.mean(valid_sharpes):.3f}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{strategy_name} — Performance Report",
        fontsize=14,
        fontweight="bold",
    )

    gs = fig.add_gridspec(2, 1, hspace=0.45)
    ax_heat = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # Heatmap
    valid_vals = pct_pivot.values[~np.isnan(pct_pivot.values)]
    vmax = max(abs(valid_vals).max(), 1.0) if len(valid_vals) > 0 else 5.0
    sns.heatmap(
        pct_pivot,
        ax=ax_heat,
        annot=True,
        fmt=".1f",
        center=0.0,
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.4,
        cbar_kws={"label": "Monthly Return (%)"},
    )
    ax_heat.set_title("Monthly Returns Heatmap (%)", fontsize=11)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("Year")

    # Annual Sharpe bar chart
    sharpe_years = list(annual_sharpe.keys())
    sharpe_vals = [annual_sharpe[y] for y in sharpe_years]
    colors = ["steelblue" if (not np.isnan(v) and v > 0) else "salmon" for v in sharpe_vals]
    bars = ax_bar.bar(sharpe_years, sharpe_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.axhline(0, color="black", linewidth=1)
    ax_bar.axhline(1.0, color="green", linewidth=0.8, linestyle="--", alpha=0.6, label="Sharpe = 1")
    ax_bar.set_title("Annual Sharpe Ratio by Year", fontsize=11)
    ax_bar.set_xlabel("Year")
    ax_bar.set_ylabel("Sharpe Ratio")
    ax_bar.legend(fontsize=9)
    ax_bar.set_xticks(sharpe_years)
    ax_bar.set_xticklabels(sharpe_years, rotation=45, ha="right", fontsize=8)

    save_path = output_dir / f"{strategy_name}_report.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [report] Saved: {save_path}")
