# Hedge Fund Backtesting Framework

A rigorous quantitative backtesting system for sector ETFs (2010–2024), implementing two core strategies with walk-forward validation, realistic cost modeling, and factor decomposition.

## Strategies

### 1. Momentum Long/Short
- 20-day z-scored momentum signal across 9 sector ETFs
- Long top 30%, short bottom 30%
- Volatility-scaled to 12% target vol

### 2. Cointegration Pairs Trading
- Statistical arbitrage on cointegrated sector ETF pairs
- Entry/exit based on z-score of spread (entry: ±2σ, exit: ±0.5σ)
- Pairs selected by Engle-Granger cointegration test (p < 0.10)

## Results (2010–2024)

| Strategy | Sharpe | Annual Return | Max DD | Alpha t-stat |
|----------|--------|---------------|--------|--------------|
| Momentum L/S | -0.85 | -1.5% | -33.4% | -0.30 (N/S) |
| **Pairs Trading** | **0.47** | **+10.1%** | -23.7% | **2.78 ✅** |

Pairs trading shows **statistically significant alpha** (p=0.005, annualized +9.3%) with near-zero market beta (β=0.07) — genuinely market-neutral.

## Structure

```
hedge-fund-backtest/
├── backtest/
│   ├── data.py          # yfinance data pipeline + cleaning
│   ├── signals.py       # momentum signal + cointegration pairs scanner
│   ├── portfolio.py     # L/S construction + volatility scaling
│   ├── costs.py         # realistic transaction cost model (bps)
│   ├── engine.py        # walk-forward validation + performance metrics
│   └── factor_model.py  # alpha vs beta decomposition (market factor)
├── main.py              # full demo: all 5 sections end-to-end
├── requirements.txt
└── output/              # equity curves, drawdowns, rolling Sharpe (gitignored)
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Output charts saved to `output/`.

## Methodology

- **No look-ahead bias**: all signals shifted 1 day before portfolio construction
- **Walk-forward validation**: 2yr train / 3mo test / 1mo step (153 windows)
- **Survivorship bias note**: uses current ETF tickers; XLRE/XLC dropped due to short history
- **Transaction costs**: ~7 bps/side (commission + spread + market impact)

## Universe

S&P 500 sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY
