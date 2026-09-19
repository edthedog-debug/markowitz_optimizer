# Markowitz Portfolio Optimizer (US Equities)

A command-line tool that builds a **mean-variance (Markowitz) optimal portfolio** from US stock tickers, compares it against the **S&P 500 (SPY)** over the identical period, and runs a **walk-forward win-rate validation** to test how often the strategy produced positive returns in historical sub-periods.

> **Disclaimer:** This is a research and education tool. All results are historical and largely in-sample. Nothing here is investment advice, and past performance does not predict future results.

---

## Features

- Downloads split/dividend-adjusted prices with `yfinance` (retry + exponential back-off, defensive parsing across yfinance versions)
- Annualised expected returns + covariance matrix (sample or Ledoit-Wolf shrinkage)
- **Maximum Sharpe ratio** portfolio (tangency portfolio) via PyPortfolioOpt, with per-asset weight bounds
- **S&P 500 benchmark comparison:** cumulative return, CAGR, volatility, Sharpe, max drawdown
- **Walk-forward win-rate screen** across lookbacks, rebalance intervals and weight caps
- High-resolution (300 dpi) efficient-frontier chart with assets, optimal portfolio, capital market line and SPY

---

## Project structure

```
markowitz_optimizer/
├── main.py            # everything: data, optimisation, backtest, reporting, plotting
├── requirements.txt
└── README.md
```

`main.py` is a single script organised into clearly separated sections, each made of small, independently testable functions:

| Section | Key functions | Responsibility |
|---|---|---|
| 1. Data | `parse_tickers`, `download_prices`, `load_market_data` | Fetch, retry, clean, align assets + benchmark on common dates |
| 2. Optimisation | `estimate_inputs`, `optimize_max_sharpe`, `compute_frontier` | μ, Σ, max-Sharpe weights, frontier curve |
| 3. Performance | `compute_performance`, `print_benchmark_comparison` | Identical metrics for portfolio and benchmark |
| 4. Validation | `rolling_win_profile`, `run_walk_forward`, `screen_configurations`, `print_win_rate_report` | Win-rate screen and holding-horizon diagnostic |
| 5. Plotting | `plot_efficient_frontier` | Saves `efficient_frontier_vs_sp500.png` |
| CLI | `parse_args`, `validate_args`, `run`, `main` | Argument handling, orchestration, error handling |

**Data flow:** tickers + SPY → cleaned price panel → (μ, Σ) → max-Sharpe weights → benchmark comparison → win-rate screen → plot.

---

## Installation

Requires Python 3.9+ (3.10+ recommended).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Defaults: AAPL MSFT GOOGL NVDA, last 10 years, SPY benchmark
python main.py

# Custom tickers and explicit dates
python main.py --tickers AAPL MSFT GOOGL NVDA AMZN --start 2018-01-01 --end 2025-12-31

# Cap any single stock at 40%, use Ledoit-Wolf covariance, 4.5% risk-free rate
python main.py --tickers AAPL,MSFT,GOOGL,NVDA --max-weight 0.4 --cov-method ledoit_wolf --risk-free-rate 0.045

# Skip the (slower) backtest screen
python main.py --skip-backtest
```

### Options

| Option | Default | Description |
|---|---|---|
| `--tickers` | `AAPL MSFT GOOGL NVDA` | Space- or comma-separated tickers (at least 2) |
| `--start` / `--end` | none / today | Date range `YYYY-MM-DD` |
| `--years` | `10` | History length when `--start` is not given |
| `--benchmark` | `SPY` | Benchmark ticker |
| `--risk-free-rate` | `0.04` | Annual risk-free rate used in Sharpe ratios |
| `--min-weight` / `--max-weight` | `0` / `1` | Per-asset weight bounds (long-only) |
| `--cov-method` | `sample` | `sample` or `ledoit_wolf` |
| `--output` | `efficient_frontier_vs_sp500.png` | Plot path |
| `--dpi` | `300` | Plot resolution |
| `--skip-backtest` | off | Skip win-rate validation |
| `--lookbacks` | `252 504` | Estimation windows (trading days) to test |
| `--rebalance-days` | `21 63 126 252` | Rebalance intervals (trading days) to test |
| `--max-weights` | `0.25 0.35 0.5 1.0` | Weight caps to test (your `--max-weight` is always added) |
| `--min-periods` | `8` | Minimum holding periods for a win rate to count |
| `--cost-bps` | `5` | Transaction cost per unit of traded notional (bps) |
| `-v` | off | Verbose logging / backtest progress |

Exit codes: `0` success, `1` data/usage error, `2` missing dependency, `130` interrupted.

### Console output (layout)

```
MARKOWITZ MEAN-VARIANCE OPTIMISATION      universe, benchmark, period, settings
OPTIMAL WEIGHTS                           one line per asset with a bar
EXPECTED PORTFOLIO METRICS                annualised return, volatility, Sharpe
PERFORMANCE vs S&P 500 (SPY)              Markowitz vs SPY vs difference
HISTORICAL WIN-RATE VALIDATION            ranked configurations, PASS / FLAG verdict, holding-period profile
OUTPUT                                    path of the saved chart
```

---

## How the S&P 500 benchmark is integrated

1. `SPY` is downloaded **in the same request** as your tickers, so both are aligned on exactly the same trading days. If a recent IPO shortens the common history, the tool warns you which ticker is limiting it.
2. If you include `SPY` in `--tickers`, it is removed from the optimisation universe and used only as the benchmark.
3. The optimised portfolio's daily return series (optimal weights held constant, i.e. daily rebalanced) and SPY's daily returns are passed through the **same** `compute_performance` function, so cumulative return, CAGR, volatility, Sharpe and max drawdown are directly comparable.
4. On the chart, SPY is plotted as a red **+** marker at its own (volatility, CAGR) point, so you can see whether the frontier dominates it in risk-return space.
5. In the win-rate screen, SPY is held over the *same* sub-periods, so each configuration is shown next to SPY's win rate.

**Note on two Sharpe figures:** the "expected" Sharpe uses the CAGR of each asset (μ) combined linearly; the "realised" Sharpe is computed from the actual daily portfolio series. They differ slightly because of compounding and rebalancing effects. This is expected.

---

## How the historical win-rate validation works

**Goal:** check whether the strategy would have produced a positive return in *every* historical sub-period, and if not, show which constraints came closest.

### Walk-forward backtest (no look-ahead)

For each configuration in a grid of **(lookback, rebalance interval, max weight per asset)**:

1. At each rebalance date *t*, μ and Σ are estimated **only from the trailing `lookback` trading days ending at *t***.
2. Max-Sharpe weights are computed under the weight cap (with fallback to minimum-variance, then equal weight, if max-Sharpe is infeasible; fallbacks are counted).
3. The weights are held buy-and-hold (they drift) for `rebalance days`. Transaction costs (`--cost-bps` on traded notional) are deducted.
4. Only **complete** holding periods are evaluated. A period is a **win** if its net return is strictly positive.
5. **Win rate = wins ÷ periods.** SPY's win rate over the same periods is reported alongside.

### Verdicts

- **`[PASS]`** at least one configuration had a 100% win rate **and** at least `--min-periods` periods. The best such configuration (highest cumulative return) is printed with the `--max-weight` value to reproduce its cap.
- **`[FLAG]`** no configuration reached 100%. The tool prints the closest one and lists exactly which constraints were tested, so you know what was ruled out.
- Configurations with fewer than `--min-periods` periods are shown but marked *too few periods* and never count as a PASS, because a 100% win rate over 3 periods is meaningless.

### Holding-horizon diagnostic

Because 100% is rarely reachable over short horizons, the report also shows, for the optimised portfolio and SPY, the share of **rolling windows** (1 month to 5 years) with a positive return and the worst window. The shortest horizon with 100% positive windows is the practical "constraint" you would need to accept (a minimum holding period) to target this metric.

### Read this before trusting a 100% result

- **A 100% historical win rate is not a forecast.** Trying 30+ configurations and reporting the one that hit 100% is data snooping; some configurations will look perfect by chance.
- Rolling-window statistics use overlapping windows, so they are not independent observations.
- The final optimal weights use the whole sample (in-sample). Only the backtest is walk-forward.
- Long horizons and few periods make 100% easy; short horizons make it very hard. The tool deliberately does **not** tune weights to force a 100% result, since that would be curve-fitting rather than validation.
- To go further, split the data in time: screen on the early part, then check the chosen configuration on a later untouched hold-out.

---

## Methodology notes

- **Expected returns:** `mean_historical_return` (annualised compounded return per asset, 252 trading days).
- **Covariance:** annualised sample covariance, or Ledoit-Wolf shrinkage (`--cov-method ledoit_wolf`), which is more stable with short histories or many assets.
- **Objective:** maximise (E[R] − r_f) / σ subject to Σw = 1 and `min-weight ≤ w ≤ max-weight` (long-only).
- **Frontier:** minimum-variance portfolio up to the highest return reachable under the weight bounds (solved as a small linear program).
- **Inputs are noisy:** mean-variance optimisation is very sensitive to expected-return estimates and tends to concentrate in past winners. Weight caps and shrinkage covariance help.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Could not download price data` | Network issue or Yahoo rate-limiting. Retry later or upgrade: `pip install -U yfinance` |
| `No data returned for the benchmark SPY` | Same as above, or a date range with no trading days |
| `max-weight is too small` | With *n* assets you need `max-weight ≥ 1/n` |
| Warning: history limited by a ticker | A recent IPO shortens the common history; drop it or accept the shorter window |
| `Max-Sharpe was infeasible` | No asset's expected return beat the risk-free rate; the tool fell back to minimum variance |
| Backtest is slow | Fewer configs: e.g. `--rebalance-days 63 252 --lookbacks 252 --max-weights 0.35 1` |
