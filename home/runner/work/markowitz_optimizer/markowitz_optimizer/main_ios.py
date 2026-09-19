#!/usr/bin/env python3
"""
Markowitz Mean-Variance Portfolio Optimizer (US equities)
=========================================================

Pipeline
--------
1. Download split/dividend-adjusted prices for the requested tickers *and* the
   S&P 500 proxy (SPY) with ``yfinance`` (retries + defensive cleaning).
2. Estimate annualised expected returns and the covariance matrix and solve the
   classic Markowitz problem: maximise the Sharpe ratio (tangency portfolio).
3. Compare the optimised portfolio with SPY over the identical period
   (cumulative return, volatility, Sharpe, max drawdown).
4. Run a walk-forward backtest "win-rate screen" over a grid of lookback
   windows, rebalance intervals and max-weight constraints, and report whether
   any configuration produced positive returns in 100% of its sub-periods
   (or exactly which constraints were tested and how close they came).
5. Save a high-resolution efficient-frontier chart with the benchmark marked.

IMPORTANT: everything here is historical / in-sample analysis. A 100% historical
win rate is a descriptive statistic, not a forecast. Not investment advice.

Usage
-----
    python main.py --tickers AAPL MSFT GOOGL NVDA --years 10
    python main.py --tickers AAPL,MSFT,GOOGL,NVDA --start 2018-01-01 --end 2025-12-31 \
                   --max-weight 0.4 --risk-free-rate 0.04
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers / CI without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

try:
    import yfinance as yf
    from pypfopt import EfficientFrontier, expected_returns, risk_models
except ImportError as _exc:  # pragma: no cover - friendly message for missing deps
    sys.stderr.write(
        f"Missing dependency: {_exc}.\n"
        "Install everything with:  pip install -r requirements.txt\n"
    )
    sys.exit(2)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TRADING_DAYS = 252
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA"]
DEFAULT_BENCHMARK = "SPY"
DEFAULT_OUTPUT = "efficient_frontier_vs_sp500.png"
# Holding horizons (trading days) used by the rolling "always positive?" diagnostic
HORIZONS = [21, 63, 126, 252, 504, 756, 1008, 1260]
HORIZON_LABELS = {
    21: "1 month", 63: "3 months", 126: "6 months", 252: "1 year",
    504: "2 years", 756: "3 years", 1008: "4 years", 1260: "5 years",
}

LOGGER = logging.getLogger("markowitz")


class DataError(RuntimeError):
    """Raised when market data cannot be fetched or is unusable."""


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def pct(value: float, digits: int = 2) -> str:
    """Format a fraction as a percentage string (NaN-safe)."""
    return "n/a" if value is None or not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def banner(title: str) -> str:
    line = "=" * 78
    return f"\n{line}\n{title}\n{line}"


# --------------------------------------------------------------------------- #
# 1. Data acquisition
# --------------------------------------------------------------------------- #
def parse_tickers(raw: Sequence[str]) -> list[str]:
    """Accept 'AAPL MSFT', 'AAPL,MSFT' or a mix; upper-case and de-duplicate."""
    tickers: list[str] = []
    for chunk in raw:
        for token in chunk.replace(";", ",").replace(" ", ",").split(","):
            token = token.strip().upper()
            if token and token not in tickers:
                tickers.append(token)
    return tickers


def _extract_close(raw: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Pull the (auto-adjusted) Close panel out of yfinance's output.

    yfinance changes its column layout between versions and between one/many
    tickers, so both MultiIndex and flat layouts are handled here.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise DataError("Downloaded data has no 'Close' field.")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise DataError("Downloaded data has no 'Close' field.")
        prices = raw[["Close"]].copy()
        prices.columns = [symbols[0]]

    index = pd.DatetimeIndex(prices.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    prices.index = index
    return prices.sort_index().astype(float)


def download_prices(
    symbols: Sequence[str], start: str, end: str, retries: int = 3, backoff: float = 2.0
) -> pd.DataFrame:
    """Download adjusted closing prices with retry / exponential back-off.

    ``auto_adjust=True`` makes yfinance's ``Close`` the split- and
    dividend-adjusted close, i.e. the "adjusted close" series.
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                list(symbols), start=start, end=end, auto_adjust=True,
                progress=False, group_by="column", threads=True,
            )
            if raw is None or raw.empty:
                raise DataError("Yahoo Finance returned an empty result.")
            return _extract_close(raw, symbols)
        except Exception as exc:  # network, rate-limit, parsing ... retry all
            last_error = exc
            LOGGER.warning("Download attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff ** attempt)
    raise DataError(f"Could not download price data after {retries} attempts: {last_error}")


def load_market_data(
    tickers: Sequence[str], benchmark: str, start: str, end: str, min_obs: int = 252
) -> tuple[pd.DataFrame, pd.Series]:
    """Download, clean and align asset prices and the benchmark on common dates."""
    universe = [t for t in tickers if t != benchmark]
    if len(universe) != len(tickers):
        LOGGER.warning("%s is the benchmark; removing it from the optimisation universe.", benchmark)
    if len(universe) < 2:
        raise DataError("Provide at least two tickers (excluding the benchmark).")

    prices = download_prices([*universe, benchmark], start, end)

    # Drop tickers that returned nothing (bad symbol, delisted, ...).
    dead = [s for s in [*universe, benchmark] if s not in prices.columns or prices[s].isna().all()]
    if benchmark in dead:
        raise DataError(f"No data returned for the benchmark {benchmark}.")
    if dead:
        LOGGER.warning("No data for %s - excluded from the analysis.", ", ".join(dead))
        universe = [t for t in universe if t not in dead]
    if len(universe) < 2:
        raise DataError("Fewer than two tickers have usable data.")
    prices = prices[[*universe, benchmark]]

    # Warn when a late-starting ticker (recent IPO) truncates the common history.
    first_valid = prices.apply(pd.Series.first_valid_index)
    limiting = first_valid.idxmax()
    if (first_valid.max() - pd.Timestamp(start)).days > 30:
        LOGGER.warning(
            "History is limited to %s onward by %s (latest first available price).",
            first_valid.max().date(), limiting,
        )

    prices = prices.ffill(limit=5).dropna(how="any")  # bridge tiny gaps, then align
    if len(prices) < min_obs:
        raise DataError(f"Only {len(prices)} common trading days found; need at least {min_obs}.")
    if (prices <= 0).any().any():
        raise DataError("Non-positive prices detected; data looks corrupted.")

    return prices[universe], prices[benchmark]


# --------------------------------------------------------------------------- #
# 2. Markowitz optimisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    method: str


def estimate_inputs(prices: pd.DataFrame, cov_method: str = "sample") -> tuple[pd.Series, pd.DataFrame]:
    """Annualised expected returns (mean historical, compounded) and covariance."""
    mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    if cov_method == "ledoit_wolf":
        cov = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    else:
        cov = risk_models.sample_cov(prices, frequency=TRADING_DAYS)
    cov = risk_models.fix_nonpositive_semidefinite(cov)  # no-op unless numerically indefinite
    return mu, cov


def _finalise_weights(raw_weights: np.ndarray, index: pd.Index) -> pd.Series:
    w = pd.Series(raw_weights, index=index, dtype=float).clip(lower=0.0)
    w[w < 1e-6] = 0.0  # strip solver noise
    return w / w.sum()


def optimize_max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, bounds: tuple[float, float], risk_free_rate: float
) -> OptimizationResult:
    """Maximise the Sharpe ratio; degrade gracefully if that problem is infeasible.

    Fallback chain: max Sharpe -> minimum volatility -> bounded equal weight.
    (max Sharpe is infeasible when no asset's expected return beats the risk-free rate.)
    """
    weights: np.ndarray | None = None
    method = "max_sharpe"
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = np.asarray(ef.weights, dtype=float)
    except Exception as exc:
        LOGGER.debug("max_sharpe failed (%s); falling back to min_volatility.", exc)
        method = "min_volatility (fallback)"
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
            ef.min_volatility()
            weights = np.asarray(ef.weights, dtype=float)
        except Exception as exc2:
            LOGGER.debug("min_volatility failed (%s); using equal weights.", exc2)
            method = "equal_weight (fallback)"
            weights = np.full(len(mu), 1.0 / len(mu))

    w = _finalise_weights(weights, mu.index)
    ret = float(w.values @ mu.values)
    vol = float(np.sqrt(w.values @ cov.values @ w.values))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return OptimizationResult(w, ret, vol, sharpe, method)


def compute_frontier(
    mu: pd.Series, cov: pd.DataFrame, bounds: tuple[float, float], n_points: int = 60
) -> pd.DataFrame:
    """Trace the efficient frontier from the min-variance portfolio to max return."""
    ef_min = EfficientFrontier(mu, cov, weight_bounds=bounds)
    ef_min.min_volatility()
    w_min = np.asarray(ef_min.weights, dtype=float)
    ret_min = float(w_min @ mu.values)

    # Highest return reachable under the bounds (a small LP) -> upper end of the curve.
    n = len(mu)
    lp = linprog(-mu.values, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[bounds] * n, method="highs")
    ret_max = float(-lp.fun) if lp.success else float(mu.max())

    rows = []
    for target in np.linspace(ret_min, ret_max - 1e-6, n_points):
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
            ef.efficient_return(float(target))
            w = np.asarray(ef.weights, dtype=float)
            rows.append((float(np.sqrt(w @ cov.values @ w)), float(w @ mu.values)))
        except Exception as exc:  # a few extreme targets may be numerically infeasible
            LOGGER.debug("Frontier point %.4f skipped: %s", target, exc)
    if len(rows) < 3:
        raise RuntimeError("Could not trace the efficient frontier (too few feasible points).")
    return pd.DataFrame(rows, columns=["volatility", "return"])


# --------------------------------------------------------------------------- #
# 3. Performance statistics and benchmark comparison
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerformanceStats:
    cumulative_return: float
    annual_return: float  # geometric (CAGR)
    annual_vol: float
    sharpe: float
    max_drawdown: float


def compute_performance(returns: pd.Series, risk_free_rate: float) -> PerformanceStats:
    """Same formulas for the portfolio and the benchmark => like-for-like comparison."""
    growth = (1.0 + returns).cumprod()
    cumulative = float(growth.iloc[-1] - 1.0)
    years = len(returns) / TRADING_DAYS
    annual_return = float(growth.iloc[-1] ** (1.0 / years) - 1.0)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else float("nan")
    drawdown = float((growth / growth.cummax() - 1.0).min())
    return PerformanceStats(cumulative, annual_return, annual_vol, sharpe, drawdown)


def print_weights(weights: pd.Series) -> None:
    print(f"  {'Ticker':<8}{'Weight':>10}")
    print("  " + "-" * 40)
    for ticker, w in weights.sort_values(ascending=False).items():
        bar = "#" * int(round(w * 30))
        print(f"  {ticker:<8}{pct(w):>10}  {bar}")


def print_benchmark_comparison(port: PerformanceStats, bench: PerformanceStats, bench_name: str) -> None:
    rows = [
        ("Cumulative return", port.cumulative_return, bench.cumulative_return, True),
        ("Annualised return (CAGR)", port.annual_return, bench.annual_return, True),
        ("Annualised volatility", port.annual_vol, bench.annual_vol, True),
        ("Sharpe ratio", port.sharpe, bench.sharpe, False),
        ("Max drawdown", port.max_drawdown, bench.max_drawdown, True),
    ]
    print(f"  {'Metric':<28}{'Markowitz':>12}{bench_name:>12}{'Difference':>14}")
    print("  " + "-" * 66)
    for name, p, b, as_pct in rows:
        fmt = pct if as_pct else num
        diff = p - b
        diff_txt = (f"{diff * 100:+.2f} pp" if as_pct else f"{diff:+.2f}") if np.isfinite(diff) else "n/a"
        print(f"  {name:<28}{fmt(p):>12}{fmt(b):>12}{diff_txt:>14}")


# --------------------------------------------------------------------------- #
# 4. Historical win-rate validation
# --------------------------------------------------------------------------- #
def rolling_win_profile(returns: pd.Series, horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    """For each holding horizon: share of rolling windows with a positive return.

    Windows overlap, so treat this as a descriptive profile, not independent trials.
    """
    growth = (1.0 + returns).cumprod().values
    rows = []
    for h in horizons:
        if h > len(growth) - 60:  # need a meaningful number of windows
            continue
        window_returns = growth[h:] / growth[:-h] - 1.0
        rows.append((h, len(window_returns), float((window_returns > 0).mean()), float(window_returns.min())))
    return pd.DataFrame(rows, columns=["horizon_days", "windows", "pct_positive", "worst_return"])


def min_horizon_for_full_win(profile: pd.DataFrame) -> int | None:
    """Shortest tested holding horizon at which *every* rolling window was positive."""
    full = profile[profile["pct_positive"] >= 1.0]
    return int(full["horizon_days"].iloc[0]) if not full.empty else None


@dataclass
class BacktestResult:
    lookback: int
    rebalance_days: int
    max_weight: float
    periods: pd.DataFrame  # one row per holding period
    fallbacks: int         # rebalances where max-Sharpe was infeasible

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def win_rate(self) -> float:
        return float((self.periods["portfolio_return"] > 0).mean()) if self.n_periods else float("nan")

    @property
    def benchmark_win_rate(self) -> float:
        return float((self.periods["benchmark_return"] > 0).mean()) if self.n_periods else float("nan")

    @property
    def cumulative_return(self) -> float:
        return float((1.0 + self.periods["portfolio_return"]).prod() - 1.0)

    @property
    def benchmark_cumulative_return(self) -> float:
        return float((1.0 + self.periods["benchmark_return"]).prod() - 1.0)

    @property
    def cagr(self) -> float:
        days = (self.periods["end_date"].iloc[-1] - self.periods["start_date"].iloc[0]).days
        years = max(days / 365.25, 1e-9)
        return float((1.0 + self.cumulative_return) ** (1.0 / years) - 1.0)

    @property
    def worst_period(self) -> float:
        return float(self.periods["portfolio_return"].min())


def run_walk_forward(
    prices: pd.DataFrame,
    bench: pd.Series,
    lookback: int,
    rebalance_days: int,
    bounds: tuple[float, float],
    risk_free_rate: float,
    cov_method: str,
    cost_bps: float,
) -> BacktestResult | None:
    """Walk-forward backtest with no look-ahead.

    At each rebalance date t the weights are estimated ONLY from the trailing
    ``lookback`` trading days ending at t, then held (buy-and-hold, weights
    drift) for ``rebalance_days`` days. Transaction costs are charged on traded
    notional. Only complete holding periods are evaluated.
    """
    n = len(prices)
    starts = list(range(lookback, n - rebalance_days, rebalance_days))
    if not starts:
        return None

    asset_px = prices.values
    bench_px = bench.values
    prev_weights: np.ndarray | None = None
    fallbacks = 0
    rows = []
    for i in starts:
        window = prices.iloc[i - lookback : i + 1]
        mu, cov = estimate_inputs(window, cov_method)
        opt = optimize_max_sharpe(mu, cov, bounds, risk_free_rate)
        fallbacks += opt.method != "max_sharpe"
        w = opt.weights.reindex(prices.columns).values

        asset_ret = asset_px[i + rebalance_days] / asset_px[i] - 1.0
        gross = float(w @ asset_ret)
        turnover = 1.0 if prev_weights is None else float(np.abs(w - prev_weights).sum())
        cost = turnover * cost_bps / 1e4
        net = (1.0 + gross) * (1.0 - cost) - 1.0
        prev_weights = w * (1.0 + asset_ret) / (1.0 + gross)  # weights after drift

        rows.append({
            "start_date": prices.index[i],
            "end_date": prices.index[i + rebalance_days],
            "portfolio_return": net,
            "benchmark_return": float(bench_px[i + rebalance_days] / bench_px[i] - 1.0),
            "turnover": turnover,
        })

    return BacktestResult(lookback, rebalance_days, bounds[1], pd.DataFrame(rows), int(fallbacks))


def screen_configurations(
    prices: pd.DataFrame,
    bench: pd.Series,
    lookbacks: Sequence[int],
    rebalance_list: Sequence[int],
    max_weights: Sequence[float],
    min_weight: float,
    risk_free_rate: float,
    cov_method: str,
    cost_bps: float,
    min_periods: int,
) -> tuple[list[BacktestResult], pd.DataFrame]:
    """Run the walk-forward backtest over the whole constraint grid."""
    n_assets = prices.shape[1]
    results: list[BacktestResult] = []
    combos = [
        (lb, rb, mw)
        for lb in lookbacks for rb in rebalance_list for mw in sorted(set(max_weights))
    ]
    LOGGER.info("Running walk-forward backtests for up to %d configurations (use -v for progress) ...", len(combos))
    for k, (lb, rb, mw) in enumerate(combos, start=1):
        if mw * n_assets < 1.0 - 1e-9:
            LOGGER.info("Skipping max_weight=%.2f: infeasible with %d assets.", mw, n_assets)
            continue
        if min_weight >= mw:
            continue
        LOGGER.debug("Backtest %d/%d  lookback=%dd rebalance=%dd max_weight=%.0f%%",
                     k, len(combos), lb, rb, mw * 100)
        result = run_walk_forward(prices, bench, lb, rb, (min_weight, mw), risk_free_rate, cov_method, cost_bps)
        if result is not None and result.n_periods > 0:
            results.append(result)

    rows = [{
        "lookback_days": r.lookback,
        "rebalance_days": r.rebalance_days,
        "max_weight": r.max_weight,
        "periods": r.n_periods,
        "win_rate": r.win_rate,
        "cum_return": r.cumulative_return,
        "cagr": r.cagr,
        "worst_period": r.worst_period,
        "bench_win_rate": r.benchmark_win_rate,
        "bench_cum_return": r.benchmark_cumulative_return,
        "fallbacks": r.fallbacks,
        "valid_sample": r.n_periods >= min_periods,
    } for r in results]
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["hits_100"] = summary["valid_sample"] & (summary["win_rate"] >= 1.0)
        summary = summary.sort_values(
            ["valid_sample", "win_rate", "cum_return"], ascending=[False, False, False]
        ).reset_index(drop=True)
    return results, summary


def print_win_rate_report(
    summary: pd.DataFrame,
    port_profile: pd.DataFrame,
    bench_profile: pd.DataFrame,
    bench_name: str,
    min_periods: int,
    top_n: int,
) -> pd.Series | None:
    """Print the validation section. Returns the winning grid row (or None)."""
    print("  Definition: win rate = share of walk-forward holding periods with a")
    print("  strictly positive net return (weights fitted only on prior data).\n")

    if summary.empty:
        print("  No configuration had enough data to evaluate. Use a longer history or")
        print("  shorter lookback / rebalance settings.")
        return None

    table = summary.head(top_n).copy()
    print(f"  Top {len(table)} configuration(s) (ranked by win rate, then cumulative return):")
    print(f"  {'Lookback':>9}{'Rebal.':>8}{'MaxW':>7}{'Periods':>9}{'Win rate':>10}"
          f"{'Cum.ret':>10}{'Worst':>9}{bench_name + ' win':>10}")
    print("  " + "-" * 72)
    for _, r in table.iterrows():
        flag = "" if r["valid_sample"] else "  (too few periods)"
        print(f"  {int(r['lookback_days']):>8}d{int(r['rebalance_days']):>7}d{r['max_weight'] * 100:>6.0f}%"
              f"{int(r['periods']):>9}{pct(r['win_rate'], 1):>10}{pct(r['cum_return'], 1):>10}"
              f"{pct(r['worst_period'], 1):>9}{pct(r['bench_win_rate'], 1):>10}{flag}")

    n_total = len(summary)
    n_hits = int(summary["hits_100"].sum())
    best = summary.iloc[0]
    print()
    if n_hits > 0:
        print(f"  [PASS] {n_hits} of {n_total} tested configurations achieved a 100% historical")
        print(f"         win rate with >= {min_periods} periods. Best of them:")
        print(f"         lookback={int(best['lookback_days'])}d, rebalance every {int(best['rebalance_days'])}d, "
              f"max weight per asset={best['max_weight'] * 100:.0f}%")
        print(f"         ({int(best['periods'])} periods, cumulative {pct(best['cum_return'], 1)} "
              f"vs {bench_name} {pct(best['bench_cum_return'], 1)}).")
        print(f"         Reproduce the weight cap with: --max-weight {best['max_weight']:.2f}")
    else:
        print(f"  [FLAG] None of the {n_total} tested configurations reached a 100% win rate")
        print(f"         (with >= {min_periods} periods). Closest: {pct(best['win_rate'], 1)} at "
              f"lookback={int(best['lookback_days'])}d, rebalance={int(best['rebalance_days'])}d, "
              f"max weight={best['max_weight'] * 100:.0f}%.")
        print("         Constraints tested: lookbacks "
              f"{sorted(summary['lookback_days'].unique().tolist())}d, rebalance intervals "
              f"{sorted(summary['rebalance_days'].unique().tolist())}d, max weights "
              f"{[f'{x:.0%}' for x in sorted(summary['max_weight'].unique())]}.")

    # Holding-horizon diagnostic: what would you have to tolerate to never lose?
    print("\n  Rolling holding-period profile (in-sample, constant-mix portfolio):")
    print(f"  {'Horizon':<14}{'Windows':>9}{'Portfolio +ve':>15}{'Worst':>9}{bench_name + ' +ve':>12}{'Worst':>9}")
    print("  " + "-" * 68)
    bench_by_h = bench_profile.set_index("horizon_days") if not bench_profile.empty else None
    for _, r in port_profile.iterrows():
        h = int(r["horizon_days"])
        b = bench_by_h.loc[h] if bench_by_h is not None and h in bench_by_h.index else None
        print(f"  {HORIZON_LABELS.get(h, f'{h}d'):<14}{int(r['windows']):>9}{pct(r['pct_positive'], 1):>15}"
              f"{pct(r['worst_return'], 1):>9}"
              f"{pct(b['pct_positive'], 1) if b is not None else 'n/a':>12}"
              f"{pct(b['worst_return'], 1) if b is not None else 'n/a':>9}")
    h_full = min_horizon_for_full_win(port_profile)
    if h_full is not None:
        print(f"\n  -> In this sample, every rolling window with a holding period of "
              f"{HORIZON_LABELS.get(h_full, str(h_full) + ' days')} or more was positive")
        print("     for the optimised portfolio (a horizon requirement, not a guarantee).")
    else:
        print("\n  -> No tested holding horizon produced positive returns in every rolling window;")
        print("     a 100% win rate is not achievable on this sample without a longer horizon.")

    print("\n  CAUTION: selecting the configuration that happens to hit 100% among many tried is")
    print("  data snooping. Treat it as a description of the past, not evidence of future results.")
    return best if n_hits > 0 else None


# --------------------------------------------------------------------------- #
# 5. Plotting
# --------------------------------------------------------------------------- #
def plot_efficient_frontier(
    frontier: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    opt: OptimizationResult,
    min_vol: OptimizationResult,
    bench_stats: PerformanceStats,
    bench_name: str,
    risk_free_rate: float,
    period_label: str,
    output_path: str,
    dpi: int = 300,
) -> None:
    """Efficient frontier, assets, optimal portfolio, capital market line and benchmark."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(frontier["volatility"], frontier["return"], color="#1f4e79", lw=2.5,
            label="Efficient frontier", zorder=2)

    asset_vol = np.sqrt(np.diag(cov.values))
    ax.scatter(asset_vol, mu.values, s=70, color="#7f8c8d", edgecolor="white", zorder=3,
               label="Individual assets")
    for ticker, x, y in zip(mu.index, asset_vol, mu.values):
        ax.annotate(ticker, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=9)

    # Capital market line through the tangency (max-Sharpe) portfolio
    x_max = max(frontier["volatility"].max(), asset_vol.max(), bench_stats.annual_vol) * 1.1
    slope = (opt.expected_return - risk_free_rate) / opt.volatility
    ax.plot([0, x_max], [risk_free_rate, risk_free_rate + slope * x_max], ls="--", lw=1.3,
            color="#c0392b", alpha=0.7, label=f"Capital market line (rf={risk_free_rate:.1%})", zorder=1)

    ax.scatter(min_vol.volatility, min_vol.expected_return, marker="D", s=90, color="#2e8b57",
               edgecolor="white", zorder=4, label="Minimum-variance portfolio")
    ax.scatter(opt.volatility, opt.expected_return, marker="*", s=420, color="#f1c40f",
               edgecolor="black", linewidth=1.0, zorder=5,
               label=f"Max-Sharpe portfolio (Sharpe {opt.sharpe:.2f})")
    ax.scatter(bench_stats.annual_vol, bench_stats.annual_return, marker="P", s=200, color="#e74c3c",
               edgecolor="black", zorder=5, label=f"{bench_name} (S&P 500) (Sharpe {bench_stats.sharpe:.2f})")

    # Frame the axes around the actual data (the CML would otherwise stretch the y-axis).
    y_values = np.concatenate([mu.values, frontier["return"].values,
                               [bench_stats.annual_return, risk_free_rate, opt.expected_return]])
    y_pad = 0.08 * (y_values.max() - y_values.min())
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(min(y_values.min(), 0.0) - y_pad, y_values.max() + y_pad)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Annualised volatility (risk)")
    ax.set_ylabel("Annualised return")
    ax.set_title(f"Markowitz Efficient Frontier vs {bench_name}\n{period_label}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=9, markerscale=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Markowitz mean-variance optimiser with S&P 500 benchmark and win-rate validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                   help="US tickers, space- or comma-separated.")
    p.add_argument("--start", help="Start date YYYY-MM-DD (overrides --years).")
    p.add_argument("--end", help="End date YYYY-MM-DD (default: today).")
    p.add_argument("--years", type=int, default=10, help="Lookback in years when --start is not given.")
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark ticker (S&P 500 proxy).")
    p.add_argument("--risk-free-rate", type=float, default=0.04, help="Annual risk-free rate (0.04 = 4%%).")
    p.add_argument("--min-weight", type=float, default=0.0, help="Minimum weight per asset (long-only: >= 0).")
    p.add_argument("--max-weight", type=float, default=1.0, help="Maximum weight per asset.")
    p.add_argument("--cov-method", choices=["sample", "ledoit_wolf"], default="sample",
                   help="Covariance estimator.")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Path of the efficient-frontier PNG.")
    p.add_argument("--dpi", type=int, default=300, help="Resolution of the saved plot.")
    p.add_argument("--frontier-points", type=int, default=60, help="Points used to trace the frontier.")

    g = p.add_argument_group("win-rate validation")
    g.add_argument("--skip-backtest", action="store_true", help="Skip the walk-forward win-rate screen.")
    g.add_argument("--lookbacks", nargs="+", type=int, default=[252, 504],
                   help="Estimation windows (trading days) to test.")
    g.add_argument("--rebalance-days", nargs="+", type=int, default=[21, 63, 126, 252],
                   help="Rebalance intervals (trading days) to test.")
    g.add_argument("--max-weights", nargs="+", type=float, default=[0.25, 0.35, 0.5, 1.0],
                   help="Per-asset weight caps to test (your --max-weight is always included).")
    g.add_argument("--min-periods", type=int, default=8,
                   help="Minimum holding periods for a win rate to count as meaningful.")
    g.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in bps of traded notional.")
    g.add_argument("--top-n", type=int, default=5, help="Rows of the ranking table to print.")

    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[str, str, list[str]]:
    """Validate CLI input and resolve the date range."""
    tickers = parse_tickers(args.tickers)
    if not (0.0 <= args.min_weight < args.max_weight <= 1.0):
        raise DataError("Weights must satisfy 0 <= min-weight < max-weight <= 1.")
    if args.years < 1 and not args.start:
        raise DataError("--years must be >= 1.")

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(dt.date.today())
    start = pd.Timestamp(args.start) if args.start else end - pd.DateOffset(years=args.years)
    if start >= end:
        raise DataError("Start date must be earlier than end date.")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), tickers


def run(args: argparse.Namespace) -> int:
    start, end, tickers = validate_args(args)
    bench_name = args.benchmark.upper()
    bounds = (args.min_weight, args.max_weight)
    if args.max_weight * len(set(tickers) - {bench_name}) < 1.0 - 1e-9:
        raise DataError("max-weight is too small: weights cannot sum to 100% with this many assets.")

    # --- Data -------------------------------------------------------------
    LOGGER.info("Downloading data for %s (+%s) from %s to %s ...",
                ", ".join(t for t in tickers if t != bench_name), bench_name, start, end)
    prices, bench_prices = load_market_data(tickers, bench_name, start, end)
    period_label = f"{prices.index[0].date()} to {prices.index[-1].date()}  ({len(prices)} trading days)"
    print(banner("MARKOWITZ MEAN-VARIANCE OPTIMISATION"))
    print(f"  Universe : {', '.join(prices.columns)}")
    print(f"  Benchmark: {bench_name}")
    print(f"  Period   : {period_label}")
    print(f"  Risk-free: {pct(args.risk_free_rate)}   Bounds: {pct(bounds[0], 0)} - {pct(bounds[1], 0)} per asset"
          f"   Covariance: {args.cov_method}")

    # --- Optimisation -----------------------------------------------------
    mu, cov = estimate_inputs(prices, args.cov_method)
    opt = optimize_max_sharpe(mu, cov, bounds, args.risk_free_rate)
    if opt.method != "max_sharpe":
        LOGGER.warning("Max-Sharpe was infeasible (no asset beat the risk-free rate?); used %s.", opt.method)
    ef_min = EfficientFrontier(mu, cov, weight_bounds=bounds)
    ef_min.min_volatility()
    w_min = _finalise_weights(np.asarray(ef_min.weights, dtype=float), mu.index)
    min_vol = OptimizationResult(
        w_min, float(w_min.values @ mu.values), float(np.sqrt(w_min.values @ cov.values @ w_min.values)),
        float("nan"), "min_volatility",
    )

    label = "maximum Sharpe ratio" if opt.method == "max_sharpe" else f"FALLBACK: {opt.method}"
    print(banner(f"OPTIMAL WEIGHTS ({label})"))
    print_weights(opt.weights)
    print(banner("EXPECTED PORTFOLIO METRICS (annualised, from historical estimates)"))
    print(f"  Expected return : {pct(opt.expected_return)}")
    print(f"  Volatility      : {pct(opt.volatility)}")
    print(f"  Sharpe ratio    : {num(opt.sharpe)}")

    # --- Benchmark comparison --------------------------------------------
    asset_returns = prices.pct_change().dropna()
    bench_returns = bench_prices.pct_change().dropna().reindex(asset_returns.index)
    port_returns = asset_returns.mul(opt.weights, axis=1).sum(axis=1)  # constant-mix (daily rebalanced)
    port_stats = compute_performance(port_returns, args.risk_free_rate)
    bench_stats = compute_performance(bench_returns, args.risk_free_rate)

    print(banner(f"PERFORMANCE vs S&P 500 ({bench_name}) - same period, in-sample"))
    print_benchmark_comparison(port_stats, bench_stats, bench_name)
    print("\n  Portfolio series = optimal weights held constant (daily rebalanced).")

    # --- Win-rate validation ---------------------------------------------
    print(banner("HISTORICAL WIN-RATE VALIDATION (walk-forward backtest)"))
    if args.skip_backtest:
        print("  Skipped (--skip-backtest).")
    else:
        port_profile = rolling_win_profile(port_returns)
        bench_profile = rolling_win_profile(bench_returns)
        _, summary = screen_configurations(
            prices, bench_prices, args.lookbacks, args.rebalance_days,
            sorted(set(args.max_weights) | {args.max_weight}), args.min_weight,
            args.risk_free_rate, args.cov_method, args.cost_bps, args.min_periods,
        )
        print_win_rate_report(summary, port_profile, bench_profile, bench_name,
                              args.min_periods, args.top_n)

    # --- Plot --------------------------------------------------------------
    frontier = compute_frontier(mu, cov, bounds, args.frontier_points)
    plot_efficient_frontier(frontier, mu, cov, opt, min_vol, bench_stats, bench_name,
                            args.risk_free_rate, period_label, args.output, args.dpi)
    print(banner("OUTPUT"))
    print(f"  Efficient-frontier chart saved to: {args.output}")
    print("\n  Disclaimer: historical, in-sample analysis for education/research only;")
    print("  not investment advice. Past performance does not predict future results.\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s", stream=sys.stderr)
    logging.getLogger("yfinance").setLevel(logging.ERROR)  # silence noisy per-ticker warnings
    try:
        return run(args)
    except DataError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130
    except Exception:  # last-resort guard: show traceback, return non-zero
        LOGGER.exception("Unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
