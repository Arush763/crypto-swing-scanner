"""
Monte Carlo Simulation for Edge Verification.

Answers the core question: "Is this strategy's performance real alpha,
or just luck on a particular sequence of trades?"

Three complementary tests are run:

1. Trade Shuffle (Permutation Test)
   Randomly shuffle the sequence of actual trade P&Ls 10,000 times.
   If the strategy's Sharpe is in the top 5% of shuffled Sharpes, the
   ORDERING of trades (trend-following timing) adds real value beyond
   the raw win/loss distribution.

2. Random Entry Baseline
   Replace every strategy entry with a random entry on the same asset
   over the same date range, keeping the same exit rules (EMA / ATR stop).
   Run 5,000 iterations. If strategy Sharpe >> random-entry distribution,
   the ENTRY SELECTION (the scanner signals) is doing meaningful work.

3. Bootstrap Confidence Intervals
   Resample trades with replacement 10,000 times to build 95% CI around
   Sharpe, total return, win rate, and max drawdown.
   Wide CIs = small sample size, results unreliable.
   Narrow CIs well above zero = robust edge.

Interpretation guide:
  p-value < 0.05  →  result is statistically significant at 95% confidence
  p-value < 0.01  →  highly significant
  p-value > 0.10  →  likely noise; do not trade this

Output is a MonteCarloResult dataclass with all statistics and a
print_report() method for human-readable output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtesting.metrics import sharpe_ratio, total_return, win_rate, max_drawdown


# ── Config ────────────────────────────────────────────────────────────────
N_SHUFFLE      = 1_000
N_BOOTSTRAP    = 2_000
N_RANDOM_ENTRY = 500
CI_LEVEL       = 0.95
SIGNIFICANCE   = 0.05
RNG_SEED       = 42


# ── Helpers ───────────────────────────────────────────────────────────────

def _equity_from_pnls(pnls: Sequence[float], initial: float = 10_000.0, risk: float = 0.02) -> pd.Series:
    """Rebuild equity curve from a list of P&L percentages."""
    equity = [initial]
    cap = initial
    for p in pnls:
        cap = max(0.01, cap * (1 + p / 100 * risk))
        equity.append(cap)
    return pd.Series(equity, dtype=float)


def _sharpe_from_pnls(pnls: Sequence[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    eq = _equity_from_pnls(pnls)
    return sharpe_ratio(eq)


def _percentile_rank(value: float, distribution: np.ndarray) -> float:
    """What fraction of the distribution is BELOW value (one-tailed p-value)."""
    return float((distribution < value).mean())


# ── Result container ──────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    # Strategy actuals
    actual_sharpe:      float
    actual_return_pct:  float
    actual_win_rate:    float
    actual_max_dd_pct:  float
    n_trades:           int

    # Test 1 — Trade shuffle
    shuffle_p_value:        float = 0.0
    shuffle_significant:    bool  = False
    shuffle_sharpe_dist:    np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    # Test 2 — Random entry baseline
    random_entry_p_value:   float = 0.0
    random_entry_significant: bool = False
    random_entry_median_sharpe: float = 0.0
    random_sharpe_dist:     np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    # Test 3 — Bootstrap CIs
    bootstrap_sharpe_ci:    Tuple[float, float] = (0.0, 0.0)
    bootstrap_return_ci:    Tuple[float, float] = (0.0, 0.0)
    bootstrap_winrate_ci:   Tuple[float, float] = (0.0, 0.0)
    bootstrap_maxdd_ci:     Tuple[float, float] = (0.0, 0.0)

    # Overall verdict
    has_edge: bool = False

    def print_report(self) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("  MONTE CARLO EDGE VERIFICATION REPORT")
        print(sep)

        print(f"\n  Strategy Performance ({self.n_trades} trades)")
        print(f"  {'Sharpe Ratio':<28} {self.actual_sharpe:>8.3f}")
        print(f"  {'Total Return':<28} {self.actual_return_pct:>8.1f}%")
        print(f"  {'Win Rate':<28} {self.actual_win_rate:>8.1f}%")
        print(f"  {'Max Drawdown':<28} {self.actual_max_dd_pct:>8.1f}%")

        print(f"\n  Test 1 — Trade Shuffle (n={N_SHUFFLE:,})")
        print(f"  {'p-value':<28} {self.shuffle_p_value:>8.4f}  {'PASS' if self.shuffle_significant else 'FAIL'}")
        print(f"  {'Shuffle median Sharpe':<28} {float(np.median(self.shuffle_sharpe_dist)):>8.3f}")
        print(f"  Interpretation: {'Trade ORDER adds value.' if self.shuffle_significant else 'Trade order NOT significant — timing may be luck.'}")

        print(f"\n  Test 2 — Random Entry Baseline (n={N_RANDOM_ENTRY:,})")
        print(f"  {'p-value':<28} {self.random_entry_p_value:>8.4f}  {'PASS' if self.random_entry_significant else 'FAIL'}")
        print(f"  {'Random-entry median Sharpe':<28} {self.random_entry_median_sharpe:>8.3f}")
        print(f"  Interpretation: {'Entry selection is generating alpha.' if self.random_entry_significant else 'Entry signals NOT better than random — revisit filters.'}")

        lo_s, hi_s = self.bootstrap_sharpe_ci
        lo_r, hi_r = self.bootstrap_return_ci
        lo_w, hi_w = self.bootstrap_winrate_ci
        lo_d, hi_d = self.bootstrap_maxdd_ci
        print(f"\n  Test 3 — Bootstrap 95% Confidence Intervals (n={N_BOOTSTRAP:,})")
        print(f"  {'Sharpe CI':<28} [{lo_s:>6.3f},  {hi_s:>6.3f}]")
        print(f"  {'Return CI':<28} [{lo_r:>6.1f}%,  {hi_r:>6.1f}%]")
        print(f"  {'Win Rate CI':<28} [{lo_w:>6.1f}%,  {hi_w:>6.1f}%]")
        print(f"  {'Max Drawdown CI':<28} [{lo_d:>6.1f}%,  {hi_d:>6.1f}%]")
        ci_positive = lo_s > 0.0
        print(f"  Interpretation: {'Lower CI > 0 - edge is real even in bad-luck scenarios.' if ci_positive else 'Lower CI <= 0 - edge evaporates under bad luck. Need more trades.'}")

        print(f"\n  {'='*30}")
        verdict = "EDGE CONFIRMED" if self.has_edge else "EDGE NOT CONFIRMED"
        print(f"  VERDICT: {verdict}")
        if self.has_edge:
            print("  All 3 tests passed. Strategy shows statistically significant")
            print("  alpha beyond random chance. Proceed with caution and live testing.")
        else:
            failed = []
            if not self.shuffle_significant:   failed.append("trade shuffle")
            if not self.random_entry_significant: failed.append("random-entry baseline")
            if not (lo_s > 0.0):               failed.append("bootstrap CI")
            print(f"  Failed: {', '.join(failed)}.")
            print("  Consider: more history, tighter filters, or wider universe.")
        print(f"{sep}\n")


# ── Test 1: Trade Shuffle ─────────────────────────────────────────────────

def _run_shuffle_test(pnls: List[float]) -> Tuple[float, np.ndarray]:
    """
    Randomly permute the trade P&L sequence N_SHUFFLE times.
    Returns (p_value, distribution_of_sharpes).
    p_value = fraction of shuffles that beat the actual Sharpe.
    """
    rng = np.random.default_rng(RNG_SEED)
    actual_sharpe = _sharpe_from_pnls(pnls)
    arr = np.array(pnls, dtype=float)
    sharpes = np.empty(N_SHUFFLE)

    for i in range(N_SHUFFLE):
        rng.shuffle(arr)
        sharpes[i] = _sharpe_from_pnls(arr.tolist())

    # p-value: fraction of shuffles that achieved >= actual Sharpe
    p_value = float((sharpes >= actual_sharpe).mean())
    return p_value, sharpes


# ── Test 2: Random Entry Baseline ─────────────────────────────────────────

def _run_random_entry_test(
    universe: dict,             # {symbol: ohlcv_df}
    actual_pnls: List[float],
    cfg,                        # BacktestConfig
) -> Tuple[float, float, np.ndarray]:
    """
    For each iteration, replace every real entry with a random long entry
    on a random day, hold until EMA20 cross-below or 2-ATR stop — same
    exit rules as the strategy. Collect Sharpes over N_RANDOM_ENTRY runs.

    Returns (p_value, median_random_sharpe, distribution).
    """
    from src.indicators.trend import ema
    from src.indicators.volatility import atr as compute_atr
    from src.config.config import ATR_TRAILING_STOP_MULTIPLIER

    rng = np.random.default_rng(RNG_SEED)
    actual_sharpe = _sharpe_from_pnls(actual_pnls)
    n_trades_per_iter = max(1, len(actual_pnls))
    sharpes = np.empty(N_RANDOM_ENTRY)

    # Pre-compute indicators for every asset
    prepared = {}
    for sym, ohlcv in universe.items():
        if len(ohlcv) < 100:
            continue
        close = ohlcv["close"]
        high  = ohlcv["high"]
        low   = ohlcv["low"]
        e20   = ema(close, cfg.ema_short).values
        atr_s = compute_atr(high, low, close).values
        prepared[sym] = {
            "close": close.values,
            "open":  ohlcv["open"].values,
            "e20":   e20,
            "atr":   atr_s,
            "n":     len(close),
        }

    symbols = list(prepared.keys())
    if not symbols:
        return 1.0, 0.0, np.zeros(N_RANDOM_ENTRY)

    for i in range(N_RANDOM_ENTRY):
        iter_pnls = []

        for _ in range(n_trades_per_iter):
            sym = rng.choice(symbols)
            d   = prepared[sym]
            n   = d["n"]
            warm = max(cfg.ema_long, 60)
            if n <= warm + 5:
                continue

            # Random entry bar in valid range (skip last 5 bars to allow exit)
            entry_bar = int(rng.integers(warm, n - 5))
            entry_price = float(d["open"][entry_bar])
            if entry_price <= 0:
                continue

            highest = entry_price
            stop = entry_price - ATR_TRAILING_STOP_MULTIPLIER * float(d["atr"][entry_bar])
            exit_pnl = None

            for b in range(entry_bar + 1, n):
                price = float(d["close"][b])
                if price > highest:
                    highest = price
                    stop = highest - ATR_TRAILING_STOP_MULTIPLIER * float(d["atr"][b])
                # Exit: EMA cross-below or ATR stop
                if price < float(d["e20"][b]) or price < stop:
                    ep = float(d["open"][min(b + 1, n - 1)])
                    exit_pnl = (ep - entry_price) / entry_price * 100
                    break

            if exit_pnl is None:
                exit_pnl = (float(d["close"][-1]) - entry_price) / entry_price * 100

            iter_pnls.append(exit_pnl)

        sharpes[i] = _sharpe_from_pnls(iter_pnls) if iter_pnls else 0.0

    p_value = float((sharpes >= actual_sharpe).mean())
    median_random = float(np.median(sharpes))
    return p_value, median_random, sharpes


# ── Test 3: Bootstrap CI ──────────────────────────────────────────────────

def _run_bootstrap(pnls: List[float]) -> dict:
    """
    Resample trades with replacement N_BOOTSTRAP times.
    Returns 95% CI dicts for Sharpe, total return, win rate, max drawdown.
    """
    rng = np.random.default_rng(RNG_SEED)
    arr = np.array(pnls, dtype=float)
    n = len(arr)
    alpha = 1.0 - CI_LEVEL

    sharpes, returns, winrates, drawdowns = [], [], [], []

    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(arr, size=n, replace=True).tolist()
        eq = _equity_from_pnls(sample)
        sharpes.append(sharpe_ratio(eq))
        returns.append(total_return(eq))
        winrates.append(win_rate(sample))
        drawdowns.append(max_drawdown(eq))

    def ci(vals):
        a = np.array(vals)
        lo = float(np.percentile(a, alpha / 2 * 100))
        hi = float(np.percentile(a, (1 - alpha / 2) * 100))
        return (round(lo, 3), round(hi, 3))

    return {
        "sharpe":   ci(sharpes),
        "return":   ci(returns),
        "winrate":  ci(winrates),
        "maxdd":    ci(drawdowns),
    }


# ── Main entry point ──────────────────────────────────────────────────────

def run_monte_carlo(
    pnls: List[float],
    universe: dict,
    cfg,
    verbose: bool = True,
) -> MonteCarloResult:
    """
    Run all three Monte Carlo tests and return a MonteCarloResult.

    Args:
        pnls     : list of per-trade P&L percentages from the backtest
        universe : {symbol: ohlcv_df} — same universe used in backtest
        cfg      : BacktestConfig instance
        verbose  : if True, print progress to stdout
    """
    if len(pnls) < 10:
        raise ValueError(f"Need at least 10 trades for Monte Carlo; got {len(pnls)}.")

    if verbose:
        print(f"Running Monte Carlo on {len(pnls)} trades...")

    # Actual metrics
    eq_actual = _equity_from_pnls(pnls)
    act_sharpe = sharpe_ratio(eq_actual)
    act_return = total_return(eq_actual)
    act_wr     = win_rate(pnls)
    act_dd     = max_drawdown(eq_actual)

    # Test 1
    if verbose: print(f"  Test 1: Trade shuffle ({N_SHUFFLE:,} iterations)...", flush=True)
    shuf_p, shuf_dist = _run_shuffle_test(pnls)

    # Test 2
    if verbose: print(f"  Test 2: Random entry baseline ({N_RANDOM_ENTRY:,} iterations)...", flush=True)
    rand_p, rand_median, rand_dist = _run_random_entry_test(universe, pnls, cfg)

    # Test 3
    if verbose: print(f"  Test 3: Bootstrap CI ({N_BOOTSTRAP:,} iterations)...", flush=True)
    boot = _run_bootstrap(pnls)

    shuf_sig  = shuf_p < SIGNIFICANCE
    rand_sig  = rand_p < SIGNIFICANCE
    ci_sig    = boot["sharpe"][0] > 0.0

    has_edge  = shuf_sig and rand_sig and ci_sig

    result = MonteCarloResult(
        actual_sharpe=round(act_sharpe, 3),
        actual_return_pct=round(act_return, 2),
        actual_win_rate=round(act_wr, 1),
        actual_max_dd_pct=round(act_dd, 2),
        n_trades=len(pnls),
        shuffle_sharpe_dist=shuf_dist,
        shuffle_p_value=round(shuf_p, 4),
        shuffle_significant=shuf_sig,
        random_sharpe_dist=rand_dist,
        random_entry_p_value=round(rand_p, 4),
        random_entry_significant=rand_sig,
        random_entry_median_sharpe=round(rand_median, 3),
        bootstrap_sharpe_ci=boot["sharpe"],
        bootstrap_return_ci=boot["return"],
        bootstrap_winrate_ci=boot["winrate"],
        bootstrap_maxdd_ci=boot["maxdd"],
        has_edge=has_edge,
    )

    if verbose:
        result.print_report()

    return result
