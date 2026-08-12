"""
Is the gated edge statistically real, or just a small sample looking lucky?

On 200 days the gated subset showed +0.360% gross and +0.125% net per trade
out of sample. Those are not the same kind of number:

    gross  t = 2.45   statistically real
    net    t = 0.85   indistinguishable from zero, 95% CI [-0.162%, +0.411%]

The fee is what moves it across that line. It takes 65% of the edge and takes
the confidence with it — which is the actual reason fees matter here, more
than the headline reduction in profit.

This script exists so that judgement is made on statistics rather than on the
mean alone, and repeatably as the sample grows. It reports:

  * t-statistic and confidence interval on NET per-trade P&L
  * the same for gross, to separate "no edge" from "edge eaten by fees"
  * sensitivity to the largest few winners — the check that caught the
    earlier sweep's overfitting, and which the 200-day run failed
    (net of top 3 winners: -0.039%/trade)
  * how many trades would be needed to reach significance at each fee level

Usage:
    python scripts/study_significance.py --days 365
    python scripts/study_significance.py --days 200 --position-usd 2000
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger("significance")
logger.setLevel(logging.INFO)

import numpy as np

from src.config.config import TapeBacktestConfig
from src.backtesting.engine import run_backtest
from src.modules.signal_filter import SignalFilter
from scripts.study_gate_as_filter import gate_series, FEE_SCENARIOS
from scripts.study_large_moves import load_bars

_DISABLED_FILTER = SignalFilter(model_path="__disabled_for_significance__.joblib")

# Two-sided 95% critical value. Using the normal approximation rather than a
# t-distribution because n is comfortably large enough for the difference to
# be immaterial, and being slightly conservative here is the right direction.
Z_95 = 1.96


def collect_gated(timeframe: str, days: int, symbols: int, atr_mult: float,
                  vol_q: float, whale_q: float, train_frac: float):
    universe = load_bars(timeframe, days, symbols)
    if not universe:
        return None, None

    cfg = replace(
        TapeBacktestConfig(), timeframe=timeframe, apply_costs=True,
        btc_regime_filter=False, atr_trailing_stop_mult=atr_mult,
    )
    result = run_backtest(universe, cfg, ml_filter=_DISABLED_FILTER)
    completed = sorted(
        (t for t in result.trades if not t.is_open), key=lambda t: t.entry_time,
    )
    if not completed:
        return None, None

    gates = {
        s: gate_series(b, int(len(b) * train_frac), vol_q, whale_q)
        for s, b in universe.items()
    }

    gated = []
    for t in completed:
        g = gates.get(t.symbol)
        if g is None:
            continue
        signal_bar = max(0, t.entry_bar - 1)
        if signal_bar < len(g) and bool(g.iloc[signal_bar]):
            gated.append(t)

    split_time = completed[int(len(completed) * train_frac)].entry_time
    return gated, [t for t in gated if t.entry_time >= split_time]


def stats_block(values: np.ndarray, label: str) -> dict:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n and sd else float("inf")
    return {
        "label": label, "n": n, "mean": mean, "sd": sd, "se": se,
        "t": mean / se if se else 0.0,
        "lo": mean - Z_95 * se, "hi": mean + Z_95 * se,
    }


def trades_needed(mean: float, sd: float, target_t: float = 2.0) -> float:
    """Sample size at which this mean/sd would reach `target_t`."""
    if mean <= 0 or sd <= 0:
        return float("inf")
    return (target_t * sd / mean) ** 2


def main() -> None:
    p = argparse.ArgumentParser(description="Significance of the gated edge")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--atr-mult", type=float, default=2.0)
    p.add_argument("--vol-q", type=float, default=0.80)
    p.add_argument("--whale-q", type=float, default=0.30)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--position-usd", type=float, default=500.0)
    args = p.parse_args()

    gated_all, gated_test = collect_gated(
        args.timeframe, args.days, args.symbols, args.atr_mult,
        args.vol_q, args.whale_q, args.train_frac,
    )
    if not gated_test:
        print("No gated out-of-sample trades — aborting.")
        return

    net = np.array([t.pnl_pct for t in gated_test])
    gross = np.array([t.gross_pnl_pct for t in gated_test])
    span = (gated_test[-1].entry_time - gated_test[0].entry_time).days or 1

    s_net = stats_block(net, "net")
    s_gross = stats_block(gross, "gross")

    print()
    print("=" * 96)
    print(f"  SIGNIFICANCE OF THE GATED EDGE   {args.timeframe}, {args.days}d, "
          f"{args.symbols} symbols")
    print(f"  Out-of-sample: {len(net)} trades over {span} days "
          f"({len(gated_all)} gated in total)")
    print("=" * 96)
    print(f"{'':<8}{'mean/tr':>10}{'sd':>9}{'SE':>9}{'t-stat':>9}"
          f"{'95% CI':>24}{'verdict':>22}")
    print("-" * 96)
    for s in (s_gross, s_net):
        real = s["t"] >= 2.0
        verdict = "statistically real" if real else "cannot reject zero"
        ci = f"[{s['lo']:+.3f}%, {s['hi']:+.3f}%]"
        print(f"{s['label']:<8}{s['mean']:>9.3f}%{s['sd']:>8.3f}%{s['se']:>8.3f}%"
              f"{s['t']:>9.2f}{ci:>24}{verdict:>22}")
    print("=" * 96)

    # --- Tail dependence -------------------------------------------------
    ordered = np.sort(net)
    ordered_gross = np.sort(gross)
    print()
    print("  Tail dependence — the check the 200-day sample failed:")
    print(f"    {'trimmed':<24}{'net':>10}{'gross':>10}")
    for k in (1, 3, 5):
        if len(ordered) > k:
            t_net = float(ordered[:-k].mean())
            t_gross = float(ordered_gross[:-k].mean())
            flag = "" if t_net > 0 else "   <-- net flips negative"
            print(f"    {'excl. top-' + str(k) + ' winners':<24}"
                  f"{t_net:>9.3f}%{t_gross:>9.3f}%{flag}")
    # Whether the tail-dependence is intrinsic or fee-induced matters for what
    # to do about it: if gross survives the trim, the body of trades does have
    # an edge and the fee is what pushes it under, which is fixable by paying
    # less. If gross also flips, the strategy genuinely rests on a few
    # outliers and no fee schedule saves it.
    if len(ordered_gross) > 3:
        if ordered_gross[:-3].mean() > 0:
            print("    -> gross survives the trim: the body of trades has an edge and the")
            print("       fee is what pushes it under. Lower fees address this directly.")
        else:
            print("    -> gross ALSO flips: the edge genuinely rests on a few outliers,")
            print("       and no fee schedule fixes that.")
    wins, losses = net[net > 0], net[net <= 0]
    if len(wins) and len(losses):
        print(f"    {len(wins)} wins avg {wins.mean():+.2f}% | "
              f"{len(losses)} losses avg {losses.mean():+.2f}%")
        print(f"    largest win {net.max():+.2f}%  largest loss {net.min():+.2f}%")

    # --- Fee levels ------------------------------------------------------
    print()
    print("  By fee level (gross edge held fixed, cost varied):")
    print(f"    {'scenario':<30}{'net/tr':>10}{'t-stat':>9}{'n for t=2':>12}{'$/yr':>10}")
    print("    " + "-" * 71)
    for label, cost, _note in FEE_SCENARIOS:
        m = s_gross["mean"] - cost
        t = m / s_net["se"] if s_net["se"] else 0.0
        need = trades_needed(m, s_net["sd"])
        per_year = m / 100 * args.position_usd * len(net) * 365 / span
        need_s = f"{need:,.0f}" if np.isfinite(need) else "never"
        print(f"    {label:<30}{m:>9.3f}%{t:>9.2f}{need_s:>12}{per_year:>10.0f}")

    print()
    print(f"  Position size ${args.position_usd:,.0f}; $/yr extrapolates the observed trade")
    print(f"  rate ({len(net)/span*365:.0f} trades/yr) — it is not a forecast, and it assumes")
    print("  the edge is real, which the t-stat above may or may not support.")
    print()


if __name__ == "__main__":
    main()
