"""
Where are moves large enough to pay the fee? A scoping measurement.

The 1,728-combo sweep established that tuning the existing signal is finished:
0 of 1,008 combos were net-positive out-of-sample, and the ceiling of the
whole parameter space (+0.148% gross) sits below the fee floor. The edge has
to change, not its parameters.

Rather than guess at a new signal, this measures where an exploitable move
even *exists*. For each timeframe and liquidity tier it reports the typical
favourable excursion available over a holding period, against the round-trip
cost of trading that tier. The ratio of the two is the ceiling on any
strategy operating there — no entry rule can capture more than the market
offers, so a cell whose ratio is near 1 cannot host a profitable strategy no
matter how good the signal is.

Two axes, both constrained early in this project for cost reasons and never
revisited:

  timeframe  Moves grow roughly with the square root of holding time while
             the fee is fixed per round trip. 15m was chosen for frequency;
             that choice caps the available move at a fraction of a percent.
  tier       Majors were chosen to minimise spread. But majors also move
             less. The cost penalty for a mid-cap is ~0.11%; the question
             never asked is whether its larger moves more than repay that.

Interpretation: `capture needed` is the fraction of the average favourable
excursion a strategy must convert into realised P&L just to break even. Above
~50% is not achievable in practice — entries are late and exits are early —
so treat that as the practical cutoff.

Usage:
    python scripts/scope_opportunity.py
    python scripts/scope_opportunity.py --horizons 8 16 32
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR, format="%(levelname)s  %(message)s")
logger = logging.getLogger("scope")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from src.execution.costs import cost_model_for
from scripts.study_large_moves import load_bars

# Practical ceiling on how much of an available move a real strategy converts
# into P&L. Entries are late, exits are early, and stops fire on noise.
REALISTIC_CAPTURE = 0.35


def resample_bars(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate 15m bars up to a coarser timeframe."""
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "buy_volume": "sum", "sell_volume": "sum",
    }
    for col in ("whale_buy_volume", "whale_sell_volume", "trade_count"):
        if col in bars.columns:
            agg[col] = "sum"
    out = bars.resample(rule).agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def excursion_stats(bars: pd.DataFrame, horizon: int) -> tuple:
    """Mean favourable / adverse excursion over the next `horizon` bars."""
    close = bars["close"]
    fwd_high = bars["high"].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    fwd_low = bars["low"].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    up = ((fwd_high - close) / close * 100).dropna()
    down = ((fwd_low - close) / close * 100).dropna()
    return float(up.mean()), float(down.mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Scope where tradeable moves exist")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--alt-cache", default="data/cache/study/micro_15m_180d_alts.pkl")
    p.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32])
    args = p.parse_args()

    groups = {}
    majors = load_bars("15m", args.days, args.symbols)
    if majors:
        groups["majors"] = majors

    alt_path = Path(args.alt_cache)
    if alt_path.exists():
        with alt_path.open("rb") as fh:
            alts = pickle.load(fh)
        if alts:
            groups["high-beta alts"] = alts
    else:
        logger.info("Alt cache not present yet (%s) — majors only", alt_path)

    # 15m base, aggregated upward. Horizons are in bars, so the holding period
    # in wall-clock time scales with the timeframe.
    timeframes = [("15m", None), ("1h", "1h"), ("4h", "4h"), ("1d", "1D")]

    print()
    print("=" * 108)
    print("  WHERE ARE MOVES LARGE ENOUGH TO PAY THE FEE?")
    print(f"  'capture needed' = share of the average favourable move a strategy must realise "
          f"to break even")
    print(f"  Practical ceiling is ~{REALISTIC_CAPTURE:.0%} capture; above that a cell cannot host "
          f"a profitable strategy")
    print("=" * 108)

    for group_name, universe in groups.items():
        sample_symbol = next(iter(universe))
        cost = cost_model_for(sample_symbol, venue="okx").round_trip_pct()

        print()
        print(f"  {group_name.upper()}  ({len(universe)} symbols, round-trip cost {cost:.3f}%)")
        print(f"    {'TF':<6}{'horizon':>9}{'hold':>10}{'avgUp':>9}{'avgDown':>10}"
              f"{'range':>9}{'capture needed':>16}{'verdict':>14}")
        print("    " + "-" * 94)

        for tf_label, rule in timeframes:
            ups, downs = [], []
            for bars in universe.values():
                b = bars if rule is None else resample_bars(bars, rule)
                if len(b) < max(args.horizons) + 30:
                    continue
                for h in args.horizons[:1]:      # headline horizon per timeframe
                    u, d = excursion_stats(b, h)
                    if np.isfinite(u):
                        ups.append(u)
                        downs.append(d)
            if not ups:
                continue

            horizon = args.horizons[0]
            up, down = float(np.mean(ups)), float(np.mean(downs))
            minutes = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf_label] * horizon
            hold = f"{minutes/60:.0f}h" if minutes < 1440 else f"{minutes/1440:.0f}d"

            needed = cost / up if up > 0 else float("inf")
            if needed <= REALISTIC_CAPTURE:
                verdict = "VIABLE"
            elif needed <= 0.6:
                verdict = "marginal"
            else:
                verdict = "impossible"

            print(f"    {tf_label:<6}{horizon:>9}{hold:>10}{up:>8.2f}%{down:>9.2f}%"
                  f"{up-down:>8.2f}%{needed:>15.0%}{verdict:>14}")

    print()
    print("=" * 108)
    print()
    print("  Reading this: at 15m a major offers ~0.8% of favourable excursion over 4h, against")
    print("  a 0.21% fee — you must convert a quarter of the theoretical move into realised P&L")
    print("  just to break even, before being wrong about entry timing. Coarser timeframes and")
    print("  higher-beta symbols both raise the numerator while the fee stays fixed.")
    print()


if __name__ == "__main__":
    main()
