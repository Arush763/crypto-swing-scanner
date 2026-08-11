"""
Is DIRECTION predictable, or only volatility?

study_large_moves.py established that two features (range_expansion,
volume_z) genuinely predict that a large move is coming — 1.26-1.32x lift,
monotone across deciles, holding out of time-sample. But it also showed the
top decile's average forward excursion was +0.80% up against -0.76% down:
almost perfectly symmetric. That is the signature of a volatility forecast,
not a directional one.

The distinction decides everything downstream. A long-only strategy fed by a
volatility forecast takes the +0.80% and the -0.76% in roughly equal measure
and pays a fee on both. A direction-agnostic strategy — react to the break
rather than predict it — only needs volatility to be forecastable, which it
demonstrably is.

This script measures directional edge directly. For each feature decile it
reports the ASYMMETRY (avg_up + avg_down; both are signed, so this is net
directional bias) rather than the hit rate, and checks whether that asymmetry
survives the train/test time split.

A feature with real directional alpha shows asymmetry that grows with the
decile and keeps its sign in test. Everything else is a volatility signal
wearing a directional costume.

Usage:
    python scripts/study_direction.py
    python scripts/study_direction.py --horizon 16 --conditional
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger("direction")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from scripts.study_large_moves import (
    FEATURE_NAMES, assemble, load_bars,
)


def decile_asymmetry(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Per decile: average up excursion, average down excursion, and their sum.

    The sum is the number that matters. If a feature only forecasts
    volatility, up grows and down shrinks (more negative) together and the
    sum stays near zero across every decile.
    """
    work = df[[feature, "fwd_max_up", "fwd_max_down"]].copy()
    try:
        work["decile"] = pd.qcut(work[feature], 10, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    grouped = work.groupby("decile").agg(
        n=("fwd_max_up", "size"),
        avg_up=("fwd_max_up", "mean"),
        avg_down=("fwd_max_down", "mean"),
    )
    grouped["asymmetry"] = grouped["avg_up"] + grouped["avg_down"]
    return grouped


def summarise(train: pd.DataFrame, test: pd.DataFrame, feature: str) -> dict:
    tr = decile_asymmetry(train, feature)
    te = decile_asymmetry(test, feature)
    if tr.empty or te.empty:
        return {}

    # Raw top-decile asymmetry is contaminated by whatever the market did
    # over that period — if the train half drifted down and the test half up,
    # every feature's asymmetry shifts with it and none of that is alpha.
    # The top-minus-bottom SPREAD cancels that common drift, so it is the
    # drift-free measure of a feature's directional information and the only
    # one worth comparing across splits.
    return {
        "feature": feature,
        "train_top_asym": tr["asymmetry"].iloc[-1],
        "test_top_asym": te["asymmetry"].iloc[-1],
        "train_bot_asym": tr["asymmetry"].iloc[0],
        "test_bot_asym": te["asymmetry"].iloc[0],
        "train_spread": tr["asymmetry"].iloc[-1] - tr["asymmetry"].iloc[0],
        "test_spread": te["asymmetry"].iloc[-1] - te["asymmetry"].iloc[0],
        "test_top_up": te["avg_up"].iloc[-1],
        "test_top_down": te["avg_down"].iloc[-1],
    }


def volatility_conditional(df: pd.DataFrame, vol_feature: str, top_decile: int = 9) -> None:
    """
    Restrict to bars where a large move is already forecast, then ask whether
    direction is predictable *within* that subset.

    This is the strongest remaining hope for a directional edge: flow might
    only be informative when something is actually happening. If direction is
    unpredictable even here, it is unpredictable.
    """
    work = df.copy()
    try:
        work["vol_decile"] = pd.qcut(work[vol_feature], 10, labels=False, duplicates="drop")
    except ValueError:
        return

    hot = work[work["vol_decile"] >= top_decile]
    if len(hot) < 500:
        print(f"  (only {len(hot)} high-volatility rows — too few to judge)")
        return

    print()
    print(f"  Within the top decile of {vol_feature}  ({len(hot):,} bars):")
    print(f"    {'direction feature':<20}{'botAsym':>10}{'topAsym':>10}{'spread':>10}")
    print("    " + "-" * 50)

    for feature in ["flow_imbalance", "whale_imbalance", "cvd_slope", "trade_size_z"]:
        d = decile_asymmetry(hot, feature)
        if d.empty:
            continue
        bot, top = d["asymmetry"].iloc[0], d["asymmetry"].iloc[-1]
        print(f"    {feature:<20}{bot:>9.3f}%{top:>9.3f}%{top - bot:>9.3f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Test for directional predictability")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--conditional", action="store_true",
                   help="Also test direction within the high-volatility subset")
    args = p.parse_args()

    universe = load_bars(args.timeframe, args.days, args.symbols)
    df = assemble(universe, args.horizon, args.lookback)
    if df.empty:
        print("No usable rows — aborting.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * args.train_frac)
    train, test = df.iloc[:split], df.iloc[split:]

    overall_asym = df["fwd_max_up"].mean() + df["fwd_max_down"].mean()

    print()
    print("=" * 96)
    print(f"  IS DIRECTION PREDICTABLE?   {args.timeframe}, {args.days}d, "
          f"{len(universe)} symbols, horizon {args.horizon}")
    print(f"  Rows: {len(df):,} (train {len(train):,} / test {len(test):,})")
    print(f"  Unconditional asymmetry: {overall_asym:+.3f}%  "
          f"(up {df['fwd_max_up'].mean():+.3f}% / down {df['fwd_max_down'].mean():+.3f}%)")
    print("=" * 96)
    print(f"{'feature':<18}{'trainSpread':>13}{'testSpread':>12}"
          f"{'testUp':>9}{'testDown':>10}{'verdict':>18}")
    print("-" * 96)

    rows = [summarise(train, test, f) for f in FEATURE_NAMES]
    rows = [r for r in rows if r]
    rows.sort(key=lambda r: r["test_spread"], reverse=True)

    for r in rows:
        # Directional edge requires the drift-free spread to (a) exceed a
        # round trip and (b) keep its sign across the split. A feature that
        # is strongly positive in test and negative in train has told us
        # about those two periods, not about the feature.
        big_enough = r["test_spread"] > 0.20
        stable = np.sign(r["test_spread"]) == np.sign(r["train_spread"])

        if big_enough and stable:
            verdict = "DIRECTIONAL"
        elif big_enough:
            verdict = "test-only (unstable)"
        else:
            verdict = "volatility only"

        print(f"{r['feature']:<18}{r['train_spread']:>12.3f}%{r['test_spread']:>11.3f}%"
              f"{r['test_top_up']:>8.2f}%{r['test_top_down']:>9.2f}%{verdict:>18}")

    print("=" * 96)
    print()
    print("  Asymmetry = avg forward up + avg forward down. Near zero means the feature")
    print("  forecasts move SIZE but not SIGN — tradeable only direction-agnostically.")
    print("  testSpread = top decile asymmetry minus bottom decile's; this is the actual")
    print("  directional edge available, and it must exceed ~0.21% to pay a round trip.")

    if args.conditional:
        volatility_conditional(test, "range_expansion")
        volatility_conditional(test, "volume_z")
    print()


if __name__ == "__main__":
    main()
