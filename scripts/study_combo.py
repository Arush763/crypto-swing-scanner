"""
Joint condition: high volume_z AND low whale_share.

Two findings survived the train/test split at every timeframe tested:

  volume_z      positive lift on large moves (1.26-1.32x at 15m/1h),
                monotone, stable sign across splits.
  whale_share   NEGATIVE lift, and it is the single most robust relationship
                in the study: 0.75x at 15m, 0.36x at 4h, with decile
                monotonicity of -0.87 to -0.99 in BOTH train and test.

The whale_share result is worth stating plainly because it is counter to the
usual intuition. A bar whose volume is dominated by a handful of very large
prints is followed by SMALLER moves, not larger ones. The reading that fits:
concentrated prints are liquidity being consumed — a block crossing, a
liquidation being absorbed — and that is the move finishing, not starting.
Broad participation, many hands, is what precedes continuation.

If both effects are real and partly independent, requiring high volume_z AND
low whale_share should select for a subset with materially larger forward
moves than either condition alone. That is the specific claim tested here,
and it is what would make the fee floor payable.

Usage:
    python scripts/study_combo.py
    python scripts/study_combo.py --timeframe 1h --horizon 12
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger("combo")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from scripts.study_large_moves import assemble, load_bars

COST_FLOOR = 0.21


def bucket_stats(df: pd.DataFrame, mask: pd.Series, label: str, base_up: float) -> dict:
    subset = df[mask]
    if len(subset) < 50:
        return {"label": label, "n": len(subset)}
    up = subset["fwd_max_up"].mean()
    down = subset["fwd_max_down"].mean()
    return {
        "label": label,
        "n": len(subset),
        "share": len(subset) / len(df) * 100,
        "up": up,
        "down": down,
        "asym": up + down,
        "up_lift": up / base_up if base_up else 0.0,
        "range": up - down,
    }


def analyse(df: pd.DataFrame, split_name: str, vol_q: float, whale_q: float) -> list:
    base_up = df["fwd_max_up"].mean()

    vol_hi = df["volume_z"] >= df["volume_z"].quantile(vol_q)
    whale_lo = df["whale_share"] <= df["whale_share"].quantile(whale_q)
    whale_hi = df["whale_share"] >= df["whale_share"].quantile(1 - whale_q)

    return [
        bucket_stats(df, pd.Series(True, index=df.index), f"{split_name}: all bars", base_up),
        bucket_stats(df, vol_hi, f"{split_name}: volume_z high", base_up),
        bucket_stats(df, whale_lo, f"{split_name}: whale_share low", base_up),
        bucket_stats(df, whale_hi, f"{split_name}: whale_share high", base_up),
        bucket_stats(df, vol_hi & whale_lo, f"{split_name}: BOTH (vol hi + whale lo)", base_up),
        bucket_stats(df, vol_hi & whale_hi, f"{split_name}: vol hi + whale HI", base_up),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Joint volume_z / whale_share condition")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--vol-q", type=float, default=0.80,
                   help="volume_z quantile above which counts as 'high'")
    p.add_argument("--whale-q", type=float, default=0.30,
                   help="whale_share quantile below which counts as 'low'")
    p.add_argument("--train-frac", type=float, default=0.7)
    args = p.parse_args()

    universe = load_bars(args.timeframe, args.days, args.symbols)
    df = assemble(universe, args.horizon, args.lookback)
    if df.empty:
        print("No usable rows — aborting.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * args.train_frac)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    print()
    print("=" * 104)
    print(f"  JOINT CONDITION   {args.timeframe}, {args.days}d, {len(universe)} symbols, "
          f"horizon {args.horizon}")
    print(f"  volume_z >= q{args.vol_q:.2f}   whale_share <= q{args.whale_q:.2f}")
    print(f"  Rows: {len(df):,} (train {len(train):,} / test {len(test):,})")
    print("=" * 104)
    print(f"{'bucket':<42}{'n':>8}{'share':>8}{'avgUp':>9}{'avgDown':>10}"
          f"{'range':>9}{'upLift':>9}{'asym':>9}")
    print("-" * 104)

    for rows in (analyse(train, "train", args.vol_q, args.whale_q),
                 analyse(test, "TEST", args.vol_q, args.whale_q)):
        for r in rows:
            if r.get("n", 0) < 50:
                continue
            print(f"{r['label']:<42}{r['n']:>8}{r['share']:>7.1f}%{r['up']:>8.2f}%"
                  f"{r['down']:>9.2f}%{r['range']:>8.2f}%{r['up_lift']:>8.2f}x"
                  f"{r['asym']:>8.3f}%")
        print("-" * 104)

    print()
    print("  range = avgUp - avgDown: the total excursion available to a trade that")
    print("          catches the move in whichever direction it goes. This is the number")
    print(f"          that has to beat the ~{COST_FLOOR:.2f}% round-trip cost, with enough margin")
    print("          left over to survive being wrong about entry timing.")
    print("  asym  = avgUp + avgDown: directional bias. Near zero = no directional edge.")
    print()


if __name__ == "__main__":
    main()
