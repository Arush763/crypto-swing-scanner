"""
Does leveraged POSITIONING predict directional moves?

Everything tested so far derives from spot price and spot volume, and that
family is exhausted: 1,728 combos at 15m gave 0/1008 net-positive
out-of-sample, 638 combos at 4h gave 8% with a median of -1.496%, and the
direction study found no feature whose drift-free directional spread reached
the 0.21% fee floor with a stable sign.

Positioning data is a different kind of information. It does not describe
what has traded — it describes what leveraged participants currently hold,
and therefore what they can be forced to do. The hypotheses tested here, each
stated before looking so the result can falsify them:

  oi_change          Open interest rising into a price move means new money
                     entering; falling means positions closing. The same
                     price action means different things under the two.
  oi_price_divergence  Price up on FALLING open interest is short-covering,
                     not accumulation — a move with nobody left to buy it.
                     Classic exhaustion setup.
  crowd_ratio_z      When the retail account ratio gets crowded on one side,
                     that side is the fuel for a liquidation cascade. This is
                     the direct "large move with a direction" hypothesis.
  smart_vs_crowd     Top accounts by position size vs the account-count
                     crowd. When they diverge, one group is positioned wrong,
                     and historically it is not the larger accounts.
  taker_ratio        Perp aggressor imbalance — the closest analogue to what
                     was already tested on spot, included as a control. If it
                     scores like the spot version, the new data adds nothing.

Method is identical to study_direction.py, so results are comparable:
train/test split by time, decile analysis, and the DRIFT-FREE spread
(top-decile asymmetry minus bottom-decile) rather than raw asymmetry, since
raw asymmetry mostly measures whatever the market did that period.

Usage:
    python scripts/study_positioning.py
    python scripts/study_positioning.py --horizon 32 --timeframe 1h
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
logger = logging.getLogger("positioning")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from src.data.futures_metrics import align_to_bars
from scripts.scope_opportunity import resample_bars
from scripts.study_large_moves import load_bars

METRICS_CACHE = ROOT / "data" / "cache" / "study" / "futures_metrics_270d.pkl"

COST_FLOOR = 0.21


def build_positioning_features(bars: pd.DataFrame, metrics: pd.DataFrame,
                               lookback: int = 20) -> pd.DataFrame:
    """
    Positioning features aligned onto `bars`. All strictly causal: metrics are
    forward-filled from the last reading at or before each bar's start.
    """
    aligned = align_to_bars(metrics, bars.index)
    f = pd.DataFrame(index=bars.index)
    if aligned.empty or "sum_open_interest" not in aligned.columns:
        return f

    oi = aligned["sum_open_interest"]
    close = bars["close"]

    oi_ret = oi.pct_change(lookback)
    price_ret = close.pct_change(lookback)

    f["oi_change"] = oi_ret
    f["oi_z"] = (oi - oi.rolling(lookback).mean()) / oi.rolling(lookback).std()

    # Price up on falling OI = short covering; price up on rising OI = new
    # longs. Sign of the product separates the two regimes.
    f["oi_price_divergence"] = np.sign(price_ret) * oi_ret

    if "count_long_short_ratio" in aligned.columns:
        crowd = aligned["count_long_short_ratio"]
        f["crowd_ratio"] = crowd
        f["crowd_ratio_z"] = (
            (crowd - crowd.rolling(lookback).mean()) / crowd.rolling(lookback).std()
        )

    if "sum_toptrader_long_short_ratio" in aligned.columns and "count_long_short_ratio" in aligned.columns:
        top = aligned["sum_toptrader_long_short_ratio"]
        f["top_ratio_z"] = (top - top.rolling(lookback).mean()) / top.rolling(lookback).std()
        # Positive = big accounts more long than the crowd is.
        f["smart_vs_crowd"] = (
            np.log(top.replace(0, np.nan))
            - np.log(aligned["count_long_short_ratio"].replace(0, np.nan))
        )

    if "sum_taker_long_short_vol_ratio" in aligned.columns:
        f["taker_ratio"] = aligned["sum_taker_long_short_vol_ratio"]

    return f.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "oi_change", "oi_z", "oi_price_divergence",
    "crowd_ratio", "crowd_ratio_z", "top_ratio_z", "smart_vs_crowd", "taker_ratio",
]


def build_targets(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    t = pd.DataFrame(index=bars.index)
    close = bars["close"]
    fwd_high = bars["high"].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    fwd_low = bars["low"].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    t["fwd_max_up"] = (fwd_high - close) / close * 100
    t["fwd_max_down"] = (fwd_low - close) / close * 100
    return t


def decile_asymmetry(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    work = df[[feature, "fwd_max_up", "fwd_max_down"]].dropna()
    if len(work) < 500:
        return pd.DataFrame()
    try:
        work["decile"] = pd.qcut(work[feature], 10, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = work.groupby("decile").agg(
        n=("fwd_max_up", "size"),
        avg_up=("fwd_max_up", "mean"),
        avg_down=("fwd_max_down", "mean"),
    )
    g["asymmetry"] = g["avg_up"] + g["avg_down"]
    return g


def main() -> None:
    p = argparse.ArgumentParser(description="Do positioning metrics predict direction?")
    p.add_argument("--timeframe", default="1h", help="Resample rule off the 15m base")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--metrics", default=str(METRICS_CACHE))
    args = p.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics cache not found at {metrics_path} — run the fetch first.")
        return
    with metrics_path.open("rb") as fh:
        metrics_by_symbol = pickle.load(fh)

    universe = load_bars("15m", args.days, args.symbols)
    rule = None if args.timeframe == "15m" else args.timeframe

    frames = []
    for symbol, bars in universe.items():
        metrics = metrics_by_symbol.get(symbol)
        if metrics is None or metrics.empty:
            continue
        b = bars if rule is None else resample_bars(bars, rule)
        if len(b) < args.lookback + args.horizon + 100:
            continue
        feats = build_positioning_features(b, metrics, args.lookback)
        if feats.empty:
            continue
        merged = pd.concat([feats, build_targets(b, args.horizon)], axis=1)
        merged["symbol"] = symbol
        merged["timestamp"] = b.index
        frames.append(merged)

    if not frames:
        print("No symbols had both bars and metrics — aborting.")
        return

    df = pd.concat(frames, ignore_index=True)
    present = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=["fwd_max_up", "fwd_max_down"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    split = int(len(df) * args.train_frac)
    train, test = df.iloc[:split], df.iloc[split:]

    print()
    print("=" * 100)
    print(f"  DOES POSITIONING PREDICT DIRECTION?   {args.timeframe}, "
          f"{len(set(df['symbol']))} symbols, horizon {args.horizon}")
    print(f"  Rows: {len(df):,} (train {len(train):,} / test {len(test):,})")
    print(f"  Unconditional: up {df['fwd_max_up'].mean():+.2f}%  "
          f"down {df['fwd_max_down'].mean():+.2f}%")
    print("=" * 100)
    print(f"{'feature':<22}{'trainSpread':>13}{'testSpread':>12}"
          f"{'testUp':>9}{'testDown':>10}{'verdict':>20}")
    print("-" * 100)

    rows = []
    for feature in present:
        tr = decile_asymmetry(train, feature)
        te = decile_asymmetry(test, feature)
        if tr.empty or te.empty:
            continue
        tr_spread = tr["asymmetry"].iloc[-1] - tr["asymmetry"].iloc[0]
        te_spread = te["asymmetry"].iloc[-1] - te["asymmetry"].iloc[0]
        rows.append({
            "feature": feature,
            "train_spread": tr_spread, "test_spread": te_spread,
            "test_up": te["avg_up"].iloc[-1], "test_down": te["avg_down"].iloc[-1],
        })

    rows.sort(key=lambda r: abs(r["test_spread"]), reverse=True)
    for r in rows:
        big = abs(r["test_spread"]) > COST_FLOOR
        stable = np.sign(r["test_spread"]) == np.sign(r["train_spread"])
        if big and stable:
            verdict = "DIRECTIONAL"
        elif big:
            verdict = "test-only (unstable)"
        else:
            verdict = "no directional edge"
        print(f"{r['feature']:<22}{r['train_spread']:>12.3f}%{r['test_spread']:>11.3f}%"
              f"{r['test_up']:>8.2f}%{r['test_down']:>9.2f}%{verdict:>20}")

    print("=" * 100)
    print()
    print("  Spread = top-decile asymmetry minus bottom-decile, which cancels market drift.")
    print(f"  It must exceed {COST_FLOOR:.2f}% (a round trip) AND keep its sign across the split.")
    print("  A spread that is large but sign-unstable is the signature of the false positives")
    print("  this project has already produced twice.")
    print()


if __name__ == "__main__":
    main()
