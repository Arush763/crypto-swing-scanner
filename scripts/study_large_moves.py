"""
What precedes a large move? A measurement study, not a strategy.

Motivation
----------
The previous approach failed in a specific way worth not repeating: a config
grid was scored on a 45-day window, the best rows were selected, and they
inverted out of sample. The mistake was selecting on the same data used to
judge, with no prior belief about *why* any config should work.

So this script does not search configs and does not trade. It asks one
question of the tape: conditional on some order-flow feature being extreme
right now, how much more likely is a large move in the next few bars than it
would be at random? That number — lift over base rate — is the entire
prerequisite for a strategy. If no feature shows lift, no amount of stop
tuning creates it, and the honest answer is that the edge is not there.

Method
------
  * Features are strictly causal: every value at bar i uses only bars <= i.
  * The target is forward maximum favourable excursion over the next N bars,
    which is what a trade with a target order can actually capture.
  * The sample is split by TIME, train first / test last, and every number is
    reported on both. A feature that shows lift in train and none in test is
    noise that happened to sort well — that is precisely the failure mode of
    the earlier sweep, so it is made visible by construction rather than
    discovered later.
  * Lift is reported per decile, so a monotone relationship (more feature ->
    more move) can be distinguished from a single lucky bucket.

Usage:
    python scripts/study_large_moves.py
    python scripts/study_large_moves.py --days 180 --symbols 12 --horizon 16
    python scripts/study_large_moves.py --move-threshold 1.5
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger("study")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from src.config.config import MAJOR_BASES
from src.data.trade_tape import TradeTapeFetcher

CACHE_DIR = ROOT / "data" / "cache" / "study"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_bars(timeframe: str, days: int, n_symbols: int, workers: int = 16) -> dict:
    """Bars including microstructure columns, cached separately from the sweep."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"micro_{timeframe}_{days}d_{n_symbols}s.pkl"

    if cache.exists():
        with cache.open("rb") as fh:
            universe = pickle.load(fh)
        logger.info("Loaded %d symbols from cache", len(universe))
        return universe

    symbols = [f"{base}/USDT" for base in MAJOR_BASES[:n_symbols]]
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)

    logger.info("Fetching %s bars for %d symbols (%s -> %s)…",
                timeframe, len(symbols), start, end)
    bars = TradeTapeFetcher().fetch_many_bars(
        symbols, start, end, timeframe=timeframe, max_workers=workers,
    )
    universe = {s: b for s, b in bars.items() if not b.empty and "whale_buy_volume" in b.columns}

    with cache.open("wb") as fh:
        pickle.dump(universe, fh)
    logger.info("Cached %d symbols", len(universe))
    return universe


# ---------------------------------------------------------------------------
# Features — all causal
# ---------------------------------------------------------------------------

def build_features(bars: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Order-flow and microstructure features, each using only data up to and
    including its own bar.

    The hypotheses being encoded, stated up front so the results can falsify
    them rather than be fitted to:

      flow_imbalance    net aggressor pressure this bar
      whale_imbalance   direction of the *large* prints specifically —
                        the informed-participant proxy
      whale_share       what fraction of volume came from large prints.
                        High share = concentrated, intentional flow
      trade_size_z      is average print size unusually large right now?
                        A regime shift in participant type
      cvd_slope         persistence of net buying over the lookback
      compression       current range vs its own recent average. Large moves
                        are widely believed to emerge from compression;
                        this is the cleanest way to check that here
      volume_z          volume relative to its own recent average
      price_vs_range    where price sits in its recent range (0=low, 1=high)
    """
    f = pd.DataFrame(index=bars.index)

    volume = bars["volume"].replace(0, np.nan)
    buy, sell = bars["buy_volume"], bars["sell_volume"]

    f["flow_imbalance"] = (buy - sell) / volume

    whale_buy = bars["whale_buy_volume"]
    whale_sell = bars["whale_sell_volume"]
    whale_total = (whale_buy + whale_sell).replace(0, np.nan)
    f["whale_imbalance"] = (whale_buy - whale_sell) / whale_total
    f["whale_share"] = (whale_buy + whale_sell) / volume

    avg_size = bars["avg_trade_usd"]
    f["trade_size_z"] = (
        (avg_size - avg_size.rolling(lookback).mean())
        / avg_size.rolling(lookback).std()
    )

    f["cvd_slope"] = f["flow_imbalance"].rolling(lookback).mean()

    bar_range = (bars["high"] - bars["low"]) / bars["close"]
    # Named for what its TOP decile means: a bar whose range is large
    # relative to its own recent average, i.e. expansion already underway.
    # (Reading this as "compression" inverts the interpretation — the
    # compressed state is the BOTTOM decile.)
    f["range_expansion"] = bar_range / bar_range.rolling(lookback).mean()

    f["volume_z"] = (
        (bars["volume"] - bars["volume"].rolling(lookback).mean())
        / bars["volume"].rolling(lookback).std()
    )

    roll_high = bars["high"].rolling(lookback).max()
    roll_low = bars["low"].rolling(lookback).min()
    span = (roll_high - roll_low).replace(0, np.nan)
    f["price_vs_range"] = (bars["close"] - roll_low) / span

    return f.replace([np.inf, -np.inf], np.nan)


FEATURE_NAMES = [
    "flow_imbalance", "whale_imbalance", "whale_share", "trade_size_z",
    "cvd_slope", "range_expansion", "volume_z", "price_vs_range",
]


def build_targets(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Forward maximum favourable / adverse excursion over the next `horizon`
    bars, as a percentage of the current close.

    Uses shift(-1) before rolling so the current bar is excluded — a target
    that includes the bar its own feature was measured on is lookahead, and
    it is the easiest way to manufacture an edge that does not exist.
    """
    t = pd.DataFrame(index=bars.index)
    close = bars["close"]

    fwd_high = bars["high"].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    fwd_low = bars["low"].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))

    t["fwd_max_up"] = (fwd_high - close) / close * 100
    t["fwd_max_down"] = (fwd_low - close) / close * 100
    return t


def assemble(universe: dict, horizon: int, lookback: int) -> pd.DataFrame:
    frames = []
    for symbol, bars in universe.items():
        if len(bars) < lookback + horizon + 50:
            continue
        feats = build_features(bars, lookback)
        targs = build_targets(bars, horizon)
        merged = pd.concat([feats, targs], axis=1)
        merged["symbol"] = symbol
        merged["timestamp"] = bars.index
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=FEATURE_NAMES + ["fwd_max_up", "fwd_max_down"])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def decile_lift(df: pd.DataFrame, feature: str, threshold: float) -> pd.DataFrame:
    """P(large move | feature decile) against the unconditional base rate."""
    work = df[[feature, "fwd_max_up", "fwd_max_down"]].copy()
    work["is_large"] = work["fwd_max_up"] >= threshold

    try:
        work["decile"] = pd.qcut(work[feature], 10, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    base = work["is_large"].mean()
    if base <= 0:
        return pd.DataFrame()

    grouped = work.groupby("decile").agg(
        n=("is_large", "size"),
        hit_rate=("is_large", "mean"),
        avg_up=("fwd_max_up", "mean"),
        avg_down=("fwd_max_down", "mean"),
    )
    grouped["lift"] = grouped["hit_rate"] / base
    grouped["base"] = base
    return grouped


def summarise_feature(train: pd.DataFrame, test: pd.DataFrame,
                      feature: str, threshold: float) -> dict:
    tr = decile_lift(train, feature, threshold)
    te = decile_lift(test, feature, threshold)
    if tr.empty or te.empty:
        return {}

    top_tr, bot_tr = tr["lift"].iloc[-1], tr["lift"].iloc[0]
    top_te, bot_te = te["lift"].iloc[-1], te["lift"].iloc[0]

    # Spearman correlation of decile index against hit rate: does the
    # relationship increase monotonically, or is one bucket carrying it?
    mono_tr = tr.reset_index()[["decile", "hit_rate"]].corr(method="spearman").iloc[0, 1]
    mono_te = te.reset_index()[["decile", "hit_rate"]].corr(method="spearman").iloc[0, 1]

    return {
        "feature": feature,
        "train_top_lift": top_tr,
        "test_top_lift": top_te,
        "train_bot_lift": bot_tr,
        "test_bot_lift": bot_te,
        "train_mono": mono_tr,
        "test_mono": mono_te,
        "test_top_avg_up": te["avg_up"].iloc[-1],
        "test_top_avg_down": te["avg_down"].iloc[-1],
        "base_rate": te["base"].iloc[0],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Measure what precedes large moves")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--horizon", type=int, default=16,
                   help="Forward bars to look for the move (16 x 15m = 4h)")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--move-threshold", type=float, default=1.0,
                   help="A 'large move' is forward max up >= this percent")
    p.add_argument("--train-frac", type=float, default=0.7)
    args = p.parse_args()

    universe = load_bars(args.timeframe, args.days, args.symbols)
    if not universe:
        print("No data with microstructure columns — aborting.")
        return

    df = assemble(universe, args.horizon, args.lookback)
    if df.empty:
        print("No usable rows after feature construction — aborting.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * args.train_frac)
    train, test = df.iloc[:split], df.iloc[split:]

    base = (df["fwd_max_up"] >= args.move_threshold).mean()

    print()
    print("=" * 104)
    print(f"  WHAT PRECEDES A LARGE MOVE?   {args.timeframe}, {args.days}d, "
          f"{len(universe)} symbols")
    print(f"  Large move = forward max up >= {args.move_threshold:.2f}% "
          f"within {args.horizon} bars")
    print(f"  Rows: {len(df):,}  (train {len(train):,} / test {len(test):,}, split by time)")
    print(f"  Unconditional base rate: {base*100:.1f}%")
    print("=" * 104)
    print(f"{'feature':<18}{'trainLift':>11}{'testLift':>10}{'trainMono':>11}"
          f"{'testMono':>10}{'testUp':>9}{'testDown':>10}{'verdict':>16}")
    print("-" * 104)

    rows = [summarise_feature(train, test, f, args.move_threshold) for f in FEATURE_NAMES]
    rows = [r for r in rows if r]
    rows.sort(key=lambda r: r["test_top_lift"], reverse=True)

    for r in rows:
        # Real signal must (a) lift in test, not just train, and (b) do it
        # monotonically rather than through one bucket.
        holds = r["test_top_lift"] >= 1.15 and r["test_mono"] >= 0.6
        train_only = r["train_top_lift"] >= 1.15 and not holds
        verdict = "PREDICTIVE" if holds else ("train-only" if train_only else "no signal")

        print(f"{r['feature']:<18}{r['train_top_lift']:>10.2f}x{r['test_top_lift']:>9.2f}x"
              f"{r['train_mono']:>11.2f}{r['test_mono']:>10.2f}"
              f"{r['test_top_avg_up']:>8.2f}%{r['test_top_avg_down']:>9.2f}%"
              f"{verdict:>16}")

    print("=" * 104)
    print()
    print("  trainLift/testLift: P(large move | top decile) / base rate.")
    print("  Mono: Spearman corr of decile vs hit rate. Near 1.0 = orderly relationship;")
    print("        near 0 = one bucket doing the work, which will not repeat.")
    print("  testUp/testDown: average forward excursion in the top decile — the size of")
    print("        the move you would be trying to catch, and what it risks first.")
    print()


if __name__ == "__main__":
    main()
