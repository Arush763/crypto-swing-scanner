"""
Backtest: short the crowd when it is most long.

Origin of the signal (measured, not searched)
---------------------------------------------
study_positioning.py found that `count_long_short_ratio` — the ratio of
retail accounts long vs short on Binance perps — has a stable directional
relationship with forward returns. Ranked WITHIN each symbol, so no
cross-sectional level differences contribute:

    bottom decile (crowd least long)   train +0.036%   test +0.030%
    top decile    (crowd most long)    train -0.831%   test -0.604%
    spread                             train -0.867%   test -0.633%

Same sign across the split, magnitude ~3x the 0.21% round-trip cost. This is
the only feature in the whole project to clear both bars — every spot-derived
feature failed one or the other.

The mechanism is not subtle: crowded leveraged longs are the fuel for a
liquidation cascade, and a cascade is directional by construction.

The confound this script exists to rule out
-------------------------------------------
The sample period has negative drift (unconditional forward asymmetry
-0.25%). A strategy that only ever shorts will look good in a falling market
regardless of whether its signal means anything. So the headline test is the
LONG/SHORT spread version — short the crowded-long decile AND long the
crowded-short decile — which is drift-neutral by construction. If only the
short leg works, the "edge" is probably just the market going down.

Usage:
    python scripts/study_crowd_short.py
    python scripts/study_crowd_short.py --timeframe 4h --hold 16
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR, format="%(levelname)s  %(message)s")
logger = logging.getLogger("crowd")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from src.execution.costs import cost_model_for
from src.data.futures_metrics import align_to_bars
from scripts.scope_opportunity import resample_bars
from scripts.study_large_moves import load_bars

METRICS_CACHE = ROOT / "data" / "cache" / "study" / "futures_metrics_270d.pkl"
Z_95 = 1.96


@dataclass
class CrowdTrade:
    symbol: str
    entry_time: pd.Timestamp
    direction: int          # -1 short the crowded-long, +1 long the crowded-short
    entry_price: float
    exit_price: float
    bars_held: int
    gross_pct: float
    cost_pct: float
    rank: float

    @property
    def net_pct(self) -> float:
        return self.gross_pct - self.cost_pct


def simulate(symbol: str, bars: pd.DataFrame, metrics: pd.DataFrame,
             hold: int, top_q: float, bot_q: float, lookback: int,
             ref_idx: int, enable_long: bool, enable_short: bool) -> list:
    """
    Rank crowd_ratio within this symbol, then take a fixed-horizon position.

    Quantile cuts come from the TRAIN portion only. A flat time-based exit is
    used rather than a tuned stop/target: the signal is a statement about
    forward return over a horizon, and adding exit parameters here would
    reintroduce exactly the search that the 1,728-combo sweep exhausted.
    """
    aligned = align_to_bars(metrics, bars.index)
    if aligned.empty or "count_long_short_ratio" not in aligned.columns:
        return []

    crowd = aligned["count_long_short_ratio"]
    if crowd.notna().sum() < lookback + hold + 100:
        return []

    ref = crowd.iloc[:ref_idx].dropna()
    if len(ref) < 200:
        return []
    hi_cut = float(np.quantile(ref, top_q))
    lo_cut = float(np.quantile(ref, bot_q))

    close = bars["close"].values
    cr = crowd.values
    cost = cost_model_for(symbol, venue="okx").round_trip_pct()

    trades = []
    n = len(bars)
    i = lookback
    while i < n - hold - 1:
        v = cr[i]
        if not np.isfinite(v):
            i += 1
            continue

        direction = 0
        if enable_short and v >= hi_cut:
            direction = -1
        elif enable_long and v <= lo_cut:
            direction = +1

        if direction == 0:
            i += 1
            continue

        entry = close[i]
        exit_i = i + hold
        exit_price = close[exit_i]
        gross = direction * (exit_price - entry) / entry * 100

        trades.append(CrowdTrade(
            symbol=symbol, entry_time=bars.index[i], direction=direction,
            entry_price=entry, exit_price=exit_price, bars_held=hold,
            gross_pct=gross, cost_pct=cost, rank=float(v),
        ))
        i = exit_i + 1          # no overlapping positions per symbol

    return trades


def summarise(trades: list, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0}
    net = np.array([t.net_pct for t in trades])
    gross = np.array([t.gross_pct for t in trades])
    sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    se = sd / np.sqrt(len(net)) if sd else float("inf")
    return {
        "label": label, "n": len(net),
        "gross": float(gross.mean()), "net": float(net.mean()),
        "median": float(np.median(net)), "sd": sd, "se": se,
        "t": float(net.mean() / se) if np.isfinite(se) and se else 0.0,
        "wr": float((net > 0).mean() * 100), "total": float(net.sum()),
        "longs": sum(1 for t in trades if t.direction > 0),
        "shorts": sum(1 for t in trades if t.direction < 0),
    }


def report(rows: list, title: str) -> None:
    print()
    print("=" * 104)
    print(f"  {title}")
    print("=" * 104)
    print(f"{'variant':<26}{'trades':>8}{'gross/tr':>11}{'NET/tr':>10}{'median':>10}"
          f"{'t':>7}{'WR':>8}{'total':>10}{'L/S':>11}")
    print("-" * 104)
    for r in rows:
        if r.get("n", 0) == 0:
            print(f"{r['label']:<26}{0:>8}")
            continue
        print(f"{r['label']:<26}{r['n']:>8}{r['gross']:>10.3f}%{r['net']:>9.3f}%"
              f"{r['median']:>9.3f}%{r['t']:>7.2f}{r['wr']:>7.1f}%{r['total']:>9.1f}%"
              f"{r['longs']:>5}/{r['shorts']:<5}")
    print("=" * 104)


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest shorting the crowded-long decile")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--hold", type=int, default=16, help="Bars held per trade")
    p.add_argument("--top-q", type=float, default=0.90)
    p.add_argument("--bot-q", type=float, default=0.10)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--metrics", default=str(METRICS_CACHE))
    args = p.parse_args()

    with open(args.metrics, "rb") as fh:
        metrics_by_symbol = pickle.load(fh)
    universe = load_bars("15m", args.days, args.symbols)
    rule = None if args.timeframe == "15m" else args.timeframe

    variants = {
        "long/short spread": (True, True),
        "short leg only": (False, True),
        "long leg only": (True, False),
    }

    all_by_variant = {}
    for name, (do_long, do_short) in variants.items():
        trades = []
        for symbol, bars in universe.items():
            metrics = metrics_by_symbol.get(symbol)
            if metrics is None or metrics.empty:
                continue
            b = bars if rule is None else resample_bars(bars, rule)
            ref_idx = int(len(b) * args.train_frac)
            trades.extend(simulate(
                symbol, b, metrics, args.hold, args.top_q, args.bot_q,
                args.lookback, ref_idx, do_long, do_short,
            ))
        trades.sort(key=lambda t: t.entry_time)
        all_by_variant[name] = trades

    # ---- Headline: train vs test for each variant ----
    rows = []
    for name, trades in all_by_variant.items():
        if not trades:
            rows.append({"label": name, "n": 0})
            continue
        split = int(len(trades) * args.train_frac)
        rows.append(summarise(trades[:split], f"{name} [train]"))
        rows.append(summarise(trades[split:], f"{name} [TEST]"))

    report(rows, f"SHORT THE CROWDED LONG — {args.timeframe}, hold {args.hold} bars, "
                 f"q{args.top_q:.2f}/q{args.bot_q:.2f}")

    # ---- Robustness on the drift-neutral variant ----
    spread_trades = all_by_variant.get("long/short spread", [])
    if not spread_trades:
        return
    split = int(len(spread_trades) * args.train_frac)
    test = spread_trades[split:]
    if len(test) < 30:
        print(f"\n  Only {len(test)} out-of-sample trades — too few to judge.\n")
        return

    net = np.array([t.net_pct for t in test])
    df = pd.DataFrame({
        "t": [t.entry_time for t in spread_trades],
        "net": [t.net_pct for t in spread_trades],
        "symbol": [t.symbol for t in spread_trades],
        "dir": [t.direction for t in spread_trades],
    })
    df["quarter"] = pd.to_datetime(df["t"]).dt.tz_localize(None).dt.to_period("Q")

    print()
    print("  Drift-neutral (long/short) variant — robustness:")
    q = df.groupby("quarter")["net"].agg(["size", "mean", "sum"])
    print(f"    {'quarter':<10}{'trades':>8}{'net/tr':>10}{'total':>10}")
    print("    " + "-" * 38)
    for period, row in q.iterrows():
        print(f"    {str(period):<10}{int(row['size']):>8}{row['mean']:>9.3f}%{row['sum']:>9.1f}%")
    q_pos = int((q["mean"] > 0).sum())
    print(f"    -> positive in {q_pos}/{len(q)} quarters")

    per_sym = df.groupby("symbol")["net"].agg(["size", "mean"])
    sym_pos = int((per_sym["mean"] > 0).sum())
    print(f"\n    Symbols positive: {sym_pos}/{len(per_sym)}")

    ordered = np.sort(net)
    print("\n    Tail dependence (test):")
    for k in (1, 3, 5):
        if len(ordered) > k:
            m = float(ordered[:-k].mean())
            flag = "" if m > 0 else "   <-- flips negative"
            print(f"      excl. top-{k}: {m:+.3f}%/trade{flag}")

    se = net.std(ddof=1) / np.sqrt(len(net))
    checks = {
        "positive out-of-sample": net.mean() > 0,
        "significant (t>=2)": net.mean() / se >= 2.0,
        "positive in >=3 quarters": q_pos >= 3,
        "positive on >=half of symbols": sym_pos >= len(per_sym) / 2,
        "survives top-3 trim": float(np.sort(net)[:-3].mean()) > 0 if len(net) > 3 else False,
    }
    print()
    print("    " + "=" * 60)
    for name, ok in checks.items():
        print(f"      [{'PASS' if ok else 'FAIL'}]  {name}")
    print("    " + "=" * 60)
    print()
    if all(checks.values()):
        print("    All checks pass. This is the first signal in the project to do so.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"    {len(failed)} failed: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    main()
