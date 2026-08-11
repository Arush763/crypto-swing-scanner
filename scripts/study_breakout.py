"""
Direction-agnostic volatility breakout — does it clear the fee floor?

Derived from measurement, not search:

  study_large_moves.py   range_expansion and volume_z predict that a large
                         move is coming: 1.26-1.32x lift over base rate,
                         monotone across deciles, holding out of time-sample.
  study_direction.py     nothing predicts WHICH WAY. Best directional spread
                         was 0.080% against a 0.21% cost floor, with no sign
                         stability between train and test.

The only strategy shape consistent with both findings is: forecast that a
move is coming, then let the market choose the direction and follow it.
Concretely — when volatility is forecast high, rest stop-entries on both
sides of the current bar; whichever is hit defines the trade.

This is deliberately NOT a parameter search. The gate comes from the study,
the entry rule has no free parameters, and the only knobs are the exit
target/stop in ATR units. Every number is reported train and test separately
so a result that only exists in the first 70% of history is visible as such.

Usage:
    python scripts/study_breakout.py
    python scripts/study_breakout.py --gate volume_z --target-atr 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger("breakout")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from src.execution.costs import cost_model_for
from src.indicators.volatility import atr as compute_atr
from scripts.study_large_moves import build_features, load_bars


@dataclass
class BreakTrade:
    symbol: str
    entry_time: pd.Timestamp
    direction: int          # +1 long, -1 short
    entry_price: float
    exit_price: float
    exit_reason: str
    bars_held: int
    gross_pct: float
    cost_pct: float

    @property
    def net_pct(self) -> float:
        return self.gross_pct - self.cost_pct


def simulate_symbol(
    symbol: str,
    bars: pd.DataFrame,
    gate_feature: str,
    gate_threshold: float,
    target_atr: float,
    stop_atr: float,
    max_hold: int,
    allow_shorts: bool,
    lookback: int = 20,
    whale_ref_idx: int = 0,
) -> list:
    """
    One symbol. At each gated bar, rest stop-entries at the bar's high and
    low; the next bar's action decides which fills.

    Ambiguity handling: if the next bar takes out BOTH sides, we cannot know
    from bar data which came first, so the trade is skipped rather than
    assumed favourable. Assuming the good side is how a backtest invents an
    edge that the live account never sees.
    """
    if len(bars) < lookback + max_hold + 10:
        return []

    feats = build_features(bars, lookback)
    atr_ser = compute_atr(bars["high"], bars["low"], bars["close"])

    high, low, close = bars["high"].values, bars["low"].values, bars["close"].values
    atr_vals = atr_ser.values

    if gate_feature == "combo":
        # High volume_z AND low whale_share — the joint condition validated in
        # study_combo.py (1.28x forward-excursion lift out of sample, with a
        # control showing high-whale-share does NOT lift, so the second term
        # is not merely proxying the first).
        vol = feats["volume_z"]
        whale = feats["whale_share"]
        vol_ok = (vol >= vol.iloc[:whale_ref_idx].quantile(0.80)).values
        whale_ok = (whale <= whale.iloc[:whale_ref_idx].quantile(0.30)).values
        gate_vals = np.where(vol_ok & whale_ok, 1.0, 0.0)
        gate_threshold = 0.5
    else:
        gate_vals = feats[gate_feature].values

    cost = cost_model_for(symbol, venue="okx").round_trip_pct()

    trades = []
    n = len(bars)
    i = lookback + 1
    while i < n - max_hold - 1:
        if not np.isfinite(gate_vals[i]) or gate_vals[i] < gate_threshold:
            i += 1
            continue

        trigger_high, trigger_low = high[i], low[i]
        nxt = i + 1
        broke_up = high[nxt] > trigger_high
        broke_down = low[nxt] < trigger_low

        if broke_up and broke_down:
            i += 1              # ambiguous ordering — skip, do not guess
            continue
        if not broke_up and not broke_down:
            i += 1
            continue
        if broke_down and not allow_shorts:
            i += 1
            continue

        direction = 1 if broke_up else -1
        entry = trigger_high if broke_up else trigger_low
        atr_at_entry = atr_vals[i]
        if not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
            i += 1
            continue

        target = entry + direction * target_atr * atr_at_entry
        stop = entry - direction * stop_atr * atr_at_entry

        exit_price, reason, held = close[min(nxt + max_hold, n - 1)], "time", max_hold
        for j in range(nxt, min(nxt + max_hold, n - 1) + 1):
            hit_stop = (low[j] <= stop) if direction == 1 else (high[j] >= stop)
            hit_target = (high[j] >= target) if direction == 1 else (low[j] <= target)

            # Stop checked first when a single bar spans both — same
            # conservative tie-break used by the live PositionMonitor.
            if hit_stop:
                exit_price, reason, held = stop, "stop", j - nxt + 1
                break
            if hit_target:
                exit_price, reason, held = target, "target", j - nxt + 1
                break

        gross = direction * (exit_price - entry) / entry * 100
        trades.append(BreakTrade(
            symbol=symbol, entry_time=bars.index[nxt], direction=direction,
            entry_price=entry, exit_price=exit_price, exit_reason=reason,
            bars_held=held, gross_pct=gross, cost_pct=cost,
        ))

        i = nxt + held + 1      # no overlapping positions in the same symbol

    return trades


def summarise(trades: list, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0, "gross": 0.0, "net": 0.0,
                "wr": 0.0, "total": 0.0, "longs": 0, "shorts": 0}
    gross = np.array([t.gross_pct for t in trades])
    net = np.array([t.net_pct for t in trades])
    return {
        "label": label,
        "n": len(trades),
        "gross": float(gross.mean()),
        "cost": float(np.mean([t.cost_pct for t in trades])),
        "net": float(net.mean()),
        "median": float(np.median(net)),
        "wr": float((net > 0).mean() * 100),
        "total": float(net.sum()),
        "longs": sum(1 for t in trades if t.direction == 1),
        "shorts": sum(1 for t in trades if t.direction == -1),
    }


def run_variant(universe: dict, gate: str, gate_pct: float, target_atr: float,
                stop_atr: float, max_hold: int, train_frac: float,
                long_only: bool = False) -> tuple:
    """Evaluate one parameter set, returning (train_summary, test_summary)."""
    trades = []
    for symbol, bars in universe.items():
        feats = build_features(bars, 20)
        split_idx = int(len(bars) * train_frac)

        if gate == "combo":
            threshold = 0.5     # set inside simulate_symbol
        else:
            train_gate = feats[gate].iloc[:split_idx].dropna()
            if train_gate.empty:
                continue
            threshold = float(np.percentile(train_gate, gate_pct))

        trades.extend(simulate_symbol(
            symbol, bars, gate, threshold, target_atr, stop_atr,
            max_hold, allow_shorts=not long_only, whale_ref_idx=split_idx,
        ))

    if not trades:
        return {"n": 0}, {"n": 0}
    trades.sort(key=lambda t: t.entry_time)
    split = int(len(trades) * train_frac)
    return summarise(trades[:split], "train"), summarise(trades[split:], "TEST")


def grid_search(universe: dict, train_frac: float) -> None:
    """
    Select on TRAIN, report TEST once.

    The discipline matters more than the grid. Scanning variants and quoting
    whichever scored best on test would reproduce exactly the error that
    invalidated the earlier sweep — the test number would then be a selection
    artifact, not an out-of-sample estimate. Here the winner is chosen blind
    to test, and its test result is reported whatever it turns out to be.
    """
    import itertools

    # Targets extended well beyond the original range. At 15m one ATR is
    # ~0.23-0.49% of price, so a 2xATR target on BTC is 0.46% against a
    # 0.21% fee — the exchange takes 46% of a perfect trade. Only targets of
    # 4xATR and up leave the fee as a minor term, so the grid has to reach
    # there or it is searching a region that cannot work by construction.
    grid = {
        "gate": ["combo", "volume_z", "range_expansion"],
        "gate_pct": [80.0, 90.0],
        "target_atr": [2.0, 4.0, 6.0, 8.0, 12.0],
        "stop_atr": [1.0, 2.0, 3.0],
        "max_hold": [16, 32, 64],
    }
    keys = list(grid)
    results = []

    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        tr, te = run_variant(universe, train_frac=train_frac, **params)
        if tr["n"] < 100:
            continue
        results.append((params, tr, te))
        logger.info("  %-58s train net %+.3f%% (n=%d)",
                    str(params), tr["net"], tr["n"])

    if not results:
        print("No variant produced enough trades.")
        return

    results.sort(key=lambda r: r[1]["net"], reverse=True)
    best_params, best_train, best_test = results[0]

    print()
    print("=" * 100)
    print("  GRID: selected on TRAIN, reported on TEST")
    print(f"  {len(results)} variants evaluated")
    print("=" * 100)
    print("  Best-on-train configuration:")
    for k, v in best_params.items():
        print(f"    {k:<14} {v}")
    print()
    print(f"{'split':<8}{'trades':>8}{'gross/tr':>11}{'cost/tr':>10}{'NET/tr':>10}"
          f"{'median':>10}{'WR':>8}{'total':>10}")
    print("-" * 100)
    for r in (best_train, best_test):
        print(f"{r['label']:<8}{r['n']:>8}{r['gross']:>10.3f}%{r['cost']:>9.3f}%"
              f"{r['net']:>9.3f}%{r['median']:>9.3f}%{r['wr']:>7.1f}%{r['total']:>9.1f}%")
    print("=" * 100)
    print()

    # How many variants were positive on train vs on test? If train produces
    # many winners and test almost none, the whole family is noise.
    train_pos = sum(1 for _, tr, _ in results if tr["net"] > 0)
    test_pos = sum(1 for _, _, te in results if te["n"] >= 30 and te["net"] > 0)
    print(f"  Variants net-positive on train: {train_pos}/{len(results)}")
    print(f"  Variants net-positive on test:  {test_pos}/{len(results)}")
    print()
    if best_test["n"] >= 30 and best_test["net"] > 0:
        print(f"  Best-on-train config holds out of sample: "
              f"{best_test['net']:+.3f}%/trade on {best_test['n']} test trades.")
    else:
        print("  Best-on-train config does NOT hold out of sample.")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Direction-agnostic breakout study")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--gate", default="range_expansion",
                   choices=["range_expansion", "volume_z", "combo"])
    p.add_argument("--gate-pct", type=float, default=90.0,
                   help="Percentile of the gate feature to trade above")
    p.add_argument("--target-atr", type=float, default=2.0)
    p.add_argument("--stop-atr", type=float, default=1.0)
    p.add_argument("--max-hold", type=int, default=16)
    p.add_argument("--long-only", action="store_true",
                   help="Disable shorts, to quantify what long-only forfeits")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--grid", action="store_true",
                   help="Search variants on train, report the winner on test once")
    args = p.parse_args()

    universe = load_bars(args.timeframe, args.days, args.symbols)
    if not universe:
        print("No data — aborting.")
        return

    if args.grid:
        grid_search(universe, args.train_frac)
        return

    # Gate threshold from the TRAIN portion of each symbol only, so the
    # percentile itself is not fitted using future data.
    all_trades = []
    for symbol, bars in universe.items():
        feats = build_features(bars, 20)
        split_idx = int(len(bars) * args.train_frac)

        if args.gate == "combo":
            threshold = 0.5     # the combo gate builds its own cut inside
        else:
            train_gate = feats[args.gate].iloc[:split_idx].dropna()
            if train_gate.empty:
                continue
            threshold = float(np.percentile(train_gate, args.gate_pct))

        all_trades.extend(simulate_symbol(
            symbol, bars, args.gate, threshold,
            args.target_atr, args.stop_atr, args.max_hold,
            allow_shorts=not args.long_only, whale_ref_idx=split_idx,
        ))

    if not all_trades:
        print("No trades generated.")
        return

    all_trades.sort(key=lambda t: t.entry_time)
    split = int(len(all_trades) * args.train_frac)
    train, test = all_trades[:split], all_trades[split:]

    rows = [summarise(train, "train"), summarise(test, "TEST"),
            summarise(all_trades, "all")]

    print()
    print("=" * 100)
    print(f"  DIRECTION-AGNOSTIC BREAKOUT   {args.timeframe}, {args.days}d, "
          f"{len(universe)} symbols")
    print(f"  Gate: {args.gate} >= p{args.gate_pct:g} (fitted on train only)   "
          f"{'LONG ONLY' if args.long_only else 'long + short'}")
    print(f"  Exit: target {args.target_atr:g}xATR / stop {args.stop_atr:g}xATR / "
          f"time {args.max_hold} bars")
    print("=" * 100)
    print(f"{'split':<8}{'trades':>8}{'gross/tr':>11}{'cost/tr':>10}{'NET/tr':>10}"
          f"{'median':>10}{'WR':>8}{'total':>10}{'L/S':>12}")
    print("-" * 100)
    for r in rows:
        if r["n"] == 0:
            continue
        print(f"{r['label']:<8}{r['n']:>8}{r['gross']:>10.3f}%{r['cost']:>9.3f}%"
              f"{r['net']:>9.3f}%{r['median']:>9.3f}%{r['wr']:>7.1f}%"
              f"{r['total']:>9.1f}%{r['longs']:>6}/{r['shorts']:<5}")
    print("=" * 100)
    print()

    test_row = rows[1]
    if test_row["n"] >= 30:
        if test_row["net"] > 0:
            print(f"  TEST net edge: {test_row['net']:+.3f}%/trade over "
                  f"{test_row['n']} out-of-sample trades.")
            print(f"  Gross {test_row['gross']:+.3f}% vs {test_row['cost']:.3f}% cost — "
                  f"cost is {test_row['cost']/max(test_row['gross'],1e-9)*100:.0f}% of gross.")
        else:
            print(f"  TEST net edge is NEGATIVE ({test_row['net']:+.3f}%/trade). "
                  f"Not tradeable as configured.")
    else:
        print(f"  Only {test_row['n']} out-of-sample trades — too few to conclude.")
    print()


if __name__ == "__main__":
    main()
