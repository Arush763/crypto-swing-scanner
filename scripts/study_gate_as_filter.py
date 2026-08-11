"""
Does the volatility forecast improve the EXISTING signal?

The forecast validated in study_combo.py (high volume_z + low whale_share ->
1.28x forward-excursion lift, out of sample, with a control) failed to make
money as a standalone breakout strategy: 0 of 270 variants were net-positive.
That failure was about the ENTRY mechanism, not the forecast — a breakout
enters at the extreme and gets stopped by the symmetric wiggle.

So the forecast is applied here the other way round: as a filter on the
repo's existing tape signal, which supplies the entry. The question is narrow
and answerable — of the trades that signal already takes, do the ones fired
while the gate was open earn materially more than the ones fired while it was
shut?

This is the cheapest possible use of the finding. It adds no new entry logic
and cannot introduce lookahead, because the gate is computed from the same
bar the signal decided on. If gated trades earn enough more to clear the fee,
the filter is worth wiring into the live scanner; if not, the forecast is
real but not monetisable through this strategy, and that is worth knowing
definitively rather than assuming either way.

Usage:
    python scripts/study_gate_as_filter.py
    python scripts/study_gate_as_filter.py --timeframe 1h --atr-mult 4
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
logger = logging.getLogger("filter")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from src.config.config import TapeBacktestConfig
from src.backtesting.engine import run_backtest
from src.modules.signal_filter import SignalFilter
from scripts.study_large_moves import build_features, load_bars

_DISABLED_FILTER = SignalFilter(model_path="__disabled_for_filter_study__.joblib")


def gate_series(bars: pd.DataFrame, train_idx: int, vol_q: float, whale_q: float) -> pd.Series:
    """
    Boolean gate per bar: volume_z high AND whale_share low.

    Quantile cuts come from the TRAIN portion only, so the threshold itself
    never sees the period it is evaluated on.
    """
    feats = build_features(bars, 20)
    vol, whale = feats["volume_z"], feats["whale_share"]

    vol_cut = vol.iloc[:train_idx].quantile(vol_q)
    whale_cut = whale.iloc[:train_idx].quantile(whale_q)

    return ((vol >= vol_cut) & (whale <= whale_cut)).fillna(False)


def summarise(trades: list, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0}
    gross = np.array([t.gross_pnl_pct for t in trades])
    net = np.array([t.pnl_pct for t in trades])
    return {
        "label": label,
        "n": len(trades),
        "gross": float(gross.mean()),
        "cost": float(np.mean([t.cost_pct for t in trades])),
        "net": float(net.mean()),
        "median": float(np.median(gross)),
        "wr": float((net > 0).mean() * 100),
        "total": float(net.sum()),
    }


# ---------------------------------------------------------------------------
# Fee sensitivity
# ---------------------------------------------------------------------------

# Round-trip costs actually reachable, cheapest to dearest. These are the
# whole decision: the strategy's gross edge is fixed by the market, so the
# only lever left is which of these rows you are paying.
FEE_SCENARIOS = [
    ("taker both legs, no tier",   0.235, "what the backtest charges — worst realistic case"),
    ("maker entry, taker exit",    0.190, "post-only entry; exit must cross to be a real stop"),
    ("maker both legs",            0.160, "requires a resting exit — will sometimes not fill"),
    ("maker both + OKB discount",  0.128, "~20% fee discount for holding exchange token"),
    ("VIP1 maker both",            0.100, "needs ~$5M/30d volume — not reachable at this size"),
]


def fee_sensitivity(gated_all: list, gated_test: list, position_usd: float) -> None:
    """
    Net edge and dollar P&L across reachable fee schedules.

    Worth separating two things that get conflated: whether the strategy has
    an edge (it does — the gross figure is positive and survived the split),
    and whether that edge is large enough to be worth trading after costs and
    at a position size this account can carry. Those have different answers.
    """
    if not gated_test:
        return

    gross_all = float(np.mean([t.gross_pnl_pct for t in gated_all]))
    gross_test = float(np.mean([t.gross_pnl_pct for t in gated_test]))
    n_test = len(gated_test)

    print()
    print("=" * 100)
    print("  FEE SENSITIVITY — what survives at each reachable cost level")
    print(f"  Gross edge: {gross_test:+.3f}%/trade out-of-sample ({n_test} trades), "
          f"{gross_all:+.3f}% all-sample")
    print(f"  Position size: ${position_usd:,.0f}")
    print("=" * 100)
    print(f"{'fee scenario':<30}{'cost':>8}{'NET/tr':>10}{'kept':>8}"
          f"{'$/trade':>10}{'$ total':>10}   note")
    print("-" * 100)

    for label, cost, note in FEE_SCENARIOS:
        net = gross_test - cost
        kept = (net / gross_test * 100) if gross_test > 0 else 0.0
        dollars = net / 100 * position_usd
        print(f"{label:<30}{cost:>7.3f}%{net:>9.3f}%{kept:>7.0f}%"
              f"{dollars:>9.2f}{dollars * n_test:>10.2f}   {note}")

    print("=" * 100)
    print()
    print("  'kept' is the share of gross edge left after fees. Below ~50% the strategy is")
    print("  working mostly for the exchange, and small errors in the edge estimate flip")
    print("  the sign of the result.")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Volatility gate as a filter on the tape signal")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--atr-mult", type=float, default=2.0)
    p.add_argument("--vol-q", type=float, default=0.80)
    p.add_argument("--whale-q", type=float, default=0.30)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--fee-sensitivity", action="store_true",
                   help="Show net edge and dollar P&L across reachable fee schedules")
    p.add_argument("--position-usd", type=float, default=500.0,
                   help="Position size used for the dollar columns (default: MAX_POSITION_USD)")
    args = p.parse_args()

    universe = load_bars(args.timeframe, args.days, args.symbols)
    if not universe:
        print("No data — aborting.")
        return

    cfg = replace(
        TapeBacktestConfig(),
        timeframe=args.timeframe,
        apply_costs=True,
        btc_regime_filter=False,
        atr_trailing_stop_mult=args.atr_mult,
    )
    result = run_backtest(universe, cfg, ml_filter=_DISABLED_FILTER)
    completed = [t for t in result.trades if not t.is_open]
    if not completed:
        print("Base strategy produced no trades.")
        return

    # Map each trade's entry bar onto the gate state of the bar the signal
    # was decided on (entry_bar - 1), matching the engine's own timing.
    gates = {}
    for symbol, bars in universe.items():
        train_idx = int(len(bars) * args.train_frac)
        gates[symbol] = gate_series(bars, train_idx, args.vol_q, args.whale_q)

    gated, ungated = [], []
    for t in completed:
        g = gates.get(t.symbol)
        if g is None:
            continue
        signal_bar = max(0, t.entry_bar - 1)
        if signal_bar >= len(g):
            continue
        (gated if bool(g.iloc[signal_bar]) else ungated).append(t)

    completed.sort(key=lambda t: t.entry_time)
    split_time = completed[int(len(completed) * args.train_frac)].entry_time
    gated_test = [t for t in gated if t.entry_time >= split_time]
    ungated_test = [t for t in ungated if t.entry_time >= split_time]

    rows = [
        summarise(completed, "all trades"),
        summarise(gated, "GATE OPEN (all)"),
        summarise(ungated, "gate shut (all)"),
        summarise(gated_test, "GATE OPEN (test only)"),
        summarise(ungated_test, "gate shut (test only)"),
    ]

    print()
    print("=" * 100)
    print(f"  VOLATILITY GATE AS A FILTER   {args.timeframe}, {args.days}d, "
          f"{len(universe)} symbols, atr={args.atr_mult:g}")
    print(f"  Gate: volume_z >= q{args.vol_q:.2f} AND whale_share <= q{args.whale_q:.2f} "
          f"(cuts fitted on train)")
    print("=" * 100)
    print(f"{'bucket':<26}{'trades':>8}{'gross/tr':>11}{'cost/tr':>10}"
          f"{'NET/tr':>10}{'median':>10}{'WR':>8}{'total':>10}")
    print("-" * 100)
    for r in rows:
        if r["n"] == 0:
            print(f"{r['label']:<26}{0:>8}{'—':>11}")
            continue
        print(f"{r['label']:<26}{r['n']:>8}{r['gross']:>10.3f}%{r['cost']:>9.3f}%"
              f"{r['net']:>9.3f}%{r['median']:>9.3f}%{r['wr']:>7.1f}%{r['total']:>9.1f}%")
    print("=" * 100)
    print()

    if args.fee_sensitivity:
        fee_sensitivity(gated, gated_test, args.position_usd)

    g_all, u_all = rows[1], rows[2]
    g_te = rows[3]
    if g_all["n"] and u_all["n"]:
        delta = g_all["gross"] - u_all["gross"]
        print(f"  Gate open vs shut, gross/trade: {delta:+.3f}% "
              f"({g_all['gross']:+.3f}% vs {u_all['gross']:+.3f}%)")
    if g_te.get("n", 0) >= 30:
        verdict = "POSITIVE" if g_te["net"] > 0 else "still negative"
        print(f"  Out-of-sample gated net: {g_te['net']:+.3f}%/trade "
              f"on {g_te['n']} trades — {verdict}.")
    else:
        print(f"  Only {g_te.get('n', 0)} gated out-of-sample trades — too few to conclude.")
    print()


if __name__ == "__main__":
    main()
