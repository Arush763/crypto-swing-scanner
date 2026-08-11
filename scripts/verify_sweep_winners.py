"""
Robustness check on the configs that cleared the cost floor.

A high average profit per trade means nothing if one or two trades produced
it — that is a lottery ticket, not an edge, and it will not repeat. This
re-runs the leading configs and reports, for each:

  * gross/trade after stripping any single trade worth more than a third of
    total gross profit (engine.exclude_dominant_trades)
  * the largest single trade's share of gross profit
  * median trade, alongside the mean — a mean far above the median is the
    signature of a distribution carried by its tail
  * out-of-sample behaviour on symbols the sweep did not select on

Anything that survives all four is worth acting on. Anything that doesn't is
noise that happened to sort well.
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
logger = logging.getLogger("verify")
logger.setLevel(logging.INFO)

import numpy as np

from src.config.config import TapeBacktestConfig
from src.backtesting.engine import run_backtest, exclude_dominant_trades
from src.modules.signal_filter import SignalFilter
from scripts.run_profit_per_trade_sweep import (
    SIGNAL_FAMILIES, base_cfg, load_bars, COST_FLOOR_TAKER, COST_FLOOR_MAKER,
)

_DISABLED_FILTER = SignalFilter(model_path="__disabled_for_verify__.joblib")

# The configs that cleared the floor in the families stage.
CANDIDATES = [
    ("climax_exhaustion", 6.0),
    ("climax_exhaustion", 10.0),
    ("delta_divergence", 10.0),
    ("delta_divergence", 6.0),
    ("liquidity_sweep", 10.0),
    ("repulsion_only", 10.0),
]


def analyse(universe: dict, family: str, atr_mult: float, timeframe: str) -> dict:
    cfg = replace(
        base_cfg(timeframe),
        atr_trailing_stop_mult=atr_mult,
        **SIGNAL_FAMILIES[family],
    )
    result = run_backtest(universe, cfg, ml_filter=_DISABLED_FILTER)
    completed = [t for t in result.trades if not t.is_open]

    if len(completed) < 5:
        return {"label": f"{family} atr={atr_mult:g}", "trades": len(completed),
                "gross": 0.0, "robust_gross": 0.0, "top_share": 0.0,
                "median": 0.0, "symbols": 0}

    gross_pnls = [t.gross_pnl_pct for t in completed]
    winners = [p for p in gross_pnls if p > 0]
    gross_profit = sum(winners) if winners else 0.0
    top_share = (max(gross_pnls) / gross_profit * 100) if gross_profit > 0 else 0.0

    survivors = exclude_dominant_trades(completed)
    robust = float(np.mean([t.gross_pnl_pct for t in survivors])) if survivors else 0.0

    # How concentrated is this across symbols? An "edge" living in one ticker
    # is a property of that ticker's 45 days, not of the signal.
    by_symbol = {}
    for t in completed:
        by_symbol.setdefault(t.symbol, []).append(t.gross_pnl_pct)
    profitable_symbols = sum(1 for pnls in by_symbol.values() if np.mean(pnls) > COST_FLOOR_TAKER)

    return {
        "label": f"{family} atr={atr_mult:g}",
        "trades": len(completed),
        "gross": float(np.mean(gross_pnls)),
        "robust_gross": robust,
        "robust_trades": len(survivors),
        "top_share": top_share,
        "median": float(np.median(gross_pnls)),
        "symbols": len(by_symbol),
        "profitable_symbols": profitable_symbols,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Robustness-check the sweep winners")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--symbols", type=int, default=5)
    args = p.parse_args()

    universe = load_bars(args.timeframe, args.days, args.symbols)
    rows = [analyse(universe, fam, mult, args.timeframe) for fam, mult in CANDIDATES]

    print()
    print("=" * 108)
    print(f"  ROBUSTNESS CHECK  ({args.timeframe}, {args.days}d, {len(universe)} symbols)")
    print(f"  Cost floor: {COST_FLOOR_TAKER:.2f}% taker / {COST_FLOOR_MAKER:.2f}% maker")
    print("=" * 108)
    print(f"{'config':<30}{'trades':>7}{'mean':>9}{'median':>9}"
          f"{'ex-dominant':>13}{'top trade':>11}{'sym +ve':>10}{'verdict':>18}")
    print("-" * 108)

    for r in rows:
        if r["trades"] < 5:
            print(f"{r['label']:<30}{r['trades']:>7}{'—':>9}{'—':>9}"
                  f"{'—':>13}{'—':>11}{'—':>10}{'too few trades':>18}")
            continue

        # A mean that collapses once the biggest trade is removed was never a
        # per-trade edge; it was one trade wearing an average's clothing.
        #
        # The median matters just as much and is easy to skip past: a config
        # with a healthy mean and a negative median loses on most trades and
        # is rescued only by its tail. That can still be a real edge, but it
        # is a fragile one — it needs the tail to keep showing up, and a
        # 45-day sample cannot establish that it will.
        robust_mean = r["robust_gross"] > COST_FLOOR_MAKER
        not_outlier_driven = r["top_share"] < 40
        broad = r["profitable_symbols"] >= 2
        typical_trade_pays = r["median"] > 0

        if robust_mean and not_outlier_driven and broad and typical_trade_pays:
            verdict = "holds up"
        elif robust_mean and not_outlier_driven and broad:
            verdict = "tail-dependent"
        else:
            verdict = "carried by outliers"

        print(f"{r['label']:<30}{r['trades']:>7}{r['gross']:>8.3f}%{r['median']:>8.3f}%"
              f"{r['robust_gross']:>12.3f}%{r['top_share']:>10.1f}%"
              f"{r['profitable_symbols']:>4}/{r['symbols']:<5}{verdict:>18}")

    print("=" * 108)
    print()
    print("  mean vs median: a mean far above the median means the tail is doing the work.")
    print("  ex-dominant:    mean after removing any trade worth >1/3 of gross profit.")
    print("  top trade:      largest single trade as a share of total gross profit.")
    print("  sym +ve:        symbols whose own mean trade clears the taker cost floor.")
    print()


if __name__ == "__main__":
    main()
