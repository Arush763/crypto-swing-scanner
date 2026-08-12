"""
Net-of-cost edge across timeframes.

Answers the question the whole high-frequency plan rests on: as the bar gets
shorter, does the strategy's per-trade edge shrink faster than the trading
cost stays fixed?

It has to shrink — a shorter bar captures a smaller price move — while the
round-trip cost does not change at all. The only question is where the two
lines cross, and that crossing point is the highest frequency this strategy
can actually be traded at. Guessing it is how accounts get spent on fees.

Usage:
    python scripts/run_timeframe_sweep.py
    python scripts/run_timeframe_sweep.py --days 90 --timeframes 4h 1h 15m 5m
    python scripts/run_timeframe_sweep.py --maker      # model post-only entries
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger("sweep")
logger.setLevel(logging.INFO)

import numpy as np

from src.config.config import MAJOR_BASES, TapeBacktestConfig
from src.backtesting.engine import run_backtest
from src.data.trade_tape import TradeTapeFetcher
from src.execution.costs import cost_model_for
from src.modules.signal_filter import SignalFilter


DEFAULT_TIMEFRAMES = ["4h", "1h", "15m", "5m"]


def build_symbols(limit: int) -> list:
    """Majors only, quoted in USDT — the venue's deepest books."""
    return [f"{base}/USDT" for base in MAJOR_BASES[:limit]]


def summarise(result, timeframe: str, bars_by_symbol: dict) -> dict:
    completed = [t for t in result.trades if not t.is_open]
    if not completed:
        return {
            "timeframe": timeframe, "trades": 0, "gross_pct": 0.0, "cost_pct": 0.0,
            "net_pct": 0.0, "win_rate": 0.0, "gross_win_rate": 0.0, "total_net": 0.0,
        }

    gross = np.mean([t.gross_pnl_pct for t in completed])
    cost = np.mean([t.cost_pct for t in completed])
    net = np.mean([t.pnl_pct for t in completed])

    return {
        "timeframe": timeframe,
        "trades": len(completed),
        "gross_pct": gross,
        "cost_pct": cost,
        "net_pct": net,
        "win_rate": sum(1 for t in completed if t.pnl_pct > 0) / len(completed) * 100,
        "gross_win_rate": sum(1 for t in completed if t.gross_pnl_pct > 0) / len(completed) * 100,
        "total_net": sum(t.pnl_pct for t in completed),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Measure net-of-cost edge by timeframe")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    p.add_argument("--symbols", type=int, default=6,
                   help="How many majors to include (default: 6)")
    p.add_argument("--maker", action="store_true",
                   help="Model post-only maker entries instead of taker")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    symbols = build_symbols(args.symbols)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days)

    logger.info("Symbols:    %s", ", ".join(symbols))
    logger.info("Window:     %s -> %s (%d days)", start, end, args.days)
    logger.info("Entry type: %s", "post-only maker" if args.maker else "taker")
    logger.info("")

    fetcher = TradeTapeFetcher()
    rows = []

    for timeframe in args.timeframes:
        logger.info("Fetching %s bars…", timeframe)
        bars_by_symbol = fetcher.fetch_many_bars(
            symbols, start, end, timeframe=timeframe, max_workers=args.workers,
        )
        universe = {s: b for s, b in bars_by_symbol.items() if not b.empty}
        if not universe:
            logger.warning("No data for %s — skipping", timeframe)
            continue

        cfg = TapeBacktestConfig(
            timeframe=timeframe,
            apply_costs=True,
            entry_is_maker=args.maker,
            exit_is_maker=False,
            btc_regime_filter=False,
        )
        # Disable the ML filter: it was trained on 4h gross-outcome labels,
        # so letting it gate a 5m net-of-cost run would confound the very
        # comparison this script exists to make.
        result = run_backtest(
            universe, cfg,
            ml_filter=SignalFilter(model_path=Path("data/models/__sweep_disabled__.joblib")),
        )
        rows.append(summarise(result, timeframe, universe))

    # ---- Report ----
    print()
    print("=" * 88)
    print(f"  NET-OF-COST EDGE BY TIMEFRAME   ({start} to {end}, "
          f"{'maker' if args.maker else 'taker'} entries)")
    print("=" * 88)
    print(f"{'TF':<6}{'trades':>8}{'gross/tr':>11}{'cost/tr':>10}"
          f"{'NET/tr':>10}{'gross WR':>10}{'net WR':>9}{'total net':>12}")
    print("-" * 88)
    for r in rows:
        print(f"{r['timeframe']:<6}{r['trades']:>8}"
              f"{r['gross_pct']:>10.3f}%{r['cost_pct']:>9.3f}%"
              f"{r['net_pct']:>9.3f}%{r['gross_win_rate']:>9.1f}%"
              f"{r['win_rate']:>8.1f}%{r['total_net']:>11.1f}%")
    print("=" * 88)

    viable = [r for r in rows if r["net_pct"] > 0 and r["trades"] >= 30]
    print()
    if viable:
        best = max(viable, key=lambda r: r["net_pct"] * r["trades"])
        print(f"  Highest-frequency timeframe with positive net edge: "
              f"{min(viable, key=lambda r: _tf_minutes(r['timeframe']))['timeframe']}")
        print(f"  Best total net return: {best['timeframe']} "
              f"({best['total_net']:+.1f}% over {best['trades']} trades)")
    else:
        print("  No timeframe shows a positive net-of-cost edge on this sample.")
        print("  Trading any of them at size would lose money on fees alone.")
    print()

    # The single most useful diagnostic: how much of the gross edge the
    # exchange takes at each frequency.
    print("  Cost as a share of gross edge:")
    for r in rows:
        if r["gross_pct"] > 0:
            share = r["cost_pct"] / r["gross_pct"] * 100
            verdict = "viable" if share < 60 else ("marginal" if share < 100 else "UNTRADEABLE")
            print(f"    {r['timeframe']:<5} {share:>6.1f}%   {verdict}")
        else:
            print(f"    {r['timeframe']:<5}      —   no gross edge to begin with")
    print()


def _tf_minutes(tf: str) -> int:
    unit = tf[-1]
    n = int(tf[:-1])
    return n * {"m": 1, "h": 60, "d": 1440}.get(unit, 1)


if __name__ == "__main__":
    main()
