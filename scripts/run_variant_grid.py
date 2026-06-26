"""
Grid-search the order-flow signal variants in src.modules.tape_signal against
the cached universe (universe_cache.pkl at the repo root), independent of the
trained ML signal filter (which was fit on the original signal's feature
distribution and would not be a fair gate for a redefined signal).

Usage:
    python scripts/run_variant_grid.py
    python scripts/run_variant_grid.py --top 15
"""

from __future__ import annotations

import argparse
import itertools
import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(levelname)-8s  %(message)s")

from src.config.config import TapeBacktestConfig
from src.backtesting.engine import run_backtest
from src.modules.signal_filter import SignalFilter

UNIVERSE_CACHE = ROOT / "universe_cache.pkl"

# Benchmarks from the original (pre-variant) raw signal, no ML filter.
BASELINE = {"total_return_pct": -2.6, "sharpe_ratio": -0.94}
BID_REPULSION_ONLY = {"total_return_pct": 1.71, "sharpe_ratio": 1.6}

_DISABLED_FILTER = SignalFilter(model_path="__disabled_for_grid_search__.joblib")

GRID = {
    "two_phase_absorption": [False, True],
    "cvd_filter": [False, True],
    "stacked_bars": [1, 2, 3],
    "enable_ask_absorption": [True, False],
}


def iter_configs():
    keys = list(GRID.keys())
    for combo in itertools.product(*GRID.values()):
        flags = dict(zip(keys, combo))
        yield flags


def label(flags: dict) -> str:
    parts = []
    parts.append("2phase" if flags["two_phase_absorption"] else "1bar")
    parts.append("cvd" if flags["cvd_filter"] else "no-cvd")
    parts.append(f"stack{flags['stacked_bars']}")
    parts.append("ask+bid" if flags["enable_ask_absorption"] else "bid-only")
    return "|".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Order-flow signal variant grid search")
    parser.add_argument("--top", type=int, default=10, help="How many rows to print")
    parser.add_argument("--min-trades", type=int, default=30,
                         help="Minimum trade count for a variant to be eligible as 'winner' (Sharpe on <30 trades is too noisy to trust)")
    args = parser.parse_args()

    if not UNIVERSE_CACHE.exists():
        print(f"No cached universe at {UNIVERSE_CACHE} — aborting.")
        return

    with UNIVERSE_CACHE.open("rb") as f:
        universe = pickle.load(f)
    print(f"Loaded cached universe: {len(universe)} symbols")

    rows = []
    for flags in iter_configs():
        cfg = TapeBacktestConfig(**flags)
        result = run_backtest(universe, cfg, ml_filter=_DISABLED_FILTER)
        rows.append({
            "variant": label(flags),
            **flags,
            "total_return_pct": round(result.total_return_pct, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 3),
            "win_rate_pct": round(result.win_rate, 1),
            "num_trades": result.num_trades,
            "max_drawdown_pct": round(result.max_drawdown_pct, 1),
        })

    rows.sort(key=lambda r: r["sharpe_ratio"], reverse=True)

    print(f"\n{'='*100}")
    print("  ORDER-FLOW SIGNAL VARIANT GRID SEARCH  (ranked by Sharpe, ML filter disabled)")
    print(f"{'='*100}")
    header = f"{'variant':45s} {'return%':>9s} {'sharpe':>8s} {'win%':>7s} {'trades':>7s} {'mdd%':>7s}"
    print(header)
    print("-" * len(header))
    for r in rows[: args.top]:
        print(f"{r['variant']:45s} {r['total_return_pct']:9.2f} {r['sharpe_ratio']:8.3f} "
              f"{r['win_rate_pct']:7.1f} {r['num_trades']:7d} {r['max_drawdown_pct']:7.1f}")

    print(f"\nBenchmarks: baseline (1bar|no-cvd|stack1|ask+bid) = "
          f"{BASELINE['total_return_pct']}% / Sharpe {BASELINE['sharpe_ratio']}")
    print(f"            bid_repulsion-only (1bar|no-cvd|stack1|bid-only) = "
          f"{BID_REPULSION_ONLY['total_return_pct']}% / Sharpe {BID_REPULSION_ONLY['sharpe_ratio']}")

    eligible = [r for r in rows if r["num_trades"] >= args.min_trades]
    print(f"\n--- Robust ranking (>= {args.min_trades} trades; Sharpe on fewer trades is too noisy to trust) ---")
    for r in eligible[:10]:
        print(f"{r['variant']:45s} {r['total_return_pct']:9.2f} {r['sharpe_ratio']:8.3f} "
              f"{r['win_rate_pct']:7.1f} {r['num_trades']:7d} {r['max_drawdown_pct']:7.1f}")

    if not eligible:
        print("No variant cleared the minimum trade count — falling back to unrestricted winner.")
        winner = rows[0]
    else:
        winner = eligible[0]

    print(f"\nWinner (robust): {winner['variant']} -> {winner['total_return_pct']}% / "
          f"Sharpe {winner['sharpe_ratio']} on {winner['num_trades']} trades")
    if winner["sharpe_ratio"] < BID_REPULSION_ONLY["sharpe_ratio"]:
        print("Winner underperforms bid_repulsion-only baseline on Sharpe.")
    else:
        print("Winner beats bid_repulsion-only baseline on Sharpe.")

    return rows


if __name__ == "__main__":
    main()
