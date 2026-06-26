"""
Backtest the four new order-flow signals (liquidity_sweep, climax_exhaustion,
delta_divergence, vwap_fade) individually, then grid-search combinations of
them — on their own and layered on top of the best absorption/repulsion
config found previously (stacked_bars=2, bid_repulsion-only) — against the
cached universe (universe_cache.pkl at the repo root). ML filter disabled
throughout to isolate the raw signal logic (see scripts/run_variant_grid.py).

Usage:
    python scripts/run_new_signals_grid.py
    python scripts/run_new_signals_grid.py --top 15 --min-trades 20
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
_DISABLED_FILTER = SignalFilter(model_path="__disabled_for_grid_search__.joblib")

PRIOR_WINNER = {"stacked_bars": 2, "enable_ask_absorption": False}  # from run_variant_grid.py
BID_REPULSION_ONLY_BENCHMARK = {"total_return_pct": 1.71, "sharpe_ratio": 1.6}

NEW_SIGNAL_FLAGS = [
    "enable_liquidity_sweep",
    "enable_climax_exhaustion",
    "enable_delta_divergence",
    "enable_vwap_fade",
]


def run_one(universe, flags: dict) -> dict:
    cfg = TapeBacktestConfig(**flags)
    result = run_backtest(universe, cfg, ml_filter=_DISABLED_FILTER)
    return {
        "total_return_pct": round(result.total_return_pct, 2),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "win_rate_pct": round(result.win_rate, 1),
        "num_trades": result.num_trades,
        "max_drawdown_pct": round(result.max_drawdown_pct, 1),
    }


def fmt_row(label: str, m: dict) -> str:
    return (f"{label:55s} {m['total_return_pct']:9.2f} {m['sharpe_ratio']:8.3f} "
            f"{m['win_rate_pct']:7.1f} {m['num_trades']:7d} {m['max_drawdown_pct']:7.1f}")


HEADER = f"{'variant':55s} {'return%':>9s} {'sharpe':>8s} {'win%':>7s} {'trades':>7s} {'mdd%':>7s}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=20)
    args = parser.parse_args()

    if not UNIVERSE_CACHE.exists():
        print(f"No cached universe at {UNIVERSE_CACHE} — aborting.")
        return
    with UNIVERSE_CACHE.open("rb") as f:
        universe = pickle.load(f)
    print(f"Loaded cached universe: {len(universe)} symbols\n")

    # --- Step 1: each new signal solo (baseline absorption/repulsion off) ---
    print("=" * 100)
    print("  SOLO BACKTESTS  (each new signal alone, original ask/bid absorption disabled)")
    print("=" * 100)
    print(HEADER)
    print("-" * len(HEADER))

    solo_rows = []
    base_off = {"enable_ask_absorption": False, "enable_bid_repulsion": False}
    for flag in NEW_SIGNAL_FLAGS:
        flags = {**base_off, flag: True}
        m = run_one(universe, flags)
        label = flag.replace("enable_", "")
        print(fmt_row(label, m))
        solo_rows.append({"variant": label, "flags": flags, **m})

    # Reference points
    m_baseline = run_one(universe, {})
    print(fmt_row("baseline (ask+bid, stack1)", m_baseline))
    m_prior_winner = run_one(universe, PRIOR_WINNER)
    print(fmt_row("prior winner (stack2, bid-only)", m_prior_winner))

    # --- Step 2: grid search new-signal combos, with and without the prior winner's base ---
    print(f"\n{'='*100}")
    print("  GRID SEARCH  (combinations of new signals, ranked by Sharpe)")
    print(f"{'='*100}")
    print(HEADER)
    print("-" * len(HEADER))

    grid_rows = []
    bases = {
        "new-signals-only": base_off,
        "+prior-winner": PRIOR_WINNER,
    }
    for base_label, base_flags in bases.items():
        for combo in itertools.product([False, True], repeat=len(NEW_SIGNAL_FLAGS)):
            toggles = dict(zip(NEW_SIGNAL_FLAGS, combo))
            if not any(toggles.values()) and base_label == "new-signals-only":
                continue  # all-off + no-base == no trades at all, skip
            flags = {**base_flags, **toggles}
            on = [f.replace("enable_", "") for f, v in toggles.items() if v]
            label = f"{base_label}|{'+'.join(on) if on else 'none'}"
            m = run_one(universe, flags)
            grid_rows.append({"variant": label, "flags": flags, **m})

    grid_rows.sort(key=lambda r: (r["sharpe_ratio"] if r["sharpe_ratio"] == r["sharpe_ratio"] else -999), reverse=True)
    for r in grid_rows[: args.top]:
        print(fmt_row(r["variant"], r))

    eligible = [r for r in grid_rows if r["num_trades"] >= args.min_trades]
    print(f"\n--- Robust ranking (>= {args.min_trades} trades) ---")
    for r in eligible[:10]:
        print(fmt_row(r["variant"], r))

    print(f"\nBenchmark: bid_repulsion-only = {BID_REPULSION_ONLY_BENCHMARK['total_return_pct']}% / "
          f"Sharpe {BID_REPULSION_ONLY_BENCHMARK['sharpe_ratio']}")
    print(f"Prior winner (stacked_bars=2, bid-only) = {m_prior_winner['total_return_pct']}% / "
          f"Sharpe {m_prior_winner['sharpe_ratio']} on {m_prior_winner['num_trades']} trades")

    if eligible:
        winner = eligible[0]
        print(f"\nNew winner (robust, >= {args.min_trades} trades): {winner['variant']} -> "
              f"{winner['total_return_pct']}% / Sharpe {winner['sharpe_ratio']} on {winner['num_trades']} trades")
        print("Flags:", winner["flags"])


if __name__ == "__main__":
    main()
