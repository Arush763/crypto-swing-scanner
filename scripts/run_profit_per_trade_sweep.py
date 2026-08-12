"""
Search for configurations whose GROSS profit per trade clears the cost floor.

The measured problem (scripts/run_timeframe_sweep.py): at 15m the strategy
earns +0.045% gross per trade against a 0.19-0.22% round-trip cost. No
execution tuning closes a gap that size. The only fix is a bigger move per
trade, so this script optimises for exactly that and ignores total return.

That distinction matters. Total return rewards *many* small-edge trades,
which is precisely the profile fees destroy — a config that takes 2,000
trades at +0.05% gross looks excellent on total return and loses money in
reality. The objective here is gross P&L per trade, with the cost floor drawn
on the chart so it is impossible to report a "winner" that cannot pay its own
commission.

Two levers actually move profit per trade:

  1. Hold longer. A wider ATR trailing stop stops harvesting noise and lets a
     winner develop. Costs nothing extra — the round trip is the same price
     whether the trade lasts four bars or forty.
  2. Be more selective. Stricter setup thresholds trade less often but each
     trade starts from a better location.

Both reduce trade count, which is the point: fewer, larger trades is the only
shape of this strategy that survives fees.

Usage:
    python scripts/run_profit_per_trade_sweep.py --stage families
    python scripts/run_profit_per_trade_sweep.py --stage selectivity
    python scripts/run_profit_per_trade_sweep.py --stage refine
"""

from __future__ import annotations

import argparse
import itertools
import logging
import pickle
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger("sweep")
logger.setLevel(logging.INFO)

import numpy as np

from src.config.config import MAJOR_BASES, TapeBacktestConfig
from src.backtesting.engine import run_backtest
from src.data.trade_tape import TradeTapeFetcher
from src.modules.signal_filter import SignalFilter

CACHE_DIR = ROOT / "data" / "cache" / "sweep"
_DISABLED_FILTER = SignalFilter(model_path="__disabled_for_sweep__.joblib")

# Round-trip cost the gross edge has to beat, from src/execution/costs.py.
# Taker both legs on a major is ~0.21%; maker entry + taker exit ~0.19%.
COST_FLOOR_TAKER = 0.21
COST_FLOOR_MAKER = 0.19


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_bars(timeframe: str, days: int, n_symbols: int, workers: int = 16) -> dict:
    """Fetch (and cache) resampled bars. Cached so a grid isn't re-downloading."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"majors_{timeframe}_{days}d_{n_symbols}s.pkl"

    if cache.exists():
        with cache.open("rb") as fh:
            universe = pickle.load(fh)
        logger.info("Loaded %d symbols of %s bars from cache", len(universe), timeframe)
        return universe

    symbols = [f"{base}/USDT" for base in MAJOR_BASES[:n_symbols]]
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)

    logger.info("Fetching %s bars for %d symbols (%s -> %s)…",
                timeframe, len(symbols), start, end)
    bars = TradeTapeFetcher().fetch_many_bars(
        symbols, start, end, timeframe=timeframe, max_workers=workers,
    )
    universe = {s: b for s, b in bars.items() if not b.empty}

    with cache.open("wb") as fh:
        pickle.dump(universe, fh)
    logger.info("Cached %d symbols", len(universe))
    return universe


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(universe: dict, cfg: TapeBacktestConfig, label: str) -> dict:
    result = run_backtest(universe, cfg, ml_filter=_DISABLED_FILTER)
    completed = [t for t in result.trades if not t.is_open]

    if not completed:
        return {"label": label, "trades": 0, "gross": 0.0, "cost": 0.0,
                "net": 0.0, "win_rate": 0.0, "total_net": 0.0, "avg_bars": 0.0}

    gross = float(np.mean([t.gross_pnl_pct for t in completed]))
    cost = float(np.mean([t.cost_pct for t in completed]))

    return {
        "label": label,
        "trades": len(completed),
        "gross": gross,
        "cost": cost,
        "net": float(np.mean([t.pnl_pct for t in completed])),
        "win_rate": sum(1 for t in completed if t.pnl_pct > 0) / len(completed) * 100,
        "total_net": float(sum(t.pnl_pct for t in completed)),
        "avg_bars": float(np.mean([t.holding_bars for t in completed])),
    }


def report(rows: list, title: str, min_trades: int, top: int) -> None:
    eligible = [r for r in rows if r["trades"] >= min_trades]
    eligible.sort(key=lambda r: r["gross"], reverse=True)

    print()
    print("=" * 104)
    print(f"  {title}")
    print(f"  Ranked by GROSS profit/trade. Cost floor: {COST_FLOOR_TAKER:.2f}% taker, "
          f"{COST_FLOOR_MAKER:.2f}% maker.")
    print("=" * 104)
    print(f"{'config':<44}{'trades':>7}{'gross/tr':>11}{'net/tr':>10}"
          f"{'WR':>7}{'held':>7}{'verdict':>16}")
    print("-" * 104)

    for r in eligible[:top]:
        if r["gross"] > COST_FLOOR_TAKER * 1.5:
            verdict = "CLEARS TAKER"
        elif r["gross"] > COST_FLOOR_MAKER * 1.5:
            verdict = "clears maker"
        elif r["gross"] > COST_FLOOR_MAKER:
            verdict = "marginal"
        else:
            verdict = "below cost"
        print(f"{r['label']:<44}{r['trades']:>7}{r['gross']:>10.3f}%"
              f"{r['net']:>9.3f}%{r['win_rate']:>6.1f}%{r['avg_bars']:>7.1f}"
              f"{verdict:>16}")

    print("=" * 104)
    skipped = len(rows) - len(eligible)
    if skipped:
        print(f"  ({skipped} configs excluded for fewer than {min_trades} trades)")
    print()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

SIGNAL_FAMILIES = {
    "absorption+repulsion": {},
    "repulsion_only": {"enable_ask_absorption": False},
    "absorption_only": {"enable_bid_repulsion": False},
    "liquidity_sweep": {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                        "enable_liquidity_sweep": True},
    "climax_exhaustion": {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                          "enable_climax_exhaustion": True},
    "vwap_fade": {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                  "enable_vwap_fade": True},
    "momentum_breakout": {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                          "enable_momentum_breakout": True},
    "delta_divergence": {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                         "enable_delta_divergence": True},
}

# The hold-length lever. 2.0 is the incumbent and harvests noise on a fast bar.
ATR_MULTS = [2.0, 4.0, 6.0, 10.0]


def base_cfg(timeframe: str) -> TapeBacktestConfig:
    return TapeBacktestConfig(
        timeframe=timeframe,
        apply_costs=True,
        btc_regime_filter=False,
        entry_is_maker=False,
        exit_is_maker=False,
    )


def stage_families(universe: dict, timeframe: str) -> list:
    """Which signal family, at which hold length, produces the biggest move?"""
    rows = []
    for name, flags in SIGNAL_FAMILIES.items():
        for atr_mult in ATR_MULTS:
            cfg = replace(base_cfg(timeframe), atr_trailing_stop_mult=atr_mult, **flags)
            rows.append(evaluate(universe, cfg, f"{name} atr={atr_mult:g}"))
            logger.info("  %-30s atr=%-5g -> %d trades, %+.3f%% gross",
                        name, atr_mult, rows[-1]["trades"], rows[-1]["gross"])
    return rows


def stage_selectivity(universe: dict, timeframe: str, family: str, atr_mult: float) -> list:
    """Given the best family, does trading less often earn more per trade?"""
    flags = SIGNAL_FAMILIES[family]
    rows = []
    grid = {
        "min_bonus_score": [0.0, 10.0, 15.0, 20.0],
        "volume_spike_mult": [2.0, 3.0, 4.0],
    }
    for bonus, vol in itertools.product(*grid.values()):
        cfg = replace(
            base_cfg(timeframe),
            atr_trailing_stop_mult=atr_mult,
            min_bonus_score=bonus,
            volume_spike_mult=vol,
            **flags,
        )
        label = f"{family} bonus>={bonus:g} vol>={vol:g}x"
        rows.append(evaluate(universe, cfg, label))
        logger.info("  %-44s -> %d trades, %+.3f%% gross",
                    label, rows[-1]["trades"], rows[-1]["gross"])
    return rows


def stage_refine(universe: dict, timeframe: str, family: str) -> list:
    """Push the hold length further and add a loss cap / cooldown."""
    flags = SIGNAL_FAMILIES[family]
    rows = []
    grid = {
        "atr_trailing_stop_mult": [6.0, 10.0, 15.0, 20.0],
        "cooldown_bars_after_loss": [0, 10],
        "max_single_trade_loss_pct": [None, 3.0],
    }
    for atr_mult, cooldown, loss_cap in itertools.product(*grid.values()):
        cfg = replace(
            base_cfg(timeframe),
            atr_trailing_stop_mult=atr_mult,
            cooldown_bars_after_loss=cooldown,
            max_single_trade_loss_pct=loss_cap,
            **flags,
        )
        cap = "none" if loss_cap is None else f"{loss_cap:g}%"
        label = f"atr={atr_mult:g} cool={cooldown} cap={cap}"
        rows.append(evaluate(universe, cfg, label))
        logger.info("  %-44s -> %d trades, %+.3f%% gross",
                    label, rows[-1]["trades"], rows[-1]["gross"])
    return rows


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Maximise gross profit per trade")
    p.add_argument("--stage", default="families",
                   choices=["families", "selectivity", "refine"])
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--symbols", type=int, default=5)
    p.add_argument("--family", default="repulsion_only",
                   help="Family to hold fixed in the selectivity/refine stages")
    p.add_argument("--atr-mult", type=float, default=6.0)
    p.add_argument("--min-trades", type=int, default=25)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    universe = load_bars(args.timeframe, args.days, args.symbols)
    if not universe:
        print("No data available — aborting.")
        return

    total_bars = sum(len(b) for b in universe.values())
    logger.info("Universe: %d symbols, %d bars total, timeframe=%s",
                len(universe), total_bars, args.timeframe)

    if args.stage == "families":
        rows = stage_families(universe, args.timeframe)
        title = f"SIGNAL FAMILY x HOLD LENGTH  ({args.timeframe}, {args.days}d)"
    elif args.stage == "selectivity":
        rows = stage_selectivity(universe, args.timeframe, args.family, args.atr_mult)
        title = f"SELECTIVITY  ({args.family}, atr={args.atr_mult:g}, {args.timeframe})"
    else:
        rows = stage_refine(universe, args.timeframe, args.family)
        title = f"REFINEMENT  ({args.family}, {args.timeframe}, {args.days}d)"

    report(rows, title, args.min_trades, args.top)


if __name__ == "__main__":
    main()
