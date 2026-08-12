"""
Exhaustive combo sweep — every parameter set, ranked, with the winner held to
an out-of-sample test.

Read this before trusting any row it prints
-------------------------------------------
Searching a large grid and reporting the best result is how this project has
already produced two false positives:

  * a 45-day sweep whose winner showed +1.958%/trade and went to -0.191% on a
    wider sample;
  * a gate filter that looked net-positive at 90 and 200 days (+0.059%,
    +0.125%) and came out at -0.099% on a full year.

The maximum of N noisy estimates is biased upward by roughly the spread of
those estimates times a factor growing with N. With ~1,700 combos that bias
is large enough to manufacture an impressive-looking winner out of pure
noise, every time, on any dataset. So the sweep is run in a way that makes
the bias visible instead of hiding it:

  1. Selection uses TRAIN only. The winner's TEST number is reported once,
     after selection, and is never used to choose.
  2. The full distribution of test results is printed, not just the top. If
     roughly half the combos are positive out-of-sample, the "winner" is a
     coin-flip that landed well.
  3. A permutation-style benchmark: how good would the best-of-N look if
     every combo had zero true edge? Any winner inside that band is noise.
  4. The winner's t-statistic on its own test trades, since a large mean on
     few trades is exactly what the two prior false positives looked like.

The honest output of a sweep like this is usually "nothing survives", and the
script is written so that answer is legible rather than buried.

Usage:
    python scripts/run_full_combo_sweep.py                 # full grid
    python scripts/run_full_combo_sweep.py --quick         # small grid, for a smoke test
    python scripts/run_full_combo_sweep.py --workers 12
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR, format="%(levelname)s  %(message)s")
logger = logging.getLogger("combo")
logger.setLevel(logging.INFO)

import numpy as np

from src.config.config import TapeBacktestConfig
from src.backtesting.engine import run_backtest
from src.modules.signal_filter import SignalFilter
from scripts.study_large_moves import load_bars

Z_95 = 1.96

# Worker-global so the (large) universe is loaded once per process rather
# than pickled into every task.
_UNIVERSE = None
_FILTER = None
_TIMEFRAME = "15m"


def _init_worker(timeframe: str, days: int, symbols: int, resample: str = ""):
    global _UNIVERSE, _FILTER, _TIMEFRAME
    logging.getLogger().setLevel(logging.ERROR)
    universe = load_bars(timeframe, days, symbols)
    if resample:
        from scripts.scope_opportunity import resample_bars
        universe = {s: resample_bars(b, resample) for s, b in universe.items()}
        universe = {s: b for s, b in universe.items() if len(b) >= 300}
    _UNIVERSE = universe
    _TIMEFRAME = resample or timeframe
    _FILTER = SignalFilter(model_path="__disabled_for_combo_sweep__.joblib")


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

SIGNAL_FAMILIES = {
    "absorb+repuls": {},
    "repulsion":     {"enable_ask_absorption": False},
    "absorption":    {"enable_bid_repulsion": False},
    "sweep":         {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                      "enable_liquidity_sweep": True},
    "climax":        {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                      "enable_climax_exhaustion": True},
    "vwap_fade":     {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                      "enable_vwap_fade": True},
    "momentum":      {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                      "enable_momentum_breakout": True},
    "delta_div":     {"enable_ask_absorption": False, "enable_bid_repulsion": False,
                      "enable_delta_divergence": True},
}

FULL_GRID = {
    "family": list(SIGNAL_FAMILIES),
    "atr_trailing_stop_mult": [1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    "min_bonus_score": [0.0, 10.0, 20.0],
    "volume_spike_mult": [1.5, 2.0, 3.0],
    "cooldown_bars_after_loss": [0, 20],
    "max_single_trade_loss_pct": [None, 3.0],
}

QUICK_GRID = {
    "family": ["absorb+repuls", "repulsion", "climax", "delta_div"],
    "atr_trailing_stop_mult": [2.0, 4.0],
    "min_bonus_score": [0.0],
    "volume_spike_mult": [2.0],
    "cooldown_bars_after_loss": [0],
    "max_single_trade_loss_pct": [None],
}


def iter_combos(grid: dict):
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def label_of(params: dict) -> str:
    cap = "none" if params["max_single_trade_loss_pct"] is None else f"{params['max_single_trade_loss_pct']:g}%"
    return (
        f"{params['family']:<14} atr={params['atr_trailing_stop_mult']:<4g} "
        f"bonus={params['min_bonus_score']:<4g} vol={params['volume_spike_mult']:<4g} "
        f"cool={params['cooldown_bars_after_loss']:<3d} cap={cap}"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(params: dict, train_frac: float = 0.7) -> dict:
    """Run one combo; split its trades by time into train and test."""
    flags = SIGNAL_FAMILIES[params["family"]]
    cfg = replace(
        TapeBacktestConfig(),
        timeframe=_TIMEFRAME,
        apply_costs=True,
        btc_regime_filter=False,
        atr_trailing_stop_mult=params["atr_trailing_stop_mult"],
        min_bonus_score=params["min_bonus_score"],
        volume_spike_mult=params["volume_spike_mult"],
        cooldown_bars_after_loss=params["cooldown_bars_after_loss"],
        max_single_trade_loss_pct=params["max_single_trade_loss_pct"],
        **flags,
    )

    try:
        result = run_backtest(_UNIVERSE, cfg, ml_filter=_FILTER)
    except Exception as exc:      # a combo that errors must not kill the sweep
        return {"label": label_of(params), "error": str(exc), "n_train": 0, "n_test": 0}

    completed = sorted(
        (t for t in result.trades if not t.is_open), key=lambda t: t.entry_time,
    )
    if len(completed) < 20:
        return {"label": label_of(params), "n_train": 0, "n_test": 0}

    split = int(len(completed) * train_frac)
    train, test = completed[:split], completed[split:]

    def block(trades):
        if not trades:
            return 0, 0.0, 0.0, 0.0
        net = np.array([t.pnl_pct for t in trades])
        gross = np.array([t.gross_pnl_pct for t in trades])
        return len(net), float(net.mean()), float(gross.mean()), float(net.std(ddof=1)) if len(net) > 1 else 0.0

    n_tr, net_tr, gross_tr, _ = block(train)
    n_te, net_te, gross_te, sd_te = block(test)
    se_te = sd_te / np.sqrt(n_te) if n_te > 1 and sd_te else float("inf")

    return {
        "label": label_of(params),
        "params": params,
        "n_train": n_tr, "net_train": net_tr, "gross_train": gross_tr,
        "n_test": n_te, "net_test": net_te, "gross_test": gross_te,
        "sd_test": sd_te, "t_test": net_te / se_te if np.isfinite(se_te) and se_te else 0.0,
        "total_test": net_te * n_te,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def best_of_n_noise_band(results: list, n_combos: int, trials: int = 2000) -> float:
    """
    If every combo had zero true edge, how good would the best-of-N look?

    Each combo's test mean has a standard error we can estimate from its own
    trades. Drawing N zero-mean values with those standard errors and taking
    the max, repeatedly, gives the distribution of "best result achievable by
    luck alone". A winner inside this band is not evidence of anything.
    """
    ses = [
        r["sd_test"] / np.sqrt(r["n_test"])
        for r in results
        if r.get("n_test", 0) > 1 and r.get("sd_test", 0) > 0
    ]
    if not ses:
        return float("nan")

    ses = np.array(ses)
    rng = np.random.default_rng(0)
    maxima = np.empty(trials)
    for i in range(trials):
        draws = rng.normal(0.0, ses)
        maxima[i] = draws.max()
    return float(np.percentile(maxima, 95))


def main() -> None:
    p = argparse.ArgumentParser(description="Exhaustive combo sweep with honest selection")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--quick", action="store_true", help="Small grid, for a smoke test")
    p.add_argument("--resample", default="",
                   help="Aggregate the 15m base bars up to this rule (e.g. 4h, 1D). "
                        "The 15m-only sweep found nothing; coarser bars offer much larger "
                        "moves against the same fixed fee.")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--min-test-trades", type=int, default=30)
    p.add_argument("--out", default="data/cache/combo_sweep_results.pkl")
    args = p.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID
    combos = list(iter_combos(grid))

    logger.info("Grid: %d combos across %d dimensions", len(combos), len(grid))
    logger.info("Workers: %d  |  data: %s %dd %d symbols",
                args.workers, args.timeframe, args.days, args.symbols)

    started = time.time()
    from concurrent.futures import ProcessPoolExecutor

    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.timeframe, args.days, args.symbols, args.resample),
    ) as pool:
        futures = {pool.submit(evaluate, c, args.train_frac): c for c in combos}
        done = 0
        for fut in futures:
            pass
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 100 == 0 or done == len(combos):
                rate = done / (time.time() - started)
                eta = (len(combos) - done) / rate if rate else 0
                logger.info("  %d/%d combos (%.1f/s, ETA %.0f min)",
                            done, len(combos), rate, eta / 60)

    elapsed = time.time() - started
    logger.info("Swept %d combos in %.1f min", len(combos), elapsed / 60)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(results, fh)

    usable = [r for r in results if r.get("n_test", 0) >= args.min_test_trades]
    if not usable:
        print("\nNo combo produced enough out-of-sample trades to judge.\n")
        return

    # ---- Selection happens on TRAIN, exactly once -----------------------
    usable.sort(key=lambda r: r["net_train"], reverse=True)
    winner = usable[0]

    print()
    print("=" * 112)
    print(f"  EXHAUSTIVE COMBO SWEEP — {len(combos)} combos, {args.days}d, "
          f"{args.symbols} symbols, {args.resample or args.timeframe}")
    print(f"  {len(usable)} combos had >= {args.min_test_trades} out-of-sample trades")
    print("=" * 112)

    print("\n  TOP 20 BY TRAIN (selection set) — with their test results alongside:\n")
    print(f"{'#':<4}{'config':<62}{'trainNet':>10}{'testNet':>10}{'testN':>7}{'testT':>8}")
    print("-" * 112)
    for i, r in enumerate(usable[:args.top], 1):
        print(f"{i:<4}{r['label']:<62}{r['net_train']:>9.3f}%{r['net_test']:>9.3f}%"
              f"{r['n_test']:>7}{r['t_test']:>8.2f}")

    # ---- Distribution of test results ----------------------------------
    test_nets = np.array([r["net_test"] for r in usable])
    pos = int((test_nets > 0).sum())

    print()
    print("=" * 112)
    print("  IS THE WINNER REAL?")
    print("=" * 112)
    print(f"  Best on train:        {winner['label']}")
    print(f"    train net           {winner['net_train']:+.3f}%/trade  ({winner['n_train']} trades)")
    print(f"    TEST net            {winner['net_test']:+.3f}%/trade  ({winner['n_test']} trades)")
    print(f"    TEST gross          {winner['gross_test']:+.3f}%/trade")
    print(f"    TEST t-stat         {winner['t_test']:.2f}   (need ~2.0)")
    se = winner["sd_test"] / np.sqrt(winner["n_test"]) if winner["n_test"] else float("nan")
    print(f"    TEST 95% CI         [{winner['net_test'] - Z_95*se:+.3f}%, "
          f"{winner['net_test'] + Z_95*se:+.3f}%]")

    print()
    print(f"  Combos net-positive out-of-sample: {pos}/{len(usable)} ({pos/len(usable)*100:.0f}%)")
    print(f"    (roughly 50% is what pure noise produces)")
    print(f"  Best TEST net among all combos:    {test_nets.max():+.3f}%")
    print(f"  Median TEST net:                   {np.median(test_nets):+.3f}%")

    band = best_of_n_noise_band(usable, len(usable))
    print()
    print(f"  Best-of-{len(usable)} under a ZERO-edge null: {band:+.3f}%/trade (95th pct)")
    if np.isfinite(band):
        if test_nets.max() > band:
            print("  -> The best test result EXCEEDS what luck alone would produce. Worth a look.")
        else:
            print("  -> The best test result is INSIDE the luck band. No combo here beats chance.")

    print()
    if winner["net_test"] > 0 and winner["t_test"] >= 2.0:
        print("  VERDICT: train-selected winner is positive and significant out of sample.")
    elif winner["net_test"] > 0:
        print("  VERDICT: train-selected winner is positive out of sample but NOT significant.")
        print("           This is exactly how the 45-day and 200-day false positives looked.")
    else:
        print("  VERDICT: the train-selected winner is NEGATIVE out of sample.")
        print("           Selecting on train did not transfer — no combo in this grid works.")
    print()


if __name__ == "__main__":
    main()
