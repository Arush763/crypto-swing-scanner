"""
Robustness check on whatever the combo sweep picked.

A single train/test split can be passed by luck, especially when the winner
was chosen from ~1,700 candidates. This applies the checks that would have
caught both of this project's earlier false positives, before any of them
reached a recommendation:

  quarters      Does the edge appear in every quarter of the year, or only in
                one favourable regime? The 45-day sweep winner failed exactly
                here — it lived entirely in one window.
  symbols       Is it spread across the universe, or one ticker's good year?
  tail trim     Does removing the best few trades flip it? The 200-day gate
                result failed this on its net series.
  significance  t-stat and CI on the test trades alone.

A config that passes all four is worth paper-trading. A config that fails any
of them is the same shape as the two results that already didn't survive.

Usage:
    python scripts/verify_combo_winner.py                    # verifies sweep's top pick
    python scripts/verify_combo_winner.py --rank 2           # second-best on train
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR, format="%(levelname)s  %(message)s")
logger = logging.getLogger("verify")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from src.config.config import TapeBacktestConfig
from src.backtesting.engine import run_backtest
from src.modules.signal_filter import SignalFilter
from scripts.run_full_combo_sweep import SIGNAL_FAMILIES, label_of
from scripts.study_large_moves import load_bars

Z_95 = 1.96


def build_cfg(params: dict) -> TapeBacktestConfig:
    return replace(
        TapeBacktestConfig(),
        timeframe="15m", apply_costs=True, btc_regime_filter=False,
        atr_trailing_stop_mult=params["atr_trailing_stop_mult"],
        min_bonus_score=params["min_bonus_score"],
        volume_spike_mult=params["volume_spike_mult"],
        cooldown_bars_after_loss=params["cooldown_bars_after_loss"],
        max_single_trade_loss_pct=params["max_single_trade_loss_pct"],
        **SIGNAL_FAMILIES[params["family"]],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Robustness-check the sweep winner")
    p.add_argument("--results", default="data/cache/combo_sweep_results.pkl")
    p.add_argument("--rank", type=int, default=1, help="1 = best on train")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--min-test-trades", type=int, default=30)
    p.add_argument("--train-frac", type=float, default=0.7)
    args = p.parse_args()

    with open(args.results, "rb") as fh:
        results = pickle.load(fh)

    usable = [r for r in results if r.get("n_test", 0) >= args.min_test_trades]
    usable.sort(key=lambda r: r["net_train"], reverse=True)
    if args.rank > len(usable):
        print(f"Only {len(usable)} usable combos.")
        return
    chosen = usable[args.rank - 1]

    universe = load_bars("15m", args.days, args.symbols)
    result = run_backtest(
        universe, build_cfg(chosen["params"]),
        ml_filter=SignalFilter(model_path="__disabled_for_verify_winner__.joblib"),
    )
    completed = sorted(
        (t for t in result.trades if not t.is_open), key=lambda t: t.entry_time,
    )
    split = int(len(completed) * args.train_frac)
    test = completed[split:]

    net = np.array([t.pnl_pct for t in test])
    gross = np.array([t.gross_pnl_pct for t in test])
    sd = float(net.std(ddof=1))
    se = sd / np.sqrt(len(net))

    print()
    print("=" * 100)
    print(f"  ROBUSTNESS CHECK — rank {args.rank} on train")
    print(f"  {chosen['label']}")
    print("=" * 100)
    print(f"  Out-of-sample: {len(net)} trades")
    print(f"    net   {net.mean():+.3f}%/trade   t={net.mean()/se:.2f}   "
          f"CI [{net.mean()-Z_95*se:+.3f}%, {net.mean()+Z_95*se:+.3f}%]")
    print(f"    gross {gross.mean():+.3f}%/trade")

    # -- Quarterly consistency -------------------------------------------
    df = pd.DataFrame({
        "t": [t.entry_time for t in completed],
        "net": [t.pnl_pct for t in completed],
        "symbol": [t.symbol for t in completed],
    })
    df["quarter"] = pd.to_datetime(df["t"]).dt.to_period("Q")

    print()
    print("  By quarter (whole year, not just test) — an edge in one quarter only is a regime:")
    print(f"    {'quarter':<10}{'trades':>8}{'net/tr':>10}{'total':>10}")
    print("    " + "-" * 38)
    q_pos = 0
    quarters = df.groupby("quarter")["net"].agg(["size", "mean", "sum"])
    for q, row in quarters.iterrows():
        if row["mean"] > 0:
            q_pos += 1
        print(f"    {str(q):<10}{int(row['size']):>8}{row['mean']:>9.3f}%{row['sum']:>9.1f}%")
    print(f"    -> positive in {q_pos}/{len(quarters)} quarters")

    # -- Symbol spread ----------------------------------------------------
    per_sym = df.groupby("symbol")["net"].agg(["size", "mean"])
    sym_pos = int((per_sym["mean"] > 0).sum())
    print()
    print(f"  Symbols with positive mean: {sym_pos}/{len(per_sym)}")
    best_sym = per_sym["mean"].idxmax()
    print(f"    best {best_sym} {per_sym.loc[best_sym,'mean']:+.3f}%  "
          f"worst {per_sym['mean'].idxmin()} {per_sym['mean'].min():+.3f}%")

    # -- Tail trim --------------------------------------------------------
    ordered = np.sort(net)
    print()
    print("  Tail dependence (test trades):")
    for k in (1, 3, 5):
        if len(ordered) > k:
            m = float(ordered[:-k].mean())
            flag = "" if m > 0 else "   <-- flips negative"
            print(f"    excl. top-{k}: {m:+.3f}%/trade{flag}")

    # -- Verdict ----------------------------------------------------------
    checks = {
        "positive out-of-sample": net.mean() > 0,
        "significant (t>=2)": net.mean() / se >= 2.0 if se else False,
        "positive in >=3 quarters": q_pos >= 3,
        "positive on >=half of symbols": sym_pos >= len(per_sym) / 2,
        "survives top-3 trim": float(np.sort(net)[:-3].mean()) > 0 if len(net) > 3 else False,
    }
    print()
    print("=" * 100)
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    print("=" * 100)
    print()
    if all(checks.values()):
        print("  All checks pass — worth paper-trading with small size.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"  {len(failed)} check(s) failed: {', '.join(failed)}")
        print("  Same shape as the earlier results that did not survive. Do not size up.")
    print()


if __name__ == "__main__":
    main()
