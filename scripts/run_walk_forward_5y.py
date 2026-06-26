"""
Run the full 5-year expanding-universe walk-forward backtest: trade week by
week starting from the earliest available history, retraining the ML signal
filter from scratch on all cumulative labelled setups after every week, then
print final performance and the loss-diagnostics report.

Usage: python scripts/run_walk_forward_5y.py
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import TapeBacktestConfig, MIN_DAILY_VOLUME_USD
from src.backtesting.walk_forward import run_walk_forward
from src.backtesting.loss_report import generate_loss_report

UNIVERSE_PATH = Path("data/walk_forward_universe_5y.pkl")
REPORT_PATH = Path("data/walk_forward_5y_report.txt")

# This strategy's avg_win/avg_loss profile implies a breakeven win rate of
# roughly avg_loss / (avg_win + avg_loss) ~= 27% -- comfortably below the
# universe's own 38% average. A symbol sitting clearly below breakeven
# (not just below the strategy's average) over a real sample of its own
# trades is more likely structurally bad for this signal than unlucky.
BLACKLIST_WIN_RATE_THRESHOLD = 0.27
BLACKLIST_MIN_SAMPLES = 15

# Reuse the live scanner's own liquidity floor (src.config.config) rather
# than hand-tuning a new number against this backtest -- several of the
# worst losers (ZEC, TAO, JTO, WLD, BICO, AVAX, DOT) sit below it for large
# stretches of their history.
MIN_DOLLAR_VOLUME = MIN_DAILY_VOLUME_USD


def main() -> None:
    with open(UNIVERSE_PATH, "rb") as f:
        universe = pickle.load(f)

    print(f"Loaded {len(universe)} symbols from {UNIVERSE_PATH}")
    for symbol, bars in sorted(universe.items()):
        print(f"  {symbol:15s} {len(bars):6d} bars  {bars.index.min()} -> {bars.index.max()}")

    cfg = TapeBacktestConfig()
    t0 = time.time()
    result = run_walk_forward(
        universe, cfg,
        blacklist_win_rate_threshold=BLACKLIST_WIN_RATE_THRESHOLD,
        blacklist_min_samples=BLACKLIST_MIN_SAMPLES,
        min_dollar_volume=MIN_DOLLAR_VOLUME,
        confidence_sizing=True,
    )
    elapsed = time.time() - t0

    print()
    print(f"Walk-forward complete in {elapsed:.1f}s — {len(result.weekly_log)} weeks simulated, "
          f"{len([t for t in result.trades if not t.is_open])} completed trades.")
    print()
    print("FINAL METRICS")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

    report = generate_loss_report(result)
    print()
    print(report)

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
