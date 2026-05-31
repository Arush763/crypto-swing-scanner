"""
Run full backtest + Monte Carlo edge verification on cached token data.

Usage:
    python scripts/run_backtest.py [--min-history 60] [--skip-mc]
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.backtesting.engine import run_backtest, BacktestResult
from src.backtesting.monte_carlo import run_monte_carlo
from src.config.config import BacktestConfig

CACHE_DIR = Path("data/cache")


def load_universe(min_history: int) -> dict:
    # Auto-detect available timeframe
    for tf_suffix in ("_4h.parquet", "_1h.parquet", "_1d.parquet"):
        files = list(CACHE_DIR.glob(f"*{tf_suffix}"))
        if files:
            print(f"Using {tf_suffix.strip('_.parquet')} timeframe data.")
            break
    print(f"Found {len(files)} cached files.")

    # Filter out non-crypto pairs (EUR/USD, GBP/USD, stablecoins)
    SKIP_BASE = {"EUR", "GBP", "USDC", "EURC", "USD1", "XAUM"}

    universe = {}
    for f in files:
        try:
            df = pd.read_parquet(f)
            parts = f.stem.split("_")
            if len(parts) < 3:
                continue
            base  = parts[1]
            quote = parts[2] if len(parts) > 2 else "USDT"
            if base in SKIP_BASE:
                continue
            symbol = f"{base}/{quote}"
            # Always include BTC for the regime filter regardless of bar count
            if base != "BTC" and len(df) < min_history:
                continue
            universe[symbol] = df
        except Exception as e:
            print(f"  WARN: could not load {f.name}: {e}")

    print(f"Loaded {len(universe)} assets (min {min_history} bars).\n")
    return universe


def main(min_history: int, skip_mc: bool) -> None:
    universe = load_universe(min_history)
    if not universe:
        print("No cached data found. Run scripts/pull_data.py first.")
        return

    # Detect timeframe from cached filenames
    sample_files = list(CACHE_DIR.glob("*.parquet"))
    detected_tf = "4h"
    if sample_files:
        name = sample_files[0].stem
        for tf in ("4h", "1h", "15m", "5m", "1d"):
            if name.endswith(f"_{tf}"):
                detected_tf = tf
                break

    print(f"Timeframe: {detected_tf}  |  EMA periods: 20 / 50 / 200 bars")
    print(f"BTC regime filter: ON\n")

    cfg = BacktestConfig(
        timeframe=detected_tf,
        ema_short=20,
        ema_mid=50,
        ema_long=200,
        volume_multiplier=2.0,
        score_threshold=80.0,
        initial_capital=10_000.0,
        risk_per_trade_pct=0.02,
        commission_pct=0.001,
        btc_regime_filter=True,
    )

    # ── Backtest ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  RUNNING BACKTEST")
    print("=" * 60)
    result: BacktestResult = run_backtest(universe, cfg)
    m = result.metrics

    print(f"\n  Assets in universe : {len(universe)}")
    print(f"  Total trades       : {m['num_trades']}")
    print(f"  Win rate           : {m['win_rate_pct']:.1f}%")
    print(f"  Total return       : {m['total_return_pct']:.1f}%")
    print(f"  CAGR               : {m.get('cagr_pct', 0):.1f}%")
    print(f"  Sharpe ratio       : {m['sharpe_ratio']:.3f}")
    print(f"  Sortino ratio      : {m['sortino_ratio']:.3f}")
    print(f"  Calmar ratio       : {m.get('calmar_ratio', 0):.3f}")
    print(f"  Max drawdown       : {m['max_drawdown_pct']:.1f}%")
    print(f"  Max DD duration    : {m.get('max_drawdown_duration_bars', 0)} bars")
    print(f"  Profit factor      : {m['profit_factor']}")
    print(f"  Avg return/trade   : {m['expectancy_pct']:.2f}%")
    print(f"  Avg win            : {m['avg_win_pct']:.2f}%")
    print(f"  Avg loss           : {m['avg_loss_pct']:.2f}%")
    print(f"  Payoff ratio       : {m['payoff_ratio']}")
    print(f"  Max consec. losses : {m['max_consecutive_losses']}")
    print(f"  Recovery factor    : {m.get('recovery_factor', 0):.3f}")

    if not result.per_symbol_stats.empty:
        print(f"\n  Top 10 symbols by total P&L:")
        top10 = result.per_symbol_stats.head(10)
        for _, row in top10.iterrows():
            print(f"    {row['symbol']:<18} {row['total_pnl_pct']:>+8.1f}%  "
                  f"WR={row['win_rate']:.0f}%  trades={int(row['trades'])}")

    # ── Monte Carlo ───────────────────────────────────────────────────────
    if skip_mc:
        print("\n  [Monte Carlo skipped via --skip-mc]\n")
        return

    completed = [t for t in result.trades if not t.is_open]
    pnls = [t.pnl_pct for t in completed]

    if len(pnls) < 10:
        print(f"\n  Only {len(pnls)} completed trades — need >= 10 for Monte Carlo.")
        print("  Try lowering --min-history or running with more token data.\n")
        return

    print(f"\n{'=' * 60}")
    print("  RUNNING MONTE CARLO EDGE VERIFICATION")
    print(f"  ({len(pnls)} completed trades)\n")
    run_monte_carlo(pnls, universe, cfg, verbose=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-history", type=int, default=60)
    ap.add_argument("--skip-mc", action="store_true")
    args = ap.parse_args()
    main(args.min_history, args.skip_mc)
