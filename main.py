"""
Entry point for the Crypto Swing Trading Scanner.

Usage
-----
Run a live scan and print the leaderboard:
    python main.py scan

Launch the Streamlit dashboard:
    python main.py dashboard
    -- or --
    streamlit run src/dashboard/app.py

Run the backtester on cached data:
    python main.py backtest

Run the parameter optimiser:
    python main.py optimise

Run unit tests:
    pytest tests/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scanner.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_scan(args) -> None:
    from src.scanner import Scanner

    logger.info("Starting live scan…")
    scanner = Scanner(
        min_volume=args.min_volume,
        score_threshold=args.threshold,
    )
    result = scanner.run()

    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE  |  {result.assets_scanned} assets  |  {result.duration_seconds}s")
    print(f"{'='*60}\n")

    if not result.signal_table.empty:
        print("🚨  SIGNALS:\n")
        print(result.signal_table.to_string(index=False))
    else:
        print("No signals above threshold this cycle.")

    print("\n🏆  TOP 10 OVERALL:\n")
    top = result.ranked_df.head(10)[
        ["symbol", "final_score", "trend_score", "momentum_score",
         "liquidity_score", "smart_money_score", "is_breakout", "is_retest", "is_squeeze"]
    ]
    print(top.to_string(index=True))


def cmd_dashboard(args) -> None:
    import subprocess, sys
    dashboard_path = str(Path(__file__).parent / "src" / "dashboard" / "app.py")
    logger.info("Launching dashboard at %s", dashboard_path)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path],
        check=True,
    )


def cmd_backtest(args) -> None:
    import pandas as pd
    from src.backtesting.engine import run_backtest, BacktestConfig
    from src.config.config import BacktestConfig as Cfg

    cache_dir = Path("data/cache")
    parquet_files = list(cache_dir.glob("*.parquet")) if cache_dir.exists() else []

    if not parquet_files:
        print("No cached data found. Run 'python main.py scan' first.")
        return

    universe = {}
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            symbol = f.stem.split("_", 1)[1].replace("_", "/")
            universe[symbol] = df
        except Exception:
            pass

    cfg = Cfg(
        ema_short=args.ema_short,
        ema_mid=args.ema_mid,
        volume_multiplier=args.vol_mult,
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk / 100,
    )

    logger.info("Running backtest on %d assets…", len(universe))
    result = run_backtest(universe, cfg)

    print(f"\n{'='*50}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"  Total Return:    {result.total_return_pct:.1f}%")
    print(f"  Win Rate:        {result.win_rate:.1f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.1f}%")
    print(f"  Profit Factor:   {result.profit_factor}")
    print(f"  Avg Return/Trade:{result.avg_return_pct:.2f}%")
    print(f"  Avg Hold (bars): {result.avg_holding_bars:.1f}")
    print(f"  Total Trades:    {result.num_trades}")
    print(f"{'='*50}\n")


def cmd_optimise(args) -> None:
    import pandas as pd
    from src.backtesting.engine import optimise_parameters

    cache_dir = Path("data/cache")
    parquet_files = list(cache_dir.glob("*.parquet")) if cache_dir.exists() else []

    if not parquet_files:
        print("No cached data. Run a scan first.")
        return

    universe = {}
    for f in parquet_files[:30]:  # limit for speed
        try:
            df = pd.read_parquet(f)
            symbol = f.stem.split("_", 1)[1].replace("_", "/")
            universe[symbol] = df
        except Exception:
            pass

    logger.info("Running grid search…")
    df = optimise_parameters(universe)
    print("\nTop 10 parameter combinations by Sharpe Ratio:\n")
    print(df.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crypto Swing Trading Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Run a live scan cycle")
    p_scan.add_argument("--min-volume", type=float, default=5_000_000,
                        help="Minimum 24h USD volume filter (default: 5M)")
    p_scan.add_argument("--threshold", type=float, default=80.0,
                        help="Minimum score to generate a signal (default: 80)")

    # dashboard
    sub.add_parser("dashboard", help="Launch the Streamlit dashboard")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run backtester on cached data")
    p_bt.add_argument("--ema-short", type=int, default=20)
    p_bt.add_argument("--ema-mid", type=int, default=50)
    p_bt.add_argument("--vol-mult", type=float, default=2.0)
    p_bt.add_argument("--capital", type=float, default=10_000.0)
    p_bt.add_argument("--risk", type=float, default=2.0,
                      help="Risk per trade in percent (default: 2)")

    # optimise
    sub.add_parser("optimise", help="Grid-search parameter optimisation")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "scan": cmd_scan,
        "dashboard": cmd_dashboard,
        "backtest": cmd_backtest,
        "optimise": cmd_optimise,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
