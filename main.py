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

Run the tape-signal backtest against free historical Binance tick data:
    python main.py backtest --symbols BTC/USDT ETH/USDT --days 60

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
         "liquidity_score", "smart_money_score", "is_wall_signal", "wall_event"]
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
    from datetime import date, timedelta
    from src.config.config import TapeBacktestConfig
    from src.data.fetcher import MarketDataFetcher
    from src.data.trade_tape import TradeTapeFetcher, dedupe_by_binance_symbol
    from src.backtesting.engine import run_backtest

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days)

    if args.all_universe:
        logger.info("Resolving live universe symbol list…")
        raw_symbols = MarketDataFetcher().list_universe_symbols()
        symbols = dedupe_by_binance_symbol(raw_symbols)
        logger.info("%d universe symbols -> %d unique Binance tickers", len(raw_symbols), len(symbols))
    else:
        symbols = args.symbols

    fetcher = TradeTapeFetcher()
    logger.info(
        "Fetching %d days of tick data for %d symbols (concurrent)…",
        args.days, len(symbols),
    )
    bars_by_symbol = fetcher.fetch_many_bars(
        symbols, start, end, timeframe=args.timeframe, max_workers=args.workers,
    )

    universe = {}
    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            logger.warning("No tick data available for %s — skipping", symbol)
            continue
        universe[symbol] = bars

    if not universe:
        print("No tick data fetched for any symbol — aborting.")
        return

    cfg = TapeBacktestConfig(timeframe=args.timeframe)
    result = run_backtest(universe, cfg)

    print(f"\n{'='*50}")
    print(f"  TAPE BACKTEST RESULTS  ({start} to {end})")
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

    if not result.per_symbol_stats.empty:
        print(result.per_symbol_stats.to_string(index=False))


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
    p_bt = sub.add_parser("backtest", help="Run the tape-signal backtest on free historical tick data")
    p_bt.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    p_bt.add_argument("--all-universe", action="store_true",
                      help="Backtest the full live-scan universe (~100 symbols) instead of --symbols")
    p_bt.add_argument("--days", type=int, default=60,
                      help="Trailing days of tick data to fetch (default: 60)")
    p_bt.add_argument("--timeframe", default="4h")
    p_bt.add_argument("--workers", type=int, default=16,
                      help="Concurrent download threads (default: 16)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "scan": cmd_scan,
        "dashboard": cmd_dashboard,
        "backtest": cmd_backtest,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
