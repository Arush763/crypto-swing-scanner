"""
Run the tape-signal backtest against free historical Binance trade-tick data.

Usage:
    python scripts/run_tape_backtest.py --symbols BTC/USDT ETH/USDT SOL/USDT --days 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

from src.config.config import TapeBacktestConfig, SCAN_TIMEFRAME
from src.data.trade_tape import TradeTapeFetcher, resample_to_bars
from src.backtesting.engine import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Tape-signal backtest")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    parser.add_argument("--days", type=int, default=60,
                        help="Trailing days of tick data to fetch (need enough bars to clear the 200-bar EMA warm-up)")
    parser.add_argument("--timeframe", default=SCAN_TIMEFRAME)
    args = parser.parse_args()

    end = date.today() - timedelta(days=1)   # yesterday — today's dump may not exist yet
    start = end - timedelta(days=args.days)

    fetcher = TradeTapeFetcher()
    universe = {}
    for symbol in args.symbols:
        logger.info("Fetching %d days of tick data for %s…", args.days, symbol)
        trades = fetcher.fetch_range(symbol, start, end)
        if trades.empty:
            logger.warning("No tick data available for %s — skipping", symbol)
            continue
        universe[symbol] = resample_to_bars(trades, timeframe=args.timeframe)

    if not universe:
        print("No data fetched for any symbol — aborting.")
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


if __name__ == "__main__":
    main()
