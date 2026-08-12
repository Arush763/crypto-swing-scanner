"""
Continuous trading loop — scan, execute, monitor, repeat.

Replaces the 4h GitHub Actions cron for high-frequency operation. The cron
model cannot work here for a structural reason: the wall signal is defined by
what changes in the order book *between* observations, so a 4h sampling
interval means nearly every absorption and repulsion event is born and dies
unseen. Running as a persistent process with a short interval is what lets
the signal be observed at the timescale it actually occurs on.

Modes
-----
    python scripts/run_trader.py                    # paper, 60s interval
    python scripts/run_trader.py --interval 30      # faster loop
    python scripts/run_trader.py --dry-run          # scan and alert only, never trade
    python scripts/run_trader.py --live             # requires LIVE_TRADING_ENABLED + keys

Live trading additionally requires the environment to carry credentials; the
--live flag alone will not send an order. See src/execution/executor.py.

Stopping
--------
Ctrl-C shuts down cleanly, leaving open positions open and reporting them.
It does NOT flatten — killing the process should not be a trading decision.
Use --flatten-on-exit if you want the opposite.
"""

from __future__ import annotations

import argparse
import logging
import signal as signal_module
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trader.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("trader")

from src.config.config import (
    EXECUTION_VENUE,
    HFT_OHLCV_LIMIT,
    HFT_SCAN_INTERVAL_SECONDS,
    HFT_SCAN_TIMEFRAME,
    HFT_SCORE_THRESHOLD,
    LIVE_TRADING_ENABLED,
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_LOSS_USD,
    MAX_DAILY_TRADES,
    MAX_POSITION_USD,
    RISK_PER_TRADE_PCT,
)
from src.execution.executor import Executor
from src.execution.monitor import Mark, PositionMonitor
from src.execution.position import PositionStore
from src.execution.risk import RiskLimits, RiskManager
from src.notifications.telegram import TelegramNotifier
from src.scanner import Scanner


class TradingLoop:
    def __init__(self, args) -> None:
        self.args = args
        self.running = True
        self.cycle = 0
        self.last_summary_date = ""

        self.notifier = TelegramNotifier()
        self.store = PositionStore()
        self.risk = RiskManager(
            limits=RiskLimits(
                max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
                max_position_usd=MAX_POSITION_USD,
                max_daily_loss_usd=MAX_DAILY_LOSS_USD,
                max_daily_trades=MAX_DAILY_TRADES,
                risk_per_trade_pct=RISK_PER_TRADE_PCT,
            ),
        )

        self.executor = None
        if not args.dry_run:
            self.executor = Executor(
                store=self.store,
                risk=self.risk,
                notifier=self.notifier,
                venue=EXECUTION_VENUE,
                live=args.live and LIVE_TRADING_ENABLED,
            )

        self.monitor = PositionMonitor(
            store=self.store,
            notifier=self.notifier,
            exit_executor=self.executor.exit_fill if self.executor else None,
            venue=EXECUTION_VENUE,
        )

        self.scanner = Scanner(
            score_threshold=args.threshold,
            enable_orderbook=True,
            timeframe=args.timeframe,
            ohlcv_limit=HFT_OHLCV_LIMIT,
        )

        signal_module.signal(signal_module.SIGINT, self._handle_shutdown)
        signal_module.signal(signal_module.SIGTERM, self._handle_shutdown)

    # ------------------------------------------------------------------

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received — finishing current cycle")
        self.running = False

    def _mode_label(self) -> str:
        if self.args.dry_run:
            return "DRY-RUN (no trading)"
        if self.executor and self.executor.live:
            return "LIVE"
        return "PAPER"

    # ------------------------------------------------------------------

    def announce_start(self) -> None:
        mode = self._mode_label()
        logger.info("=" * 62)
        logger.info("  Trading loop starting — %s", mode)
        logger.info("  Venue:     %s", EXECUTION_VENUE)
        logger.info("  Interval:  %ds", self.args.interval)
        logger.info("  Timeframe: %s", self.args.timeframe)
        logger.info("  Threshold: %.1f", self.args.threshold)
        logger.info("  Limits:    %d concurrent | $%.0f/position | $%.0f daily loss",
                    MAX_CONCURRENT_POSITIONS, MAX_POSITION_USD, MAX_DAILY_LOSS_USD)
        logger.info("=" * 62)

        if self.notifier.enabled:
            self.notifier.send(
                f"🤖 <b>Trading bot started</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Mode:      <b>{mode}</b>\n"
                f"Venue:     <code>{EXECUTION_VENUE}</code>\n"
                f"Interval:  <code>{self.args.interval}s</code>  "
                f"Timeframe: <code>{self.args.timeframe}</code>\n"
                f"Max risk:  <code>${MAX_DAILY_LOSS_USD:,.0f}/day</code>, "
                f"<code>{MAX_CONCURRENT_POSITIONS}</code> positions\n"
            )

    # ------------------------------------------------------------------

    def marks_from_universe(self, universe: dict) -> dict:
        """
        Build marks for open positions from this cycle's OHLCV.

        Uses the last bar's high/low rather than only its close, so a stop
        touched between polls is still detected. A position whose symbol
        dropped out of the scanned universe gets no mark and is deliberately
        left alone rather than marked to a stale price.
        """
        marks = {}
        for symbol, ohlcv in universe.items():
            if ohlcv is None or ohlcv.empty:
                continue
            last = ohlcv.iloc[-1]
            marks[symbol] = Mark(
                last=float(last["close"]),
                low=float(last["low"]),
                high=float(last["high"]),
            )
        return marks

    # ------------------------------------------------------------------

    def run_cycle(self) -> None:
        self.cycle += 1
        started = time.time()

        result = self.scanner.run()

        # 1) Manage what's already open, before opening anything new. An open
        #    position that has hit its stop must be closed before this cycle
        #    considers adding more risk.
        marks = self.marks_from_universe(getattr(self.scanner, "last_universe", {}) or {})
        if marks:
            events = self.monitor.check(marks)
            for event in events:
                if event.closed:
                    halt = self.risk.record_close(event.pnl_usd)
                    if halt and self.notifier.enabled:
                        self.notifier.send_risk_halt("Daily loss limit", halt)

        # 2) Consider new entries.
        for sig in result.signals:
            if self.notifier.enabled:
                self.notifier.send_signal(sig)

            if self.args.dry_run or self.executor is None:
                continue

            mark = marks.get(sig.symbol)
            price = mark.last if mark else sig.current_price
            equity = self.executor.account_equity_usd(self.args.paper_equity)
            self.executor.open_position(sig, equity_usd=equity, mark_price=price)

        duration = time.time() - started
        logger.info(
            "Cycle %d — %d assets | %d signals | %d open | %.1fs",
            self.cycle, result.assets_scanned, len(result.signals),
            self.store.open_count, duration,
        )

        self._maybe_send_daily_summary()

    def _maybe_send_daily_summary(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        hour = datetime.now(timezone.utc).hour
        if hour == 23 and self.last_summary_date != today:
            self.last_summary_date = today
            if self.notifier.enabled:
                self.notifier.send_daily_summary(self.store.daily_stats(today))

    # ------------------------------------------------------------------

    def run(self) -> None:
        self.announce_start()

        while self.running:
            cycle_start = time.time()
            try:
                self.run_cycle()
            except Exception as exc:
                # A scan failure must not kill the loop — open positions still
                # need monitoring on the next pass.
                logger.exception("Cycle %d failed: %s", self.cycle, exc)

            elapsed = time.time() - cycle_start
            sleep_for = max(0.0, self.args.interval - elapsed)
            if elapsed > self.args.interval:
                logger.warning(
                    "Cycle took %.1fs, longer than the %ds interval — running continuously",
                    elapsed, self.args.interval,
                )
            deadline = time.time() + sleep_for
            while self.running and time.time() < deadline:
                time.sleep(min(1.0, deadline - time.time()))

        self.shutdown()

    def shutdown(self) -> None:
        open_positions = self.store.all_open()

        if self.args.flatten_on_exit and open_positions and self.executor:
            logger.info("Flattening %d open position(s) on exit", len(open_positions))
            universe = getattr(self.scanner, "last_universe", {}) or {}
            self.monitor.force_close_all(self.marks_from_universe(universe), "shutdown")
            open_positions = []

        self.store.save()
        self.risk.save()

        logger.info("Loop stopped after %d cycles — %d position(s) left open",
                    self.cycle, len(open_positions))

        if self.notifier.enabled:
            detail = ""
            if open_positions:
                detail = "\n⚠️ Still open: " + ", ".join(p.symbol for p in open_positions)
                detail += "\nThese are no longer being monitored."
            self.notifier.send(
                f"🛑 <b>Trading bot stopped</b>\n"
                f"Cycles: <code>{self.cycle}</code>{detail}"
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Continuous crypto trading loop")
    p.add_argument("--interval", type=int, default=HFT_SCAN_INTERVAL_SECONDS,
                   help=f"Seconds between scan cycles (default: {HFT_SCAN_INTERVAL_SECONDS})")
    p.add_argument("--timeframe", default=HFT_SCAN_TIMEFRAME,
                   help=f"Candle timeframe (default: {HFT_SCAN_TIMEFRAME})")
    p.add_argument("--threshold", type=float, default=HFT_SCORE_THRESHOLD,
                   help=f"Composite score gate (default: {HFT_SCORE_THRESHOLD})")
    p.add_argument("--paper-equity", type=float, default=10_000.0,
                   help="Simulated account equity for paper sizing (default: 10000)")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and send alerts, but never open a position")
    p.add_argument("--live", action="store_true",
                   help="Route real orders. Also requires LIVE_TRADING_ENABLED and API keys.")
    p.add_argument("--flatten-on-exit", action="store_true",
                   help="Close all open positions on shutdown (default: leave them open)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.live and not LIVE_TRADING_ENABLED:
        logger.error(
            "--live was passed but LIVE_TRADING_ENABLED is False in config.py. "
            "Running in PAPER mode. Set it deliberately, after reviewing paper results."
        )

    TradingLoop(args).run()


if __name__ == "__main__":
    main()
