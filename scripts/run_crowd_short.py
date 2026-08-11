"""
Live runner for the crowd-short strategy.

Polls leveraged-positioning data, emits short signals when a symbol's retail
long/short ratio is at an extreme against its own history, and (optionally)
routes them through the executor. Also records every reading so that history
accumulates for venues whose archive is too short to validate against today.

Modes
-----
    python scripts/run_crowd_short.py --record-only    # accumulate history, no signals
    python scripts/run_crowd_short.py                  # signals + Telegram, no trading
    python scripts/run_crowd_short.py --paper          # simulated fills
    python scripts/run_crowd_short.py --live           # real orders (see below)

Deployment constraint — the thing to fix before this earns anything
-------------------------------------------------------------------
The strategy was validated on Binance positioning data across 12 majors.
Binance's live API returns 451 from this environment. OKX is reachable but
its ratio only tracks Binance's for BTC, ETH and SOL, and those three alone
do NOT carry the edge (58 out-of-sample trades, t=1.14, tail trim collapses
it to +0.008%).

So on this machine the runner will correctly refuse to trade most symbols.
Running it from a host where Binance's API is reachable — a VPS outside the
restricted region — makes the validated signal deploy as tested, and is the
single change that turns this from a refusal into a strategy. Until then
--record-only is the useful mode: it accumulates OKX history so the signal
can eventually be re-validated on OKX's own data rather than assumed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/crowd_short.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("crowd")

from src.config.config import (
    MAJOR_BASES,
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_LOSS_USD,
    MAX_DAILY_TRADES,
    MAX_POSITION_USD,
    RISK_PER_TRADE_PCT,
)
from src.data.positioning import PositioningFetcher
from src.modules.crowd_signal import CrowdShortSignal, HOLD_HOURS
from src.notifications.telegram import TelegramNotifier


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crowd-short strategy runner")
    p.add_argument("--source", default="okx", choices=["okx", "binance"],
                   help="Positioning data source (binance requires an unblocked host)")
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--interval", type=int, default=3600,
                   help="Seconds between polls (default: 1h, matching the signal's bar)")
    p.add_argument("--hold-hours", type=int, default=HOLD_HOURS)
    p.add_argument("--record-only", action="store_true",
                   help="Accumulate positioning history without generating signals")
    p.add_argument("--paper", action="store_true", help="Simulated fills")
    p.add_argument("--live", action="store_true", help="Route real orders")
    p.add_argument("--allow-unvalidated", action="store_true",
                   help="Trade symbols whose source was not validated. Off by default "
                        "because it means trading a different signal than the tested one.")
    p.add_argument("--once", action="store_true", help="Single pass, then exit")
    return p


def main() -> None:
    args = build_parser().parse_args()

    symbols = [f"{b}/USDT" for b in MAJOR_BASES[:args.symbols]]
    fetcher = PositioningFetcher()
    notifier = TelegramNotifier()

    signal = CrowdShortSignal(
        source=args.source,
        hold_hours=args.hold_hours,
        require_validated_source=not args.allow_unvalidated,
    )

    mode = ("RECORD-ONLY" if args.record_only
            else "LIVE" if args.live
            else "PAPER" if args.paper
            else "SIGNALS-ONLY")

    logger.info("=" * 62)
    logger.info("  Crowd-short runner — %s", mode)
    logger.info("  Source:    %s", args.source)
    logger.info("  Symbols:   %d", len(symbols))
    logger.info("  Hold:      %dh", args.hold_hours)
    logger.info("  Interval:  %ds", args.interval)
    if not args.record_only:
        logger.info("  Limits:    %d concurrent | $%.0f/position | $%.0f daily loss",
                    MAX_CONCURRENT_POSITIONS, MAX_POSITION_USD, MAX_DAILY_LOSS_USD)
    logger.info("=" * 62)

    while True:
        started = time.time()
        try:
            snapshots = fetcher.fetch_many(symbols, period="1H")
            logger.info("Fetched positioning for %d/%d symbols", len(snapshots), len(symbols))

            if args.record_only:
                for sym, snap in sorted(snapshots.items()):
                    logger.info("  %-12s ratio %.3f  p%.0f  (%d obs)",
                                sym, snap.long_short_ratio,
                                snap.percentile_of_current() * 100, snap.observations)
            else:
                signals, verdicts = signal.generate(snapshots)

                blocked = [v for v in verdicts if not v.fired and "not validated" in v.reason]
                if blocked:
                    logger.warning(
                        "%d symbol(s) refused — source not validated for them: %s",
                        len(blocked), ", ".join(v.symbol for v in blocked),
                    )

                if not signals:
                    logger.info("No crowd extremes this cycle.")
                for sig in signals:
                    logger.info("SIGNAL: SHORT %s  ratio=%.2f p%.0f  hold %dh",
                                sig.symbol, sig.ratio, sig.percentile * 100, sig.hold_hours)
                    if notifier.enabled:
                        notifier.send(
                            f"🔻 <b>CROWD SHORT</b> — {sig.symbol}\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"Long/short ratio: <code>{sig.ratio:.2f}</code>  "
                            f"(p{sig.percentile*100:.0f} of {sig.observations} obs)\n"
                            f"Hold: <code>{sig.hold_hours}h</code>   "
                            f"Source: <code>{sig.source}</code>\n"
                            f"<i>{sig.reason}</i>"
                        )
                    if args.paper or args.live:
                        logger.warning(
                            "Order routing for perp shorts is not wired up — signal logged "
                            "only. The existing executor is spot-long-only; see notes in "
                            "src/execution/executor.py.",
                        )

        except Exception as exc:
            logger.exception("Cycle failed: %s", exc)

        fetcher.save()
        if args.once:
            break
        time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
