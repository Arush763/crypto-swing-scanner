"""
Automated trading into Coinbase.

Runs the one validated strategy in this repo — the crowd-short signal — against
a Coinbase venue, with the same risk breakers, position store and trade log the
rest of the execution layer uses. Paper by default; live requires two separate
opt-ins.

What this actually does each cycle
----------------------------------
  1. Read retail long/short positioning (OKX rubik; Binance where reachable).
  2. Ask CrowdShortSignal which symbols are at a crowding extreme.
  3. Resolve those to this venue's contract (BTC/USDT -> BTC/USD:USD-301220).
  4. Refuse any trade whose measured edge does not clear this venue's fees.
  5. Size, place, and stamp a 16-hour deadline on each accepted signal.
  6. Flatten anything that has reached its deadline.

Which Coinbase venue
--------------------
The strategy is SHORT-ONLY, so the venue must be able to hold a short. Coinbase
Advanced Trade and Coinbase Exchange are spot: no borrow, so the position is not
expressible there at all, and the runner refuses rather than quietly inverting
the trade (the mirror long measured -0.240%/trade). Their retail fees would also
exceed the entire +0.655% gross edge.

The default venue is `coinbasederivatives` — CFTC-regulated CDE futures, reached
through the same ccxt client as Advanced Trade, shortable, and open to US
persons. Contracts are nano-sized and dated; the furthest series is the
perpetual-style product and resolve_contract picks it.

Three things differ from spot and are handled here: size is a whole number of
contracts, commission is charged per contract rather than per dollar, and the
market has trading sessions. See src/execution/contracts.py.

Modes
-----
    python scripts/run_coinbase_trader.py --dry-run    # signals only, no fills
    python scripts/run_coinbase_trader.py              # paper fills (default)
    python scripts/run_coinbase_trader.py --live       # real orders; see below

--live additionally requires LIVE_TRADING_ENABLED in config.py and credentials
in the environment. Either one missing falls back to paper rather than failing
open.
"""

from __future__ import annotations

import argparse
import logging
import signal as signal_module
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/coinbase_trader.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("coinbase")

from src.config.config import (
    COINBASE_VENUE,
    ENTRY_POST_ONLY,
    LIVE_TRADING_ENABLED,
    MAJOR_BASES,
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_LOSS_USD,
    MAX_DAILY_TRADES,
    MAX_POSITION_USD,
    RISK_PER_TRADE_PCT,
)
from src.data.positioning import PositioningFetcher
from src.execution.contracts import (
    ContractSpec,
    contract_fee_usd,
    exit_falls_in_break,
    fee_is_measured,
    session_allows_entry,
    session_from_market,
    size_in_contracts,
    spec_from_market,
)
from src.execution.edge_guard import EdgeGuard
from src.execution.executor import Executor
from src.execution.monitor import Mark, PositionMonitor
from src.execution.position import PositionStore
from src.execution.risk import RiskLimits, RiskManager
from src.execution.venues import (
    VENUES,
    exchange_class,
    resolve_contract,
    venue_spec,
)
from src.modules.crowd_signal import HOLD_HOURS, CrowdShortSignal
from src.notifications.telegram import TelegramNotifier


# ---------------------------------------------------------------------------
# Order request
# ---------------------------------------------------------------------------

@dataclass
class ShortOrder:
    """
    What the executor needs to open a crowd-short position.

    Deliberately not reusing signals.generator.Signal: that class carries a
    stop, two targets and a risk/reward ratio, none of which this strategy
    has. Populating them with zeros to satisfy a type would invite exactly the
    parameter search the combo sweep already exhausted.
    """
    symbol: str
    signal_type: str = "crowd_short"
    side: str = "short"
    hold_hours: float = HOLD_HOURS

    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    current_price: float = 0.0


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

class MarkFeed:
    """
    Current prices from the execution venue itself.

    Marking a Coinbase position against an OKX price would be wrong in the way
    that matters most: an exit is priced by the book it executes on, and the
    two venues can differ by more than this strategy's per-trade edge during a
    fast move.
    """

    def __init__(self, venue: str) -> None:
        spec = venue_spec(venue)
        self.spec = spec
        self.client = exchange_class(spec)({"enableRateLimit": True})
        self.client.load_markets()

    def available(self, symbols: List[str]) -> List[str]:
        return [s for s in symbols if s in self.client.markets]

    def resolve(self, base: str) -> Optional[str]:
        """
        The venue symbol to trade for `base`.

        Templated for spot and perps; resolved against live markets for dated
        futures, whose symbol carries an expiry that no template can know.
        """
        if not self.spec.dynamic_symbols:
            symbol = self.spec.symbol_for(base)
            return symbol if symbol in self.client.markets else None
        return resolve_contract(
            self.client, base, venue=self.spec.name,
            now_ms=int(time.time() * 1000),
        )

    @staticmethod
    def _price_from(ticker: dict) -> Optional[float]:
        """
        Pull a usable price out of a ticker, tolerating a thin parser.

        ccxt's coinbaseinternational parser returns `last`, `close`, `high` and
        `low` as None and populates only bid/ask and the raw payload — so the
        obvious `ticker["last"]` yields nothing and every mark silently goes
        missing, which reads as "no data" rather than as a bug. Falling back
        through the venue's own field and finally to the bid/ask midpoint keeps
        this working across venues whose parsers differ.
        """
        for key in ("last", "close"):
            value = ticker.get(key)
            if value:
                return float(value)

        raw = ticker.get("info") or {}
        for key in ("trade_price", "mark_price", "index_price"):
            value = raw.get(key)
            if value:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        bid, ask = ticker.get("bid"), ticker.get("ask")
        if bid and ask:
            return (float(bid) + float(ask)) / 2
        return None

    def fetch(self, symbols: List[str]) -> Dict[str, Mark]:
        """
        Current marks, one per symbol.

        `high`/`low` are the 24h extremes where the venue reports them, and
        absent otherwise — in which case Mark falls back to the last price for
        both. That degrades the monitor's ability to catch a level touched
        between polls, which is why the fixed-hold strategy this runner drives
        carries no stop to miss. A stop-bearing strategy on such a venue needs
        a real interval feed, not this.
        """
        marks: Dict[str, Mark] = {}
        for symbol in symbols:
            try:
                ticker = self.client.fetch_ticker(symbol)
            except Exception as exc:
                logger.warning("No mark for %s: %s", symbol, exc)
                continue

            price = self._price_from(ticker)
            if price is None:
                logger.warning("Ticker for %s carried no usable price", symbol)
                continue

            marks[symbol] = Mark(
                last=price,
                low=float(ticker["low"]) if ticker.get("low") else None,
                high=float(ticker["high"]) if ticker.get("high") else None,
            )
        return marks


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

class CoinbaseCrowdShortLoop:

    def __init__(self, args) -> None:
        self.args = args
        self.running = True
        self.cycle = 0
        self._warned_exposure = False

        self.spec = venue_spec(args.venue)
        # Contract venues size in whole contracts and charge per contract, so
        # both the sizing and the cost model take a different path.
        self.is_contract_venue = self.spec.product == "future"
        self.notifier = TelegramNotifier()
        self.store = PositionStore(path=f"data/state/positions_{args.venue}.json")
        self.risk = RiskManager(
            limits=RiskLimits(
                max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
                max_position_usd=MAX_POSITION_USD,
                max_daily_loss_usd=MAX_DAILY_LOSS_USD,
                max_daily_trades=MAX_DAILY_TRADES,
                risk_per_trade_pct=RISK_PER_TRADE_PCT,
            ),
            state_path=f"data/state/risk_{args.venue}.json",
        )

        self.fetcher = PositioningFetcher()
        self.signal = CrowdShortSignal(
            source=args.source,
            hold_hours=args.hold_hours,
            require_validated_source=not args.allow_unvalidated,
        )
        self.guard = EdgeGuard(
            venue=args.venue,
            entry_is_maker=ENTRY_POST_ONLY,
            safety_margin_pct=args.safety_margin,
        )

        self.executor: Optional[Executor] = None
        if not args.dry_run:
            self.executor = Executor(
                store=self.store,
                risk=self.risk,
                notifier=self.notifier,
                venue=args.venue,
                live=args.live and LIVE_TRADING_ENABLED,
            )

        self.monitor = PositionMonitor(
            store=self.store,
            notifier=self.notifier,
            exit_executor=self.executor.exit_fill if self.executor else None,
            venue=args.venue,
        )

        self.marks = MarkFeed(args.venue)

        signal_module.signal(signal_module.SIGINT, self._shutdown)
        signal_module.signal(signal_module.SIGTERM, self._shutdown)

    # ------------------------------------------------------------------

    def _shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received — finishing current cycle")
        self.running = False

    def mode(self) -> str:
        if self.args.dry_run:
            return "DRY-RUN"
        if self.executor and self.executor.live:
            return "LIVE"
        return "PAPER"

    # ------------------------------------------------------------------

    def preflight(self) -> bool:
        """
        Refuse to start on a venue where the strategy cannot work.

        This duplicates part of scripts/coinbase_preflight.py on purpose. The
        preflight script is something a person chooses to run; this is the
        check that happens whether or not they did.
        """
        logger.info("=" * 62)
        logger.info("  Coinbase crowd-short — %s", self.mode())
        logger.info("  Venue:     %s (%s)", self.spec.name, self.spec.description)
        logger.info("  Source:    %s", self.args.source)
        logger.info("  Hold:      %dh", self.args.hold_hours)
        logger.info("  Limits:    %d concurrent | $%.0f/position | $%.0f daily loss",
                    MAX_CONCURRENT_POSITIONS, MAX_POSITION_USD, MAX_DAILY_LOSS_USD)
        logger.info("=" * 62)

        if not self.spec.can_short:
            logger.error(
                "REFUSING TO START. %s is %s and cannot hold a short position, and "
                "the crowd-short signal is the only validated strategy here. Use "
                "--venue coinbaseinternational (perps) if you are eligible for it, "
                "or run with --dry-run to receive signals without trading.",
                self.spec.name, self.spec.product,
            )
            return False

        # Cost-check a representative symbol. BTC is the most favourable case
        # on a per-contract venue (largest notional, so the flat fee is the
        # smallest fraction of it), which is exactly why passing here is a
        # necessary and not a sufficient condition — each symbol is re-checked
        # at entry.
        probe_symbol = self.marks.resolve("BTC")
        if not probe_symbol:
            logger.error("REFUSING TO START. %s lists no BTC contract to price against.",
                         self.spec.name)
            return False

        fee_pct = None
        if self.is_contract_venue:
            spec = spec_from_market(probe_symbol, self.marks.client.markets[probe_symbol])
            price = (self.marks.fetch([probe_symbol]).get(probe_symbol) or Mark(last=0)).last
            fee_pct = spec.round_trip_fee_pct(price, self.spec.name)
            if not fee_is_measured(self.spec.name):
                logger.warning(
                    "Per-contract fee is a PLACEHOLDER ($%.2f/side), not your account's "
                    "real rate. Run scripts/coinbase_preflight.py --venue %s with "
                    "credentials to pin it. Every cost check below inherits this guess.",
                    contract_fee_usd(self.spec.name),
                    self.spec.name,
                )

        verdict = self.guard.evaluate(
            probe_symbol, "crowd_short",
            hold_hours=self.args.hold_hours, fee_pct_round_trip=fee_pct,
        )
        if not verdict.allowed:
            logger.error(
                "REFUSING TO START. %s Run scripts/coinbase_preflight.py --venue %s "
                "for the full breakdown.",
                verdict.reason, self.spec.name,
            )
            return False

        logger.info("Cost check (%s): %s", probe_symbol, verdict.explain())
        return True

    # ------------------------------------------------------------------

    def run_cycle(self) -> None:
        self.cycle += 1

        # 1) Manage open risk before adding any. An expired position must be
        #    flattened this cycle regardless of what the signal says now.
        held = [p.symbol for p in self.store.all_open()]
        if held:
            for event in self.monitor.check(self.marks.fetch(held)):
                if event.closed:
                    halt = self.risk.record_close(event.pnl_usd)
                    if halt and self.notifier.enabled:
                        self.notifier.send_risk_halt("Daily loss limit", halt)

        # 2) Fresh signals.
        source_symbols = [f"{b}/USDT" for b in MAJOR_BASES[:self.args.symbols]]
        snapshots = self.fetcher.fetch_many(source_symbols, period="1H")
        signals, verdicts = self.signal.generate(snapshots)

        blocked = [v for v in verdicts if not v.fired and "not validated" in v.reason]
        if blocked:
            logger.warning(
                "%d symbol(s) refused — the positioning source was not validated "
                "for them: %s",
                len(blocked), ", ".join(v.symbol for v in blocked),
            )

        if not signals:
            logger.info("Cycle %d — no crowd extremes, %d open",
                        self.cycle, self.store.open_count)
            self.fetcher.save()
            return

        for sig in signals:
            self._consider(sig)

        self.fetcher.save()
        logger.info("Cycle %d — %d signal(s), %d open",
                    self.cycle, len(signals), self.store.open_count)

    def _budget_for(self, equity_usd: float) -> float:
        """
        Dollars available for one position.

        On a divisible instrument this is the smaller of the notional cap and
        the risk-per-trade fraction, and the fraction usually binds. On an
        indivisible contract that rule quietly disables the strategy: 1% of a
        $10k account is $100, no contract costs less than $72, and the five
        largest-notional contracts — which are also the CHEAPEST once the flat
        per-contract fee is expressed as a percentage — all cost more than $500.
        Applying the fraction there means refusing every good symbol and keeping
        only the expensive ones, which is the opposite of prudent.

        So for contract venues the notional cap governs, and the resulting
        exposure is reported rather than hidden. The honest statement is that
        contract size sets a minimum account size: holding true 1%-per-trade
        risk against a $773 BTC contract needs roughly $77k of equity.
        """
        if not self.is_contract_venue:
            return min(MAX_POSITION_USD, equity_usd * RISK_PER_TRADE_PCT)

        budget = MAX_POSITION_USD
        implied = (budget / equity_usd * 100) if equity_usd > 0 else float("inf")
        if implied > RISK_PER_TRADE_PCT * 100 * 2 and not self._warned_exposure:
            self._warned_exposure = True
            logger.warning(
                "One position is up to $%.0f on $%.0f of equity — %.1f%% notional "
                "exposure, against a configured risk-per-trade of %.1f%%. Contracts "
                "are indivisible, so this is the floor, not a setting: holding %.1f%% "
                "would need about $%.0f of equity.",
                budget, equity_usd, implied, RISK_PER_TRADE_PCT * 100,
                RISK_PER_TRADE_PCT * 100, budget / RISK_PER_TRADE_PCT,
            )
        return budget

    def _alert(
        self,
        sig,
        venue_symbol: str,
        blocked: str = "",
        contracts: Optional[float] = None,
        notional: Optional[float] = None,
        verdict=None,
    ) -> None:
        """
        Send one Telegram message describing the trade the signal implies.

        Blocked signals are reported too, with the reason. Crowd extremes are
        rare by construction — this only fires on the top decile — so the
        volume is low enough that silence would be more confusing than a
        "here's why not" line. A reader who is told a symbol is crowded but not
        told it was refused will assume the bot simply missed it.
        """
        if not self.notifier.enabled:
            return

        head = "⛔ <b>CROWD SHORT (blocked)</b>" if blocked else "🔻 <b>CROWD SHORT</b>"
        lines = [
            f"{head} — {venue_symbol}",
            "━━━━━━━━━━━━━━━━━━━━",
            f"Long/short ratio: <code>{sig.ratio:.2f}</code>  "
            f"(p{sig.percentile*100:.0f} of {sig.observations} obs)",
            f"Hold: <code>{sig.hold_hours}h</code>   Mode: <code>{self.mode()}</code>",
        ]

        if contracts:
            lines.append(
                f"Size: <code>{int(contracts)}</code> contract(s) = "
                f"<code>${notional:,.0f}</code> notional"
            )
        if verdict is not None:
            lines.append(
                f"Cost: <code>{verdict.round_trip_cost_pct:.3f}%</code> round trip → "
                f"net <code>{verdict.net_edge_pct:+.3f}%</code>"
            )
            if self.is_contract_venue and not fee_is_measured(self.spec.name):
                lines.append("⚠️ fee is a placeholder, not your real rate")
        if blocked:
            lines.append(f"<i>{blocked}</i>")

        self.notifier.send("\n".join(lines))

    def _consider(self, sig) -> None:
        """Translate one crowd signal into a Coinbase order, or explain why not."""
        base = sig.symbol.split("/")[0]
        venue_symbol = self.marks.resolve(base)

        if not venue_symbol:
            logger.info("Skipping %s — %s lists no tradeable contract for it",
                        sig.symbol, self.spec.name)
            return

        logger.info("SIGNAL: SHORT %s -> %s  ratio=%.2f p%.0f  hold %dh",
                    sig.symbol, venue_symbol, sig.ratio, sig.percentile * 100,
                    sig.hold_hours)

        market = self.marks.client.markets[venue_symbol]

        # Everything from here to the order itself is pure computation, so it
        # runs in dry-run too. An alert that says only "short BTC" leaves the
        # reader to redo the contract lookup, the sizing and the cost check by
        # hand — which is the work that decides whether the trade is worth
        # taking at all.
        if self.is_contract_venue:
            session = session_from_market(market)
            allowed, why = session_allows_entry(session, sig.hold_hours)
            if not allowed:
                logger.info("Entry deferred for %s — %s", venue_symbol, why)
                self._alert(sig, venue_symbol, blocked=why)
                return
            delayed, note = exit_falls_in_break(session, sig.hold_hours)
            if delayed:
                logger.warning("%s: %s", venue_symbol, note)

        mark = self.marks.fetch([venue_symbol]).get(venue_symbol)
        if mark is None:
            logger.warning("No mark for %s — skipping entry", venue_symbol)
            self._alert(sig, venue_symbol, blocked="no price available")
            return

        equity = (self.executor.account_equity_usd(self.args.paper_equity)
                  if self.executor else self.args.paper_equity)
        budget = self._budget_for(equity)

        quantity = notional = None
        fee_pct = None
        contract_size, fee_per_contract = 1.0, 0.0
        if self.is_contract_venue:
            spec = spec_from_market(venue_symbol, market)
            sizing = size_in_contracts(spec, mark.last, budget, self.spec.name)
            if not sizing.tradeable:
                logger.info("Entry refused for %s — %s", venue_symbol, sizing.reason)
                self._alert(sig, venue_symbol, blocked=sizing.reason)
                return
            quantity, notional = float(sizing.contracts), sizing.notional_usd
            fee_pct = sizing.fee_pct_round_trip
            contract_size = spec.contract_size
            fee_per_contract = contract_fee_usd(self.spec.name)
            logger.info(
                "%s: %d contract(s) x %s = $%.2f notional (fee %.3f%% round trip%s)",
                venue_symbol, sizing.contracts, spec.display_name or spec.contract_id,
                notional, fee_pct,
                "" if fee_is_measured(self.spec.name) else ", PLACEHOLDER fee",
            )
            budget = notional

        verdict = self.guard.evaluate(
            venue_symbol, "crowd_short",
            hold_hours=sig.hold_hours,
            order_size_usd=budget,
            fee_pct_round_trip=fee_pct,
        )
        if not verdict.allowed:
            logger.info("Entry refused for %s — %s", venue_symbol, verdict.reason)
            self._alert(sig, venue_symbol, blocked=verdict.reason,
                        contracts=quantity, notional=notional, verdict=verdict)
            return
        logger.info("Cost check %s: %s", venue_symbol, verdict.explain())

        self._alert(sig, venue_symbol, contracts=quantity,
                    notional=notional, verdict=verdict)

        if self.args.dry_run or self.executor is None:
            return

        order = ShortOrder(
            symbol=venue_symbol,
            hold_hours=sig.hold_hours,
            current_price=mark.last,
        )
        self.executor.open_position(
            order,
            equity_usd=self.executor.account_equity_usd(self.args.paper_equity),
            mark_price=mark.last,
            quantity_override=quantity,
            notional_override=notional,
            contract_size=contract_size,
            fee_per_contract_usd=fee_per_contract,
        )

    # ------------------------------------------------------------------

    def run(self) -> int:
        if not self.preflight():
            return 1

        if self.notifier.enabled:
            self.notifier.send(
                f"🤖 <b>Coinbase crowd-short started</b>\n"
                f"Mode: <b>{self.mode()}</b>  Venue: <code>{self.spec.name}</code>"
            )

        while self.running:
            started = time.time()
            try:
                self.run_cycle()
            except Exception as exc:
                logger.exception("Cycle %d failed: %s", self.cycle, exc)

            if self.args.once:
                break

            deadline = started + self.args.interval
            while self.running and time.time() < deadline:
                time.sleep(min(1.0, deadline - time.time()))

        open_now = self.store.all_open()
        self.store.save()
        self.risk.save()
        logger.info("Stopped after %d cycle(s) — %d position(s) left open",
                    self.cycle, len(open_now))
        if open_now:
            logger.warning(
                "Still open and no longer monitored: %s. These have hold deadlines; "
                "restart the runner or close them manually.",
                ", ".join(p.symbol for p in open_now),
            )
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crowd-short strategy on Coinbase")
    p.add_argument("--venue", default=COINBASE_VENUE,
                   choices=sorted(v for v in VENUES if v.startswith("coinbase")),
                   help=f"Coinbase product to trade (default: {COINBASE_VENUE})")
    p.add_argument("--source", default="okx", choices=["okx", "binance"],
                   help="Positioning data source (binance requires an unblocked host)")
    p.add_argument("--symbols", type=int, default=12)
    p.add_argument("--interval", type=int, default=3600,
                   help="Seconds between polls (default: 1h, matching the signal's bar)")
    p.add_argument("--hold-hours", type=int, default=HOLD_HOURS)
    p.add_argument("--paper-equity", type=float, default=10_000.0)
    p.add_argument("--safety-margin", type=float, default=0.05,
                   help="Net edge, in percent, a trade must clear beyond breakeven")
    p.add_argument("--dry-run", action="store_true",
                   help="Emit signals and alerts, never open a position")
    p.add_argument("--live", action="store_true",
                   help="Route real orders. Also requires LIVE_TRADING_ENABLED and keys.")
    p.add_argument("--allow-unvalidated", action="store_true",
                   help="Trade symbols whose positioning source was not validated "
                        "against the backtested series. Off by default.")
    p.add_argument("--once", action="store_true", help="Single cycle, then exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.live and not LIVE_TRADING_ENABLED:
        logger.error(
            "--live was passed but LIVE_TRADING_ENABLED is False in config.py. "
            "Running in PAPER mode. Set it deliberately, after reviewing paper results."
        )
    return CoinbaseCrowdShortLoop(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
