"""
Position monitor — turns price movement into lifecycle events and alerts.

Called once per scan cycle with fresh marks. For each open position it checks,
in strict priority order, whether the stop, TP1, or TP2 has been reached, and
emits the corresponding Telegram trigger.

Ordering is not arbitrary. The stop is evaluated *before* the targets because
when a single cycle's price range spans both — which is common on a fast
timeframe with a 60s polling interval — assuming the favourable one filled
first is exactly the bias that makes a backtest look better than the account
it is supposed to describe. Resolving ambiguity against yourself is the only
defensible default when the data can't say which came first.

The same reasoning applies to marking: positions are checked against the
low/high of the interval where available, not just the last trade price, so a
stop that was touched between polls is not silently missed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.execution.costs import cost_model_for
from src.execution.position import Position, PositionStore, log_closed_trade

logger = logging.getLogger(__name__)


@dataclass
class Mark:
    """Current price observation for one symbol over the last interval."""
    last: float
    low: Optional[float] = None      # interval low; defaults to `last`
    high: Optional[float] = None     # interval high; defaults to `last`

    @property
    def _low(self) -> float:
        return self.low if self.low is not None else self.last

    @property
    def _high(self) -> float:
        return self.high if self.high is not None else self.last

    def adverse(self, direction: int) -> float:
        """
        The extreme of the interval that hurts a position in `direction`.

        A long is hurt by the low; a short is hurt by the high. Naming these
        by what they do to the trade rather than by which end of the bar they
        are is what keeps the stop check correct when a short is added — the
        original `worst`/`best` pair silently meant "low"/"high", which is only
        the same thing for a long.
        """
        return self._low if direction > 0 else self._high

    def favourable(self, direction: int) -> float:
        """The extreme of the interval that helps a position in `direction`."""
        return self._high if direction > 0 else self._low

    # Retained for callers that predate the short side; both assume a long.
    @property
    def worst(self) -> float:
        return self._low

    @property
    def best(self) -> float:
        return self._high


@dataclass
class LifecycleEvent:
    position: Position
    kind: str             # "tp1" | "tp2" | "stop" | "trail" | "expiry"
    price: float
    pnl_pct: float
    pnl_usd: float
    closed: bool


class PositionMonitor:
    """
    Evaluates open positions against fresh marks and emits lifecycle events.

    `exit_executor` is called to actually flatten (or partially flatten) a
    position — in paper mode it's a no-op that just reports the fill price,
    in live mode it sends the order. Injecting it keeps this class free of
    exchange concerns and directly testable.
    """

    def __init__(
        self,
        store: PositionStore,
        notifier=None,
        exit_executor: Optional[Callable[[Position, float, str], float]] = None,
        venue: str = "okx",
    ) -> None:
        self.store = store
        self.notifier = notifier
        self.exit_executor = exit_executor
        self.venue = venue

    # ------------------------------------------------------------------

    def _exit_fee_usd(self, position: Position, price: float, quantity: float) -> float:
        """
        Fee charged on an exit leg. Exits are taker orders — a stop that waits
        politely in the queue is not a stop — so this uses the taker rate
        regardless of how the entry was placed.

        Delegated to the position so contract size and flat per-contract
        commissions are applied in one place rather than re-derived here.
        """
        model = cost_model_for(position.symbol, venue=self.venue, exit_is_maker=False)
        fee_pct = model.leg_cost_pct(is_maker=False)
        return position.leg_fee_usd(price, quantity, fee_pct)

    def _fill(self, position: Position, price: float, reason: str) -> float:
        """Route an exit through the executor, returning the achieved price."""
        if self.exit_executor is None:
            return price
        try:
            return self.exit_executor(position, price, reason)
        except Exception as exc:
            logger.error("Exit execution failed for %s: %s", position.symbol, exc)
            return price

    # ------------------------------------------------------------------

    def check(self, marks: Dict[str, Mark]) -> List[LifecycleEvent]:
        """
        Evaluate every open position against `marks`.

        Symbols absent from `marks` are left untouched rather than assumed
        unchanged — a missing mark means "no information", and acting on a
        stale price is worse than waiting for the next cycle.
        """
        events: List[LifecycleEvent] = []

        for position in self.store.all_open():
            mark = marks.get(position.symbol)
            if mark is None:
                # An expired position is the one case where "no information"
                # does not justify waiting: the horizon the strategy was
                # measured over has passed, and leaving it open converts a
                # tested trade into an untested one. Flatten it, and say
                # plainly that the fill price is a fallback.
                if position.is_expired():
                    logger.warning(
                        "%s is past its hold horizon but has no fresh mark — "
                        "closing at the entry price as a fallback",
                        position.symbol,
                    )
                    events.append(
                        self._close_out(position, position.entry_price, "expiry", "expiry")
                    )
                continue
            event = self._check_one(position, mark)
            if event:
                events.extend(event)

        return events

    @staticmethod
    def _stop_breached(position: Position, mark: Mark) -> bool:
        """
        Whether the interval reached the stop, in the direction that hurts.

        A stop of zero means "no stop" — the crowd-short signal is a fixed-hold
        statement with neither stop nor target, and treating an unset level as
        a price would close a short instantly, since every price is above zero.
        """
        if position.stop_loss <= 0:
            return False
        adverse = mark.adverse(position.direction)
        return adverse <= position.stop_loss if position.direction > 0 else adverse >= position.stop_loss

    @staticmethod
    def _target_reached(position: Position, mark: Mark, target: float) -> bool:
        if target <= 0:
            return False
        favourable = mark.favourable(position.direction)
        return favourable >= target if position.direction > 0 else favourable <= target

    def _close_out(
        self, position: Position, price: float, reason: str, kind: str,
    ) -> LifecycleEvent:
        """Flatten a position and emit its terminal event."""
        fill = self._fill(position, price, reason)
        fee = self._exit_fee_usd(position, fill, position.remaining_quantity)
        position.close(fill, reason, fee)
        pnl_pct = position.total_pnl_pct()

        self.store.retire(position)
        log_closed_trade(position)
        if self.notifier:
            self.notifier.send_position_closed(
                position, reason, fill, pnl_pct, position.realised_pnl_usd,
            )
        logger.info("%s %s %s @ %.8g (%.2f%%)",
                    reason.upper(), position.side, position.symbol, fill, pnl_pct)
        return LifecycleEvent(position, kind, fill, pnl_pct, position.realised_pnl_usd, True)

    def _check_one(self, position: Position, mark: Mark) -> List[LifecycleEvent]:
        events: List[LifecycleEvent] = []

        # --- Stop first, deliberately. See module docstring. ---
        if self._stop_breached(position, mark):
            price = self._fill(position, position.stop_loss, "stop_loss")
            fee = self._exit_fee_usd(position, price, position.remaining_quantity)
            position.close(price, "stop_loss", fee)
            pnl_pct = position.total_pnl_pct()

            self.store.retire(position)
            log_closed_trade(position)
            if self.notifier:
                self.notifier.send_stop_hit(position, price, pnl_pct, position.realised_pnl_usd)

            logger.info("STOP %s @ %.8g (%.2f%%)", position.symbol, price, pnl_pct)
            return [LifecycleEvent(position, "stop", price, pnl_pct, position.realised_pnl_usd, True)]

        # --- Time-based exit, before the targets. ---
        # A fixed-hold signal that has run out of horizon should be flattened
        # even if the same interval also touched a target: the study measured
        # return at the horizon, and preferring the target here would report a
        # fill the strategy is not entitled to.
        if position.is_expired():
            return [self._close_out(position, mark.last, "expiry", "expiry")]

        # --- TP1: scale out, move stop to breakeven ---
        if not position.tp1_hit and self._target_reached(position, mark, position.take_profit_1):
            qty = position.quantity * position.tp1_scale_out_fraction
            price = self._fill(position, position.take_profit_1, "tp1")
            fee = self._exit_fee_usd(position, price, qty)
            leg_pnl = position.record_partial_exit(price, qty, fee)
            position.tp1_hit = True

            # Breakeven stop after a scale-out is what converts a partial win
            # into a trade that can no longer lose — but only nominally: the
            # entry fee is already paid and unrecoverable, so "breakeven" here
            # means flat on price, still slightly down on cost.
            position.stop_loss = position.entry_price
            self.store.save()

            pct = leg_pnl / (position.entry_price * qty) * 100 if qty else 0.0
            if self.notifier:
                self.notifier.send_target_hit(position, "TP1", price, pct, leg_pnl)
            logger.info("TP1 %s @ %.8g — scaled out %.4g, stop -> breakeven",
                        position.symbol, price, qty)
            events.append(LifecycleEvent(position, "tp1", price, pct, leg_pnl, False))

            # A single cycle can carry price through both targets; fall
            # through so TP2 is evaluated against the same mark rather than
            # waiting a cycle and reporting a fill that never existed.

        # --- TP2: close the remainder ---
        if position.is_open and self._target_reached(position, mark, position.take_profit_2):
            price = self._fill(position, position.take_profit_2, "tp2")
            fee = self._exit_fee_usd(position, price, position.remaining_quantity)
            position.close(price, "take_profit_2", fee)
            pnl_pct = position.total_pnl_pct()

            self.store.retire(position)
            log_closed_trade(position)
            if self.notifier:
                self.notifier.send_target_hit(position, "TP2", price, pnl_pct, position.realised_pnl_usd)
                self.notifier.send_position_closed(
                    position, "TP2", price, pnl_pct, position.realised_pnl_usd,
                )
            logger.info("TP2 %s @ %.8g (%.2f%%)", position.symbol, price, pnl_pct)
            events.append(LifecycleEvent(position, "tp2", price, pnl_pct, position.realised_pnl_usd, True))

        return events

    # ------------------------------------------------------------------

    def force_close_all(self, marks: Dict[str, Mark], reason: str = "manual") -> List[LifecycleEvent]:
        """Flatten everything — used by the kill switch and on shutdown."""
        events: List[LifecycleEvent] = []
        for position in self.store.all_open():
            mark = marks.get(position.symbol)
            price = mark.last if mark else position.entry_price
            price = self._fill(position, price, reason)
            fee = self._exit_fee_usd(position, price, position.remaining_quantity)
            position.close(price, reason, fee)
            pnl_pct = position.total_pnl_pct()

            self.store.retire(position)
            log_closed_trade(position)
            if self.notifier:
                self.notifier.send_position_closed(
                    position, reason, price, pnl_pct, position.realised_pnl_usd,
                )
            events.append(LifecycleEvent(position, "close", price, pnl_pct, position.realised_pnl_usd, True))
        return events
