"""
Order execution — paper and live.

Two modes share one interface:

  paper : orders are simulated against the *live* order book, charged the
          same modelled fees and spread a real order would pay. Nothing is
          sent to the exchange. This is the default and the only mode that
          runs without explicit opt-in.
  live  : orders are routed via ccxt. Requires LIVE_TRADING_ENABLED plus
          credentials in the environment; either one missing falls back to
          paper rather than failing open.

PAPER_FILL_NOTE
---------------
Paper mode's honesty has a hard limit worth stating plainly, because it is
the thing most likely to make these results diverge from a real account.

A post-only entry is modelled as filling at the touch. In reality it joins
the back of a queue at that price and fills only if enough volume trades
through to reach it — and the times it *doesn't* fill are not random. A
resting bid gets filled precisely when sellers keep coming, i.e. when price
is about to continue down through it; it gets missed precisely when the
market turns up without it, which is the move the signal was trying to
catch. That adverse selection is a real cost, it is not in the model here,
and it falls hardest on exactly the maker-entry strategy that the fee
arithmetic otherwise recommends.

So: paper results here should be read as an optimistic bound on maker-entry
performance. The only way to measure the gap is to run small live size and
compare realised fills against this simulation — which is what the trade log
in src/execution/position.py exists to enable.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from src.config.config import (
    ENTRY_LIMIT_TIMEOUT_SECONDS,
    ENTRY_POST_ONLY,
    EXECUTION_VENUE,
    LIVE_TRADING_ENABLED,
)
from src.execution.costs import cost_model_for
from src.execution.position import Position, PositionStore, log_closed_trade
from src.execution.risk import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    """Result of an order attempt."""
    filled: bool
    price: float = 0.0
    quantity: float = 0.0
    fee_usd: float = 0.0
    order_id: str = ""
    reason: str = ""


class Executor:
    """
    Places entries and exits, enforcing risk limits on every entry.

    The risk check happens here rather than in the caller so there is exactly
    one path to opening a position, and it is not possible to add a new
    caller that forgets to ask permission first.
    """

    def __init__(
        self,
        store: PositionStore,
        risk: RiskManager,
        notifier=None,
        venue: str = EXECUTION_VENUE,
        live: bool = LIVE_TRADING_ENABLED,
        post_only: bool = ENTRY_POST_ONLY,
    ) -> None:
        self.store = store
        self.risk = risk
        self.notifier = notifier
        self.venue = venue
        self.post_only = post_only

        self.client = None
        self.live = False
        if live:
            self.live = self._init_live_client()
            if not self.live:
                logger.error(
                    "LIVE_TRADING_ENABLED is set but the exchange client could not be "
                    "initialised — falling back to PAPER mode. No orders will be sent."
                )

        logger.info(
            "Executor ready — venue=%s mode=%s post_only=%s",
            self.venue, "LIVE" if self.live else "PAPER", self.post_only,
        )

    # ------------------------------------------------------------------
    # Live client
    # ------------------------------------------------------------------

    def _init_live_client(self) -> bool:
        """
        Bring up an authenticated ccxt client. Returns False (staying in
        paper mode) on any problem — a half-configured live client is more
        dangerous than no live client.
        """
        api_key = os.environ.get("EXCHANGE_API_KEY", "")
        secret = os.environ.get("EXCHANGE_API_SECRET", "")
        password = os.environ.get("EXCHANGE_API_PASSWORD", "")   # OKX requires this

        if not api_key or not secret:
            logger.error("Live trading requested but EXCHANGE_API_KEY/SECRET are not set")
            return False

        try:
            import ccxt
            klass = getattr(ccxt, self.venue)
            options = {"apiKey": api_key, "secret": secret, "enableRateLimit": True}
            if password:
                options["password"] = password
            self.client = klass(options)

            # Demo/sandbox mode when requested — the intended way to exercise
            # this path before risking anything.
            if os.environ.get("EXCHANGE_SANDBOX", "").lower() in ("1", "true", "yes"):
                self.client.set_sandbox_mode(True)
                logger.warning("Exchange client is in SANDBOX mode")

            self.client.load_markets()
            return True
        except Exception as exc:
            logger.error("Could not initialise live client for %s: %s", self.venue, exc)
            return False

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def open_position(
        self,
        signal,
        equity_usd: float,
        mark_price: float,
    ) -> Optional[Position]:
        """
        Attempt to open a position from a Signal. Returns the Position on a
        fill, or None if refused by risk limits or not filled.
        """
        symbol = signal.symbol

        quantity, notional = self.risk.position_size(
            equity_usd, mark_price, signal.stop_loss,
        )

        allowed, reason = self.risk.can_open(
            symbol, notional,
            open_count=self.store.open_count,
            already_open_symbol=self.store.has_open(symbol),
        )
        if not allowed:
            logger.info("Entry refused for %s — %s", symbol, reason)
            return None

        fill = self._place_entry(symbol, quantity, mark_price)
        if not fill.filled:
            logger.info("Entry not filled for %s — %s", symbol, fill.reason)
            return None

        entry_cost_pct = cost_model_for(
            symbol, venue=self.venue, entry_is_maker=self.post_only,
        ).leg_cost_pct(is_maker=self.post_only)

        position = Position(
            symbol=symbol,
            side="long",
            entry_price=fill.price,
            quantity=fill.quantity,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            signal_type=signal.signal_type,
            is_paper=not self.live,
            entry_cost_pct=entry_cost_pct,
            fees_paid_usd=fill.fee_usd,
            entry_order_id=fill.order_id,
        )

        self.store.add(position)
        self.risk.record_open()

        if self.live:
            self._place_protective_stop(position)

        if self.notifier:
            self.notifier.send_entry_filled(position)

        logger.info(
            "OPENED %s %s qty=%.8g @ %.8g (notional $%.2f)",
            "LIVE" if self.live else "PAPER", symbol, fill.quantity, fill.price, notional,
        )
        return position

    def _place_entry(self, symbol: str, quantity: float, mark_price: float) -> Fill:
        if quantity <= 0:
            return Fill(filled=False, reason="zero quantity")

        if not self.live:
            return self._simulate_entry(symbol, quantity, mark_price)

        try:
            params = {"postOnly": True} if self.post_only else {}
            order_type = "limit" if self.post_only else "market"

            if self.post_only:
                order = self.client.create_order(
                    symbol, order_type, "buy", quantity, mark_price, params,
                )
                filled = self._await_fill(symbol, order["id"])
                if not filled:
                    self._cancel_quietly(symbol, order["id"])
                    return Fill(filled=False, reason="post-only entry timed out unfilled")
                order = filled
            else:
                order = self.client.create_order(symbol, "market", "buy", quantity)

            avg = float(order.get("average") or order.get("price") or mark_price)
            filled_qty = float(order.get("filled") or quantity)
            fee_usd = self._fee_from_order(order, avg, filled_qty)
            return Fill(True, avg, filled_qty, fee_usd, str(order.get("id", "")))

        except Exception as exc:
            logger.error("Entry order failed for %s: %s", symbol, exc)
            return Fill(filled=False, reason=str(exc))

    def _simulate_entry(self, symbol: str, quantity: float, mark_price: float) -> Fill:
        """
        Paper fill. See PAPER_FILL_NOTE in the module docstring for why the
        post-only case is optimistic.
        """
        model = cost_model_for(symbol, venue=self.venue, entry_is_maker=self.post_only)
        cost_pct = model.leg_cost_pct(is_maker=self.post_only)

        # A taker entry crosses the spread, so it fills worse than the mark.
        # A maker entry rests at the mark and pays no spread — but see the
        # note above about the fills it therefore misses.
        fill_price = mark_price if self.post_only else mark_price * (1 + cost_pct / 100)
        fee_usd = fill_price * quantity * cost_pct / 100

        return Fill(
            filled=True,
            price=fill_price,
            quantity=quantity,
            fee_usd=fee_usd,
            order_id=f"paper-{int(time.time()*1000)}",
        )

    def _await_fill(self, symbol: str, order_id: str) -> Optional[dict]:
        """Poll a resting post-only order until filled or timed out."""
        deadline = time.time() + ENTRY_LIMIT_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                order = self.client.fetch_order(order_id, symbol)
            except Exception as exc:
                logger.warning("Could not poll order %s: %s", order_id, exc)
                time.sleep(2)
                continue

            status = order.get("status")
            if status == "closed":
                return order
            if status in ("canceled", "expired", "rejected"):
                return None
            time.sleep(2)
        return None

    def _cancel_quietly(self, symbol: str, order_id: str) -> None:
        try:
            self.client.cancel_order(order_id, symbol)
        except Exception as exc:
            logger.warning("Could not cancel order %s: %s", order_id, exc)

    def _place_protective_stop(self, position: Position) -> None:
        """
        Rest a stop order on the exchange immediately after entry.

        This matters more than it looks: without an exchange-side stop, the
        position's only protection is this process staying alive. If the
        machine sleeps, the network drops, or the loop throws, an unprotected
        position stays open with nothing watching it.
        """
        try:
            order = self.client.create_order(
                position.symbol, "stop", "sell", position.quantity, None,
                {"stopPrice": position.stop_loss, "reduceOnly": True},
            )
            position.stop_order_id = str(order.get("id", ""))
            self.store.save()
            logger.info("Protective stop resting for %s @ %.8g",
                        position.symbol, position.stop_loss)
        except Exception as exc:
            # Loud, because the position is now open and unprotected at the
            # exchange. The monitor still guards it, but only while this
            # process is running.
            logger.error(
                "COULD NOT PLACE PROTECTIVE STOP for %s: %s — position is open and "
                "relies on the local monitor alone",
                position.symbol, exc,
            )
            if self.notifier:
                self.notifier.send_risk_halt(
                    "Protective stop failed",
                    f"{position.symbol} is open without an exchange-side stop. "
                    f"Local monitor is still active. Error: {exc}",
                )

    def _fee_from_order(self, order: dict, price: float, quantity: float) -> float:
        """Prefer the exchange's reported fee; fall back to the model."""
        fee = order.get("fee") or {}
        cost = fee.get("cost")
        if cost is not None:
            try:
                return abs(float(cost))
            except (TypeError, ValueError):
                pass
        model = cost_model_for(order.get("symbol", ""), venue=self.venue)
        return abs(price * quantity) * model.leg_cost_pct(is_maker=self.post_only) / 100

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def exit_fill(self, position: Position, target_price: float, reason: str) -> float:
        """
        Executor callback used by PositionMonitor. Returns the achieved price.

        Exits are market orders — a stop that rests politely in the queue is
        not a stop.
        """
        if not self.live:
            model = cost_model_for(position.symbol, venue=self.venue)
            slip_pct = model.leg_cost_pct(is_maker=False) - model._fee_pct(False)
            # Exiting a long is a sell, so slippage works against you: you
            # receive less than the trigger price, not more.
            return target_price * (1 - slip_pct / 100)

        try:
            if position.stop_order_id and reason == "stop_loss":
                self._cancel_quietly(position.symbol, position.stop_order_id)
            order = self.client.create_order(
                position.symbol, "market", "sell", position.remaining_quantity,
                None, {"reduceOnly": True},
            )
            return float(order.get("average") or order.get("price") or target_price)
        except Exception as exc:
            logger.error("Exit order failed for %s: %s", position.symbol, exc)
            return target_price

    # ------------------------------------------------------------------

    def account_equity_usd(self, fallback: float) -> float:
        """Live free balance, or `fallback` in paper mode / on error."""
        if not self.live:
            return fallback
        try:
            balance = self.client.fetch_balance()
            for ccy in ("USDT", "USD"):
                total = balance.get("total", {}).get(ccy)
                if total:
                    return float(total)
        except Exception as exc:
            logger.warning("Could not fetch balance: %s", exc)
        return fallback
