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
from src.execution.costs import cost_model_for, register_fee_schedule
from src.execution.position import Position, PositionStore, log_closed_trade
from src.execution.risk import RiskManager
from src.execution.venues import (
    DirectionUnsupported,
    ccxt_options,
    check_direction,
    load_credentials,
    venue_spec,
)

logger = logging.getLogger(__name__)

# ccxt order sides, by position side and leg. Written as a table rather than
# inline conditionals because getting one of the four wrong produces an order
# that doubles the position instead of closing it.
_ORDER_SIDE = {
    ("long", "entry"): "buy",
    ("long", "exit"): "sell",
    ("short", "entry"): "sell",
    ("short", "exit"): "buy",
}


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
        self.spec = venue_spec(venue)
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
            "Executor ready — venue=%s (%s, short=%s) mode=%s post_only=%s",
            self.venue, self.spec.product, self.spec.can_short,
            "LIVE" if self.live else "PAPER", self.post_only,
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
        creds = load_credentials(self.spec)
        if not creds.present:
            logger.error(
                "Live trading requested but credentials are missing — set %s",
                " and ".join(creds.missing(self.spec)),
            )
            return False

        try:
            import ccxt
            klass = getattr(ccxt, self.spec.ccxt_id)
            self.client = klass(ccxt_options(self.spec, creds))

            # Demo/sandbox mode when requested — the intended way to exercise
            # this path before risking anything.
            if os.environ.get("EXCHANGE_SANDBOX", "").lower() in ("1", "true", "yes"):
                if not self.spec.has_sandbox:
                    # Refusing rather than warning: a caller who asked for a
                    # sandbox and got a live account is the exact mistake that
                    # spends real money on a rehearsal.
                    logger.error(
                        "EXCHANGE_SANDBOX is set but %s has no sandbox — refusing to "
                        "connect, because this would have traded live instead.",
                        self.venue,
                    )
                    return False
                self.client.set_sandbox_mode(True)
                logger.warning("Exchange client is in SANDBOX mode")

            self.client.load_markets()
            self._register_live_fees()
            return True
        except Exception as exc:
            logger.error("Could not initialise live client for %s: %s", self.venue, exc)
            return False

    def _register_live_fees(self) -> None:
        """
        Replace the table's conservative fee guess with the account's real tier.

        Coinbase's retail spot fees vary by an order of magnitude across volume
        tiers, so the difference between the assumed rate and the charged rate
        can be larger than the entire edge being traded. Every downstream cost
        calculation — sizing, the edge guard, paper fills — reads through
        `resolve_fee_schedule`, so registering here fixes all of them at once.
        """
        try:
            fees = self.client.fetch_trading_fees()
        except Exception as exc:
            logger.warning(
                "Could not read live fee tier for %s (%s) — using the conservative "
                "table in costs.py, which may overstate or understate your real cost",
                self.venue, exc,
            )
            return

        rates = [f for f in fees.values() if isinstance(f, dict)
                 and f.get("maker") is not None and f.get("taker") is not None]
        if not rates:
            return

        # The worst tier across markets, not the best: fees differ per market
        # on some venues and assuming the cheapest one is how a cost model
        # ends up flattering the strategy.
        maker = max(float(f["maker"]) for f in rates) * 10_000
        taker = max(float(f["taker"]) for f in rates) * 10_000
        register_fee_schedule(self.venue, maker_bps=maker, taker_bps=taker)
        logger.info(
            "Live fee tier for %s: maker %.1fbps, taker %.1fbps (round trip %.3f%%)",
            self.venue, maker, taker, (maker + taker) / 100,
        )

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def open_position(
        self,
        signal,
        equity_usd: float,
        mark_price: float,
        quantity_override: Optional[float] = None,
        notional_override: Optional[float] = None,
        contract_size: float = 1.0,
        fee_per_contract_usd: float = 0.0,
    ) -> Optional[Position]:
        """
        Attempt to open a position from a Signal. Returns the Position on a
        fill, or None if refused by risk limits or not filled.

        `quantity_override` bypasses this class's sizing for instruments whose
        size is not a free variable. A futures contract trades in whole units
        of a fixed size, so the caller has already had to solve "how many
        contracts fit in the budget" (see contracts.size_in_contracts) and
        re-deriving a fractional quantity here would only discard that answer.
        The risk limits below still apply to the resulting notional — the
        override changes who computes the size, not whether it is checked.
        """
        symbol = signal.symbol
        side = getattr(signal, "side", "long").lower()

        # Capability check first: a venue that cannot express this direction
        # must refuse before any sizing happens, so there is no code path where
        # a size exists for a trade that was never placeable.
        try:
            check_direction(self.spec, side)
        except DirectionUnsupported as exc:
            logger.error("Entry refused for %s — %s", symbol, exc)
            if self.notifier:
                self.notifier.send_risk_halt("Direction unsupported", str(exc))
            return None

        hold_hours = float(getattr(signal, "hold_hours", 0.0) or 0.0)

        if quantity_override is not None:
            quantity = quantity_override
            notional = (notional_override if notional_override is not None
                        else quantity * mark_price)
        elif signal.stop_loss and signal.stop_loss > 0:
            quantity, notional = self.risk.position_size(
                equity_usd, mark_price, signal.stop_loss, side=side,
            )
        else:
            # No stop: a fixed-hold signal. See RiskManager.notional_size for
            # why max_position_usd becomes the operative risk limit here.
            quantity, notional = self.risk.notional_size(equity_usd, mark_price)

        allowed, reason = self.risk.can_open(
            symbol, notional,
            open_count=self.store.open_count,
            already_open_symbol=self.store.has_open(symbol),
        )
        if not allowed:
            logger.info("Entry refused for %s — %s", symbol, reason)
            return None

        fill = self._place_entry(symbol, quantity, mark_price, side,
                                 contract_size, fee_per_contract_usd)
        if not fill.filled:
            logger.info("Entry not filled for %s — %s", symbol, fill.reason)
            return None

        entry_cost_pct = cost_model_for(
            symbol, venue=self.venue, entry_is_maker=self.post_only,
        ).leg_cost_pct(is_maker=self.post_only)

        position = Position(
            symbol=symbol,
            side=side,
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
            contract_size=contract_size,
            fee_per_contract_usd=fee_per_contract_usd,
        )
        if hold_hours > 0:
            position.set_hold_hours(hold_hours)

        self.store.add(position)
        self.risk.record_open()

        if self.live and position.stop_loss > 0:
            self._place_protective_stop(position)

        if self.notifier:
            self.notifier.send_entry_filled(position)

        logger.info(
            "OPENED %s %s %s qty=%.8g @ %.8g (notional $%.2f)%s",
            "LIVE" if self.live else "PAPER", side.upper(), symbol,
            fill.quantity, fill.price, notional,
            f" expires {position.expires_at}" if position.expires_at else "",
        )
        return position

    def _place_entry(
        self, symbol: str, quantity: float, mark_price: float, side: str = "long",
        contract_size: float = 1.0, fee_per_contract_usd: float = 0.0,
    ) -> Fill:
        if quantity <= 0:
            return Fill(filled=False, reason="zero quantity")

        order_side = _ORDER_SIDE[(side, "entry")]

        if not self.live:
            return self._simulate_entry(symbol, quantity, mark_price, side,
                                        contract_size, fee_per_contract_usd)

        try:
            params = {"postOnly": True} if self.post_only else {}
            order_type = "limit" if self.post_only else "market"
            quantity, limit_price = self._to_precision(symbol, quantity, mark_price)

            if self.post_only:
                order = self.client.create_order(
                    symbol, order_type, order_side, quantity, limit_price, params,
                )
                filled = self._await_fill(symbol, order["id"])
                if not filled:
                    self._cancel_quietly(symbol, order["id"])
                    return Fill(filled=False, reason="post-only entry timed out unfilled")
                order = filled
            else:
                order = self.client.create_order(symbol, "market", order_side, quantity)

            avg = float(order.get("average") or order.get("price") or mark_price)
            filled_qty = float(order.get("filled") or quantity)
            fee_usd = self._fee_from_order(order, avg, filled_qty)
            return Fill(True, avg, filled_qty, fee_usd, str(order.get("id", "")))

        except Exception as exc:
            logger.error("Entry order failed for %s: %s", symbol, exc)
            return Fill(filled=False, reason=str(exc))

    def _simulate_entry(
        self, symbol: str, quantity: float, mark_price: float, side: str = "long",
        contract_size: float = 1.0, fee_per_contract_usd: float = 0.0,
    ) -> Fill:
        """
        Paper fill. See PAPER_FILL_NOTE in the module docstring for why the
        post-only case is optimistic.
        """
        model = cost_model_for(symbol, venue=self.venue, entry_is_maker=self.post_only)
        cost_pct = model.leg_cost_pct(is_maker=self.post_only)

        # A taker entry crosses the spread, so it fills worse than the mark —
        # which for a short means selling *below* the mark, not above it. A
        # maker entry rests at the mark and pays no spread, but see the note
        # above about the fills it therefore misses.
        direction = -1 if side == "short" else 1
        fill_price = (
            mark_price if self.post_only
            else mark_price * (1 + direction * cost_pct / 100)
        )
        # Charge on true notional (contracts x contract size x price), or the
        # flat per-contract commission where the venue works that way.
        fee_usd = (
            quantity * fee_per_contract_usd if fee_per_contract_usd > 0
            else fill_price * quantity * contract_size * cost_pct / 100
        )

        return Fill(
            filled=True,
            price=fill_price,
            quantity=quantity,
            fee_usd=fee_usd,
            order_id=f"paper-{int(time.time()*1000)}",
        )

    def _to_precision(self, symbol: str, quantity: float, price: float) -> Tuple[float, float]:
        """
        Round size and price to what the venue will actually accept.

        Not cosmetic. CDE's BTC contract quotes a `price_increment` of 5, so a
        limit order at 77,382 is rejected outright — and a post-only entry is
        the ONLY order type this strategy uses to get in, meaning an unrounded
        price does not degrade the fill, it silently prevents every trade. The
        same applies to size: contracts have a `base_increment` of 1, and a
        fractional quantity is not a smaller position, it is an error.

        Falls back to the raw values if the venue has no precision data, since
        sending something is better than crashing on a missing field.
        """
        try:
            quantity = float(self.client.amount_to_precision(symbol, quantity))
            price = float(self.client.price_to_precision(symbol, price))
        except Exception as exc:
            logger.warning("Could not apply %s precision for %s (%s) — sending raw values",
                           self.venue, symbol, exc)
        return quantity, price

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
                position.symbol, "stop", _ORDER_SIDE[(position.side, "exit")],
                position.quantity, None,
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
            # Slippage always works against the trade: exiting a long is a
            # sell and you receive less than the trigger price; exiting a short
            # is a buy and you pay more. Multiplying by the direction keeps
            # both cases pessimistic rather than making one of them a bonus.
            return target_price * (1 - position.direction * slip_pct / 100)

        if position.stop_order_id and reason == "stop_loss":
            self._cancel_quietly(position.symbol, position.stop_order_id)

        exit_side = _ORDER_SIDE[(position.side, "exit")]
        quantity, _ = self._to_precision(
            position.symbol, position.remaining_quantity, target_price,
        )

        # reduceOnly is the safer request — it cannot accidentally open an
        # opposing position — but not every venue accepts it, and ccxt reports
        # createReduceOnlyOrder=False for Coinbase. A rejected exit would leave
        # a position open with nothing closing it, which is far worse than an
        # exit without the flag, so fall back rather than give up.
        for params in ({"reduceOnly": True}, {}):
            try:
                order = self.client.create_order(
                    position.symbol, "market", exit_side, quantity, None, params,
                )
                if not params:
                    logger.warning(
                        "%s exited without reduceOnly — the venue rejected it. "
                        "Verify the position is flat rather than reversed.",
                        position.symbol,
                    )
                return float(order.get("average") or order.get("price") or target_price)
            except Exception as exc:
                logger.error("Exit order failed for %s (params=%s): %s",
                             position.symbol, params or "none", exc)

        logger.error(
            "COULD NOT EXIT %s — both attempts failed. The position is STILL OPEN "
            "and must be closed manually.", position.symbol,
        )
        if self.notifier:
            self.notifier.send_risk_halt(
                "Exit failed",
                f"{position.symbol} could not be flattened. It is still open.",
            )
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
