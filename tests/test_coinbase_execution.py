"""
Tests for Coinbase routing, short-side economics, and the pre-trade edge gate.

The short side is new, and it is the kind of change where a sign error does not
crash — it produces a plausible-looking number that points the wrong way. So
most of what is asserted here is signs: that a short profits when price falls,
that its stop is above its entry, that slippage costs it on both legs, and that
a venue which cannot short refuses rather than inverting the trade.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.costs import (
    FEE_SCHEDULES,
    register_fee_schedule,
    resolve_fee_schedule,
)
from src.execution.edge_guard import EdgeGuard, measured_edge
from src.execution.executor import _ORDER_SIDE, Executor
from src.execution.monitor import Mark, PositionMonitor
from src.execution.position import Position, PositionStore
from src.execution.risk import RiskLimits, RiskManager
from src.execution.venues import (
    DirectionUnsupported,
    Credentials,
    ExchangeUnavailable,
    resolve_ccxt_id,
    ccxt_options,
    check_direction,
    repair_pem,
    venue_spec,
)


# ---------------------------------------------------------------------------
# Venue registry
# ---------------------------------------------------------------------------

def test_coinbase_spot_cannot_short():
    with pytest.raises(DirectionUnsupported) as exc:
        check_direction(venue_spec("coinbaseadvanced"), "short")
    # The message must say why inverting is not a fallback, since that is the
    # workaround someone would otherwise reach for.
    assert "-0.240%" in str(exc.value)


def test_coinbase_perps_can_short():
    check_direction(venue_spec("coinbaseinternational"), "short")


def test_every_venue_allows_long():
    for name in ("coinbaseadvanced", "coinbaseexchange", "coinbaseinternational", "okx"):
        check_direction(venue_spec(name), "long")


def test_coinbase_venues_declare_a_ccxt_fallback():
    """
    ccxt renames exchange classes between releases — 4.5.56 has
    `coinbaseadvanced`, 4.5.77 does not — and requirements.txt pins only
    `ccxt>=4.3.0`. A CI runner therefore resolves a different set of ids than a
    developer machine, which is invisible locally and fatal in CI.
    """
    for name in ("coinbaseadvanced", "coinbasederivatives"):
        assert "coinbase" in venue_spec(name).ccxt_id_fallbacks


def test_ccxt_id_resolves_to_something_installed():
    import ccxt
    for name in ("coinbaseadvanced", "coinbasederivatives", "coinbaseinternational"):
        assert hasattr(ccxt, resolve_ccxt_id(venue_spec(name)))


def test_a_venue_with_no_available_class_raises_a_readable_error():
    from dataclasses import replace
    spec = replace(venue_spec("coinbasederivatives"),
                   ccxt_id="nonesuch", ccxt_id_fallbacks=("alsomissing",))
    with pytest.raises(ExchangeUnavailable) as exc:
        resolve_ccxt_id(spec)
    # Must name every id it tried, not just the first.
    assert "nonesuch" in str(exc.value) and "alsomissing" in str(exc.value)


def test_symbol_mapping_is_product_specific():
    assert venue_spec("coinbaseadvanced").symbol_for("BTC") == "BTC/USD"
    assert venue_spec("coinbaseinternational").symbol_for("BTC") == "BTC/USDC:USDC"
    # Tolerates being handed a full pair from the positioning source.
    assert venue_spec("coinbaseinternational").symbol_for("BTC/USDT") == "BTC/USDC:USDC"


def test_pem_newlines_are_repaired():
    mangled = "-----BEGIN EC PRIVATE KEY-----\\nMHc=\\n-----END EC PRIVATE KEY-----"
    assert "\n" in repair_pem(mangled)
    assert "\\n" not in repair_pem(mangled)


def test_pem_repair_leaves_hmac_secrets_alone():
    assert repair_pem("abc\\ndef") == "abc\\ndef"


def test_cdp_key_gets_no_passphrase():
    """A CDP key carries no passphrase; passing one triggers legacy signing."""
    creds = Credentials(api_key="organizations/x/apiKeys/y", secret="-----BEGIN EC-----",
                        passphrase="leftover")
    assert "password" not in ccxt_options(venue_spec("coinbaseadvanced"), creds)


def test_legacy_key_keeps_its_passphrase():
    creds = Credentials(api_key="plainkey", secret="plainsecret", passphrase="pass")
    assert ccxt_options(venue_spec("coinbaseexchange"), creds)["password"] == "pass"


def test_perp_venue_requests_swap_market_type():
    creds = Credentials(api_key="k", secret="s")
    options = ccxt_options(venue_spec("coinbaseinternational"), creds)
    assert options["options"]["defaultType"] == "swap"


# ---------------------------------------------------------------------------
# Order sides
# ---------------------------------------------------------------------------

def test_order_side_table_is_complete_and_opposed():
    assert _ORDER_SIDE[("long", "entry")] == "buy"
    assert _ORDER_SIDE[("long", "exit")] == "sell"
    assert _ORDER_SIDE[("short", "entry")] == "sell"
    assert _ORDER_SIDE[("short", "exit")] == "buy"
    for side in ("long", "short"):
        assert _ORDER_SIDE[(side, "entry")] != _ORDER_SIDE[(side, "exit")]


# ---------------------------------------------------------------------------
# Short-side P&L
# ---------------------------------------------------------------------------

def short_position(**overrides) -> Position:
    defaults = dict(
        symbol="BTC/USDC:USDC",
        side="short",
        entry_price=100.0,
        quantity=1.0,
        stop_loss=0.0,
        take_profit_1=0.0,
        take_profit_2=0.0,
    )
    defaults.update(overrides)
    return Position(**defaults)


def test_short_profits_when_price_falls():
    pos = short_position()
    assert pos.unrealised_pnl_pct(90.0) == pytest.approx(10.0)
    assert pos.unrealised_pnl_usd(90.0) == pytest.approx(10.0)


def test_short_loses_when_price_rises():
    pos = short_position()
    assert pos.unrealised_pnl_pct(110.0) == pytest.approx(-10.0)
    assert pos.unrealised_pnl_usd(110.0) == pytest.approx(-10.0)


def test_long_pnl_is_unchanged_by_the_direction_refactor():
    pos = short_position(side="long")
    assert pos.unrealised_pnl_pct(110.0) == pytest.approx(10.0)
    assert pos.unrealised_pnl_usd(90.0) == pytest.approx(-10.0)


def test_short_realised_pnl_is_net_of_fees():
    pos = short_position()
    net = pos.close(price=90.0, reason="expiry", fee_usd=0.5)
    assert net == pytest.approx(9.5)
    assert pos.total_pnl_pct() == pytest.approx(9.5)


def test_short_partial_exit_reduces_remaining_quantity():
    pos = short_position(quantity=2.0)
    pos.record_partial_exit(price=95.0, quantity=1.0, fee_usd=0.0)
    assert pos.remaining_quantity == pytest.approx(1.0)
    assert pos.realised_pnl_usd == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Time-based exit
# ---------------------------------------------------------------------------

def test_hold_deadline_is_stamped_from_the_open_time():
    pos = short_position()
    pos.set_hold_hours(16)
    opened = datetime.fromisoformat(pos.opened_at)
    assert datetime.fromisoformat(pos.expires_at) - opened == timedelta(hours=16)


def test_position_without_a_deadline_never_expires():
    assert short_position().is_expired() is False


def test_expiry_is_detected_after_the_horizon():
    pos = short_position()
    pos.set_hold_hours(16)
    later = datetime.fromisoformat(pos.expires_at) + timedelta(minutes=1)
    assert pos.is_expired(now=later) is True
    earlier = datetime.fromisoformat(pos.expires_at) - timedelta(minutes=1)
    assert pos.is_expired(now=earlier) is False


def test_unparseable_deadline_does_not_close_the_position():
    pos = short_position(expires_at="not a timestamp")
    assert pos.is_expired() is False


# ---------------------------------------------------------------------------
# Monitor: side-aware triggers
# ---------------------------------------------------------------------------

def monitor_for(tmp_path, position: Position) -> tuple:
    store = PositionStore(path=str(tmp_path / "positions.json"))
    store.add(position)
    return store, PositionMonitor(store=store, venue="coinbaseinternational")


def test_mark_extremes_flip_with_direction():
    mark = Mark(last=100.0, low=95.0, high=105.0)
    assert mark.adverse(1) == 95.0 and mark.favourable(1) == 105.0
    assert mark.adverse(-1) == 105.0 and mark.favourable(-1) == 95.0


def test_short_stop_triggers_on_the_high(tmp_path):
    pos = short_position(stop_loss=110.0)
    store, monitor = monitor_for(tmp_path, pos)

    # A low that would stop a long must not stop a short.
    assert monitor.check({pos.symbol: Mark(last=100.0, low=80.0, high=101.0)}) == []
    events = monitor.check({pos.symbol: Mark(last=108.0, low=100.0, high=112.0)})
    assert [e.kind for e in events] == ["stop"]
    assert store.open_count == 0


def test_short_target_triggers_on_the_low(tmp_path):
    pos = short_position(take_profit_2=90.0)
    store, monitor = monitor_for(tmp_path, pos)
    events = monitor.check({pos.symbol: Mark(last=92.0, low=89.0, high=95.0)})
    assert [e.kind for e in events] == ["tp2"]
    assert events[0].pnl_pct > 0


def test_a_stopless_short_is_not_closed_instantly(tmp_path):
    """A stop of zero means 'no stop' — every price is above zero."""
    pos = short_position(stop_loss=0.0, take_profit_1=0.0, take_profit_2=0.0)
    store, monitor = monitor_for(tmp_path, pos)
    assert monitor.check({pos.symbol: Mark(last=100.0, low=1.0, high=1e9)}) == []
    assert store.open_count == 1


def test_expiry_closes_the_position(tmp_path):
    pos = short_position()
    pos.expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store, monitor = monitor_for(tmp_path, pos)

    events = monitor.check({pos.symbol: Mark(last=95.0)})
    assert [e.kind for e in events] == ["expiry"]
    assert store.open_count == 0
    assert events[0].pnl_pct > 0          # shorted at 100, covered at 95


def test_expiry_beats_a_target_reached_in_the_same_interval(tmp_path):
    """The study measured return at the horizon, not at a target."""
    pos = short_position(take_profit_2=90.0)
    pos.expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store, monitor = monitor_for(tmp_path, pos)

    events = monitor.check({pos.symbol: Mark(last=95.0, low=85.0, high=96.0)})
    assert [e.kind for e in events] == ["expiry"]


def test_a_breached_stop_still_beats_expiry(tmp_path):
    """Losses are resolved against the trade; that ordering does not change."""
    pos = short_position(stop_loss=110.0)
    pos.expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store, monitor = monitor_for(tmp_path, pos)

    events = monitor.check({pos.symbol: Mark(last=108.0, low=100.0, high=115.0)})
    assert [e.kind for e in events] == ["stop"]


def test_expired_position_without_a_mark_is_still_closed(tmp_path):
    pos = short_position()
    pos.expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store, monitor = monitor_for(tmp_path, pos)

    events = monitor.check({})
    assert [e.kind for e in events] == ["expiry"]
    assert store.open_count == 0


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def risk_manager(tmp_path) -> RiskManager:
    return RiskManager(
        limits=RiskLimits(max_position_usd=500.0, risk_per_trade_pct=0.01),
        state_path=str(tmp_path / "risk.json"),
    )


def test_short_sizing_uses_a_stop_above_entry(tmp_path):
    """$100 of risk over a $10 stop distance is 10 units — the same arithmetic
    a long would get from a stop the same distance below its entry."""
    risk = risk_manager(tmp_path)
    risk.limits.max_position_usd = 100_000     # out of the way for this check

    short_qty, _ = risk.position_size(10_000, 100.0, 110.0, side="short")
    long_qty, _ = risk.position_size(10_000, 100.0, 90.0, side="long")
    assert short_qty == pytest.approx(10.0)
    assert short_qty == pytest.approx(long_qty)


def test_short_sizing_is_clamped_by_the_position_cap(tmp_path):
    risk = risk_manager(tmp_path)               # max_position_usd = 500
    qty, notional = risk.position_size(10_000, 100.0, 110.0, side="short")
    assert notional == pytest.approx(500.0)
    assert qty == pytest.approx(5.0)


def test_short_sizing_rejects_a_stop_below_entry(tmp_path):
    """A stop on the wrong side means the caller has the direction confused."""
    risk = risk_manager(tmp_path)
    assert risk.position_size(10_000, 100.0, 90.0, side="short") == (0.0, 0.0)


def test_long_sizing_rejects_a_stop_above_entry(tmp_path):
    risk = risk_manager(tmp_path)
    assert risk.position_size(10_000, 100.0, 110.0, side="long") == (0.0, 0.0)


def test_stopless_sizing_is_capped_by_notional(tmp_path):
    risk = risk_manager(tmp_path)
    qty, notional = risk.notional_size(10_000, entry_price=100.0)
    assert notional == pytest.approx(100.0)     # 1% of equity, under the $500 cap
    assert qty == pytest.approx(1.0)


def test_stopless_sizing_respects_the_position_cap(tmp_path):
    risk = risk_manager(tmp_path)
    _, notional = risk.notional_size(1_000_000, entry_price=100.0)
    assert notional == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def test_coinbase_products_have_distinct_fee_schedules():
    for name in ("coinbaseadvanced", "coinbaseexchange", "coinbaseinternational"):
        assert name in FEE_SCHEDULES
    assert (FEE_SCHEDULES["coinbaseadvanced"].taker_bps
            > FEE_SCHEDULES["coinbaseinternational"].taker_bps)


def test_a_measured_fee_tier_overrides_the_table():
    original = resolve_fee_schedule("coinbaseadvanced")
    try:
        register_fee_schedule("coinbaseadvanced", maker_bps=5.0, taker_bps=10.0)
        assert resolve_fee_schedule("coinbaseadvanced").taker_bps == 10.0
    finally:
        register_fee_schedule("coinbaseadvanced", original.maker_bps, original.taker_bps)


# ---------------------------------------------------------------------------
# Edge guard
# ---------------------------------------------------------------------------

def test_coinbase_spot_fees_exceed_the_measured_edge():
    verdict = EdgeGuard(venue="coinbaseadvanced").evaluate("BTC/USD", "crowd_short")
    assert verdict.allowed is False
    assert verdict.round_trip_cost_pct > measured_edge("crowd_short").gross_pct_per_trade


def test_coinbase_perp_fees_leave_the_edge_intact():
    verdict = EdgeGuard(venue="coinbaseinternational").evaluate(
        "BTC/USDC:USDC", "crowd_short", hold_hours=16,
    )
    assert verdict.allowed is True
    assert verdict.net_edge_pct > 0.5


def test_an_unmeasured_signal_type_is_refused():
    verdict = EdgeGuard(venue="coinbaseinternational").evaluate("BTC/USDC:USDC", "hunch")
    assert verdict.allowed is False
    assert "no out-of-sample edge" in verdict.reason


def test_the_legacy_wall_signal_is_refused_as_insignificant():
    verdict = EdgeGuard(venue="coinbaseinternational").evaluate(
        "BTC/USDC:USDC", "ob_wall_bid_repulsion",
    )
    assert verdict.allowed is False
    assert "zero" in verdict.reason


def test_negative_funding_is_charged_and_positive_funding_is_not_credited():
    guard = EdgeGuard(venue="coinbaseinternational")
    charged = guard.evaluate("BTC/USDC:USDC", "crowd_short",
                             hold_hours=16, funding_rate_pct_per_hour=-0.01)
    assert charged.funding_pct == pytest.approx(0.16)

    credited = guard.evaluate("BTC/USDC:USDC", "crowd_short",
                              hold_hours=16, funding_rate_pct_per_hour=+0.01)
    assert credited.funding_pct == 0.0


def test_safety_margin_is_enforced():
    edge = measured_edge("crowd_short").gross_pct_per_trade
    strict = EdgeGuard(venue="coinbaseinternational", safety_margin_pct=edge + 1)
    assert strict.evaluate("BTC/USDC:USDC", "crowd_short").allowed is False


# ---------------------------------------------------------------------------
# Executor refusals (paper mode — no network)
# ---------------------------------------------------------------------------

class _Order:
    symbol = "BTC/USD"
    signal_type = "crowd_short"
    side = "short"
    hold_hours = 16
    stop_loss = 0.0
    take_profit_1 = 0.0
    take_profit_2 = 0.0


def executor_for(tmp_path, venue: str) -> Executor:
    return Executor(
        store=PositionStore(path=str(tmp_path / "positions.json")),
        risk=risk_manager(tmp_path),
        venue=venue,
        live=False,
    )


def test_executor_refuses_a_short_on_a_spot_venue(tmp_path):
    executor = executor_for(tmp_path, "coinbaseadvanced")
    assert executor.open_position(_Order(), equity_usd=10_000, mark_price=100.0) is None
    assert executor.store.open_count == 0


def test_executor_opens_a_paper_short_on_perps(tmp_path):
    executor = executor_for(tmp_path, "coinbaseinternational")
    order = _Order()
    order.symbol = "BTC/USDC:USDC"

    position = executor.open_position(order, equity_usd=10_000, mark_price=100.0)
    assert position is not None
    assert position.side == "short"
    assert position.expires_at            # the 16h horizon was stamped
    assert position.quantity > 0


def test_a_taker_short_entry_fills_below_the_mark(tmp_path):
    """Crossing the spread to sell means receiving less, not more."""
    executor = executor_for(tmp_path, "coinbaseinternational")
    executor.post_only = False
    fill = executor._simulate_entry("BTC/USDC:USDC", 1.0, 100.0, side="short")
    assert fill.price < 100.0

    long_fill = executor._simulate_entry("BTC/USDC:USDC", 1.0, 100.0, side="long")
    assert long_fill.price > 100.0


# ---------------------------------------------------------------------------
# Mark extraction
# ---------------------------------------------------------------------------

def price_from(ticker: dict):
    from scripts.run_coinbase_trader import MarkFeed
    return MarkFeed._price_from(ticker)


def test_mark_prefers_last_price():
    assert price_from({"last": 100.0, "bid": 90.0, "ask": 92.0}) == 100.0


def test_mark_falls_back_to_the_venue_field_when_last_is_empty():
    """ccxt's coinbaseinternational parser returns last/close as None."""
    ticker = {"last": None, "close": None, "info": {"trade_price": "77262"}}
    assert price_from(ticker) == pytest.approx(77262.0)


def test_mark_falls_back_to_the_bid_ask_midpoint():
    ticker = {"last": None, "close": None, "info": {}, "bid": 100.0, "ask": 102.0}
    assert price_from(ticker) == pytest.approx(101.0)


def test_mark_reports_nothing_rather_than_guessing():
    assert price_from({"last": None, "close": None, "info": {}}) is None


# ---------------------------------------------------------------------------
# Order precision — a rejected order is not a smaller order
# ---------------------------------------------------------------------------

class _PrecisionClient:
    """Stands in for ccxt, applying CDE-like increments."""
    def amount_to_precision(self, symbol, amount):
        return str(int(amount))                       # whole contracts
    def price_to_precision(self, symbol, price):
        return str(int(price // 5) * 5)               # $5 tick


def test_limit_price_is_rounded_to_the_venue_tick(tmp_path):
    """CDE quotes BTC in $5 increments; an unrounded limit is rejected outright,
    and post-only is the only way this strategy enters."""
    executor = executor_for(tmp_path, "coinbasederivatives")
    executor.client = _PrecisionClient()
    qty, price = executor._to_precision("BTC/USD:USD-301220", 1.0, 77_382.37)
    assert price == 77_380.0
    assert price % 5 == 0


def test_quantity_is_rounded_to_whole_contracts(tmp_path):
    executor = executor_for(tmp_path, "coinbasederivatives")
    executor.client = _PrecisionClient()
    qty, _ = executor._to_precision("BTC/USD:USD-301220", 3.4, 77_380.0)
    assert qty == 3.0


def test_precision_failure_falls_back_to_raw_values(tmp_path):
    """Sending something beats crashing on a missing precision field."""
    class Broken:
        def amount_to_precision(self, *a): raise ValueError("no precision data")
        def price_to_precision(self, *a): raise ValueError("no precision data")

    executor = executor_for(tmp_path, "coinbasederivatives")
    executor.client = Broken()
    assert executor._to_precision("X/Y", 1.5, 99.9) == (1.5, 99.9)


def test_exit_retries_without_reduce_only(tmp_path):
    """
    ccxt reports createReduceOnlyOrder=False for Coinbase. A rejected exit
    would strand an open position, so the fallback matters more than the flag.
    """
    attempts = []

    class PickyClient(_PrecisionClient):
        def create_order(self, symbol, type_, side, amount, price=None, params=None):
            attempts.append(dict(params or {}))
            if params and params.get("reduceOnly"):
                raise ccxt_error("reduceOnly is not supported")
            return {"average": 76_500.0}

    def ccxt_error(msg):
        return ValueError(msg)

    executor = executor_for(tmp_path, "coinbasederivatives")
    executor.live = True
    executor.client = PickyClient()

    position = short_position(symbol="BTC/USD:USD-301220", quantity=1.0)
    price = executor.exit_fill(position, 76_600.0, "expiry")

    assert price == 76_500.0
    assert attempts == [{"reduceOnly": True}, {}]     # tried, then fell back


def test_exit_returning_target_price_when_both_attempts_fail(tmp_path):
    class DeadClient(_PrecisionClient):
        def create_order(self, *a, **k):
            raise ValueError("venue down")

    executor = executor_for(tmp_path, "coinbasederivatives")
    executor.live = True
    executor.client = DeadClient()

    position = short_position(symbol="BTC/USD:USD-301220", quantity=1.0)
    assert executor.exit_fill(position, 76_600.0, "expiry") == 76_600.0


def test_paper_exit_slippage_hurts_both_directions(tmp_path):
    executor = executor_for(tmp_path, "coinbaseinternational")
    short = short_position(symbol="BTC/USDC:USDC")
    long_ = short_position(symbol="BTC/USDC:USDC", side="long")

    # Covering a short is a buy: you pay above the trigger.
    assert executor.exit_fill(short, 100.0, "expiry") > 100.0
    # Selling a long: you receive below the trigger.
    assert executor.exit_fill(long_, 100.0, "expiry") < 100.0
