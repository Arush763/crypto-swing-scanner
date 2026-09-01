"""
Tests for CDE futures contract mechanics.

The bug these exist to prevent already happened once during development: a
position of one nano BTC contract (0.01 BTC) was booked as one whole BTC, which
overstated notional, P&L and fees by 100x — in the direction that makes a
mediocre strategy look extraordinary. Nothing threw; the numbers were simply
wrong. Most of what is asserted here is scale.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.contracts import (
    ContractSpec,
    contract_fee_usd,
    exit_falls_in_break,
    fee_is_measured,
    register_contract_fee,
    session_allows_entry,
    session_from_market,
    size_in_contracts,
    spec_from_market,
)
from src.execution.position import Position

VENUE = "test_cde"


def nano_btc(**overrides) -> ContractSpec:
    defaults = dict(
        symbol="BTC/USD:USD-301220",
        contract_id="BIP-20DEC30-CDE",
        base="BTC",
        contract_size=0.01,
        min_contracts=1,
        max_contracts=5000,
        display_name="BTC PERP",
    )
    defaults.update(overrides)
    return ContractSpec(**defaults)


@pytest.fixture(autouse=True)
def _clean_fee():
    register_contract_fee(VENUE, 1.00)
    yield


# ---------------------------------------------------------------------------
# Notional and fee conversion
# ---------------------------------------------------------------------------

def test_one_nano_contract_is_a_hundredth_of_the_underlying():
    assert nano_btc().notional_usd(80_000) == pytest.approx(800.0)


def test_flat_fee_becomes_a_smaller_percentage_on_a_bigger_contract():
    """The counter-intuitive core of per-contract pricing."""
    big = nano_btc().fee_pct_per_side(80_000, VENUE)          # $800 notional
    small = nano_btc(contract_size=0.001).fee_pct_per_side(80_000, VENUE)  # $80
    assert small > big
    assert big == pytest.approx(0.125)     # $1 / $800
    assert small == pytest.approx(1.25)    # $1 / $80


def test_round_trip_is_two_sides():
    spec = nano_btc()
    assert spec.round_trip_fee_pct(80_000, VENUE) == pytest.approx(
        2 * spec.fee_pct_per_side(80_000, VENUE))


def test_a_registered_fee_replaces_the_placeholder():
    register_contract_fee(VENUE, 0.25)
    assert contract_fee_usd(VENUE) == 0.25
    assert fee_is_measured(VENUE) is True
    assert fee_is_measured("a_venue_never_registered") is False


def test_spec_is_read_from_a_ccxt_market():
    market = {
        "id": "BIP-20DEC30-CDE", "base": "BTC", "contractSize": 0.01,
        "limits": {"amount": {"min": 1, "max": 5000}},
        "expiryDatetime": "2030-12-20T16:00:00.000Z",
        "info": {"display_name": "BTC PERP", "future_product_details": {}},
    }
    spec = spec_from_market("BTC/USD:USD-301220", market)
    assert spec.contract_size == 0.01
    assert spec.min_contracts == 1
    assert spec.display_name == "BTC PERP"


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def test_sizing_rounds_down_to_whole_contracts():
    sizing = size_in_contracts(nano_btc(), 80_000, max_notional_usd=2_500, venue=VENUE)
    assert sizing.contracts == 3           # 3 x $800 = $2400, 4 would be $3200
    assert sizing.notional_usd == pytest.approx(2_400.0)
    assert sizing.base_quantity == pytest.approx(0.03)


def test_sizing_refuses_when_one_contract_exceeds_the_cap():
    """Never silently take 54% more risk than authorised."""
    sizing = size_in_contracts(nano_btc(), 80_000, max_notional_usd=500, venue=VENUE)
    assert sizing.tradeable is False
    assert sizing.contracts == 0
    assert "raise MAX_POSITION_USD" in sizing.reason


def test_overshoot_is_possible_but_must_be_asked_for():
    sizing = size_in_contracts(nano_btc(), 80_000, max_notional_usd=500,
                               venue=VENUE, allow_overshoot=True)
    assert sizing.contracts == 1
    assert sizing.notional_usd == pytest.approx(800.0)


def test_sizing_respects_the_venue_maximum():
    sizing = size_in_contracts(nano_btc(max_contracts=2), 80_000,
                               max_notional_usd=1_000_000, venue=VENUE)
    assert sizing.contracts == 2


def test_sizing_needs_a_price():
    assert size_in_contracts(nano_btc(), 0, 1000, VENUE).tradeable is False


# ---------------------------------------------------------------------------
# Position scale — the 100x bug
# ---------------------------------------------------------------------------

def contract_position(**overrides) -> Position:
    defaults = dict(
        symbol="BTC/USD:USD-301220", side="short",
        entry_price=80_000.0, quantity=1.0,
        stop_loss=0.0, take_profit_1=0.0, take_profit_2=0.0,
        contract_size=0.01, fee_per_contract_usd=1.00,
    )
    defaults.update(overrides)
    return Position(**defaults)


def test_notional_uses_contract_size():
    """One nano contract is $800 of exposure, not $80,000."""
    assert contract_position().notional_usd == pytest.approx(800.0)


def test_base_quantity_is_contracts_times_size():
    assert contract_position(quantity=3).base_quantity == pytest.approx(0.03)


def test_unrealised_pnl_is_scaled_to_the_contract():
    pos = contract_position()                       # short 1 contract @ 80,000
    assert pos.unrealised_pnl_usd(79_200) == pytest.approx(8.0)   # 1% of $800
    assert pos.unrealised_pnl_pct(79_200) == pytest.approx(1.0)


def test_realised_pnl_is_scaled_to_the_contract():
    pos = contract_position()
    net = pos.close(price=79_200, reason="expiry", fee_usd=1.0)
    assert net == pytest.approx(7.0)                # $8 gross - $1 exit fee


def test_a_spot_position_is_unaffected_by_the_contract_fields():
    """contract_size defaults to 1.0, so existing behaviour is unchanged."""
    pos = Position(symbol="BTC/USD", side="long", entry_price=100.0, quantity=2.0,
                   stop_loss=0.0, take_profit_1=0.0, take_profit_2=0.0)
    assert pos.contract_size == 1.0
    assert pos.notional_usd == pytest.approx(200.0)
    assert pos.unrealised_pnl_usd(110.0) == pytest.approx(20.0)


def test_flat_fee_is_charged_per_contract_not_per_dollar():
    pos = contract_position(quantity=3)
    # Percentage argument is ignored when a flat per-contract fee is set.
    assert pos.leg_fee_usd(80_000, 3, fee_pct=99.0) == pytest.approx(3.0)


def test_percentage_fee_applies_when_no_flat_fee_is_set():
    pos = contract_position(fee_per_contract_usd=0.0)
    # 1 contract, 0.01 BTC @ 80,000 = $800 notional, 0.5% = $4
    assert pos.leg_fee_usd(80_000, 1, fee_pct=0.5) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def market_with_session(**overrides) -> dict:
    details = dict(is_session_open=True, session_state="FCM_TRADING_SESSION_STATE_OPEN",
                   open_time="2026-08-31T21:00:00Z", close_time="2026-09-01T21:00:00Z",
                   closed_reason="FCM_TRADING_SESSION_CLOSED_REASON_UNDEFINED")
    details.update(overrides)
    return {"info": {"fcm_trading_session_details": details}}


def test_open_session_allows_entry():
    session = session_from_market(market_with_session())
    allowed, reason = session_allows_entry(session, hold_hours=16)
    assert allowed is True and reason == ""


def test_closed_session_blocks_entry():
    session = session_from_market(market_with_session(
        is_session_open=False, session_state="FCM_TRADING_SESSION_STATE_CLOSED"))
    allowed, reason = session_allows_entry(session, hold_hours=16)
    assert allowed is False
    assert "closed" in reason


def test_a_long_hold_does_not_block_entry_by_itself():
    """
    Sessions run 24h with a short break; refusing every entry within 16h of the
    close would reject most signals for no real reason.
    """
    session = session_from_market(market_with_session())
    assert session_allows_entry(session, hold_hours=16)[0] is True
    assert session_allows_entry(session, hold_hours=200)[0] is True


def test_a_market_without_a_session_block_trades_continuously():
    session = session_from_market({"info": {}})
    assert session.is_open is True
    assert session_allows_entry(session, hold_hours=16)[0] is True


def test_exit_landing_in_the_break_is_flagged():
    close = datetime.now(timezone.utc) + timedelta(hours=10)
    session = session_from_market(market_with_session(
        close_time=close.isoformat().replace("+00:00", "Z")))
    # A 10.5h hold lands half an hour into the one-hour break.
    delayed, note = exit_falls_in_break(session, hold_hours=10.5)
    assert delayed is True
    assert "delayed" in note


def test_exit_clear_of_the_break_is_not_flagged():
    close = datetime.now(timezone.utc) + timedelta(hours=10)
    session = session_from_market(market_with_session(
        close_time=close.isoformat().replace("+00:00", "Z")))
    assert exit_falls_in_break(session, hold_hours=5)[0] is False    # before
    assert exit_falls_in_break(session, hold_hours=12)[0] is False   # after


def test_unparseable_close_time_is_not_treated_as_a_break():
    session = session_from_market(market_with_session(close_time="soon"))
    assert exit_falls_in_break(session, hold_hours=16)[0] is False
