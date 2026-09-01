"""
Coinbase preflight — everything that should be checked before the first order.

Read-only. Places no orders, cancels nothing, and can be run against a live
account safely. It answers, in order, the five questions that decide whether
automated trading into Coinbase can work at all:

  1. Does ccxt have this venue, and do the credentials authenticate?
  2. Can this venue hold the direction the strategy needs?  (spot: no)
  3. Which of the strategy's symbols actually list here?
  4. What fee tier is this account really on?
  5. At that tier, does the measured edge survive the round trip?

Question 5 is the one that matters and the one usually skipped. The strategy
this repo validated has a gross edge of +0.655%/trade. Coinbase's retail spot
tier can charge more than that in fees alone, which means the answer to "should
this be automated into Coinbase spot" is arithmetic, not opinion — and the
arithmetic is printed below rather than described.

Usage
-----
    python scripts/coinbase_preflight.py                          # default venue
    python scripts/coinbase_preflight.py --venue coinbaseadvanced
    python scripts/coinbase_preflight.py --venue coinbaseinternational
    python scripts/coinbase_preflight.py --public-only            # skip auth

Exit code is 0 only if the venue is usable for the strategy end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config.config import COINBASE_VENUE, MAJOR_BASES
from src.execution.costs import register_fee_schedule, resolve_fee_schedule
from src.execution.edge_guard import EdgeGuard, measured_edge
from src.execution.contracts import (
    contract_fee_usd,
    fee_is_measured,
    register_contract_fee,
    session_allows_entry,
    session_from_market,
    spec_from_market,
)
from src.execution.venues import (
    VENUES,
    ccxt_options,
    load_credentials,
    resolve_contract,
    venue_spec,
)

OK = "  OK  "
WARN = " WARN "
FAIL = " FAIL "


class Report:
    """Collects check results so the summary can be a verdict, not a scroll."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def line(self, status: str, message: str) -> None:
        print(f"[{status}] {message}")
        if status == FAIL:
            self.failures.append(message)
        elif status == WARN:
            self.warnings.append(message)

    def section(self, title: str) -> None:
        print(f"\n{title}\n{'-' * len(title)}")


def check_venue(report: Report, venue: str):
    report.section(f"1. Venue — {venue}")
    try:
        spec = venue_spec(venue)
    except KeyError as exc:
        report.line(FAIL, str(exc))
        return None

    report.line(OK, f"{spec.description}")
    print(f"       product={spec.product}  quote={spec.quote}  "
          f"sandbox={'yes' if spec.has_sandbox else 'no'}")
    if spec.notes:
        print(f"       note: {spec.notes}")

    try:
        import ccxt
    except ImportError:
        report.line(FAIL, "ccxt is not installed — pip install -r requirements.txt")
        return None

    if not hasattr(ccxt, spec.ccxt_id):
        report.line(FAIL, f"ccxt {ccxt.__version__} has no exchange {spec.ccxt_id!r}")
        return None
    report.line(OK, f"ccxt {ccxt.__version__} provides {spec.ccxt_id}")

    if not spec.us_available:
        report.line(WARN, f"{venue} is not available to US persons — confirm eligibility "
                          f"before funding an account")
    return spec


def check_direction_support(report: Report, spec) -> None:
    report.section("2. Direction — can this venue hold the strategy's side?")
    edge = measured_edge("crowd_short")
    print(f"       The validated strategy is SHORT-ONLY "
          f"(+{edge.gross_pct_per_trade:.3f}%/trade gross, t={edge.t_stat}).")
    if spec.can_short:
        report.line(OK, f"{spec.name} supports short positions ({spec.product})")
    else:
        report.line(
            FAIL,
            f"{spec.name} is {spec.product} and cannot short. The strategy is not "
            f"expressible here — its mirror long trade measured -0.240%/trade, so "
            f"inverting it is not a workaround.",
        )


def connect(report: Report, spec, public_only: bool):
    report.section("3. Credentials and connectivity")
    import ccxt

    creds = load_credentials(spec)
    if public_only:
        report.line(WARN, "--public-only: skipping authentication")
        client = getattr(ccxt, spec.ccxt_id)({"enableRateLimit": True})
    elif not creds.present:
        report.line(
            FAIL,
            "credentials missing — set " + " and ".join(creds.missing(spec)),
        )
        print("       See docs/COINBASE.md for how to mint a CDP key.")
        return None
    else:
        kind = "CDP/Cloud key" if creds.is_cdp_key else "legacy HMAC key"
        report.line(OK, f"credentials present ({kind})")
        if creds.is_cdp_key and "BEGIN" in creds.secret and "\n" not in creds.secret:
            report.line(FAIL, "the PEM secret has no newlines — it will not parse")
        client = getattr(ccxt, spec.ccxt_id)(ccxt_options(spec, creds))

    try:
        markets = client.load_markets()
        report.line(OK, f"loaded {len(markets)} markets")
    except Exception as exc:
        report.line(FAIL, f"could not load markets: {exc}")
        return None

    if not public_only:
        try:
            balance = client.fetch_balance()
            free = balance.get("free", {})
            cash = {c: v for c, v in free.items()
                    if c in ("USD", "USDC", "USDT") and v}
            report.line(OK, f"authenticated — free cash: {cash or 'none'}")
        except Exception as exc:
            report.line(FAIL, f"authentication failed: {exc}")
            return None

    return client


def resolve_symbols(spec, client, limit: int) -> tuple:
    """Map each major to its venue symbol; dated futures resolve dynamically."""
    import time as _time
    available, missing = [], []
    for base in MAJOR_BASES[:limit]:
        if spec.dynamic_symbols:
            symbol = resolve_contract(client, base, venue=spec.name,
                                      now_ms=int(_time.time() * 1000))
            (available if symbol else missing).append(symbol or base)
        else:
            symbol = spec.symbol_for(base)
            (available if symbol in client.markets else missing).append(symbol)
    return available, missing


def check_symbols(report: Report, spec, client, limit: int) -> list[str]:
    report.section("4. Symbol coverage")
    available, missing = resolve_symbols(spec, client, limit)

    report.line(
        OK if len(available) >= 8 else WARN,
        f"{len(available)}/{limit} of the strategy's symbols list here",
    )
    print("       available:", ", ".join(available) or "none")
    if missing:
        print("       missing:  ", ", ".join(missing))
    if len(available) < 8:
        report.line(
            WARN,
            "the edge was measured across 12 symbols; a much smaller set was "
            "exactly the condition under which earlier candidates failed "
            "(BTC/ETH/SOL alone: t=1.14)",
        )
    return available


def check_contract_fees(report: Report, spec, client, public_only: bool) -> None:
    """
    Read the account's real per-contract futures commission.

    Coinbase quotes FCM futures commission per contract, not in basis points,
    so `fetch_trading_fees` (which speaks percentages) does not answer this.
    The figure lives in the brokerage transaction summary under the FUTURE
    product type. It is the single number that decides which contracts are
    worth trading, and it is the one thing preflight cannot infer without
    credentials.
    """
    report.section("5. Per-contract commission — the number that decides this")
    print(f"       current assumption: ${contract_fee_usd(spec.name):.4f}/contract/side")

    if public_only:
        report.line(WARN, "--public-only: cannot read your real commission; the "
                          "placeholder below is a guess, not your rate")
        return

    summary = None
    for params in ({"product_type": "FUTURE"}, {}):
        try:
            summary = client.privateGetBrokerageTransactionSummary(params)
            if summary:
                break
        except Exception as exc:
            last = exc
            summary = None

    if not summary:
        report.line(
            WARN,
            "could not read the futures commission from the API — find it in "
            "Coinbase's fee schedule for CDE contracts and pin it with "
            "register_contract_fee(), or trade only the large-notional contracts "
            "where even a $1.00/side fee is affordable",
        )
        return

    fee_tier = summary.get("fee_tier") or {}
    print(f"       raw fee_tier: {fee_tier}")
    for key in ("taker_fee_rate", "maker_fee_rate"):
        if fee_tier.get(key) is not None:
            print(f"       {key}: {fee_tier[key]}")

    per_contract = (summary.get("total_fees") or fee_tier.get("usd_from")
                    or None)
    if per_contract is None:
        report.line(WARN, "the summary carried no per-contract commission field; "
                          "confirm the rate against Coinbase's published CDE "
                          "schedule and pin it with register_contract_fee()")
        return

    try:
        register_contract_fee(spec.name, float(per_contract))
        report.line(OK, f"per-contract commission set to ${float(per_contract):.4f}/side")
    except (TypeError, ValueError):
        report.line(WARN, f"unparseable commission value {per_contract!r}")


def check_fees(report: Report, spec, client, public_only: bool) -> None:
    report.section("5. Fee tier — the number that decides this")
    assumed = resolve_fee_schedule(spec.name)
    print(f"       assumed (costs.py): maker {assumed.maker_bps:.1f}bps  "
          f"taker {assumed.taker_bps:.1f}bps")

    if not public_only:
        try:
            fees = client.fetch_trading_fees()
            rates = [f for f in fees.values() if isinstance(f, dict)
                     and f.get("maker") is not None and f.get("taker") is not None]
            if rates:
                maker = max(float(f["maker"]) for f in rates) * 10_000
                taker = max(float(f["taker"]) for f in rates) * 10_000
                register_fee_schedule(spec.name, maker, taker)
                report.line(OK, f"live tier: maker {maker:.1f}bps  taker {taker:.1f}bps")
                print(f"       to pin this, add to costs.py FEE_SCHEDULES:")
                print(f'         "{spec.name}": FeeSchedule("{spec.name}", '
                      f"maker_bps={maker:.1f}, taker_bps={taker:.1f}),")
            else:
                report.line(WARN, "the venue returned no usable fee rates")
        except Exception as exc:
            report.line(WARN, f"could not read the live fee tier ({exc}) — "
                              f"using the assumed schedule above")


def check_sessions(report: Report, spec, client, symbols: list) -> None:
    report.section("5b. Trading sessions")
    print("       FCM futures are not continuously traded: they close daily and")
    print("       for maintenance. Orders into a closed session are rejected.")
    print()
    probe = symbols[0] if symbols else None
    if not probe or probe not in client.markets:
        report.line(WARN, "no contract available to read session state from")
        return
    session = session_from_market(client.markets[probe])
    allowed, why = session_allows_entry(session, 16)
    print(f"       state:  {session.state}")
    print(f"       open:   {session.open_time_iso}")
    print(f"       close:  {session.close_time_iso}")
    report.line(OK if allowed else WARN,
                "session is open — entries allowed now" if allowed else why)


def check_contract_viability(report: Report, spec, client, symbols: list[str]) -> None:
    """
    Per-contract venues need their own arithmetic.

    A flat per-contract commission is a percentage that depends on the
    contract's notional, so viability is decided symbol by symbol and the
    ranking is the opposite of intuition: the BIG contracts are the cheap ones.
    """
    report.section("6. Does the edge survive the round trip? (per-contract fees)")
    edge = measured_edge("crowd_short")
    fee = contract_fee_usd(spec.name)
    measured = fee_is_measured(spec.name)

    print(f"       measured gross edge:  +{edge.gross_pct_per_trade:.3f}%/trade")
    print(f"       per-contract fee:     ${fee:.4f}/side "
          f"({'from your account' if measured else 'PLACEHOLDER — not your real rate'})")
    if not measured:
        report.line(WARN, "per-contract fee is a placeholder; every figure below "
                          "inherits that guess")
    print()

    guard = EdgeGuard(venue=spec.name)
    viable, blocked = [], []
    print(f"       {'symbol':26s} {'1 contract':>11s} {'fee RT':>8s} {'net edge':>10s}")
    for symbol in symbols:
        market = client.markets.get(symbol)
        if not market:
            continue
        cspec = spec_from_market(symbol, market)
        try:
            price = float(client.fetch_ticker(f"{cspec.base}/USD").get("last") or 0)
        except Exception:
            price = 0.0
        if price <= 0:
            continue
        fee_pct = cspec.round_trip_fee_pct(price, spec.name)
        verdict = guard.evaluate(symbol, "crowd_short", hold_hours=16,
                                 fee_pct_round_trip=fee_pct)
        (viable if verdict.allowed else blocked).append(cspec.base)
        flag = "PASS" if verdict.allowed else "FAIL"
        print(f"  {flag} {symbol:26s} ${cspec.notional_usd(price):10,.0f} "
              f"{fee_pct:7.3f}% {verdict.net_edge_pct:+9.3f}%")

    print()
    if viable:
        report.line(OK, f"{len(viable)}/{len(viable)+len(blocked)} contracts clear "
                        f"the edge: {', '.join(viable)}")
    if blocked:
        report.line(WARN, f"too expensive at this fee: {', '.join(blocked)} — "
                          f"their contracts are small, so a flat fee is a large "
                          f"fraction of notional")
    if not viable:
        report.line(FAIL, "no contract clears the measured edge at this fee")


def check_viability(report: Report, spec, symbols: list[str]) -> None:
    report.section("6. Does the edge survive the round trip?")
    edge = measured_edge("crowd_short")
    print(f"       measured gross edge: +{edge.gross_pct_per_trade:.3f}%/trade")
    print(f"       source: {edge.source}\n")

    any_viable = False
    for entry_is_maker in (True, False):
        guard = EdgeGuard(venue=spec.name, entry_is_maker=entry_is_maker)
        label = "post-only entry" if entry_is_maker else "market entry"
        print(f"       -- {label} --")
        for symbol in (symbols or [spec.symbol_for("BTC")])[:6]:
            verdict = guard.evaluate(symbol, "crowd_short", hold_hours=16)
            any_viable = any_viable or verdict.allowed
            print(f"         {'PASS' if verdict.allowed else 'FAIL'}  {verdict.explain()}")
        print()

    if any_viable:
        report.line(OK, "the measured edge clears modelled costs on this venue")
    else:
        report.line(
            FAIL,
            "modelled costs exceed the entire measured edge — automating this "
            "strategy here loses money on every trade by construction",
        )

    if spec.product == "perp":
        report.line(
            WARN,
            "perp funding is NOT in the cost model. A crowd-short is usually on "
            "the receiving side of funding when the crowd is long, so this most "
            "likely understates the edge — but the rate can flip inside a 16h "
            "hold and the strategy was never validated on funding income.",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Coinbase preflight checks")
    parser.add_argument("--venue", default=COINBASE_VENUE, choices=sorted(VENUES))
    parser.add_argument("--symbols", type=int, default=12,
                        help="How many of MAJOR_BASES to check (default: 12)")
    parser.add_argument("--public-only", action="store_true",
                        help="Skip authentication; check markets and fees only")
    args = parser.parse_args()

    report = Report()
    print("=" * 72)
    print(f"  COINBASE PREFLIGHT — {args.venue}")
    print("=" * 72)

    spec = check_venue(report, args.venue)
    if spec is None:
        return summarise(report)

    check_direction_support(report, spec)

    client = connect(report, spec, args.public_only)
    if client is None:
        return summarise(report)

    symbols = check_symbols(report, spec, client, args.symbols)
    if spec.product == "future":
        check_contract_fees(report, spec, client, args.public_only)
        check_sessions(report, spec, client, symbols)
        check_contract_viability(report, spec, client, symbols)
    else:
        check_fees(report, spec, client, args.public_only)
        check_viability(report, spec, symbols)

    return summarise(report)


def summarise(report: Report) -> int:
    print("\n" + "=" * 72)
    if report.failures:
        print(f"  RESULT: NOT READY — {len(report.failures)} blocking issue(s)")
        for item in report.failures:
            print(f"    - {item}")
    else:
        print("  RESULT: READY — no blocking issues")
    if report.warnings:
        print(f"\n  {len(report.warnings)} warning(s):")
        for item in report.warnings:
            print(f"    - {item}")
    print("=" * 72)
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
