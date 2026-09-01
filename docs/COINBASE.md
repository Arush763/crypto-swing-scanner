# Automated trading into Coinbase

Everything here targets **Coinbase only**. This document is what was built, what
works, and what still needs a number from your account before real money moves.

---

## The short version

The one validated strategy in this repo is **short-only** (`src/modules/crowd_signal.py`).
That decides which Coinbase product can run it:

| Coinbase product | ccxt route | Short? | US? | Round trip | Verdict |
|---|---|---|---|---|---|
| Advanced Trade (spot) | `coinbaseadvanced` | **No** — no borrow | Yes | ~1.8–2.4% | Unusable |
| Exchange (ex-Pro, spot) | `coinbaseexchange` | **No** | Yes | ~1.0% | Unusable |
| International (perps) | `coinbaseinternational` | Yes | **No** | ~0.09–0.15% | Closed to US persons |
| **Derivatives (CDE futures)** | `coinbaseadvanced` | **Yes** | **Yes** | **~0.26–0.49%** | **This is the one** |

**Coinbase Derivatives Exchange** is the answer: CFTC-regulated futures, open to
US persons, shortable, and reached through the same ccxt client as spot. The
default venue is now `coinbasederivatives`.

Contracts are *nano*-sized and dated. The furthest-dated series (Dec 2030) is
Coinbase's perpetual-style product — `display_name` is literally `BTC PERP` —
and `venues.resolve_contract` picks it automatically. A near-dated quarterly
would expire underneath a 16-hour hold.

---

## What is genuinely working

Verified against live Coinbase markets:

- All **12/12** of the strategy's majors have a tradeable CDE contract.
- Signal → contract resolution → session gate → cost gate → short entry → timed
  cover runs end to end in paper mode.
- Position scale is correct: 1 nano BTC contract = 0.01 BTC ≈ $773 notional,
  ~$2 round-trip commission. (This was wrong at first — see *Contract size* below.)
- 267 tests pass.

```bash
python scripts/coinbase_preflight.py            # → RESULT: READY
python scripts/run_coinbase_trader.py --once    # → runs a full cycle
```

---

## Three things a futures contract changes

Spot lets you buy any fraction, pay a percentage, and trade whenever. A CDE
contract breaks all three, and each break produces a *wrong number* rather than
an error.

### 1. Contract size — the 100× trap

`quantity` counts **contracts**, not coins. One nano BTC contract is 0.01 BTC.
During development a position of 1 contract was booked as 1 whole BTC, which
overstated notional, P&L and fees by 100× — in the direction that makes a
mediocre strategy look extraordinary. Nothing threw.

`Position.contract_size` now carries the multiplier and every economic quantity
multiplies by it. `tests/test_contracts.py` pins this.

### 2. Fees are per contract, not per dollar

This inverts the usual intuition: a flat fee is proportionally **cheaper on a
big contract**. At $1.00/side:

| | 1 contract | round-trip fee | net edge |
|---|---|---|---|
| BTC | $773 | 0.259% | **+0.391%** |
| BNB | $680 | 0.294% | **+0.346%** |
| XRP | $676 | 0.296% | **+0.344%** |
| LINK | $562 | 0.356% | **+0.284%** |
| SOL | $500 | 0.400% | **+0.240%** |
| DOGE | $409 | 0.488% | **+0.152%** |
| LTC / BCH / ETH | ~$245 | ~0.81% | −0.17% |
| ADA | $196 | 1.020% | −0.380% |
| DOT | $87 | 2.295% | −1.655% |
| AVAX | $72 | 2.765% | −2.125% |

Sensitivity to the real fee:

| fee/side | viable contracts |
|---|---|
| $0.10 | **12 / 12** |
| $0.25 | 11 / 12 |
| $0.50 | 10 / 12 |
| $1.00 | 6 / 12 |

**$1.00/side is a deliberately pessimistic placeholder, not your rate.** Nano
futures are typically well below it. Run preflight with credentials to pin the
real number — see *The one open question* below.

### 3. Trading sessions

FCM futures close daily (21:00 UTC) and for maintenance. Orders into a closed
session are rejected, so `session_allows_entry` blocks entries while closed.

Crossing the daily break mid-position is normal and is **not** blocked — an
earlier version refused any entry within 16h of the close, which would have
rejected roughly two thirds of all signals. What is flagged instead is an exit
deadline landing inside the break, which delays the flatten to the reopen.

---

## Position size and account size

Contracts are indivisible, so contract size sets a **minimum viable account**.

`MAX_POSITION_USD` was raised **$500 → $800**. At $500 the five largest-notional
contracts (BTC, XRP, BNB, LINK, SOL) could not be sized at all — and those are
precisely the *cheapest* ones per fee, so the old cap kept only the expensive
symbols.

The trade-off is explicit: on a $10k account, one $800 position is **8% notional
exposure** against a configured `RISK_PER_TRADE_PCT` of 1%. Holding a true 1%
against a $773 BTC contract needs roughly **$77k** of equity. The runner logs
this on the first sizing decision rather than hiding it. This strategy carries
**no stop**, so that exposure is bounded only by how far price runs in 16 hours.

Lower the cap and you lose symbols, largest-contract-first.

---

## Getting API keys

From <https://portal.cdp.coinbase.com> → API keys → Create:

```bash
export COINBASE_API_KEY='organizations/.../apiKeys/...'
export COINBASE_API_SECRET='-----BEGIN EC PRIVATE KEY-----
MHcCAQ...
-----END EC PRIVATE KEY-----'
```

Grant **trade** permission only — never withdrawals. Nothing here withdraws, so
a key that can is pure downside.

> **The newline trap.** Most `.env` parsers collapse the PEM's newlines into the
> literal characters `\n`, and ccxt then fails with an error that never mentions
> newlines. `venues.repair_pem` fixes it automatically and preflight reports it.

The account must additionally be **futures-enabled** in Coinbase — spot access
alone will not trade CDE.

---

## The go-live sequence

```bash
# 1. Public checks — no credentials needed
python scripts/coinbase_preflight.py --public-only

# 2. Authenticated — reads your REAL per-contract commission
python scripts/coinbase_preflight.py

# 3. Signals only, no fills
python scripts/run_coinbase_trader.py --dry-run

# 4. Paper fills at real prices, real contract sizes
python scripts/run_coinbase_trader.py

# 5. Live — after setting LIVE_TRADING_ENABLED = True in config.py
python scripts/run_coinbase_trader.py --live
```

Exit code from preflight is 0 only if the venue is usable end to end. If it says
`NOT READY`, the runner will refuse for the same reasons.

The runner is a **long-running process** — it polls hourly and holds positions
for 16 hours. It is not suitable for ephemeral CI runners, whose filesystem
would discard the position store mid-trade. Run it somewhere that stays up.

---

## The one open question

**Your real per-contract commission.** It is the number that decides how many of
the 12 symbols are worth trading, and it is the one thing that cannot be read
without credentials. Everything is built to accept it:

```python
from src.execution.contracts import register_contract_fee
register_contract_fee("coinbasederivatives", 0.25)   # your rate, USD/contract/side
```

Preflight attempts this automatically from the brokerage transaction summary.
Until it succeeds, every cost figure inherits the pessimistic $1.00 placeholder,
and the runner says so on every entry.

---

## The remaining blocker: the signal feed

Execution on Coinbase is solved. The **data** is not.

The crowd-short signal reads Binance's retail long/short positioning. Verified
today from this machine:

```
fapi.binance.com          451  "restricted location"
fapi1-4.binance.com       202  empty (CDN challenge)
data-api.binance.vision   404  spot mirror only
data.binance.vision       200  ARCHIVE REACHABLE (~2-day lag)
okx.com rubik             200  works
```

- The **archive** is reachable and carries the exact validated field
  (`count_long_short_ratio`), but publishes ~2 days late — useless for a live
  16-hour signal, valuable for re-validation.
- **OKX** is live but caps at 720 points (30 days) and will not page back, and
  its ratio only tracks Binance's for **BTC, ETH and SOL**. Those three alone
  measure t=1.14 and collapse to +0.008% under the tail trim.

So the runner currently refuses 9 of 12 symbols by design, and the 3 it allows
carry a weaker edge than the headline number. `--allow-unvalidated` exists but
trades a different signal than the one that was measured.

**The cheapest fix:** restart hourly OKX recording so an OKX-native history
accumulates and the signal can be re-validated on the feed it will actually
trade from. `.github/workflows/crowd_short.yml` does this; it last ran
2026-08-12.

**The complete fix:** run from a host where Binance's API is reachable. Then the
validated signal deploys as tested, against Coinbase execution that already works.
