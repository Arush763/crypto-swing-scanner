"""
Pull real OHLCV data across US-accessible exchanges.

Supports any timeframe. For 1h, paginates to fetch up to `--days` days
of history (default 730 days = 2 years = ~17,520 hourly bars).

Volume filter uses KuCoin + Coinbase tickers (global volumes).
OHLCV fetched in priority order: Binance.US → KuCoin → Coinbase → Kraken.

Usage:
    python scripts/pull_data.py --timeframe 1h --days 730 --limit 300
    python scripts/pull_data.py --timeframe 1d --days 1000 --limit 300
"""

import sys, time, argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
import pandas as pd

CACHE_DIR     = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MIN_VOL_USD   = 5_000_000
REQUEST_DELAY = 0.3
PAGE_SIZE     = 1000   # max bars per ccxt request

VOLUME_SOURCES = ["kucoin", "coinbase"]
OHLCV_SOURCES  = ["binanceus", "kucoin", "coinbase", "kraken"]
SKIP_BASE      = {"EUR", "GBP", "USDC", "EURC", "USD1", "XAUM", "USDT",
                  "BUSD", "TUSD", "DAI", "USDUC", "USELESS"}

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
              "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}


def init(exchange_id: str) -> ccxt.Exchange:
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def fetch_paginated(client: ccxt.Exchange, symbol: str, timeframe: str, days: int) -> list:
    """
    Fetch up to `days` days of OHLCV by paginating backwards from now.
    Returns a flat list of [ts, o, h, l, c, v] rows, sorted ascending.
    """
    tf_mins   = TF_MINUTES.get(timeframe, 60)
    bars_needed = int(days * 1440 / tf_mins)
    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms  = now_ms - bars_needed * tf_mins * 60 * 1000

    all_rows = []
    fetch_since = since_ms

    while True:
        try:
            time.sleep(REQUEST_DELAY)
            rows = client.fetch_ohlcv(symbol, timeframe=timeframe,
                                       since=fetch_since, limit=PAGE_SIZE)
        except Exception:
            break

        if not rows:
            break

        all_rows.extend(rows)

        last_ts = rows[-1][0]
        # Advance past the last returned bar
        fetch_since = last_ts + tf_mins * 60 * 1000

        # Stop if we've reached the current time or got a short page
        if last_ts >= now_ms or len(rows) < PAGE_SIZE:
            break

    # Deduplicate and sort
    seen = {}
    for r in all_rows:
        seen[r[0]] = r
    return sorted(seen.values(), key=lambda x: x[0])


def main(limit: int, days: int, timeframe: str) -> None:
    tf_mins = TF_MINUTES.get(timeframe)
    if tf_mins is None:
        print(f"Unknown timeframe: {timeframe}. Choose from {list(TF_MINUTES.keys())}")
        return

    bars_expected = int(days * 1440 / tf_mins)
    print(f"Timeframe : {timeframe}  ({bars_expected:,} bars per asset for {days} days)")
    print(f"Min volume: ${MIN_VOL_USD/1e6:.0f}M\n")

    # ── Step 1: volume-filtered candidates ───────────────────────────────
    best_map = {}
    for eid in VOLUME_SOURCES:
        print(f"Fetching tickers from {eid}...", flush=True)
        try:
            ex = init(eid)
            tickers = ex.fetch_tickers()
        except Exception as e:
            print(f"  WARN {eid}: {e}")
            continue

        count = 0
        for sym, t in tickers.items():
            parts = sym.split("/")
            if len(parts) != 2:
                continue
            base, quote = parts
            if quote not in ("USDT", "USD", "USDC"):
                continue
            if base in SKIP_BASE:
                continue
            vol = t.get("quoteVolume") or 0.0
            if vol < MIN_VOL_USD:
                continue
            if vol > best_map.get(base, {}).get("volume", 0.0):
                best_map[base] = {"volume": vol, "symbol": sym, "exchange": eid}
            count += 1
        print(f"  {eid}: {count} pairs >= ${MIN_VOL_USD/1e6:.0f}M")

    # Add Binance.US pairs — only for coins already in best_map (verified volume)
    # This gives us Binance.US's longer OHLCV history for coins we already trust
    print("Loading Binance.US market list for history depth...", flush=True)
    try:
        bnus = init("binanceus")
        bnus_markets = bnus.load_markets()
        upgraded = 0
        for sym, mkt in bnus_markets.items():
            if not sym.endswith("/USDT") or not mkt.get("active"):
                continue
            base = sym.split("/")[0]
            # Only upgrade existing volume-verified coins to use Binance.US for OHLCV
            if base in best_map and best_map[base]["volume"] > 0:
                best_map[base]["binanceus_symbol"] = sym
                upgraded += 1
        print(f"  Mapped {upgraded} volume-verified coins to Binance.US for deeper history")
    except Exception as e:
        print(f"  WARN binanceus: {e}")

    ranked = sorted(best_map.values(), key=lambda x: x["volume"], reverse=True)[:limit]
    print(f"\nTotal candidates: {len(ranked)}")
    print(f"Pulling {timeframe} OHLCV ({days} days each)...\n")

    # ── Step 2: fetch OHLCV ──────────────────────────────────────────────
    clients = {}
    for eid in OHLCV_SOURCES:
        try:
            clients[eid] = init(eid)
        except Exception:
            pass

    saved = skipped = 0
    min_bars = max(60, int(days * 1440 / tf_mins * 0.05))  # need at least 5% of target

    for idx, info in enumerate(ranked, 1):
        base       = info["symbol"].split("/")[0]
        vol_m      = info["volume"] / 1_000_000
        primary_ex = info["exchange"]

        print(f"[{idx:>3}/{len(ranked)}] {base:<10} ${vol_m:>8.1f}M ... ", end="", flush=True)

        df = None
        used_ex = None

        # If this coin has a Binance.US mapping, try it first for deeper history
        bnus_sym = info.get("binanceus_symbol")
        preferred_order = (["binanceus"] + [e for e in OHLCV_SOURCES if e != "binanceus"]) if bnus_sym else ([primary_ex] + [e for e in OHLCV_SOURCES if e != primary_ex])

        for eid in preferred_order:
            client = clients.get(eid)
            if client is None:
                continue
            candidates = [bnus_sym] if (eid == "binanceus" and bnus_sym) else [f"{base}/{q}" for q in ("USDT", "USD", "USDC")]
            for sym_try in candidates:
                try:
                    rows = fetch_paginated(client, sym_try, timeframe, days)
                    if rows and len(rows) >= min_bars:
                        tmp = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
                        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], unit="ms", utc=True)
                        tmp = tmp.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
                        df = tmp
                        used_ex = eid
                        break
                except Exception:
                    continue
            if df is not None:
                break

        if df is None or len(df) < min_bars:
            print("SKIP")
            skipped += 1
            continue

        safe = f"{base}_USDT"
        out  = CACHE_DIR / f"{used_ex}_{safe}_{timeframe}.parquet"
        df.to_parquet(out, index=False)
        print(f"OK ({len(df):,} bars, {used_ex})")
        saved += 1

    print(f"\n{'='*52}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Saved:     {saved}")
    print(f"  Skipped:   {skipped}")
    print(f"  Cache:     {CACHE_DIR.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",     type=int,   default=300)
    ap.add_argument("--days",      type=int,   default=730)
    ap.add_argument("--timeframe", type=str,   default="1h")
    args = ap.parse_args()
    main(args.limit, args.days, args.timeframe)
