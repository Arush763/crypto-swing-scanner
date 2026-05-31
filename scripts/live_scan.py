"""
Live scanner — runs one full scan cycle, sends Telegram alerts,
and writes results JSON for the GitHub Pages dashboard.

Called by GitHub Actions on a schedule. Can also be run locally:
    python scripts/live_scan.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

from src.scanner import Scanner
from src.notifications.telegram import TelegramNotifier

# Top liquid coins — same universe validated in backtest
UNIVERSE_COINS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "BNB/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "NEAR/USDT", "AAVE/USDT", "XLM/USDT", "XMR/USDT", "ZEC/USDT",
    "LTC/USDT", "BCH/USDT", "UNI/USDT", "INJ/USDT", "TON/USDT",
    "HBAR/USDT", "SUI/USDT", "FET/USDT", "STG/USDT", "SHIB/USDT",
]

OUTPUT_DIR  = Path("dashboard")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_JSON = OUTPUT_DIR / "results.json"


def run() -> None:
    notifier = TelegramNotifier()

    logger.info("Starting live scan (%d coins)…", len(UNIVERSE_COINS))

    scanner = Scanner(
        exchange_ids=["coinbase", "kucoin", "binanceus", "kraken"],
        min_volume=5_000_000,
        score_threshold=80.0,
        enable_orderbook=False,   # keep fast for scheduled runs
    )

    result = scanner.run()

    # ── Write JSON for dashboard ─────────────────────────────────────
    signals_data = []
    for s in result.signals:
        signals_data.append({
            "symbol":          s.symbol,
            "type":            s.signal_type,
            "strength":        s.strength,
            "score":           s.final_score,
            "trend_score":     s.trend_score,
            "momentum_score":  s.momentum_score,
            "liquidity_score": s.liquidity_score,
            "smart_money":     s.smart_money_score,
            "price":           s.current_price,
            "entry_low":       s.entry_zone_low,
            "entry_high":      s.entry_zone_high,
            "stop_loss":       s.stop_loss,
            "risk_pct":        s.risk_pct,
            "reward_pct":      s.reward_pct,
            "rr":              s.risk_reward,
            "resistance":      s.resistance_level,
            "timestamp":       s.timestamp.isoformat(),
        })

    leaderboard = []
    if not result.ranked_df.empty:
        for _, row in result.ranked_df.head(20).iterrows():
            leaderboard.append({
                "symbol":     row["symbol"],
                "score":      row["final_score"],
                "trend":      row["trend_score"],
                "momentum":   row["momentum_score"],
                "liquidity":  row["liquidity_score"],
                "breakout":   bool(row.get("is_breakout", False)),
                "retest":     bool(row.get("is_retest", False)),
                "squeeze":    bool(row.get("is_squeeze", False)),
                "price":      row.get("latest_price", 0),
            })

    output = {
        "scan_time":      result.timestamp.isoformat(),
        "assets_scanned": result.assets_scanned,
        "signals_count":  len(result.signals),
        "duration_s":     result.duration_seconds,
        "signals":        signals_data,
        "leaderboard":    leaderboard,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", RESULTS_JSON)

    # ── Send Telegram alerts ─────────────────────────────────────────
    if notifier.enabled:
        notifier.send_scan_summary(result)
        for signal in result.signals:
            notifier.send_signal(signal)
        if not result.signals:
            ts = result.timestamp.strftime("%Y-%m-%d %H:%M UTC")
            notifier.send_no_signal_ping(ts)
    else:
        logger.info("Telegram not configured — skipping notifications")

    logger.info("Scan complete: %d signals in %.1fs", len(result.signals), result.duration_seconds)


if __name__ == "__main__":
    run()
