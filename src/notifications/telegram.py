"""
Telegram notification module.

Sends formatted scan alerts and signal cards to a Telegram channel/chat.
Uses the Bot API via simple HTTP — no heavy library needed.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Add the bot to a channel or get your chat ID via @userinfobot
  3. Set env vars: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_MSG_LEN  = 4096
RETRY_DELAY  = 3   # seconds between retries on rate-limit


class TelegramNotifier:
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.token   = token   or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        if not self.token or not self.chat_id:
            logger.warning("Telegram token/chat_id not set — notifications disabled")

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message, returns True on success."""
        if not self.enabled:
            return False

        url = TELEGRAM_API.format(token=self.token, method="sendMessage")
        # Chunk if over limit
        chunks = [text[i:i + MAX_MSG_LEN] for i in range(0, len(text), MAX_MSG_LEN)]

        for chunk in chunks:
            for attempt in range(3):
                try:
                    resp = requests.post(url, json={
                        "chat_id": self.chat_id,
                        "text": chunk,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    }, timeout=10)
                    if resp.status_code == 429:   # rate limited
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    break
                except Exception as exc:
                    logger.warning("Telegram send failed (attempt %d): %s", attempt + 1, exc)
                    time.sleep(RETRY_DELAY)

        return True

    # ------------------------------------------------------------------
    # Formatted message builders
    # ------------------------------------------------------------------

    def send_signal(self, signal) -> None:
        """Send a formatted signal card for one trading alert."""
        strength_icon = "🔥" if signal.strength == "strong" else "📈"
        type_icon = {
            "breakout":        "💥",
            "retest":          "🔄",
            "squeeze_breakout":"🌀",
            "trend_continuation": "➡️",
        }.get(signal.signal_type, "📊")

        msg = (
            f"{strength_icon} <b>{signal.symbol}</b> — {type_icon} {signal.signal_type.replace('_',' ').title()}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price:       <code>${signal.current_price:,.6g}</code>\n"
            f"🎯 Entry Zone:  <code>${signal.entry_zone_low:,.6g} – ${signal.entry_zone_high:,.6g}</code>\n"
            f"🛑 Stop Loss:   <code>${signal.stop_loss:,.6g}</code>\n"
            f"📏 Risk:        <code>{signal.risk_pct:.2f}%</code>\n"
            f"🏆 R:R Ratio:   <code>{signal.risk_reward:.2f}x</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Score:       <code>{signal.final_score:.1f}/100</code>\n"
            f"   Trend:       <code>{signal.trend_score:.0f}</code>  "
            f"Momentum: <code>{signal.momentum_score:.0f}</code>\n"
            f"   Liquidity:   <code>{signal.liquidity_score:.0f}</code>  "
            f"SmartMoney: <code>{signal.smart_money_score:.0f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🚪 Exit:        {signal.exit_primary}\n"
        )
        if signal.max_safe_position_usd > 0:
            msg += f"💼 Max Position: <code>${signal.max_safe_position_usd:,.0f}</code>\n"

        self.send(msg)


    def send_scan_summary(self, result) -> None:
        """Send a brief scan summary with top assets."""
        ts = result.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        n_signals = len(result.signals)

        msg = (
            f"🔍 <b>Scan Complete</b> — {ts}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Assets scanned: <code>{result.assets_scanned}</code>\n"
            f"Signals fired:  <code>{n_signals}</code>\n"
            f"Duration:       <code>{result.duration_seconds}s</code>\n"
        )

        if not result.ranked_df.empty:
            top5 = result.ranked_df.head(5)
            msg += "\n🏆 <b>Top 5 Assets</b>\n"
            for _, row in top5.iterrows():
                flags = ""
                if row.get("is_breakout"): flags += "💥"
                if row.get("is_retest"):   flags += "🔄"
                if row.get("is_squeeze"):  flags += "🌀"
                msg += f"  {flags} <code>{row['symbol']:<14}</code> Score: <code>{row['final_score']:.0f}</code>\n"

        if n_signals == 0:
            msg += "\n⚪ No signals this cycle — markets consolidating."

        self.send(msg)

    def send_no_signal_ping(self, timestamp: str) -> None:
        """Lightweight heartbeat when no signals fire — confirms bot is alive."""
        self.send(f"⚪ Scan complete {timestamp} — no signals.")
