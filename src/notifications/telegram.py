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
            "ob_wall_ask_absorption": "🧱",
            "ob_wall_bid_repulsion":  "🏐",
        }.get(signal.signal_type, "📊")

        msg = (
            f"{strength_icon} <b>{signal.symbol}</b> — {type_icon} {signal.signal_type.replace('_',' ').title()}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price:       <code>${signal.current_price:,.6g}</code>\n"
            f"🎯 Entry Zone:  <code>${signal.entry_zone_low:,.6g} – ${signal.entry_zone_high:,.6g}</code>\n"
            f"🥇 Target 1:    <code>${signal.take_profit_1:,.6g}</code>  (1R)\n"
            f"🥈 Target 2:    <code>${signal.take_profit_2:,.6g}</code>  ({signal.risk_reward:.1f}R)\n"
            f"🛑 Stop Loss:   <code>${signal.stop_loss:,.6g}</code>\n"
            f"📏 Risk:        <code>{signal.risk_pct:.2f}%</code>\n"
            f"🏆 R:R Ratio:   <code>{signal.risk_reward:.2f}x</code>\n"
            f"💸 Est. Cost:   <code>{signal.est_cost_pct:.3f}%</code>  "
            f"→ net <code>{signal.net_reward_pct:+.2f}%</code>\n"
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
                flags = "🧱" if row.get("is_wall_signal") else ""
                msg += f"  {flags} <code>{row['symbol']:<14}</code> Score: <code>{row['final_score']:.0f}</code>\n"

        if n_signals == 0:
            msg += "\n⚪ No signals this cycle — markets consolidating."

        self.send(msg)

    def send_no_signal_ping(self, timestamp: str) -> None:
        """Lightweight heartbeat when no signals fire — confirms bot is alive."""
        self.send(f"⚪ Scan complete {timestamp} — no signals.")

    # ------------------------------------------------------------------
    # Position lifecycle triggers
    #
    # The scanner only ever announced entries. These close the loop by
    # reporting what actually happened to the position afterwards, which is
    # the difference between an alert feed and a trading record.
    # ------------------------------------------------------------------

    def send_entry_filled(self, position) -> None:
        mode = "PAPER" if position.is_paper else "LIVE"
        self.send(
            f"✅ <b>ENTRY FILLED</b> [{mode}] — {position.symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📥 Fill:        <code>${position.entry_price:,.6g}</code>\n"
            f"📦 Size:        <code>{position.quantity:,.6g}</code>  "
            f"(<code>${position.notional_usd:,.2f}</code>)\n"
            f"🥇 TP1:         <code>${position.take_profit_1:,.6g}</code>\n"
            f"🥈 TP2:         <code>${position.take_profit_2:,.6g}</code>\n"
            f"🛑 SL:          <code>${position.stop_loss:,.6g}</code>\n"
            f"💸 Entry cost:  <code>{position.entry_cost_pct:.3f}%</code>\n"
        )

    def send_target_hit(self, position, level: str, price: float, pnl_pct: float, pnl_usd: float) -> None:
        icon = "🥇" if level == "TP1" else "🥈"
        self.send(
            f"{icon} <b>{level} HIT</b> — {position.symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📤 Exit:        <code>${price:,.6g}</code>\n"
            f"📈 P&amp;L:        <code>{pnl_pct:+.2f}%</code>  "
            f"(<code>${pnl_usd:+,.2f}</code>)  <i>net of fees</i>\n"
        )

    def send_stop_hit(self, position, price: float, pnl_pct: float, pnl_usd: float) -> None:
        self.send(
            f"🛑 <b>STOP LOSS HIT</b> — {position.symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📤 Exit:        <code>${price:,.6g}</code>\n"
            f"📉 P&amp;L:        <code>{pnl_pct:+.2f}%</code>  "
            f"(<code>${pnl_usd:+,.2f}</code>)  <i>net of fees</i>\n"
        )

    def send_position_closed(self, position, reason: str, price: float, pnl_pct: float, pnl_usd: float) -> None:
        icon = "🟢" if pnl_pct >= 0 else "🔴"
        self.send(
            f"{icon} <b>CLOSED</b> — {position.symbol}  ({reason})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📥 Entry:       <code>${position.entry_price:,.6g}</code>\n"
            f"📤 Exit:        <code>${price:,.6g}</code>\n"
            f"⏱ Held:        {position.holding_description()}\n"
            f"📊 P&amp;L:        <code>{pnl_pct:+.2f}%</code>  "
            f"(<code>${pnl_usd:+,.2f}</code>)  <i>net of fees</i>\n"
        )

    def send_risk_halt(self, reason: str, detail: str) -> None:
        """Circuit breaker tripped — new entries are blocked."""
        self.send(
            f"🚨 <b>TRADING HALTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Reason: <b>{reason}</b>\n"
            f"{detail}\n\n"
            f"No new positions will open until this resets. "
            f"Existing positions are still managed."
        )

    def send_daily_summary(self, stats: dict) -> None:
        pnl = stats.get("realised_pnl_usd", 0.0)
        icon = "🟢" if pnl >= 0 else "🔴"
        self.send(
            f"{icon} <b>Daily Summary</b> — {stats.get('date','')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Trades:      <code>{stats.get('trades', 0)}</code>\n"
            f"Win rate:    <code>{stats.get('win_rate_pct', 0):.1f}%</code>\n"
            f"Fees paid:   <code>${stats.get('fees_usd', 0):,.2f}</code>\n"
            f"Net P&amp;L:    <code>${pnl:+,.2f}</code>\n"
            f"Open now:    <code>{stats.get('open_positions', 0)}</code>\n"
        )
