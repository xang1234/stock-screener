"""Slack notification module (ported from Repo A)."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Send screening results to Slack via webhook or bot token."""

    def __init__(self, webhook_url: Optional[str] = None, bot_token: Optional[str] = None,
                 channel: Optional[str] = None) -> None:
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.channel = channel or os.getenv("SLACK_CHANNEL", "#stock-alerts")
        self.client = None
        if self.bot_token:
            try:
                from slack_sdk import WebClient
                self.client = WebClient(token=self.bot_token)
            except ImportError:
                logger.warning("slack-sdk not installed")

        if not self.webhook_url and not self.bot_token:
            logger.warning("Slack not configured. Set SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN.")

    def send_message(self, text: str) -> bool:
        """Send a plain text message."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
        if self.webhook_url:
            return self._send_via_webhook(blocks)
        elif self.client:
            return self._send_via_bot(blocks)
        logger.warning("Slack not configured, skipping send_message")
        return False

    def send_screening_results(self, results: list, top_n: int = 5) -> bool:
        """Send top screener signals to Slack. results is a list of dicts with signal data."""
        if not results:
            logger.warning("No screening results to send")
            return False

        today = datetime.now().strftime("%B %d, %Y")
        top = results[:top_n]

        blocks: List[Dict] = [
            {"type": "header", "text": {"type": "plain_text", "text": f"📊 Stock Screening Results - {today}"}},
            {"type": "section", "text": {"type": "mrkdwn",
                                         "text": f"*{len(results)}* stocks screened. Top *{len(top)}* candidates:"}},
            {"type": "divider"},
        ]

        for i, sig in enumerate(top, 1):
            score = sig.get("composite_score", sig.get("score", 0)) or 0
            ticker = sig.get("symbol", sig.get("ticker", "?"))
            if score >= 80:
                emoji, label = "🔥", "STRONG BUY"
            elif score >= 65:
                emoji, label = "✅", "BUY"
            elif score >= 50:
                emoji, label = "⚡", "CONSIDER"
            else:
                emoji, label = "⏸️", "WATCH"

            text = f"*#{i}: {ticker}*\n"
            text += f"{emoji} *{label}* — Score: *{score:.1f}*\n"
            if sig.get("entry_price"):
                text += f"• Entry: ${sig['entry_price']:.2f}"
            if sig.get("stop_loss"):
                text += f"  |  Stop: ${sig['stop_loss']:.2f}"
            if sig.get("stage"):
                text += f"  |  Stage {sig['stage']}"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})

        blocks.extend([
            {"type": "divider"},
            {"type": "context", "elements": [
                {"type": "mrkdwn",
                 "text": "*Legend:* 🔥 Strong Buy (80+) | ✅ Buy (65-79) | ⚡ Consider (50-64) | ⏸️ Watch (<50)"}
            ]},
            {"type": "context", "elements": [
                {"type": "mrkdwn", "text": "⚠️ _Not financial advice. Always do your own research._"}
            ]},
        ])

        if self.webhook_url:
            return self._send_via_webhook(blocks)
        elif self.client:
            return self._send_via_bot(blocks)
        logger.warning("Slack not configured, skipping send")
        return False

    def _send_via_webhook(self, blocks: List[Dict]) -> bool:
        try:
            import requests
            response = requests.post(self.webhook_url, json={"blocks": blocks}, timeout=10)
            if response.status_code == 200:
                logger.info("Slack message sent via webhook")
                return True
            logger.error("Slack webhook failed: %s %s", response.status_code, response.text)
            return False
        except Exception as e:
            logger.error("Webhook send failed: %s", e)
            return False

    def _send_via_bot(self, blocks: List[Dict]) -> bool:
        try:
            response = self.client.chat_postMessage(
                channel=self.channel, blocks=blocks, text="Stock Screening Results"
            )
            if response["ok"]:
                logger.info("Slack message sent to %s", self.channel)
                return True
            logger.error("Slack bot send failed: %s", response)
            return False
        except Exception as e:
            logger.error("Bot send failed: %s", e)
            return False
