"""Email notification module for morning digest."""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send screening results via SMTP email."""

    def __init__(self) -> None:
        self.smtp_host = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self.smtp_user = os.getenv("EMAIL_SMTP_USER", "")
        self.smtp_password = os.getenv("EMAIL_SMTP_PASSWORD", "")
        self.from_addr = os.getenv("EMAIL_FROM", self.smtp_user)
        self.to_addrs = [a.strip() for a in os.getenv("EMAIL_TO", "").split(",") if a.strip()]

        self._configured = bool(self.smtp_user and self.smtp_password and self.to_addrs)
        if not self._configured:
            logger.warning(
                "Email not configured. Set EMAIL_SMTP_USER, EMAIL_SMTP_PASSWORD, EMAIL_TO."
            )

    @property
    def configured(self) -> bool:
        return self._configured

    def send_morning_digest(
        self,
        top_signals: list,
        market: str = "US",
        regime: Optional[dict] = None,
        breadth: Optional[dict] = None,
    ) -> bool:
        if not self._configured:
            logger.warning("Email not configured, skipping digest")
            return False

        today = datetime.now().strftime("%B %d, %Y")
        subject = f"📊 Morning Stock Digest — {market} — {today}"
        html = _build_digest_html(top_signals, market, today, regime, breadth)
        return self._send(subject, html)

    def _send(self, subject: str, html: str) -> bool:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        msg.attach(MIMEText(html, "html"))
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            logger.info("Email digest sent to %s", self.to_addrs)
            return True
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return False


def _build_digest_html(
    signals: list,
    market: str,
    date: str,
    regime: Optional[dict],
    breadth: Optional[dict],
) -> str:
    regime_html = ""
    if regime:
        buy_allowed = regime.get("buy_allowed", False)
        spy_phase = regime.get("spy_phase", "?")
        pct = regime.get("pct_stocks_phase2")
        pct_str = f"{pct * 100:.0f}%" if pct is not None else "—"
        color = "#2e7d32" if buy_allowed else "#d32f2f"
        label = "BUYS ON ✅" if buy_allowed else "BUYS OFF ⛔"
        regime_html = f"""
        <tr><td colspan="6" style="background:{color};color:#fff;padding:8px 12px;font-weight:700">
            Market Regime: {label} — SPY Phase {spy_phase}, {pct_str} stocks in Phase 2
        </td></tr>"""

    rows_html = ""
    for i, sig in enumerate(signals[:20], 1):
        score = sig.get("composite_score") or sig.get("signal_score") or 0
        ticker = sig.get("symbol", "?")
        stop = sig.get("stop_loss")
        stage = sig.get("stage") or sig.get("weinstein_stage")
        breakout = sig.get("breakout_type", "")
        rs = sig.get("rs_rating") or sig.get("rs_rating_1m")

        score_color = "#2e7d32" if score >= 75 else "#f57c00" if score >= 55 else "#555"
        stop_str = f"${stop:.2f}" if stop else "—"
        stage_str = str(stage) if stage else "—"
        rs_str = str(int(rs)) if rs else "—"

        rows_html += f"""
        <tr style="background:{'#f9f9f9' if i % 2 == 0 else '#fff'}">
            <td style="padding:6px 10px;font-weight:700">{i}</td>
            <td style="padding:6px 10px;font-weight:700;color:#1565c0">{ticker}</td>
            <td style="padding:6px 10px;text-align:center;font-weight:700;color:{score_color}">{score:.1f}</td>
            <td style="padding:6px 10px;text-align:center">{stage_str}</td>
            <td style="padding:6px 10px;text-align:center">{rs_str}</td>
            <td style="padding:6px 10px;text-align:center">{stop_str}</td>
        </tr>"""

    if not rows_html:
        rows_html = '<tr><td colspan="6" style="padding:16px;text-align:center;color:#888">No signals today</td></tr>'

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto">
    <h2 style="color:#1a1a2e">📊 Morning Stock Digest — {market} — {date}</h2>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead>
            <tr style="background:#1a1a2e;color:#fff">
                <th style="padding:8px 10px">#</th>
                <th style="padding:8px 10px">Ticker</th>
                <th style="padding:8px 10px">Score</th>
                <th style="padding:8px 10px">Stage</th>
                <th style="padding:8px 10px">RS</th>
                <th style="padding:8px 10px">Stop</th>
            </tr>
        </thead>
        <tbody>
            {regime_html}
            {rows_html}
        </tbody>
    </table>
    <p style="color:#888;font-size:11px;margin-top:20px">
        ⚠️ Not financial advice. Always do your own research.<br>
        Generated by Stock Scanner — {date}
    </p>
    </body></html>"""
