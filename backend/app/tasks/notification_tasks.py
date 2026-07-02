"""Morning digest notification tasks — Slack and email."""
from __future__ import annotations

import logging
from datetime import date

from ..celery_app import celery_app
from ..database import SessionLocal

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.notification_tasks.send_morning_digest",
    max_retries=2,
    default_retry_delay=300,
)
def send_morning_digest(self, market: str = "us") -> dict:
    """
    Send the morning stock digest via Slack and/or email.

    Pulls the top scan candidates from the latest daily snapshot and
    the current market regime, then pushes to all configured channels.
    Runs at 06:30 ET on trading days (scheduled via Celery beat).
    """
    market_upper = market.upper()
    logger.info("Sending morning digest for market=%s", market_upper)

    results: dict = {"market": market_upper, "slack": False, "email": False, "signals": 0}

    try:
        signals, regime, breadth = _fetch_digest_data(market_upper)
        results["signals"] = len(signals)

        from ..notifications.slack_notifier import SlackNotifier
        slack = SlackNotifier()
        if slack.webhook_url or slack.client:
            results["slack"] = slack.send_screening_results(signals, top_n=10)
            if results["slack"]:
                _send_regime_to_slack(slack, regime)
        else:
            logger.info("Slack not configured, skipping")

        from ..notifications.email_notifier import EmailNotifier
        email = EmailNotifier()
        if email.configured:
            results["email"] = email.send_morning_digest(
                signals, market=market_upper, regime=regime, breadth=breadth
            )
        else:
            logger.info("Email not configured, skipping")

    except Exception as exc:
        logger.exception("Morning digest failed: %s", exc)
        raise self.retry(exc=exc)

    logger.info("Morning digest sent: %s", results)
    return results


def _fetch_digest_data(market: str) -> tuple[list, dict | None, dict | None]:
    """Fetch top signals, regime, and breadth from DB."""
    from ..services.market_regime import get_market_regime

    db = SessionLocal()
    try:
        regime = None
        try:
            regime = get_market_regime(db)
        except Exception:
            pass

        signals = _fallback_scan_results(db, market)
        return signals, regime, None
    finally:
        db.close()


def _fallback_scan_results(db, market: str) -> list:
    """Fall back to latest scan results if snapshot is unavailable."""
    try:
        from sqlalchemy import text
        rows = db.execute(text("""
            SELECT sr.symbol, sr.composite_score, sr.rs_rating,
                   sr.weinstein_stage, sr.signal_score, sr.stop_loss
            FROM scan_results sr
            JOIN scans s ON sr.scan_id = s.scan_id
            WHERE s.universe_market = :market
              AND s.status = 'completed'
            ORDER BY s.completed_at DESC, sr.composite_score DESC
            LIMIT 20
        """), {"market": market.upper()}).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as exc:
        logger.warning("Fallback scan results failed: %s", exc)
        return []


def _send_regime_to_slack(slack, regime: dict | None) -> None:
    if not regime:
        return
    buy_allowed = regime.get("buy_allowed", False)
    spy_phase = regime.get("spy_phase", "?")
    pct = regime.get("pct_stocks_phase2")
    pct_str = f"{pct * 100:.0f}%" if pct is not None else "—"
    icon = "✅" if buy_allowed else "⛔"
    status = "BUYS ON" if buy_allowed else "BUYS OFF"
    slack.send_message(
        f"{icon} *Market Regime: {status}* — SPY Phase {spy_phase}, "
        f"{pct_str} stocks in Phase 2"
    )
