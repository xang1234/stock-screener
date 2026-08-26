"""Upsert helper for MaxPainSnapshot/GexSnapshot rows.

The natural key for these tables is (ticker, trading_date, expiration) -- one
row per ticker per trading day per expiration analyzed. Without this, a
manual re-trigger of the daily batch (the chain_next=False API paths added
alongside the scheduling fixes) or the on-demand per-expiration term
structure feature would insert a second row for the same (ticker,
trading_date), which corrupts the history endpoints: their DISTINCT ON
queries would pick an arbitrary one of the duplicates rather than the
intended single EOD value for that day.

No DB-level unique constraint is added for this -- production data may
already contain duplicate (ticker, trading_date) rows from prior manual
re-runs made before this upsert existed, and a migration adding a hard
UNIQUE constraint would fail outright against any such existing duplicates.
This is an application-level check-then-write instead: safe for the
low-concurrency way these rows are actually written (one Celery task at a
time per ticker), not meant to defend against concurrent writers racing on
the exact same key.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional, Type
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

_ET = ZoneInfo("America/New_York")


def trading_date_for(fetched_at_utc: datetime) -> date:
    """US/Eastern trading day for a naive-UTC timestamp.

    Same helper duplicated locally in max_pain_tasks.py/gex_tasks.py (kept
    local there to avoid touching already-working code); this copy is for
    new call sites like the on-demand term-structure endpoint.
    """
    return fetched_at_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(_ET).date()


def upsert_snapshot(
    db: Session,
    model: Type,
    *,
    ticker: str,
    trading_date: date,
    expiration: Optional[date],
    values: dict[str, Any],
):
    """Insert a new row, or update in place if one already exists for this
    (ticker, trading_date, expiration). Returns the row (not yet committed
    or flushed -- caller controls the transaction)."""
    existing = (
        db.query(model)
        .filter(
            model.ticker == ticker,
            model.trading_date == trading_date,
            model.expiration == expiration,
        )
        .first()
    )
    if existing is not None:
        for key, value in values.items():
            setattr(existing, key, value)
        return existing

    row = model(ticker=ticker, trading_date=trading_date, expiration=expiration, **values)
    db.add(row)
    return row
