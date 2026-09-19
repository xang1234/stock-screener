"""Shared Yahoo earnings-calendar normalization.

Single owner for turning ``ticker.earnings_dates`` into the persisted
event-calendar evidence consumed by the scan pipeline's fail-closed
survivor gate. It preserves the distinction between:

- a successful lookup with no future earnings date (fresh observation
  stamp with ``next_earnings_date=None`` — the row stays evidence-complete),
  and
- a provider failure (no observation stamp — the row stays explicitly
  unavailable and scheduled producers retry on the next run).

Every producer that persists fundamentals (US snapshot hydration,
non-US bulk fundamentals ingestion, per-symbol snapshots) shares this
module so the observation semantics cannot drift, and the persisted
freshness window consumed by ``DataPreparationLayer._persisted_event_calendar``
is owned here as well.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# How long a persisted calendar observation stays usable by the scan
# pipeline's persisted-only event gate. Weekly producers must refresh at
# least this often; the window is intentionally wider than one week so a
# hydration run does not strand daily snapshots with stale evidence
# (age is measured against each snapshot's price as_of_date).
EVENT_CALENDAR_MAX_AGE_DAYS = 14
EVENT_CALENDAR_FUTURE_TOLERANCE_DAYS = 3
# Producer-side refresh threshold, strictly shorter than the consumer TTL
# above. Producers run weekly: an observation is only "fresh enough" to
# skip re-hydration while younger than one cadence week, so the next
# weekly run always refreshes before the consumer TTL can reject it.
# (With a 7-day cadence and 14-day TTL, max observation age stays ≤ 14.)
EVENT_CALENDAR_PRODUCER_REFRESH_AFTER_DAYS = 7


def normalize_yahoo_earnings_dates(
    earnings_dates: Any,
    *,
    symbol: str | None = None,
    limit: int = 4,
) -> tuple[list[date], bool]:
    """Normalize one ``ticker.earnings_dates`` lookup.

    Returns a sorted, de-duplicated date list and an availability flag.
    A successful empty response is ``([], True)``; provider failure is
    ``([], False)``.
    """
    try:
        # NOTE: ``DataFrame.empty`` is True whenever any axis is empty —
        # an all-index frame (no columns) reports empty even with rows.
        # Test row count explicitly so such frames are still normalized.
        row_count = 0 if earnings_dates is None else len(earnings_dates)
        if earnings_dates is None or row_count == 0:
            return [], True

        normalized = earnings_dates.reset_index()
        # Only date-carrying columns are eligible. A missing/NA cell in a
        # recognized column must NOT fall back to the positional "index"
        # column — that would parse row numbers as epoch dates (1970). The
        # positional "index" fallback applies only when the frame carries
        # its dates in the index itself (unnamed DatetimeIndex).
        date_column = None
        for column_name in ("Earnings Date", "Date"):
            if column_name in normalized.columns:
                date_column = column_name
                break
        if date_column is None and isinstance(
            getattr(earnings_dates, "index", None), pd.DatetimeIndex
        ):
            date_column = "index"

        result: list[date] = []
        for row in normalized.head(limit).to_dict("records"):
            if date_column is None:
                continue
            raw_value = row.get(date_column)
            # Only guard on None here: pd.isna on a non-scalar cell (e.g. a
            # multi-element list) returns an array whose Boolean evaluation
            # raises and would abort the whole lookup via the outer handler.
            # Let pd.Timestamp reject malformed values instead — all missing
            # scalars (NA/NaT/NaN/None) parse to NaT and are skipped by the
            # NaT check below.
            if raw_value is None:
                continue
            try:
                timestamp = pd.Timestamp(raw_value)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping unparsable earnings date for %s: %r",
                    symbol,
                    raw_value,
                )
                continue
            if pd.isna(timestamp):
                continue
            result.append(timestamp.date())
        if not result and row_count > 0:
            # A nonempty response that yields no parseable dates is malformed
            # provider data, not a known no-upcoming-earnings result. Stay
            # fail-closed so the producer retries instead of stamping a
            # fresh observation over garbage input.
            logger.warning(
                "Calendar response for %s yielded no parseable earnings dates "
                "from %d rows; treating as provider failure",
                symbol,
                row_count,
            )
            return [], False
        return sorted(set(result)), True
    except Exception as exc:
        logger.error("Error fetching earnings dates for %s: %s", symbol, exc)
        return [], False


def stamp_event_calendar_observation(
    upcoming_dates: list[date],
    available: bool,
    *,
    observed_at: date | None = None,
) -> dict[str, Any]:
    """Return persisted fundamentals keys for one calendar observation.

    Returns ``{}`` when the provider lookup failed so failed producers
    never stamp a marker, keeping the row explicitly unavailable for the
    fail-closed gate and eligible for retry. A successful observation
    stamps ``event_calendar_as_of_date`` (the freshness owner) and the
    first known-future ``next_earnings_date`` (or ``None`` when Yahoo
    lists no upcoming earnings).
    """
    if not available:
        return {}
    observation_date = observed_at or datetime.now(UTC).date()
    next_earnings_date = next(
        (value for value in upcoming_dates if value >= observation_date),
        None,
    )
    return {
        "event_calendar_as_of_date": observation_date,
        "next_earnings_date": next_earnings_date,
    }


def is_event_calendar_observation_fresh(
    observed_at: Any,
    *,
    reference_date: date | None = None,
) -> bool:
    """Return whether an observation timestamp satisfies the freshness gate.

    The consumer side (``_persisted_event_calendar``) and the producer
    completeness check share this rule so both sides age evidence out on
    the same schedule.
    """
    return _observation_age_in_window(
        observed_at,
        reference_date,
        max_age_days=EVENT_CALENDAR_MAX_AGE_DAYS,
    )


def is_event_calendar_observation_due_for_refresh(
    observed_at: Any,
    *,
    reference_date: date | None = None,
) -> bool:
    """Return whether a producer must re-observe this calendar.

    Producer completeness uses a threshold strictly shorter than the
    consumer TTL: an observation older than one weekly cadence must be
    refreshed on the next scheduled run so consumers never see evidence
    that has already aged out of ``EVENT_CALENDAR_MAX_AGE_DAYS``.
    """
    return not _observation_age_in_window(
        observed_at,
        reference_date,
        max_age_days=EVENT_CALENDAR_PRODUCER_REFRESH_AFTER_DAYS,
    )


def _observation_age_in_window(
    observed_at: Any,
    reference_date: date | None,
    *,
    max_age_days: int,
) -> bool:
    if observed_at is None:
        return False
    try:
        timestamp = pd.Timestamp(observed_at)
    except (TypeError, ValueError):
        return False
    if pd.isna(timestamp):
        return False
    reference = reference_date or datetime.now(UTC).date()
    age_days = (reference - timestamp.date()).days
    return (
        -EVENT_CALENDAR_FUTURE_TOLERANCE_DAYS
        <= age_days
        <= max_age_days
    )


__all__ = [
    "EVENT_CALENDAR_FUTURE_TOLERANCE_DAYS",
    "EVENT_CALENDAR_MAX_AGE_DAYS",
    "EVENT_CALENDAR_PRODUCER_REFRESH_AFTER_DAYS",
    "is_event_calendar_observation_due_for_refresh",
    "is_event_calendar_observation_fresh",
    "normalize_yahoo_earnings_dates",
    "stamp_event_calendar_observation",
]
