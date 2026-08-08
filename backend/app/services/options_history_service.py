"""Time-series history for Max Pain / GEX snapshots, by timeframe.

Financial-logic note: these metrics are point-in-time levels (dealer
positioning as of a given day for a given expiration), not flows -- they
must never be averaged or summed across a period the way price OHLC bars
are. "Weekly/Monthly/Quarterly/Yearly" here means a lookback *window* on
the x-axis, plotting every trading day within it. Only once the window
gets long enough that plotting every day would be too dense (Quarterly,
Yearly) do we downsample -- and even then only by taking the *last*
snapshot of each period (a real, dated value), never an average.

trading_date (not fetched_at) is the query column throughout -- see
migration 20260805_0028 for why.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session


class Timeframe(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# lookback_days is calendar days (buffer over trading days for weekends/
# holidays). period is the SQL date_trunc() bucket used for downsampling to
# "last snapshot of period" -- None means plot every trading day, no
# downsampling.
_TIMEFRAME_CONFIG: dict[Timeframe, dict] = {
    Timeframe.DAILY: {"lookback_days": 14, "period": None},
    Timeframe.WEEKLY: {"lookback_days": 35, "period": None},
    Timeframe.MONTHLY: {"lookback_days": 45, "period": None},
    Timeframe.QUARTERLY: {"lookback_days": 95, "period": "week"},
    Timeframe.YEARLY: {"lookback_days": 370, "period": "month"},
}


def resolve_window(
    timeframe: Timeframe,
    start: Optional[date],
    end: Optional[date],
) -> tuple[date, date, Optional[str]]:
    """Return (start, end, period) for a timeframe, honoring explicit overrides."""
    config = _TIMEFRAME_CONFIG[timeframe]
    resolved_end = end or datetime.utcnow().date()
    resolved_start = start or (resolved_end - timedelta(days=config["lookback_days"]))
    return resolved_start, resolved_end, config["period"]


def _fetch_rows(
    db: Session,
    *,
    table: str,
    columns: list[str],
    ticker: str,
    start: date,
    end: date,
    period: Optional[str],
    expiration: Optional[date],
) -> list[dict]:
    """Fetch history rows for one ticker.

    Since the on-demand term-structure endpoint (options.py::get_options_term_structure)
    can now write additional rows for the SAME (ticker, trading_date) under a
    DIFFERENT expiration than the daily batch's nearest-expiration row, a
    plain "one row per trading_date" query is no longer guaranteed without
    disambiguating which expiration series is wanted:

    - `expiration` given: pin to that specific expiration's series (works
      whether the rows came from the daily batch or an on-demand view).
    - `expiration` omitted (the default/backward-compatible call, used by
      every existing caller of these history endpoints): restrict to
      `batch_id IS NOT NULL`, i.e. daily-batch-authored rows only. The
      term-structure endpoint always writes with batch_id=None, so this
      keeps the default series exactly what it was before expiration-based
      snapshots existed -- the nearest-expiration daily series -- rather
      than silently jumping between expirations on days a user happened to
      also view a different one.
    """
    col_list = ", ".join(columns)
    expiration_clause = "AND expiration = :expiration" if expiration is not None else "AND batch_id IS NOT NULL"

    if period is None:
        query = text(
            f"""
            SELECT trading_date, {col_list}
            FROM {table}
            WHERE ticker = :ticker
              AND status = 'OK'
              AND trading_date BETWEEN :start AND :end
              {expiration_clause}
            ORDER BY trading_date ASC
            """
        )
    else:
        # DISTINCT ON + a matching ORDER BY is the Postgres-idiomatic "last
        # value per bucket" -- picks the row with the max trading_date in
        # each date_trunc(period) group, and the result set is already in
        # period-ascending order since that's the primary sort key.
        query = text(
            f"""
            SELECT DISTINCT ON (date_trunc(:period, trading_date))
                trading_date, {col_list}
            FROM {table}
            WHERE ticker = :ticker
              AND status = 'OK'
              AND trading_date BETWEEN :start AND :end
              {expiration_clause}
            ORDER BY date_trunc(:period, trading_date), trading_date DESC
            """
        )

    params = {"ticker": ticker, "start": start, "end": end}
    if period is not None:
        params["period"] = period
    if expiration is not None:
        params["expiration"] = expiration

    rows = db.execute(query, params).mappings().all()
    if period is not None:
        # DISTINCT ON's own ordering is by (period, trading_date DESC); re-sort
        # ascending by trading_date for a clean chart-ready series.
        rows = sorted(rows, key=lambda r: r["trading_date"])
    return [dict(r) for r in rows]


def fetch_max_pain_history(
    db: Session,
    *,
    symbol: str,
    timeframe: Timeframe,
    start: Optional[date] = None,
    end: Optional[date] = None,
    expiration: Optional[date] = None,
) -> dict:
    resolved_start, resolved_end, period = resolve_window(timeframe, start, end)
    rows = _fetch_rows(
        db,
        table="max_pain_snapshots",
        columns=[
            "max_pain_strike",
            "last_price",
            "distance_pct",
            "call_oi",
            "put_oi",
            "put_call_ratio",
        ],
        ticker=symbol,
        start=resolved_start,
        end=resolved_end,
        period=period,
        expiration=expiration,
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe.value,
        "expiration": expiration.isoformat() if expiration else None,
        "sampling": "point_in_time_eod" if period is None else "last_of_period",
        "period": period,
        "start": resolved_start.isoformat(),
        "end": resolved_end.isoformat(),
        "as_of": datetime.utcnow().isoformat(),
        "points": [
            {
                "date": r["trading_date"].isoformat(),
                "max_pain_strike": r["max_pain_strike"],
                "last_price": r["last_price"],
                "distance_pct": r["distance_pct"],
                "call_oi": r["call_oi"],
                "put_oi": r["put_oi"],
                "put_call_ratio": r["put_call_ratio"],
            }
            for r in rows
        ],
    }


def fetch_gex_history(
    db: Session,
    *,
    symbol: str,
    timeframe: Timeframe,
    start: Optional[date] = None,
    end: Optional[date] = None,
    expiration: Optional[date] = None,
) -> dict:
    resolved_start, resolved_end, period = resolve_window(timeframe, start, end)
    rows = _fetch_rows(
        db,
        table="gex_snapshots",
        columns=[
            "spot_price",
            "call_oi",
            "put_oi",
            "call_gex",
            "put_gex",
            "total_gex",
            "flip_level",
            "distance_to_flip_pct",
        ],
        ticker=symbol,
        start=resolved_start,
        end=resolved_end,
        period=period,
        expiration=expiration,
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe.value,
        "expiration": expiration.isoformat() if expiration else None,
        "sampling": "point_in_time_eod" if period is None else "last_of_period",
        "period": period,
        "start": resolved_start.isoformat(),
        "end": resolved_end.isoformat(),
        "as_of": datetime.utcnow().isoformat(),
        "points": [
            {
                "date": r["trading_date"].isoformat(),
                "spot_price": r["spot_price"],
                "call_oi": r["call_oi"],
                "put_oi": r["put_oi"],
                "call_gex": r["call_gex"],
                "put_gex": r["put_gex"],
                "total_gex": r["total_gex"],
                "flip_level": r["flip_level"],
                "distance_to_flip_pct": r["distance_to_flip_pct"],
            }
            for r in rows
        ],
    }
