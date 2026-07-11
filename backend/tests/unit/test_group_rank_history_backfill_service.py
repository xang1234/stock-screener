"""Tests for canonical recent group-rank history backfill."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.group_rank_history_backfill_service import (
    DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS,
    GroupRankHistoryBackfillService,
)


def test_backfill_uses_canonical_market_session_range():
    query = MagicMock()
    query.filter.return_value = query
    query.distinct.return_value = query
    query.all.return_value = []
    db = MagicMock()
    db.query.return_value = query

    @contextmanager
    def session_factory():
        yield db

    range_calls = []
    trading_dates = [date(2026, 4, 3), date(2026, 4, 7)]

    def resolve_trading_days(market, start, end):
        range_calls.append((market, start, end))
        return trading_dates

    fill_calls = []

    def fill_gaps_optimized(db_arg, missing_dates, *, market):
        fill_calls.append((db_arg, list(missing_dates), market))
        return {"processed": len(missing_dates), "errors": 0}

    service = GroupRankHistoryBackfillService(
        session_factory=session_factory,
        calendar_service=SimpleNamespace(trading_days=resolve_trading_days),
        group_rank_service=SimpleNamespace(fill_gaps_optimized=fill_gaps_optimized),
    )

    result = service.backfill(as_of_date=date(2026, 4, 7), market="hk")

    assert range_calls == [
        (
            "HK",
            date(2026, 4, 7) - timedelta(days=DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS),
            date(2026, 4, 7),
        )
    ]
    assert fill_calls == [(db, trading_dates, "HK")]
    assert result["status"] == "completed"
    assert result["processed"] == 2
    assert result["missing_dates"] == 2
