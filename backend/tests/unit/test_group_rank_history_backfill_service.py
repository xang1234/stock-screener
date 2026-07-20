"""Tests for canonical recent group-rank history backfill."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.group_rank_history_backfill_service import (
    DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS,
    GroupRankHistoryBackfillService,
    GroupRankHistoryBackfillStatus,
)
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.services.group_rank_snapshot_coordinator import (
    GroupBackfillReport,
    GroupSnapshotResult,
    GroupSnapshotStatus,
)


def test_backfill_skips_market_without_group_rankings():
    calendar = MagicMock()
    coordinator = MagicMock()

    def session_factory():
        raise AssertionError("group-less backfill must not open a database session")

    service = GroupRankHistoryBackfillService(
        session_factory=session_factory,
        calendar_service=calendar,
        group_snapshot_coordinator=coordinator,
    )

    result = service.backfill(
        as_of_date=date(2026, 4, 7),
        market="DE",
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert result.status is GroupRankHistoryBackfillStatus.SKIPPED
    assert result.reason == "group_rankings_not_supported"
    assert result.ready_for_enrichment is True
    calendar.trading_days.assert_not_called()
    coordinator.backfill.assert_not_called()


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

    def backfill(db_arg, *, identities, continue_on_error):
        identities = tuple(identities)
        fill_calls.append((db_arg, identities, continue_on_error))
        return GroupBackfillReport(
            results=tuple(
                GroupSnapshotResult(
                    identity=identity,
                    status=GroupSnapshotStatus.PROCESSED,
                    row_count=1,
                    market_rs_run_id=42,
                )
                for identity in identities
            )
        )

    service = GroupRankHistoryBackfillService(
        session_factory=session_factory,
        calendar_service=SimpleNamespace(trading_days=resolve_trading_days),
        group_snapshot_coordinator=SimpleNamespace(backfill=backfill),
    )

    result = service.backfill(
        as_of_date=date(2026, 4, 7),
        market="hk",
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert range_calls == [
        (
            "HK",
            date(2026, 4, 7) - timedelta(days=DEFAULT_GROUP_RANK_HISTORY_LOOKBACK_DAYS),
            date(2026, 4, 7),
        )
    ]
    assert fill_calls[0][0] is db
    assert [identity.as_of_date for identity in fill_calls[0][1]] == trading_dates
    assert {identity.formula_version for identity in fill_calls[0][1]} == {
        BALANCED_RS_FORMULA_VERSION
    }
    assert fill_calls[0][2] is True
    assert result.status is GroupRankHistoryBackfillStatus.COMPLETED
    assert result.processed == 2
    assert result.missing_dates == 2


def test_backfill_treats_returned_gap_fill_errors_as_errored():
    query = MagicMock()
    query.filter.return_value = query
    query.distinct.return_value = query
    query.all.return_value = []
    db = MagicMock()
    db.query.return_value = query

    @contextmanager
    def session_factory():
        yield db

    service = GroupRankHistoryBackfillService(
        session_factory=session_factory,
        calendar_service=SimpleNamespace(
            trading_days=lambda _market, _start, end: [end]
        ),
        group_snapshot_coordinator=SimpleNamespace(
            backfill=lambda _db, *, identities, continue_on_error: GroupBackfillReport(
                results=tuple(
                    GroupSnapshotResult(
                        identity=identity,
                        status=GroupSnapshotStatus.ERRORED,
                        row_count=0,
                        market_rs_run_id=None,
                        error="benchmark unavailable",
                    )
                    for identity in identities
                )
            )
        ),
    )

    result = service.backfill(
        as_of_date=date(2026, 4, 7),
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert result.status is GroupRankHistoryBackfillStatus.ERRORED
    assert result.ready_for_enrichment is False
    assert result.as_dict()["error"] == "benchmark unavailable"
