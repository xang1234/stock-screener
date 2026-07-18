from datetime import UTC, date, datetime
import hashlib

import pytest

from app.models.stock_universe import (
    StockUniverse,
    StockUniverseStatusEvent,
    UNIVERSE_EVENT_STATUS_CHANGED,
    UNIVERSE_STATUS_ACTIVE,
    UNIVERSE_STATUS_INACTIVE_MANUAL,
)
from app.services.point_in_time_universe_service import (
    PointInTimeUniverseService,
    PointInTimeUniverseUnavailable,
)


class _CalendarStub:
    def __init__(self, latest: date):
        self.latest = latest

    @staticmethod
    def normalize_market(market: str) -> str:
        return market.upper()

    def last_completed_trading_day(self, _market: str) -> date:
        return self.latest


def _universe_row(
    symbol: str,
    *,
    first_seen_at: datetime,
    status: str = UNIVERSE_STATUS_ACTIVE,
    is_active: bool = True,
) -> StockUniverse:
    return StockUniverse(
        symbol=symbol,
        name=symbol,
        market="US",
        exchange="NASDAQ",
        currency="USD",
        timezone="America/New_York",
        status=status,
        is_active=is_active,
        first_seen_at=first_seen_at,
    )


def _status_event(
    symbol: str,
    status: str,
    *,
    created_at: datetime,
) -> StockUniverseStatusEvent:
    return StockUniverseStatusEvent(
        symbol=symbol,
        event_type=UNIVERSE_EVENT_STATUS_CHANGED,
        new_status=status,
        trigger_source="test",
        created_at=created_at,
    )


def test_resolve_reconstructs_historical_active_membership(db_session):
    db_session.add_all(
        [
            _universe_row(
                "ACTIVE", first_seen_at=datetime(2025, 1, 2, tzinfo=UTC)
            ),
            _universe_row(
                "DEACTIVATED",
                first_seen_at=datetime(2025, 1, 2, tzinfo=UTC),
                status=UNIVERSE_STATUS_INACTIVE_MANUAL,
                is_active=False,
            ),
            _universe_row(
                "FUTURE", first_seen_at=datetime(2026, 4, 12, tzinfo=UTC)
            ),
            _status_event(
                "ACTIVE",
                UNIVERSE_STATUS_ACTIVE,
                created_at=datetime(2025, 1, 2, 12, tzinfo=UTC),
            ),
            _status_event(
                "DEACTIVATED",
                UNIVERSE_STATUS_ACTIVE,
                created_at=datetime(2025, 1, 2, 12, tzinfo=UTC),
            ),
            _status_event(
                "DEACTIVATED",
                UNIVERSE_STATUS_INACTIVE_MANUAL,
                created_at=datetime(2026, 4, 1, 12, tzinfo=UTC),
            ),
            _status_event(
                "FUTURE",
                UNIVERSE_STATUS_ACTIVE,
                created_at=datetime(2026, 4, 12, 12, tzinfo=UTC),
            ),
        ]
    )
    db_session.commit()
    service = PointInTimeUniverseService(
        market_calendar=_CalendarStub(latest=date(2026, 4, 17))
    )

    snapshot = service.resolve(
        db_session, market="US", as_of_date=date(2026, 4, 10)
    )

    assert snapshot.symbols == ("ACTIVE",)
    assert snapshot.universe_hash == hashlib.sha256(b"ACTIVE\n").hexdigest()


def test_historical_resolve_fails_closed_when_a_candidate_has_no_status_event(
    db_session,
):
    db_session.add(
        _universe_row("UNKNOWN", first_seen_at=datetime(2025, 1, 2, tzinfo=UTC))
    )
    db_session.commit()
    service = PointInTimeUniverseService(
        market_calendar=_CalendarStub(latest=date(2026, 4, 17))
    )

    with pytest.raises(PointInTimeUniverseUnavailable, match="UNKNOWN"):
        service.resolve(db_session, market="US", as_of_date=date(2026, 4, 10))


def test_current_resolve_uses_authoritative_active_filter_without_event_history(
    db_session,
):
    db_session.add_all(
        [
            _universe_row(
                "CURRENT", first_seen_at=datetime(2025, 1, 2, tzinfo=UTC)
            ),
            _universe_row(
                "INACTIVE",
                first_seen_at=datetime(2025, 1, 2, tzinfo=UTC),
                status=UNIVERSE_STATUS_INACTIVE_MANUAL,
                is_active=False,
            ),
        ]
    )
    db_session.commit()
    service = PointInTimeUniverseService(
        market_calendar=_CalendarStub(latest=date(2026, 4, 10))
    )

    snapshot = service.resolve(
        db_session, market="US", as_of_date=date(2026, 4, 10)
    )

    assert snapshot.symbols == ("CURRENT",)
