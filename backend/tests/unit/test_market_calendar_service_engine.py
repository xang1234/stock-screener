from datetime import date, datetime

import pytest

from app.services import market_calendar_service as module
from app.services.market_calendar_service import MarketCalendarService


pytestmark = pytest.mark.skipif(
    module.xcals is None,
    reason="exchange_calendars is required for engine-level calendar tests",
)


def test_calendar_ids_are_canonical_for_supported_markets():
    service = MarketCalendarService()

    assert service.calendar_id("US") == "XNYS"
    assert service.calendar_id("HK") == "XHKG"
    assert service.calendar_id("IN") == "XNSE"
    assert service.calendar_id("JP") == "XTKS"
    assert service.calendar_id("TW") == "XTAI"
    assert service.calendar_id("SG") == "XSES"


def test_weekend_is_non_trading_day_for_all_supported_markets():
    service = MarketCalendarService()
    saturday = date(2026, 4, 11)

    for market in ("US", "HK", "JP", "TW", "SG"):
        assert service.is_trading_day(market, saturday) is False


def test_last_completed_session_on_sunday_returns_previous_friday():
    service = MarketCalendarService()
    sunday_utc = datetime.fromisoformat("2026-04-12T12:00:00+00:00")

    for market in ("US", "HK", "JP", "TW", "SG"):
        assert service.last_completed_trading_day(market, now=sunday_utc) == date(2026, 4, 10)
