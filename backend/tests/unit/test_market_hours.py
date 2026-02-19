"""
Unit tests for market_hours utility module.

Tests the NYSE calendar integration via pandas_market_calendars,
ensuring trading day detection, holiday handling, and timezone
correctness.
"""
import pytest
from datetime import date, datetime, time, timedelta
from unittest.mock import patch

from app.utils.market_hours import (
    is_trading_day,
    get_last_trading_day,
    is_market_open,
    get_eastern_now,
    get_last_market_close,
    get_next_market_close,
    _get_trading_days_set,
    EASTERN,
)


class TestIsTradingDay:
    """Tests for is_trading_day() with pandas_market_calendars."""

    def test_weekdays_are_trading_days(self):
        """Normal weekdays should be trading days (barring holidays)."""
        # 2025-02-10 is a Monday
        assert is_trading_day(date(2025, 2, 10)) is True
        # 2025-02-14 is a Friday
        assert is_trading_day(date(2025, 2, 14)) is True

    def test_saturdays_are_not_trading_days(self):
        assert is_trading_day(date(2025, 2, 15)) is False

    def test_sundays_are_not_trading_days(self):
        assert is_trading_day(date(2025, 2, 16)) is False

    def test_christmas_2025_not_trading_day(self):
        assert is_trading_day(date(2025, 12, 25)) is False

    def test_new_years_2026_not_trading_day(self):
        assert is_trading_day(date(2026, 1, 1)) is False

    def test_good_friday_2026_not_trading_day(self):
        """Good Friday 2026 is April 3."""
        assert is_trading_day(date(2026, 4, 3)) is False

    def test_mlk_day_2026_not_trading_day(self):
        """MLK Day 2026 is January 19."""
        assert is_trading_day(date(2026, 1, 19)) is False

    def test_juneteenth_2025_not_trading_day(self):
        assert is_trading_day(date(2025, 6, 19)) is False

    def test_thanksgiving_2025_not_trading_day(self):
        assert is_trading_day(date(2025, 11, 27)) is False

    def test_works_for_2027_and_beyond(self):
        """pandas_market_calendars should cover future years automatically."""
        # Christmas 2027 is Saturday, observed Friday Dec 24
        assert is_trading_day(date(2027, 12, 24)) is False
        # New Year's 2028
        assert is_trading_day(date(2027, 12, 31)) is True  # Friday before New Year

    def test_defaults_to_today(self):
        """When called with no args, should not raise."""
        result = is_trading_day()
        assert isinstance(result, bool)


class TestGetLastTradingDay:
    """Tests for get_last_trading_day()."""

    def test_returns_same_day_if_trading_day(self):
        """A normal Tuesday should return itself."""
        tuesday = date(2025, 2, 11)  # Tuesday
        assert get_last_trading_day(tuesday) == tuesday

    def test_saturday_returns_friday(self):
        saturday = date(2025, 2, 15)
        friday = date(2025, 2, 14)
        assert get_last_trading_day(saturday) == friday

    def test_sunday_returns_friday(self):
        sunday = date(2025, 2, 16)
        friday = date(2025, 2, 14)
        assert get_last_trading_day(sunday) == friday

    def test_three_day_weekend_returns_friday(self):
        """MLK Day 2026 (Monday Jan 19) should return Friday Jan 16."""
        mlk_day = date(2026, 1, 19)
        friday_before = date(2026, 1, 16)
        assert get_last_trading_day(mlk_day) == friday_before

    def test_day_after_holiday_with_weekend(self):
        """Tuesday after MLK Day 2026 should return itself (it's a trading day)."""
        tuesday = date(2026, 1, 20)
        assert get_last_trading_day(tuesday) == tuesday

    def test_defaults_to_today(self):
        result = get_last_trading_day()
        assert isinstance(result, date)


class TestIsMarketOpen:
    """Tests for is_market_open() with holiday awareness."""

    def test_open_during_market_hours(self):
        """10:00 AM ET on a normal Tuesday should be open."""
        dt = EASTERN.localize(datetime(2025, 2, 11, 10, 0))
        assert is_market_open(dt) is True

    def test_closed_before_open(self):
        dt = EASTERN.localize(datetime(2025, 2, 11, 9, 0))
        assert is_market_open(dt) is False

    def test_closed_after_close(self):
        dt = EASTERN.localize(datetime(2025, 2, 11, 16, 30))
        assert is_market_open(dt) is False

    def test_closed_on_weekend(self):
        dt = EASTERN.localize(datetime(2025, 2, 15, 12, 0))  # Saturday noon
        assert is_market_open(dt) is False

    def test_closed_on_holiday(self):
        """Christmas 2025 (Thursday) at 10 AM should be closed."""
        dt = EASTERN.localize(datetime(2025, 12, 25, 10, 0))
        assert is_market_open(dt) is False

    def test_closed_on_good_friday_2026(self):
        dt = EASTERN.localize(datetime(2026, 4, 3, 10, 0))
        assert is_market_open(dt) is False


class TestGetLastMarketClose:
    """Tests for get_last_market_close()."""

    def test_after_close_on_trading_day(self):
        """After 4 PM on a trading day should return today's close."""
        dt = EASTERN.localize(datetime(2025, 2, 11, 17, 0))  # 5 PM Tuesday
        result = get_last_market_close(dt)
        assert result.date() == date(2025, 2, 11)
        assert result.time() == time(16, 0)

    def test_on_saturday_returns_friday_close(self):
        dt = EASTERN.localize(datetime(2025, 2, 15, 12, 0))  # Saturday noon
        result = get_last_market_close(dt)
        assert result.date() == date(2025, 2, 14)  # Friday


class TestGetNextMarketClose:
    """Tests for get_next_market_close()."""

    def test_during_market_hours_returns_today(self):
        dt = EASTERN.localize(datetime(2025, 2, 11, 10, 0))
        result = get_next_market_close(dt)
        assert result.date() == date(2025, 2, 11)
        assert result.time() == time(16, 0)

    def test_on_weekend_skips_holiday(self):
        """Saturday Feb 15 -> Monday Feb 17 is Presidents' Day -> Tuesday Feb 18."""
        dt = EASTERN.localize(datetime(2025, 2, 15, 12, 0))  # Saturday
        result = get_next_market_close(dt)
        # Presidents Day 2025 is Feb 17 (Monday) — a holiday!
        # So next trading day is Tuesday Feb 18
        assert result.date() == date(2025, 2, 18)


class TestTradingDaysCache:
    """Tests for the internal caching mechanism."""

    def test_cache_returns_set(self):
        result = _get_trading_days_set()
        assert isinstance(result, set)
        assert len(result) > 200  # At least 200 trading days in a 3-year window

    def test_cache_recomputes_on_year_change(self):
        """Verify cache invalidates when year changes."""
        import app.utils.market_hours as mh

        # Get initial cache
        _get_trading_days_set()
        initial_year = mh._cache_year

        # Simulate year change
        old_year = mh._cache_year
        mh._cache_year = old_year - 1  # Force a "stale" year

        # Should recompute
        _get_trading_days_set()
        assert mh._cache_year == initial_year  # Restored to current year
