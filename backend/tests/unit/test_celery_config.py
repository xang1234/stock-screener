"""
Tests for Celery schedule helpers and related settings validators.
"""
import pytest
from pydantic import ValidationError

from app.celery_app import _offset_schedule


# ---------------------------------------------------------------------------
# _offset_schedule helper
# ---------------------------------------------------------------------------

class TestOffsetSchedule:
    """Tests for minute-carry arithmetic in _offset_schedule."""

    @pytest.mark.parametrize(
        "hour, minute, offset, expected",
        [
            # No carry
            (17, 30, 5, (17, 35)),
            (17, 30, 10, (17, 40)),
            (9, 0, 15, (9, 15)),
            # Exact boundary (no carry at 60)
            (17, 50, 10, (18, 0)),
            # Single carry
            (17, 55, 5, (18, 0)),
            (17, 55, 10, (18, 5)),
            (17, 59, 1, (18, 0)),
            (17, 59, 2, (18, 1)),
            # Large offset crossing multiple hours: 59 + 61 = 120 min = 2h 0m
            (17, 59, 61, (19, 0)),
            # Midnight wrap
            (23, 55, 10, (0, 5)),
            (23, 50, 70, (1, 0)),
            # Zero offset
            (12, 30, 0, (12, 30)),
        ],
        ids=[
            "no-carry-5",
            "no-carry-10",
            "no-carry-from-zero",
            "exact-60",
            "carry-55+5",
            "carry-55+10",
            "carry-59+1",
            "carry-59+2",
            "large-offset-multi-hour",
            "midnight-wrap",
            "midnight-wrap-large",
            "zero-offset",
        ],
    )
    def test_offset_schedule(self, hour, minute, offset, expected):
        assert _offset_schedule(hour, minute, offset) == expected

    def test_full_day_wrap(self):
        """Offset that wraps past 24 hours."""
        # 23:30 + 1440 (24 hours) = 23:30 next day → (23, 30)
        assert _offset_schedule(23, 30, 1440) == (23, 30)

    def test_double_carry(self):
        """59 minutes + 121 offset = 180 total = 3 hours exactly."""
        h, m = _offset_schedule(10, 59, 121)
        assert (h, m) == (13, 0)


# ---------------------------------------------------------------------------
# Settings validators for cache_warm_hour / cache_warm_minute
# ---------------------------------------------------------------------------

class TestSettingsValidators:
    """Tests for Pydantic field validators on schedule fields."""

    def _make_settings(self, **overrides):
        """Build a Settings instance with env overrides (no .env file)."""
        from app.config.settings import Settings

        defaults = {
            "database_url": "sqlite:///test.db",
            "cache_warm_hour": 17,
            "cache_warm_minute": 30,
        }
        defaults.update(overrides)
        return Settings(**defaults)

    def test_valid_hour_boundaries(self):
        s = self._make_settings(cache_warm_hour=0)
        assert s.cache_warm_hour == 0
        s = self._make_settings(cache_warm_hour=23)
        assert s.cache_warm_hour == 23

    def test_valid_minute_boundaries(self):
        s = self._make_settings(cache_warm_minute=0)
        assert s.cache_warm_minute == 0
        s = self._make_settings(cache_warm_minute=59)
        assert s.cache_warm_minute == 59

    def test_invalid_hour_too_high(self):
        with pytest.raises(ValidationError, match="cache_warm_hour must be 0-23"):
            self._make_settings(cache_warm_hour=24)

    def test_invalid_hour_negative(self):
        with pytest.raises(ValidationError, match="cache_warm_hour must be 0-23"):
            self._make_settings(cache_warm_hour=-1)

    def test_invalid_minute_too_high(self):
        with pytest.raises(ValidationError, match="cache_warm_minute must be 0-59"):
            self._make_settings(cache_warm_minute=60)

    def test_invalid_minute_negative(self):
        with pytest.raises(ValidationError, match="cache_warm_minute must be 0-59"):
            self._make_settings(cache_warm_minute=-1)
