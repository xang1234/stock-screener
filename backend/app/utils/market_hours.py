"""
Market hours utilities for cache staleness detection.

Provides functions to determine if US stock market is open,
when it closes, and if cached data is stale based on market hours.

Uses pandas_market_calendars for authoritative NYSE holiday/schedule data.
"""
from datetime import datetime, date, time, timedelta
from typing import Optional
import pytz
import pandas_market_calendars as mcal


# US Eastern timezone
EASTERN = pytz.timezone('US/Eastern')

# Market hours (all times in Eastern)
MARKET_OPEN_TIME = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = time(16, 0)  # 4:00 PM ET

# NYSE calendar singleton
_NYSE = mcal.get_calendar("NYSE")

# Cache: pre-compute 3-year window of valid trading days for O(1) lookups
_trading_days_cache: Optional[set] = None
_cache_year: Optional[int] = None


def _get_trading_days_set() -> set:
    """Get set of trading days for current year +/- 1 year. Recomputes on year change."""
    global _trading_days_cache, _cache_year
    current_year = get_eastern_now().year
    if _trading_days_cache is None or _cache_year != current_year:
        start = f"{current_year - 1}-01-01"
        end = f"{current_year + 1}-12-31"
        schedule = _NYSE.schedule(start_date=start, end_date=end)
        _trading_days_cache = set(schedule.index.date)
        _cache_year = current_year
    return _trading_days_cache


def get_eastern_now() -> datetime:
    """Get current time in US Eastern timezone."""
    return datetime.now(EASTERN)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if US stock market is currently open.

    Args:
        dt: Datetime to check (defaults to now in ET)

    Returns:
        True if market is open, False otherwise

    Market is open:
    - Monday-Friday
    - 9:30 AM - 4:00 PM ET
    - Not on market holidays
    """
    if dt is None:
        dt = get_eastern_now()
    elif dt.tzinfo is None:
        # Convert naive datetime to Eastern
        dt = EASTERN.localize(dt)
    else:
        # Convert to Eastern timezone
        dt = dt.astimezone(EASTERN)

    # Check if this is a trading day (handles weekends + holidays)
    if not is_trading_day(dt.date()):
        return False

    # Check if within market hours
    current_time = dt.time()
    return MARKET_OPEN_TIME <= current_time < MARKET_CLOSE_TIME


def get_next_market_close(dt: Optional[datetime] = None) -> datetime:
    """
    Get the next market close time.

    Args:
        dt: Starting datetime (defaults to now in ET)

    Returns:
        Datetime of next market close (4:00 PM ET on next trading day)
    """
    if dt is None:
        dt = get_eastern_now()
    elif dt.tzinfo is None:
        dt = EASTERN.localize(dt)
    else:
        dt = dt.astimezone(EASTERN)

    # If market is currently open, return today's close
    if is_market_open(dt):
        return EASTERN.localize(
            datetime.combine(dt.date(), MARKET_CLOSE_TIME)
        )

    # Otherwise, find next trading day
    current_date = dt.date()
    max_days = 10  # Look ahead maximum 10 days

    for i in range(1, max_days):
        next_date = current_date + timedelta(days=i)
        next_dt = EASTERN.localize(
            datetime.combine(next_date, MARKET_CLOSE_TIME)
        )

        # Check if this is a trading day (handles weekends + holidays)
        if is_trading_day(next_date):
            return next_dt

    # Fallback: return tomorrow at close
    return EASTERN.localize(
        datetime.combine(
            current_date + timedelta(days=1),
            MARKET_CLOSE_TIME
        )
    )


def get_last_market_close(dt: Optional[datetime] = None) -> datetime:
    """
    Get the most recent market close time.

    Args:
        dt: Reference datetime (defaults to now in ET)

    Returns:
        Datetime of most recent market close
    """
    if dt is None:
        dt = get_eastern_now()
    elif dt.tzinfo is None:
        dt = EASTERN.localize(dt)
    else:
        dt = dt.astimezone(EASTERN)

    current_date = dt.date()
    current_time = dt.time()

    # If today is a trading day and market has closed, return today's close
    if is_trading_day(current_date) and current_time >= MARKET_CLOSE_TIME:
        return EASTERN.localize(
            datetime.combine(current_date, MARKET_CLOSE_TIME)
        )

    # Otherwise, look backwards for last trading day
    max_days = 10

    for i in range(1, max_days):
        prev_date = current_date - timedelta(days=i)

        if is_trading_day(prev_date):
            return EASTERN.localize(
                datetime.combine(prev_date, MARKET_CLOSE_TIME)
            )

    # Fallback: return yesterday at close
    return EASTERN.localize(
        datetime.combine(
            current_date - timedelta(days=1),
            MARKET_CLOSE_TIME
        )
    )


def is_data_stale(
    last_update: datetime,
    reference_time: Optional[datetime] = None,
    max_age_hours: int = 24
) -> bool:
    """
    Determine if cached data is stale based on market hours.

    Args:
        last_update: When data was last updated
        reference_time: Current time (defaults to now in ET)
        max_age_hours: Maximum age in hours before considered stale

    Returns:
        True if data is stale and should be refreshed

    Staleness rules:
    - During market hours: Stale if older than 1 hour
    - After market close: Valid until next market close + 1 hour
    - Weekends: Friday's close data is fresh until Monday 5 PM ET
    - Holidays: Previous trading day's data is fresh
    """
    if reference_time is None:
        reference_time = get_eastern_now()
    elif reference_time.tzinfo is None:
        reference_time = EASTERN.localize(reference_time)
    else:
        reference_time = reference_time.astimezone(EASTERN)

    # Ensure last_update is timezone-aware
    if last_update.tzinfo is None:
        last_update = EASTERN.localize(last_update)
    else:
        last_update = last_update.astimezone(EASTERN)

    # Calculate age
    age = reference_time - last_update

    # If market is currently open, use stricter staleness check
    if is_market_open(reference_time):
        # During market hours, data older than 1 hour is stale
        return age > timedelta(hours=1)

    # Market is closed - check if data is from last market close or later
    last_close = get_last_market_close(reference_time)

    # If data is newer than last close, it's fresh
    if last_update >= last_close:
        return False

    # If data is older than last close, check max age
    return age > timedelta(hours=max_age_hours)


def should_refresh_cache(
    last_update: Optional[datetime] = None,
    reference_time: Optional[datetime] = None
) -> bool:
    """
    Determine if cache should be refreshed now.

    Args:
        last_update: When cache was last updated (None means never)
        reference_time: Current time (defaults to now)

    Returns:
        True if cache should be refreshed

    Refresh strategy:
    - Never updated: Always refresh
    - Market open: Refresh every hour
    - Market closed: Refresh once after close
    - Weekend/Holiday: No refresh needed
    """
    if last_update is None:
        return True  # Never cached - refresh now

    if reference_time is None:
        reference_time = get_eastern_now()

    # Use staleness check
    return is_data_stale(last_update, reference_time)


def is_trading_day(d: date = None) -> bool:
    """Check if a date is a valid trading day (not weekend, not holiday)."""
    if d is None:
        d = get_eastern_now().date()
    return d in _get_trading_days_set()


def get_last_trading_day(d: date = None) -> date:
    """Get the most recent trading day on or before the given date."""
    if d is None:
        d = get_eastern_now().date()

    if is_trading_day(d):
        return d

    for i in range(1, 10):
        prev_date = d - timedelta(days=i)
        if is_trading_day(prev_date):
            return prev_date

    return d - timedelta(days=1)


def get_seconds_until_market_close(dt: Optional[datetime] = None) -> int:
    """
    Get seconds until next market close.

    Args:
        dt: Reference time (defaults to now)

    Returns:
        Seconds until market close (0 if market is closed)
    """
    if dt is None:
        dt = get_eastern_now()

    if not is_market_open(dt):
        return 0

    next_close = get_next_market_close(dt)
    delta = next_close - dt
    return int(delta.total_seconds())


def format_market_status() -> str:
    """
    Get human-readable market status.

    Returns:
        String describing current market status
    """
    now = get_eastern_now()

    if is_market_open(now):
        seconds_left = get_seconds_until_market_close(now)
        minutes_left = seconds_left // 60
        return f"Market OPEN - Closes in {minutes_left} minutes"
    else:
        next_close = get_next_market_close(now)
        return f"Market CLOSED - Next close: {next_close.strftime('%Y-%m-%d %I:%M %p ET')}"
