"""Golden regression tests for the RRG (Relative Rotation Graph) math.

These exercise the pure functions in ``app.services.rrg_service`` that turn a
daily ``avg_rs_rating`` series into RS-Ratio (x) / RS-Momentum (y) coordinates
plus a weekly tail. The math is deterministic float arithmetic with no DB and no
RNG, so the snapshots below are stable.

The module is intentionally free of heavy (sqlalchemy/DB) imports at module top,
so this file imports only the pure surface and runs without the app bootstrap.
"""

from __future__ import annotations

from datetime import date, timedelta

from app.services.rrg_service import (
    MIN_TAIL_WEEKS,
    RRGParams,
    _bucket_weekly,
    _ema,
    classify_quadrant,
    compute_group_rrg,
)


def _daily(n_days: int, value_fn, start: date = date(2024, 1, 1)):
    """Build a deterministic daily (date, value) series over ``n_days`` business
    days, value = ``value_fn(business_day_index)``."""
    out = []
    d = start
    i = 0
    while len(out) < n_days:
        if d.weekday() < 5:  # Mon..Fri
            out.append((d, float(value_fn(i))))
            i += 1
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Quadrant classification — the four RRG quadrants split at the 100/100 cross
# ---------------------------------------------------------------------------


def test_classify_quadrant_maps_each_corner():
    # x = RS-Ratio, y = RS-Momentum; boundary value 100 belongs to the
    # strong/rising side (>=).
    assert classify_quadrant(105.0, 103.0) == "Leading"     # strong + rising
    assert classify_quadrant(105.0, 97.0) == "Weakening"    # strong + falling
    assert classify_quadrant(95.0, 97.0) == "Lagging"       # weak + falling
    assert classify_quadrant(95.0, 103.0) == "Improving"    # weak + rising


def test_classify_quadrant_boundaries_are_inclusive_on_strong_side():
    # Exactly on the cross counts as Leading (>=100 on both axes).
    assert classify_quadrant(100.0, 100.0) == "Leading"
    assert classify_quadrant(100.0, 99.9) == "Weakening"
    assert classify_quadrant(99.9, 100.0) == "Improving"


# ---------------------------------------------------------------------------
# Weekly bucketing — UTC Sunday-origin week start, close-of-week value
# (mirrors the frontend aggregateToWeekly rule fixed in commit e4406584).
# ---------------------------------------------------------------------------


def test_bucket_weekly_uses_utc_sunday_start_and_close_of_week():
    # Mon 2024-01-01 .. Fri 2024-01-05 all fall in the week starting Sun 2023-12-31.
    days = [
        (date(2024, 1, 1), 10.0),
        (date(2024, 1, 2), 11.0),
        (date(2024, 1, 3), 12.0),
        (date(2024, 1, 4), 13.0),
        (date(2024, 1, 5), 14.0),
    ]
    assert _bucket_weekly(days) == [(date(2023, 12, 31), 14.0)]


def test_bucket_weekly_one_point_per_week_ascending():
    # Unsorted input spanning two weeks -> one close-of-week point per week,
    # ascending by week start.
    days = [
        (date(2024, 1, 12), 22.0),  # Fri, week of 2024-01-07
        (date(2024, 1, 5), 14.0),   # Fri, week of 2023-12-31
        (date(2024, 1, 8), 20.0),   # Mon, week of 2024-01-07
    ]
    assert _bucket_weekly(days) == [
        (date(2023, 12, 31), 14.0),
        (date(2024, 1, 7), 22.0),
    ]


def test_bucket_weekly_sunday_maps_to_itself():
    # A Sunday is its own week start (getUTCDay() == 0).
    assert _bucket_weekly([(date(2024, 1, 7), 5.0)]) == [(date(2024, 1, 7), 5.0)]


# ---------------------------------------------------------------------------
# EMA — adjust=False recursion (seed = first value), matches pandas ewm.
# ---------------------------------------------------------------------------


def test_ema_seeds_on_first_value_and_recurses():
    # span=2 -> alpha = 2/3. Hand-computed:
    #   e0 = 1
    #   e1 = 2/3*2 + 1/3*1 = 5/3
    #   e2 = 2/3*3 + 1/3*5/3 = 23/9
    out = _ema([1.0, 2.0, 3.0], span=2)
    assert len(out) == 3
    assert abs(out[0] - 1.0) < 1e-12
    assert abs(out[1] - 5.0 / 3.0) < 1e-12
    assert abs(out[2] - 23.0 / 9.0) < 1e-12


def test_ema_constant_series_is_flat():
    assert _ema([7.0, 7.0, 7.0, 7.0], span=5) == [7.0, 7.0, 7.0, 7.0]


def test_ema_empty_returns_empty():
    assert _ema([], span=3) == []


# ---------------------------------------------------------------------------
# Full pipeline behavior — compute_group_rrg
# ---------------------------------------------------------------------------


def test_compute_group_rrg_flat_series_centers_at_100():
    # A perfectly flat RS series sits dead-center: x == y == 100 (EPS guard on
    # the zero-variance window) -> Leading by the inclusive boundary rule.
    series = _daily(200, lambda i: 50.0)  # ~40 weeks
    result = compute_group_rrg(series, RRGParams())
    assert result is not None
    cur = result["current"]
    assert cur["x"] == 100.0
    assert cur["y"] == 100.0
    assert result["is_provisional"] is False  # ~40 weeks >= MIN_WEEKS
    assert classify_quadrant(cur["x"], cur["y"]) == "Leading"


def test_compute_group_rrg_returns_none_for_short_history():
    # Fewer than MIN_TAIL_WEEKS weekly points -> not enough to plot.
    series = _daily(5 * (MIN_TAIL_WEEKS - 2), lambda i: 50.0)  # ~10 weeks
    assert compute_group_rrg(series, RRGParams()) is None


def test_compute_group_rrg_default_tail_length():
    series = _daily(200, lambda i: 50.0)
    result = compute_group_rrg(series, RRGParams())
    assert len(result["tail"]) == RRGParams().tail_weeks  # default 8


def test_compute_group_rrg_rising_series_is_leading():
    # Steadily rising RS -> recent value above its own trailing mean (x>100) and
    # still climbing (y>100) -> Leading.
    series = _daily(200, lambda i: 40.0 + 0.1 * i)
    result = compute_group_rrg(series, RRGParams())
    cur = result["current"]
    assert cur["x"] > 100.0
    assert cur["y"] > 100.0
    assert classify_quadrant(cur["x"], cur["y"]) == "Leading"


def test_compute_group_rrg_falling_series_is_lagging():
    # Steadily falling RS -> below own trailing mean (x<100) and still dropping
    # (y<100) -> Lagging.
    series = _daily(200, lambda i: 60.0 - 0.1 * i)
    result = compute_group_rrg(series, RRGParams())
    cur = result["current"]
    assert cur["x"] < 100.0
    assert cur["y"] < 100.0
    assert classify_quadrant(cur["x"], cur["y"]) == "Lagging"


def test_compute_group_rrg_tail_points_are_chronological():
    series = _daily(200, lambda i: 40.0 + 0.1 * i)
    tail = compute_group_rrg(series, RRGParams())["tail"]
    dates = [p["date"] for p in tail]
    assert dates == sorted(dates)
    assert tail[-1] == compute_group_rrg(series, RRGParams())["current"]


# ---------------------------------------------------------------------------
# Golden numeric pins — lock the exact transform output so a future refactor
# can't silently shift the math. Captured from the reference implementation;
# regenerate intentionally if the methodology changes.
# ---------------------------------------------------------------------------


def test_golden_leading_archetype_pinned():
    # Steady linear RS rise from 40 -> ~60 over ~40 weeks.
    series = _daily(200, lambda i: 40.0 + 0.1 * i)
    result = compute_group_rrg(series, RRGParams())
    assert result["current"] == {"date": "2024-09-29", "x": 108.334, "y": 106.0963}
    assert result["tail"][0] == {"date": "2024-08-11", "x": 108.3453, "y": 97.7609}
    assert result["is_provisional"] is False


def test_golden_lagging_archetype_pinned():
    series = _daily(200, lambda i: 60.0 - 0.1 * i)
    result = compute_group_rrg(series, RRGParams())
    assert result["current"] == {"date": "2024-09-29", "x": 91.666, "y": 93.9037}


def test_golden_provisional_short_history_flagged():
    # ~20 weeks: enough to plot (>= MIN_TAIL_WEEKS) but below MIN_WEEKS, so the
    # most-recent point used a shortened window -> provisional.
    series = _daily(100, lambda i: 40.0 + 0.1 * i)
    result = compute_group_rrg(series, RRGParams())
    assert result["is_provisional"] is True
    assert result["current"] == {"date": "2024-05-12", "x": 108.5403, "y": 96.4545}
