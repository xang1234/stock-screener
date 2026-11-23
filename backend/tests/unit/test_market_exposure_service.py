"""Unit tests for the market exposure rubric, the market cycle and DB round-trip."""
from datetime import date

import pandas as pd
import pytest

import app.services.market_exposure_service as svc
from app.domain.markets.market_cycle import CycleBar, CycleState, MarketCycle, market_cycle
from app.services.benchmark_cache_service import BenchmarkFallbackPolicy, BenchmarkResolution
from app.services.market_exposure_service import (
    CAP_BELOW_200DMA,
    CAP_HEAVY_DISTRIBUTION,
    FTD_FLOOR,
    backfill_exposure,
    build_exposure_payload,
    compute_and_store,
    count_distribution_days,
    compute_trend,
    refresh_market_exposure_for_date,
    _score,
    _stance,
)

# Trend dicts at the extremes (price relative to 50/200-DMA).
_STRONG_UPTREND = {"price": 120.0, "ma50": 110.0, "ma200": 100.0}  # well above rising MAs
_DEEP_DOWNTREND = {"price": 85.0, "ma50": 95.0, "ma200": 110.0}    # below falling MAs


def _df(closes, volumes, end=None):
    """Build a minimal OHLCV DataFrame with a DatetimeIndex.

    ``end`` pins the last bar's date (so a backfill's trading days fall inside
    the frame); otherwise the frame starts at a fixed 2024 date.
    """
    if end is not None:
        idx = pd.date_range(end=end, periods=len(closes), freq="D")
    else:
        idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": volumes},
        index=idx,
    )


UP, CORR, ATT = CycleState.UPTREND, CycleState.CORRECTION, CycleState.RALLY_ATTEMPT


def _bars(closes, lows=None, volumes=None):
    """Daily bars from 2024-01-01; Low defaults to Close and Volume to 1000."""
    lows = closes if lows is None else lows
    volumes = [1000] * len(closes) if volumes is None else volumes
    priors = [float("nan"), *volumes[:-1]]
    return [CycleBar(_day(i), *bar) for i, bar in enumerate(zip(closes, lows, volumes, priors))]


def _day(i):
    return date(2024, 1, 1 + i)


# Peak 100, correction on bar 1 (rally low 91), then three rally days.
_ATTEMPT_TO_DAY_3 = [100, 91, 91.5, 91.6, 91.7]


def test_cycle_enters_correction_on_a_close_8pct_below_the_highest_close():
    # 110 * 0.92 = 101.2: 101.3 holds, 101.1 enters at that bar's intraday low.
    cycles = market_cycle(_bars([100, 110, 101.3, 101.1], lows=[99, 109, 101, 100.5]))
    assert [c.state for c in cycles] == [UP, UP, UP, CORR]
    assert cycles[3] == MarketCycle(CORR, rally_low=100.5, rally_low_date=_day(3))


def test_cycle_undercut_in_correction_moves_the_rally_low():
    cycles = market_cycle(_bars([100, 91, 90, 89.5, 89.4], lows=[100, 91, 90, 89, 89.2]))
    assert [c.state for c in cycles] == [UP, CORR, CORR, CORR, CORR]
    assert [(c.rally_low, c.rally_low_date) for c in cycles[1:]] == [
        (91, _day(1)), (90, _day(2)), (89, _day(3)), (89, _day(3)),
    ]


def test_cycle_rally_day_1_is_the_first_up_close_and_every_later_session_counts():
    cycles = market_cycle(_bars([100, 91, 91.5, 91.2, 91.3]))
    assert [c.state for c in cycles] == [UP, CORR, ATT, ATT, ATT]
    assert [c.rally_day for c in cycles] == [None, None, 1, 2, 3]


def test_cycle_undercut_resets_the_rally_attempt():
    cycles = market_cycle(_bars(_ATTEMPT_TO_DAY_3 + [91.8], lows=_ATTEMPT_TO_DAY_3 + [90.5]))
    assert cycles[-1] == MarketCycle(CORR, rally_low=90.5, rally_low_date=_day(5))


def test_cycle_follow_through_on_day_4_with_gain_and_rising_volume():
    cycles = market_cycle(_bars(_ATTEMPT_TO_DAY_3 + [93.2], volumes=[1000] * 5 + [1500]))
    assert cycles[4].rally_day == 3
    assert cycles[5] == MarketCycle(UP, rally_low=91, rally_low_date=_day(1), ftd_date=_day(5))


@pytest.mark.parametrize(
    "prior, close, confirms",
    [(1000, 1014.99, False), (1000, 1015, True), (1000, 1015.01, True), (100, 101.5, True)],
)
def test_cycle_follow_through_gain_is_inclusive_at_1_5_pct(prior, close, confirms):
    # Peak, correction, three rally days ending at ``prior``, then day 4.
    closes = [prior * 1.2, prior * 0.995, prior * 0.998, prior * 0.999, prior, close]
    last = market_cycle(_bars(closes, volumes=[1000] * 5 + [1500]))[-1]
    assert (last.ftd_date is not None) is confirms


def test_cycle_no_follow_through_on_day_3():
    cycles = market_cycle(_bars([100, 91, 91.5, 91.6, 93.1], volumes=[1000] * 4 + [1500]))
    assert (cycles[-1].state, cycles[-1].rally_day, cycles[-1].ftd_date) == (ATT, 3, None)


@pytest.mark.parametrize(
    "prior_volume, volume",
    [(1000, 1000), (1000, float("nan")), (float("nan"), 1500), (0, 1500)],
)
def test_cycle_flat_missing_or_zero_volume_never_confirms(prior_volume, volume):
    cycles = market_cycle(_bars(_ATTEMPT_TO_DAY_3 + [93.2], volumes=[1000] * 4 + [prior_volume, volume]))
    assert (cycles[-1].state, cycles[-1].rally_day, cycles[-1].ftd_date) == (ATT, 4, None)


def test_cycle_follow_through_fails_when_a_close_breaks_the_rally_low():
    closes = _ATTEMPT_TO_DAY_3 + [93.2, 92.0, 90.9, 90.6]
    lows = _ATTEMPT_TO_DAY_3 + [93.2, 90.8, 90.5, 90.6]
    cycles = market_cycle(_bars(closes, lows=lows, volumes=[1000] * 5 + [1500, 1000, 1000, 1000]))
    # An intraday undercut that closes above the rally low still holds.
    assert cycles[6] == MarketCycle(UP, rally_low=91, rally_low_date=_day(1), ftd_date=_day(5))
    assert cycles[7] == MarketCycle(CORR, rally_low=90.5, rally_low_date=_day(7), failed_ftd_date=_day(7))
    assert cycles[8] == MarketCycle(CORR, rally_low=90.5, rally_low_date=_day(7))


def test_cycle_rally_attempt_expires_on_day_25_and_the_next_up_close_is_day_1():
    # Up closes of +0.05 never make a 1.5% day, so the attempt runs out.
    closes = [100, 91] + [round(91 + 0.05 * k, 2) for k in range(1, 28)]
    cycles = market_cycle(_bars(closes))
    assert (cycles[25].state, cycles[25].rally_day) == (ATT, 24)
    assert cycles[26] == MarketCycle(CORR, rally_low=91, rally_low_date=_day(1))
    assert (cycles[27].state, cycles[27].rally_day) == (ATT, 1)


def test_cycle_recovers_on_a_close_above_the_pre_correction_peak():
    cycles = market_cycle(_bars([100, 91, 92, 100.5]))
    assert cycles[3] == MarketCycle(UP)


def test_cycle_recovers_straight_from_correction():
    cycles = market_cycle(_bars([100, 91, 101]))
    assert [c.state for c in cycles] == [UP, CORR, UP]


def test_cycle_follow_through_on_day_25_confirms_before_the_attempt_expires():
    closes = [100, 91] + [round(91 + 0.05 * k, 2) for k in range(1, 25)]
    closes.append(round(closes[-1] * 1.02, 2))
    volumes = [1000] * (len(closes) - 1) + [1500]
    last = market_cycle(_bars(closes, volumes=volumes))[-1]
    assert (last.state, last.ftd_date) == (UP, _day(len(closes) - 1))


def test_count_distribution_days_counts_down_days_on_higher_volume():
    # 3 down days (-0.3%) each on rising volume -> distribution days.
    # A 4th down day on FALLING volume must NOT count; the final up day must not.
    closes = [100.0, 99.7, 99.4, 99.1, 98.8, 99.5]
    volumes = [1000, 1100, 1200, 1300, 1200, 1400]
    assert count_distribution_days(_df(closes, volumes)) == 3


def test_count_distribution_days_empty_is_zero():
    assert count_distribution_days(_df([100.0], [1000])) == 0


def test_compute_trend_downtrend_is_bearish():
    closes = list(range(250, 0, -1))  # strictly decreasing -> price < ma50 < ma200
    trend = compute_trend(_df(closes, [1000] * len(closes)))
    assert trend["trend"] == "bearish"
    assert trend["price"] < trend["ma50"] < trend["ma200"]


def test_score_tracks_trend():
    # Defining property: above rising MAs -> high; below falling MAs -> low,
    # with a wide spread (not pinned to a couple of values).
    up, _ = _score(_STRONG_UPTREND, dist_count=0, ftd=False, vix=None, net_4pct=None)
    down, comps = _score(_DEEP_DOWNTREND, dist_count=0, ftd=False, vix=None, net_4pct=None)
    assert up >= 85                       # Power Trend
    assert down <= CAP_BELOW_200DMA       # below the 200-DMA -> capped low
    assert "below_200dma_cap" in comps
    assert up - down >= 40


def test_distribution_is_a_drag_not_a_gate():
    # Distribution days lower the score within the regime, but a strong uptrend
    # with baseline-ish distribution still scores well — not pinned to 40.
    clean, _ = _score(_STRONG_UPTREND, dist_count=0, ftd=False, vix=None, net_4pct=None)
    pressured, _ = _score(_STRONG_UPTREND, dist_count=6, ftd=False, vix=None, net_4pct=None)
    assert pressured < clean
    assert pressured >= 65


def test_heavy_distribution_caps():
    # >=8 distribution days is a genuine risk overlay even in an uptrend.
    score, comps = _score(_STRONG_UPTREND, dist_count=8, ftd=False, vix=None, net_4pct=None)
    assert score <= CAP_HEAVY_DISTRIBUTION
    assert "heavy_distribution_cap" in comps


def test_stance_bands():
    assert _stance(90) == "Power Trend"
    assert _stance(10) == "Correction — In Cash"


class _FakeBundle:
    def __init__(self, df, symbol):
        self.data = df
        self.benchmark_symbol = symbol
        self.candidate_statuses = ()


def _fake_benchmark_factory(df, symbol="SPY"):
    class _FakeBenchmarkCacheService:
        def __init__(self, *a, **k):
            pass

        def resolve_benchmark_bundle(
            self,
            market="US",
            period="2y",
            force_refresh=False,
            fallback_policy=BenchmarkFallbackPolicy.ALLOW,
            required_as_of_date=None,
        ):
            if df is None:
                return BenchmarkResolution(bundle=None, error="no_benchmark_data")
            return BenchmarkResolution(bundle=_FakeBundle(df, symbol))

        def get_benchmark_bundle(self, **kwargs):
            return self.resolve_benchmark_bundle(**kwargs).bundle

    return _FakeBenchmarkCacheService


def test_refresh_market_exposure_for_date_can_use_primary_only_benchmark_policy(monkeypatch):
    from app.database import SessionLocal

    calls = []
    df = _df(list(range(100, 350)), [1000] * 250)

    class _FakeBenchmarkCacheService:
        def __init__(self, *a, **k):
            pass

        def resolve_benchmark_bundle(self, *, market, period, fallback_policy, required_as_of_date):
            calls.append(
                {
                    "market": market,
                    "period": period,
                    "fallback_policy": fallback_policy,
                    "required_as_of_date": required_as_of_date,
                }
            )
            return BenchmarkResolution(bundle=_FakeBundle(df, "SPY"))

    monkeypatch.setattr(
        "app.services.benchmark_cache_service.BenchmarkCacheService",
        _FakeBenchmarkCacheService,
    )

    as_of = df.index[-1].date()
    db = SessionLocal()
    try:
        result = refresh_market_exposure_for_date(
            db,
            "us",
            as_of,
            seed_history=False,
            benchmark_fallback_policy=BenchmarkFallbackPolicy.PRIMARY_ONLY,
        )
        assert "error" not in result
        assert result["benchmark_symbol"] == "SPY"
        assert calls == [
            {
                "market": "US",
                "period": "2y",
                "fallback_policy": BenchmarkFallbackPolicy.PRIMARY_ONLY,
                "required_as_of_date": as_of,
            }
        ]
    finally:
        db.close()


def test_refresh_market_exposure_for_date_seeds_history_with_same_benchmark_policy(monkeypatch):
    db = object()
    as_of = date(2026, 6, 25)
    calls: list[tuple] = []

    def fake_compute(
        market,
        as_of_date,
        db_arg,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        calls.append(("compute", market, as_of_date, db_arg, benchmark_fallback_policy))
        return {
            "market": market,
            "date": as_of_date,
            "exposure_score": 78.0,
            "stance": "Confirmed Uptrend",
        }

    def fake_seed(
        db_arg,
        market,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        calls.append(("seed", db_arg, market, benchmark_fallback_policy))
        return {"seeded": 4, "failed": 0}

    monkeypatch.setattr(svc, "compute_and_store", fake_compute)
    monkeypatch.setattr(svc, "ensure_exposure_history", fake_seed)

    result = refresh_market_exposure_for_date(
        db,
        "us",
        as_of,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.PRIMARY_ONLY,
    )

    assert calls == [
        ("compute", "US", as_of, db, BenchmarkFallbackPolicy.PRIMARY_ONLY),
        ("seed", db, "US", BenchmarkFallbackPolicy.PRIMARY_ONLY),
    ]
    assert result["history_seed"] == {"seeded": 4, "failed": 0}


def test_backfill_exposure_uses_requested_benchmark_policy_for_each_day(monkeypatch):
    from app.services.market_calendar_service import MarketCalendarService

    days = [date(2026, 6, 24), date(2026, 6, 25)]
    calls: list[tuple] = []
    db = object()

    monkeypatch.setattr(
        MarketCalendarService,
        "trading_days",
        lambda self, market, start, end: days,
    )

    def fake_compute(
        market,
        day,
        db_arg,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        calls.append((market, day, db_arg, benchmark_fallback_policy))
        return {"market": market, "date": day, "exposure_score": 70.0}

    monkeypatch.setattr(svc, "compute_and_store", fake_compute)

    result = backfill_exposure(
        db,
        "us",
        days[0],
        days[-1],
        benchmark_fallback_policy=BenchmarkFallbackPolicy.PRIMARY_ONLY,
    )

    assert result == {"seeded": 2, "failed": 0}
    assert calls == [
        ("US", days[0], db, BenchmarkFallbackPolicy.PRIMARY_ONLY),
        ("US", days[1], db, BenchmarkFallbackPolicy.PRIMARY_ONLY),
    ]


def test_ensure_exposure_history_passes_benchmark_policy_to_backfill(monkeypatch):
    from app.database import SessionLocal
    from app.services.market_calendar_service import MarketCalendarService

    end = date(2026, 6, 25)
    calls: list[tuple] = []

    monkeypatch.setattr(
        MarketCalendarService,
        "last_completed_trading_day",
        lambda self, market: end,
    )

    def fake_backfill(
        db_arg,
        market,
        start,
        end_arg,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        calls.append((db_arg, market, start, end_arg, benchmark_fallback_policy))
        return {"seeded": 2, "failed": 0}

    monkeypatch.setattr(svc, "backfill_exposure", fake_backfill)

    db = SessionLocal()
    try:
        result = svc.ensure_exposure_history(
            db,
            "us",
            min_rows=2,
            days=12,
            benchmark_fallback_policy=BenchmarkFallbackPolicy.PRIMARY_ONLY,
        )
    finally:
        db.close()

    assert result == {"seeded": 2, "failed": 0}
    assert calls == [
        (
            db,
            "US",
            date(2026, 6, 13),
            end,
            BenchmarkFallbackPolicy.PRIMARY_ONLY,
        )
    ]


def test_ensure_exposure_history_rebuilds_non_primary_history_for_primary_only_policy(monkeypatch):
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure
    from app.services.market_calendar_service import MarketCalendarService

    end = date(2026, 6, 25)
    calls: list[tuple] = []

    monkeypatch.setattr(
        MarketCalendarService,
        "last_completed_trading_day",
        lambda self, market: end,
    )

    def fake_backfill(
        db_arg,
        market,
        start,
        end_arg,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        calls.append((db_arg, market, start, end_arg, benchmark_fallback_policy))
        for row in db_arg.query(MarketExposure).filter(MarketExposure.market == market).all():
            row.benchmark_symbol = "SPY"
        db_arg.commit()
        return {"seeded": 2, "failed": 0}

    monkeypatch.setattr(svc, "backfill_exposure", fake_backfill)

    db = SessionLocal()
    try:
        db.add_all(
            [
                MarketExposure(
                    market="US",
                    date=date(2026, 6, 23),
                    exposure_score=70.0,
                    stance="Confirmed Uptrend",
                    benchmark_symbol="IVV",
                ),
                MarketExposure(
                    market="US",
                    date=date(2026, 6, 24),
                    exposure_score=71.0,
                    stance="Confirmed Uptrend",
                    benchmark_symbol="IVV",
                ),
            ]
        )
        db.commit()

        result = svc.ensure_exposure_history(
            db,
            "US",
            min_rows=2,
            days=12,
            benchmark_fallback_policy=BenchmarkFallbackPolicy.PRIMARY_ONLY,
        )
    finally:
        db.close()

    assert result == {"seeded": 2, "failed": 0}
    assert calls == [
        (
            db,
            "US",
            date(2026, 6, 13),
            end,
            BenchmarkFallbackPolicy.PRIMARY_ONLY,
        )
    ]


def test_ensure_exposure_history_reports_error_when_strict_rebuild_leaves_non_primary_rows(monkeypatch):
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure
    from app.services.market_calendar_service import MarketCalendarService

    end = date(2026, 6, 25)

    monkeypatch.setattr(
        MarketCalendarService,
        "last_completed_trading_day",
        lambda self, market: end,
    )

    def fake_backfill(
        db_arg,
        market,
        start,
        end_arg,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        return {"seeded": 0, "failed": 1}

    monkeypatch.setattr(svc, "backfill_exposure", fake_backfill)

    db = SessionLocal()
    try:
        db.add(
            MarketExposure(
                market="US",
                date=date(2026, 6, 24),
                exposure_score=71.0,
                stance="Confirmed Uptrend",
                benchmark_symbol="IVV",
            )
        )
        db.commit()

        result = svc.ensure_exposure_history(
            db,
            "US",
            min_rows=1,
            days=12,
            benchmark_fallback_policy=BenchmarkFallbackPolicy.PRIMARY_ONLY,
        )
    finally:
        db.close()

    assert result["error"] == "strict_benchmark_history_incomplete"
    assert result["failed"] == 1
    assert result["non_primary_benchmark_rows"] == 1
    assert result["primary_benchmark_symbol"] == "SPY"


def test_refresh_market_exposure_for_date_promotes_strict_history_seed_error(monkeypatch):
    db = object()
    as_of = date(2026, 6, 25)
    history_seed = {
        "error": "strict_benchmark_history_incomplete",
        "failed": 1,
        "non_primary_benchmark_rows": 1,
    }

    def fake_compute(
        market,
        as_of_date,
        db_arg,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        return {
            "market": market,
            "date": as_of_date,
            "exposure_score": 78.0,
            "stance": "Confirmed Uptrend",
        }

    def fake_seed(
        db_arg,
        market,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        return history_seed

    monkeypatch.setattr(svc, "compute_and_store", fake_compute)
    monkeypatch.setattr(svc, "ensure_exposure_history", fake_seed)

    result = refresh_market_exposure_for_date(
        db,
        "us",
        as_of,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.PRIMARY_ONLY,
    )

    assert result["error"] == "strict_benchmark_history_incomplete"
    assert result["history_seed"] == history_seed


def test_compute_and_store_round_trip_validates_against_schema(monkeypatch):
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure
    from app.schemas.market_scan import MarketHealthExposure

    # 250-session uptrend -> price > ma50 > ma200, no distribution days.
    closes = list(range(100, 350))
    df = _df(closes, [1000] * len(closes))
    monkeypatch.setattr(
        "app.services.benchmark_cache_service.BenchmarkCacheService",
        _fake_benchmark_factory(df),
    )

    as_of = df.index[-1].date()
    db = SessionLocal()
    try:
        result = compute_and_store("US", as_of, db)
        assert "error" not in result
        assert result["stance"] == "Power Trend"  # clean uptrend, no penalties

        row = db.query(MarketExposure).filter(MarketExposure.date == as_of, MarketExposure.market == "US").one()
        assert row.exposure_score == 100.0
        assert row.benchmark_symbol == "SPY"

        payload = build_exposure_payload(db, "US")
        # The strict (extra="forbid") schema is the live + static contract.
        MarketHealthExposure.model_validate(payload)
        assert payload["exposure_score"] == 100.0
        assert payload["history"][-1]["date"] == as_of.isoformat()
    finally:
        db.close()


_FTD_BAR = 78


def _ftd_then_hold(sessions_after_ftd, tail=()):
    """Flat at 100, a slide to 85, an FTD on rally day 4, then a hold at 87.

    The hold sits far enough under the 50-DMA that the core score is 38,
    below FTD_FLOOR, so only the floor can lift it.
    """
    closes = (
        [100.0] * 60
        + [float(c) for c in range(99, 84, -1)]
        + [85.3, 85.5, 85.6, 87.0]
        + [87.0] * sessions_after_ftd
        + list(tail)
    )
    volumes = [1000] * len(closes)
    volumes[_FTD_BAR] = 1500
    return _df(closes, volumes)


def _compute(df, monkeypatch, fn=svc.compute_exposure):
    from app.database import SessionLocal

    monkeypatch.setattr(
        "app.services.benchmark_cache_service.BenchmarkCacheService",
        _fake_benchmark_factory(df),
    )
    db = SessionLocal()
    try:
        return fn("US", df.index[-1].date(), db)
    finally:
        db.close()


@pytest.mark.parametrize("sessions_after_ftd, floored", [(14, True), (15, False)])
def test_ftd_floor_applies_for_15_sessions_after_the_ftd(monkeypatch, sessions_after_ftd, floored):
    df = _ftd_then_hold(sessions_after_ftd)
    result = _compute(df, monkeypatch)

    ftd_date = df.index[_FTD_BAR].date()
    assert (result["cycle_state"], result["follow_through_day"], result["follow_through_date"]) == (
        "uptrend", True, ftd_date,
    )
    assert result["benchmark_price"] < result["benchmark_ma50"]
    assert ("ftd_floor" in result["components"]) is floored
    assert result["exposure_score"] == (FTD_FLOOR if floored else 38.0)


def test_failed_ftd_drops_the_floor_and_is_stored(monkeypatch):
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure

    df = _ftd_then_hold(3, tail=[84.5])
    as_of = df.index[-1].date()
    result = _compute(df, monkeypatch, fn=compute_and_store)

    assert "ftd_floor" not in result["components"]
    db = SessionLocal()
    try:
        row = db.query(MarketExposure).filter(MarketExposure.date == as_of).one()
        stored = (row.cycle_state, row.rally_day, row.rally_low_date, row.failed_ftd_date,
                  row.follow_through_day, row.follow_through_date)
    finally:
        db.close()
    assert stored == ("correction", None, as_of, as_of, False, None)


_CYCLE_FIELDS = ("exposure_score", "cycle_state", "rally_day", "follow_through_date", "failed_ftd_date")


def test_cycle_for_a_date_does_not_depend_on_where_the_frame_starts(monkeypatch):
    # A daily run and a later backfill load frames that start on different days.
    # The 110 high falls inside one frame only; it must not decide the FTD.
    closes = [110.0] + [100.0] * 300 + [94.0, 94.2, 94.3, 94.4, 96.0]
    volumes = [1000] * (len(closes) - 1) + [1500]
    full = _df(closes, volumes)
    later_start = full.iloc[1:]

    def fields(df):
        result = _compute(df, monkeypatch)
        return tuple(result[key] for key in _CYCLE_FIELDS)

    assert fields(full) == fields(later_start)


def test_compute_exposure_keeps_one_bar_per_session(monkeypatch):
    # A repeated session would otherwise count as an extra rally day and
    # confirm an FTD on what is really day 3. The later bar for a date wins.
    df = _df([100, 91, 91.5, 91.6, 91.7, 93.2], [1000] * 5 + [1500])
    df.index = df.index[:4].append(df.index[3:5])
    result = _compute(df, monkeypatch)
    assert (result["cycle_state"], result["rally_day"], result["follow_through_day"]) == (
        "rally_attempt", 3, False,
    )


def test_compute_exposure_never_confirms_an_ftd_across_a_skipped_bar(monkeypatch):
    # The skipped session traded 2000. Against it, 1500 is lower volume; against
    # the earlier retained bar's 1000 it would look higher and confirm.
    df = _df([100, 91, 91.5, 91.6, 91.7, 0.0, 93.2], [1000] * 5 + [2000, 1500])
    result = _compute(df, monkeypatch)
    assert (result["cycle_state"], result["follow_through_day"]) == ("rally_attempt", False)


def test_compute_exposure_confirms_an_ftd_on_the_second_session_after_a_skipped_bar(monkeypatch):
    # 93.2 follows 91.7 directly, so their volumes compare even though a bar
    # before 91.7 was skipped.
    df = _df([100, 91, 91.5, 91.6, 0.0, 91.7, 93.2], [1000] * 6 + [1500])
    result = _compute(df, monkeypatch)
    assert (result["cycle_state"], result["follow_through_date"]) == ("uptrend", df.index[-1].date())


def test_compute_exposure_counts_a_distribution_day_on_the_second_session_after_a_skipped_bar(monkeypatch):
    df = _df([100, 99, 0.0, 98.0, 97.5], [1000, 1000, 1000, 1000, 1500])
    assert _compute(df, monkeypatch)["distribution_day_count"] == 1


@pytest.mark.parametrize("bad_close", [0.0, -5.0, float("nan"), float("inf")])
def test_compute_exposure_skips_bars_without_a_positive_finite_price(monkeypatch, bad_close):
    df = _df([100, 91, 91.5, bad_close, 91.6, 91.7, 93.2], [1000] * 6 + [1500])
    result = _compute(df, monkeypatch)
    assert (result["cycle_state"], result["follow_through_date"]) == ("uptrend", df.index[-1].date())


def test_ensure_exposure_history_seeds_then_skips(monkeypatch):
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure
    from app.services.market_calendar_service import MarketCalendarService
    from app.services.market_exposure_service import ensure_exposure_history

    # The benchmark frame must contain a bar on each backfilled trading day
    # (compute_exposure now requires the latest bar to equal the date), so end
    # the frame at the market's last completed trading day.
    end = MarketCalendarService().last_completed_trading_day("US")
    df = _df(list(range(100, 350)), [1000] * 250, end=pd.Timestamp(end))
    monkeypatch.setattr(
        "app.services.benchmark_cache_service.BenchmarkCacheService",
        _fake_benchmark_factory(df),
    )
    db = SessionLocal()
    try:
        first = ensure_exposure_history(db, "US", min_rows=2, days=12)
        assert first["seeded"] >= 1
        count = db.query(MarketExposure).filter(MarketExposure.market == "US").count()
        assert count == first["seeded"]

        # Second call is a no-op: history is now above the threshold.
        second = ensure_exposure_history(db, "US", min_rows=2, days=12)
        assert second.get("skipped") is True
        assert second["seeded"] == 0
        assert db.query(MarketExposure).filter(MarketExposure.market == "US").count() == count
    finally:
        db.close()


def test_refresh_market_exposure_for_date_computes_and_seeds_history(monkeypatch):
    db = object()
    as_of = date(2026, 6, 25)
    calls: list[tuple] = []

    def fake_compute(
        market,
        as_of_date,
        db_arg,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        calls.append(("compute", market, as_of_date, db_arg))
        return {
            "market": market,
            "date": as_of_date,
            "exposure_score": 78.0,
            "stance": "Confirmed Uptrend",
        }

    def fake_seed(
        db_arg,
        market,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        calls.append(("seed", db_arg, market))
        return {"seeded": 4, "failed": 0}

    monkeypatch.setattr(svc, "compute_and_store", fake_compute)
    monkeypatch.setattr(svc, "ensure_exposure_history", fake_seed)

    result = refresh_market_exposure_for_date(db, "us", as_of)

    assert calls == [
        ("compute", "US", as_of, db),
        ("seed", db, "US"),
    ]
    assert result == {
        "market": "US",
        "date": as_of,
        "exposure_score": 78.0,
        "stance": "Confirmed Uptrend",
        "history_seed": {"seeded": 4, "failed": 0},
    }


def test_refresh_market_exposure_for_date_does_not_seed_after_compute_error(monkeypatch):
    db = object()
    as_of = date(2026, 6, 25)
    calls: list[tuple] = []

    def fake_compute(
        market,
        as_of_date,
        db_arg,
        *,
        benchmark_fallback_policy=BenchmarkFallbackPolicy.ALLOW,
    ):
        calls.append(("compute", market, as_of_date, db_arg))
        return {"market": market, "date": as_of_date.isoformat(), "error": "no_benchmark_data"}

    monkeypatch.setattr(svc, "compute_and_store", fake_compute)
    monkeypatch.setattr(
        svc,
        "ensure_exposure_history",
        lambda *_args, **_kwargs: calls.append(("seed",)),
    )

    result = refresh_market_exposure_for_date(db, "us", as_of)

    assert calls == [("compute", "US", as_of, db)]
    assert result == {"market": "US", "date": "2026-06-25", "error": "no_benchmark_data"}


def test_build_exposure_payload_carries_the_cycle_and_marks_event_days():
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure
    from app.schemas.market_scan import MarketHealthExposure

    d1, d2, d3, d4 = date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3), date(2026, 6, 4)
    db = SessionLocal()
    try:
        db.add_all([
            MarketExposure(market="US", date=d1, exposure_score=70.0, stance="Confirmed Uptrend",
                           cycle_state="uptrend", follow_through_day=True, follow_through_date=d1),
            MarketExposure(market="US", date=d2, exposure_score=72.0, stance="Confirmed Uptrend",
                           cycle_state="uptrend", follow_through_day=True, follow_through_date=d1),
            MarketExposure(market="US", date=d3, exposure_score=48.0, stance="Downtrend/Caution",
                           cycle_state="correction", rally_low_date=d3, failed_ftd_date=d3),
            MarketExposure(market="US", date=d4, exposure_score=49.0, stance="Downtrend/Caution",
                           cycle_state="rally_attempt", rally_day=1, rally_low_date=d3),
        ])
        db.commit()

        payload = build_exposure_payload(db, "US")
        MarketHealthExposure.model_validate(payload)
        assert [
            (p["date"], p["cycle_state"], p["rally_day"], p["follow_through"], p["failed_ftd"])
            for p in payload["history"]
        ] == [
            ("2026-06-01", "uptrend", None, True, False),
            ("2026-06-02", "uptrend", None, False, False),
            ("2026-06-03", "correction", None, False, True),
            ("2026-06-04", "rally_attempt", 1, False, False),
        ]
        headline = (payload["cycle_state"], payload["rally_day"], payload["follow_through_date"])
        assert headline == ("rally_attempt", 1, None)

        pinned = build_exposure_payload(db, "US", as_of_date=d2)
        headline = (pinned["cycle_state"], pinned["follow_through_day"], pinned["follow_through_date"])
        assert headline == ("uptrend", True, "2026-06-01")
    finally:
        db.close()


def test_build_exposure_payload_shows_no_v1_ftd_marker_on_rows_without_a_cycle():
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure

    day = date(2026, 6, 1)
    db = SessionLocal()
    try:
        db.add(MarketExposure(market="US", date=day, exposure_score=45.0, stance="Downtrend/Caution",
                              follow_through_day=True, follow_through_date=day))
        db.commit()
        point = build_exposure_payload(db, "US")["history"][0]
        assert (point["cycle_state"], point["follow_through"]) == (None, False)
    finally:
        db.close()


def test_build_exposure_payload_pins_to_as_of_date():
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure
    from app.schemas.market_scan import MarketHealthExposure

    d1, d2, d3 = date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)
    db = SessionLocal()
    try:
        # Rows stored before the market cycle have no cycle columns.
        for d, score in [(d1, 60.0), (d2, 70.0), (d3, 80.0)]:
            db.add(MarketExposure(market="US", date=d, exposure_score=score, stance="Confirmed Uptrend"))
        db.commit()

        latest = build_exposure_payload(db, "US")
        MarketHealthExposure.model_validate(latest)
        assert latest["date"] == d3.isoformat()  # absolute latest
        assert latest["cycle_state"] is None and latest["history"][-1]["failed_ftd"] is False

        pinned = build_exposure_payload(db, "US", as_of_date=d2)
        assert pinned["date"] == d2.isoformat()
        assert all(p["date"] <= d2.isoformat() for p in pinned["history"])  # excludes newer d3

        # Pinned to a date with no exposure row -> omit (no stale earlier-row fallback)
        assert build_exposure_payload(db, "US", as_of_date=date(2026, 6, 10)) is None
    finally:
        db.close()


def test_compute_and_store_skips_write_when_no_benchmark(monkeypatch):
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure

    monkeypatch.setattr(
        "app.services.benchmark_cache_service.BenchmarkCacheService",
        _fake_benchmark_factory(None),
    )
    db = SessionLocal()
    try:
        result = compute_and_store("US", date(2026, 6, 16), db)
        assert result.get("error") == "no_benchmark_data"
        assert db.query(MarketExposure).count() == 0
    finally:
        db.close()


def _breadth(day, **values):
    from app.models.market_breadth import MarketBreadth

    return MarketBreadth(
        market="US", date=day, stocks_up_4pct=values.pop("stocks_up_4pct", 0),
        stocks_down_4pct=0, total_stocks_scanned=100, calculation_revision=3, **values,
    )


def _exposure(day, **values):
    from app.models.market_exposure import MarketExposure

    return MarketExposure(market="US", date=day, exposure_score=70.0, stance="Confirmed Uptrend", **values)


def _payload(*rows, as_of_date=None):
    from app.database import SessionLocal
    from app.schemas.market_scan import MarketHealthExposure

    db = SessionLocal()
    try:
        db.add_all(rows)
        db.commit()
        payload = build_exposure_payload(db, "US", as_of_date=as_of_date)
        MarketHealthExposure.model_validate(payload)
        return payload
    finally:
        db.close()


@pytest.mark.parametrize(
    "index_off_high_pct, above_sma200_pct, highs, lows, flagged",
    [
        (-1.2, 34.0, 10, 5, True),    # near the high, most stocks below the 200-day
        (-3.0, 49.99, 10, 5, True),   # exactly 3% off still counts as near the high
        (-1.2, 60.0, 5, 10, True),    # new lows outnumber new highs
        (-1.2, 60.0, 20, 5, False),   # healthy participation
        (-3.01, 34.0, 5, 10, False),  # index not near its high
    ],
)
def test_breadth_divergence_flags_an_index_near_its_high_with_weak_participation(
    index_off_high_pct, above_sma200_pct, highs, lows, flagged,
):
    day = date(2026, 10, 30)
    payload = _payload(
        _exposure(day, index_off_high_pct=index_off_high_pct),
        _breadth(day, above_sma200_pct=above_sma200_pct, new_high_52week_count=highs, new_low_52week_count=lows),
    )

    assert payload["breadth_divergence"] == {
        "index_off_high_pct": index_off_high_pct,
        "above_sma200_pct": above_sma200_pct,
        "new_high_52week_count": highs,
        "new_low_52week_count": lows,
        "flagged": flagged,
    }


@pytest.mark.parametrize(
    "exposure_values, breadth_values",
    [
        ({"index_off_high_pct": -1.2}, None),  # no breadth row that day
        ({"index_off_high_pct": -1.2}, {"new_high_52week_count": 10, "new_low_52week_count": 5}),  # row predates participation
        ({}, {"above_sma200_pct": 34.0, "new_high_52week_count": 5, "new_low_52week_count": 10}),  # exposure predates the column
    ],
)
def test_breadth_divergence_is_unknown_not_false_when_an_input_is_missing(exposure_values, breadth_values):
    day = date(2026, 10, 30)
    rows = [_exposure(day, **exposure_values)]
    if breadth_values is not None:
        rows.append(_breadth(day, **breadth_values))

    assert _payload(*rows)["breadth_divergence"]["flagged"] is None


def test_breadth_divergence_counts_an_index_exactly_3pct_off_its_high_as_near():
    from app.services.market_exposure_service import divergence_flagged

    exactly_3pct_off = (97.0 / 100.0 - 1) * 100  # -3.0000000000000027 in floating point
    assert divergence_flagged(exactly_3pct_off, 34.0, 10, 5) is True


@pytest.mark.parametrize("above_sma200_pct, flagged", [(60.0, None), (34.0, True)])
def test_breadth_divergence_treats_highs_and_lows_without_eligible_stocks_as_unknown(above_sma200_pct, flagged):
    day = date(2026, 10, 30)
    divergence = _payload(
        _exposure(day, index_off_high_pct=-1.2),
        _breadth(
            day, above_sma200_pct=above_sma200_pct, high_low_52week_eligible_count=0,
            new_high_52week_count=0, new_low_52week_count=0,
        ),
    )["breadth_divergence"]

    assert (divergence["new_high_52week_count"], divergence["new_low_52week_count"]) == (None, None)
    assert divergence["flagged"] is flagged


def test_ftd_quality_reads_the_breadth_row_of_the_ftd_session():
    ftd, today = date(2026, 11, 11), date(2026, 11, 20)
    payload = _payload(
        _exposure(today, cycle_state="uptrend", follow_through_day=True, follow_through_date=ftd),
        _breadth(ftd, stocks_up_4pct=137, up_4pct_near_high_count=52,
                 new_high_52week_count=30, new_low_52week_count=12),
        _breadth(today, stocks_up_4pct=40, up_4pct_near_high_count=35),
    )

    assert payload["ftd_quality"] == {
        "ftd_date": "2026-11-11",
        "stocks_up_4pct": 137,
        "up_4pct_near_high_count": 52,
        "near_high_share": pytest.approx(52 / 137),
        "new_high_52week_count": 30,
        "new_low_52week_count": 12,
    }


@pytest.mark.parametrize(
    "ftd_breadth, expected",
    [
        ({"stocks_up_4pct": 0, "up_4pct_near_high_count": 0}, (0, 0, None)),
        (None, (None, None, None)),
    ],
)
def test_ftd_quality_has_no_share_without_gainers_or_a_breadth_row(ftd_breadth, expected):
    ftd, today = date(2026, 11, 11), date(2026, 11, 20)
    rows = [_exposure(today, cycle_state="uptrend", follow_through_day=True, follow_through_date=ftd)]
    if ftd_breadth is not None:
        rows.append(_breadth(ftd, **ftd_breadth))

    quality = _payload(*rows)["ftd_quality"]

    assert quality["ftd_date"] == "2026-11-11"
    assert (quality["stocks_up_4pct"], quality["up_4pct_near_high_count"], quality["near_high_share"]) == expected


def test_ftd_quality_treats_highs_and_lows_without_eligible_stocks_as_unknown():
    ftd, today = date(2026, 11, 11), date(2026, 11, 20)
    quality = _payload(
        _exposure(today, cycle_state="uptrend", follow_through_day=True, follow_through_date=ftd),
        _breadth(ftd, stocks_up_4pct=10, up_4pct_near_high_count=4, high_low_52week_eligible_count=0,
                 new_high_52week_count=0, new_low_52week_count=0),
    )["ftd_quality"]

    assert (quality["new_high_52week_count"], quality["new_low_52week_count"]) == (None, None)
    assert quality["near_high_share"] == pytest.approx(0.4)


def test_ftd_quality_is_absent_outside_a_confirmed_uptrend():
    ftd = date(2026, 11, 11)
    assert _payload(_exposure(date(2026, 11, 20), cycle_state="correction"))["ftd_quality"] is None
    # A row stored before the market cycle has no cycle_state.
    legacy = _exposure(ftd, follow_through_day=True, follow_through_date=ftd)
    assert _payload(legacy, as_of_date=ftd)["ftd_quality"] is None


def test_compute_exposure_stores_how_far_the_index_sits_below_its_cycle_high(monkeypatch):
    closes = [float(c) for c in range(100, 350)] + [340.0]
    result = _compute(_df(closes, [1000] * len(closes)), monkeypatch)

    # Full precision, so the 3% test isn't decided by rounding; the UI rounds.
    assert result["index_off_high_pct"] == pytest.approx((340 / 349 - 1) * 100, abs=1e-9)
    assert type(result["index_off_high_pct"]) is float  # psycopg2 and Celery JSON reject numpy floats


def test_breadth_participation_never_moves_the_exposure_score(monkeypatch):
    from app.database import SessionLocal

    closes = [float(c) for c in range(100, 350)]
    df = _df(closes, [1000] * len(closes))
    as_of = df.index[-1].date()

    def score_with(**participation):
        db = SessionLocal()
        try:
            db.query(svc.MarketBreadth).delete()
            db.add(_breadth(as_of, **participation))
            db.commit()
        finally:
            db.close()
        return _compute(df, monkeypatch)["exposure_score"]

    assert score_with() == score_with(
        above_sma200_pct=10.0, down_20pct_from_high_pct=70.0,
        new_high_52week_count=1, new_low_52week_count=90,
    )


def test_index_off_high_migration_follows_the_participation_migration():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    from app.models.market_exposure import MarketExposure

    path = Path(__file__).resolve().parents[2] / "alembic/versions/20261123_0078_index_off_high_column.py"
    spec = spec_from_file_location("exposure_off_high", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (module.revision, module.down_revision) == ("20261123_0078", "20261123_0077")
    assert MarketExposure.__table__.c.index_off_high_pct.nullable is True
