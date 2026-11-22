"""Unit tests for the market exposure rubric, the market cycle and DB round-trip."""
from datetime import date

import pandas as pd
import pytest

import app.services.market_exposure_service as svc
from app.domain.markets.market_cycle import CycleState, MarketCycle, market_cycle
from app.services.benchmark_cache_service import BenchmarkFallbackPolicy, BenchmarkResolution
from app.services.market_exposure_service import (
    CAP_BELOW_200DMA,
    CAP_HEAVY_DISTRIBUTION,
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
    return pd.DataFrame(
        {
            "Close": closes,
            "Low": closes if lows is None else lows,
            "Volume": [1000] * len(closes) if volumes is None else volumes,
        },
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


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


def test_build_exposure_payload_marks_follow_through_event_day():
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure
    from app.schemas.market_scan import MarketHealthExposure

    d1, d2, d3 = date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)
    db = SessionLocal()
    try:
        db.add_all([
            # event day: the detected FTD date is this row's own date
            MarketExposure(market="US", date=d1, exposure_score=70.0, stance="Confirmed Uptrend",
                           follow_through_day=True, follow_through_date=d1),
            # post-FTD day: still sees d1's FTD in its window, but is not the event
            MarketExposure(market="US", date=d2, exposure_score=72.0, stance="Confirmed Uptrend",
                           follow_through_day=True, follow_through_date=d1),
            MarketExposure(market="US", date=d3, exposure_score=68.0, stance="Confirmed Uptrend",
                           follow_through_day=False, follow_through_date=None),
        ])
        db.commit()

        payload = build_exposure_payload(db, "US")
        MarketHealthExposure.model_validate(payload)  # schema now carries follow_through
        flags = {p["date"]: p["follow_through"] for p in payload["history"]}
        assert flags[d1.isoformat()] is True
        assert flags[d2.isoformat()] is False
        assert flags[d3.isoformat()] is False
    finally:
        db.close()


def test_build_exposure_payload_pins_to_as_of_date():
    from app.database import SessionLocal
    from app.models.market_exposure import MarketExposure

    d1, d2, d3 = date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)
    db = SessionLocal()
    try:
        for d, score in [(d1, 60.0), (d2, 70.0), (d3, 80.0)]:
            db.add(MarketExposure(market="US", date=d, exposure_score=score, stance="Confirmed Uptrend"))
        db.commit()

        assert build_exposure_payload(db, "US")["date"] == d3.isoformat()  # absolute latest

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
