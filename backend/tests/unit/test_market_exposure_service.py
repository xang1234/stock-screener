"""Unit tests for the market exposure rubric and DB round-trip."""
from datetime import date

import pandas as pd

import app.services.market_exposure_service as svc
from app.services.market_exposure_service import (
    CAP_BELOW_200DMA,
    CAP_HEAVY_DISTRIBUTION,
    build_exposure_payload,
    compute_and_store,
    count_distribution_days,
    compute_trend,
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


def _fake_benchmark_factory(df, symbol="SPY"):
    class _FakeBenchmarkCacheService:
        def __init__(self, *a, **k):
            pass

        def get_benchmark_bundle(self, market="US", period="2y", force_refresh=False):
            return _FakeBundle(df, symbol)

    return _FakeBenchmarkCacheService


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
