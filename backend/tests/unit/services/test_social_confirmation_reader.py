"""Fixed-clock MIC freshness and coherent saved feature/exposure joins."""
from datetime import date, datetime, timezone

import pytest


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


@pytest.fixture
def calendar(monkeypatch):
    from app.services.market_calendar_adapters import RawMarketCalendarAdapter
    from app.services.market_calendar_service import MarketCalendarService
    monkeypatch.setattr(RawMarketCalendarAdapter, "_session_range_cache_client", lambda self: None)
    return MarketCalendarService()


@pytest.mark.parametrize("clock,want", [
    ("2026-07-02T16:00:00", "2026-07-01"),  # intraday
    ("2026-11-27T19:59:59", "2026-11-25"),  # early close + grace not reached
    ("2026-11-27T20:00:00", "2026-11-27"),  # 13:00 EST early close + two hours
    ("2026-07-03T22:00:00", "2026-07-02"),  # holiday
    ("2026-07-05T22:00:00", "2026-07-02"),  # weekend
    ("2026-03-06T22:59:59", "2026-03-05"),  # EST
    ("2026-03-06T23:00:00", "2026-03-06"),
    ("2026-03-09T21:59:59", "2026-03-06"),  # EDT
    ("2026-03-09T22:00:00", "2026-03-09"),
])
def test_required_session_respects_close_grace(calendar, clock, want):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    fact = SocialConfirmationReader(None, calendar=calendar).freshness(
        "US", utc(clock), date.fromisoformat(want), mic="XNYS")
    assert fact.required_session == date.fromisoformat(want)
    assert fact.actual_session == date.fromisoformat(want)
    assert fact.fresh is True


def test_grace_is_configurable_and_missing_calendar_fails_closed(calendar):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    reader = SocialConfirmationReader(None, calendar=calendar, grace_minutes=0)
    assert reader.freshness("US", utc("2026-11-27T18:00:00"), date(2026, 11, 27), mic="XNYS").fresh
    assert reader.freshness("US", utc("2026-11-27T18:00:00"), date(2026, 11, 27), mic="UNKNOWN").reason == "calendar_unavailable"


@pytest.mark.parametrize("actual,reason", [(None, "missing_session"), (date(2026, 7, 1), "stale_session"), (date(2026, 7, 6), "future_session")])
def test_missing_stale_future_dates_are_not_fresh(calendar, actual, reason):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    fact = SocialConfirmationReader(None, calendar=calendar).freshness("US", utc("2026-07-05T22:00:00"), actual, mic="XNYS")
    assert fact.fresh is False
    assert fact.reason == reason


def test_listing_and_market_calendars_are_independent():
    from app.services.social_confirmation_reader import SocialConfirmationReader
    class Calendar:
        def market_now(self, market, now, *, mic=None):
            return now
        def trading_days(self, market, start, end, *, mic=None):
            return [date(2026, 7, 1), date(2026, 7, 2)]
        def session_close(self, market, day, *, mic=None):
            return utc(f"{day}T{'20' if mic == 'XNAS' else '17'}:00:00")
    reader = SocialConfirmationReader(None, calendar=Calendar())
    assert reader.freshness("US", utc("2026-07-02T19:00:00"), date(2026, 7, 1), mic="XNAS").fresh
    assert reader.freshness("US", utc("2026-07-02T19:00:00"), date(2026, 7, 2)).fresh


def seed_snapshot(db):
    from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer, StockFeatureDaily
    from app.models.market_exposure import MarketExposure
    run = FeatureRun(as_of_date=date(2026, 7, 2), run_type="daily_snapshot", status="published",
                     config_json={"universe": {"market": "US"}},
                     completed_at=utc("2026-07-02T18:00:00"), published_at=utc("2026-07-02T18:01:00"))
    db.add(run)
    db.flush()
    db.add(FeatureRunPointer(key="latest_published_market:US", run_id=run.id))
    feature = StockFeatureDaily(run_id=run.id, symbol="AAA", as_of_date=date(2026, 7, 2), details_json={"rs_rating": 80})
    exposure = MarketExposure(market="US", date=date(2026, 7, 2), exposure_score=60, stance="uptrend", benchmark_symbol="SPY",
                              created_at=utc("2026-07-02T18:00:00"), updated_at=utc("2026-07-02T18:00:00"))
    db.add_all([feature, exposure])
    db.flush()
    return run, feature, exposure


def test_snapshot_pins_one_run_and_market_exposure(db_session, calendar):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    run, _, exposure = seed_snapshot(db_session)
    facts = SocialConfirmationReader(db_session, calendar=calendar).read("AAA", "US", utc("2026-07-02T23:00:00"), mic="XNYS")
    assert facts.feature_run_id == run.id
    assert facts.market_exposure_id == exposure.id
    assert facts.feature_freshness.fresh and facts.market_freshness.fresh
    assert facts.rs_rating == 80 and facts.market_exposure == 60


@pytest.mark.parametrize("mutation,reason", [("row_date", "feature_run_date_mismatch"), ("market", "feature_market_mismatch"),
    ("future_publish", "feature_not_available"), ("future_exposure", "market_not_available"), ("benchmark", "market_benchmark_mismatch")])
def test_incoherent_or_future_snapshots_are_unavailable(db_session, calendar, mutation, reason):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    run, feature, exposure = seed_snapshot(db_session)
    if mutation == "row_date": feature.as_of_date = date(2026, 7, 1)
    if mutation == "market": run.config_json = {"universe": {"market": "HK"}}
    if mutation == "future_publish": run.published_at = utc("2026-07-03T00:00:00")
    if mutation == "future_exposure": exposure.updated_at = utc("2026-07-03T00:00:00")
    if mutation == "benchmark":
        exposure.benchmark_symbol = "^HSI"
        exposure.updated_at = utc("2026-07-02T18:01:00")
    db_session.flush()
    facts = SocialConfirmationReader(db_session, calendar=calendar).read("AAA", "US", utc("2026-07-02T23:00:00"), mic="XNYS")
    assert reason in facts.reasons
    assert not (facts.feature_freshness.fresh and facts.market_freshness.fresh)
