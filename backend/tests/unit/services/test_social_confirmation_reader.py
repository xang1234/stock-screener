"""Fixed-clock MIC freshness and coherent saved feature/exposure joins."""
from datetime import date, datetime, timezone

import pytest
from tests.unit.services.test_social_theme_market_service import basket


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
    facts = SocialConfirmationReader(db_session, calendar=calendar).read_facts("AAA", "US", utc("2026-07-02T23:00:00"), mic="XNYS")
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
    facts = SocialConfirmationReader(db_session, calendar=calendar).read_facts("AAA", "US", utc("2026-07-02T23:00:00"), mic="XNYS")
    assert reason in facts.reasons
    assert not (facts.feature_freshness.fresh and facts.market_freshness.fresh)


def seed_batch(db):
    from app.infra.db.models.feature_store import StockFeatureDaily
    from app.models.stock_universe import StockUniverse
    from app.models.industry import IBDGroupRank
    run, feature, exposure = seed_snapshot(db)
    feature.details_json = {"rs_rating": 80, "rs_rating_1m": 70, "rs_rating_3m": 90,
        "setup_engine": {"setup_score": 75, "setup_ready": True},
        "avg_dollar_volume": 100_000_000, "ibd_industry_group": "Chips",
        "ibd_group_rank": 2, "ibd_group_rank_date": "2026-07-02"}
    db.add_all([StockUniverse(symbol=s, market="US", exchange="NYSE", is_active=True) for s in ("AAA", "BBB")])
    db.add(StockFeatureDaily(run_id=run.id, symbol="BBB", as_of_date=run.as_of_date,
        details_json={**feature.details_json, "setup_engine": {"setup_score": 55, "setup_ready": False}}))
    for market, day, groups in (("US", date(2026, 7, 2), ("Energy", "Chips", "Retail")),
                               ("HK", date(2026, 7, 2), ("Chips",)),
                               ("US", date(2026, 7, 1), ("Chips",))):
        for rank, group in enumerate(groups, 1):
            db.add(IBDGroupRank(market=market, date=day, industry_group=group, rank=rank,
                avg_rs_rating=80, rs_formula_version="legacy-linear-v1", created_at=utc("2026-07-02T18:00:00")))
    db.flush()
    return run, feature, exposure


def test_batch_protocol_returns_feature_fields_and_exact_group_cohort(db_session, calendar):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    run, _, exposure = seed_batch(db_session)
    reader = SocialConfirmationReader(db_session, calendar=calendar)
    method = getattr(reader, "read_market", None)
    assert callable(method), "coherent batch evidence API is missing"
    batch = method("US", ("BBB", "AAA"), utc("2026-07-02T23:00:00"))
    assert tuple(i.candidate_key for i in batch.inputs) == ("US:AAA", "US:BBB")
    a, b = batch.inputs
    assert (a.setup_score, a.rs_rating_1m, a.rs_rating_3m) == (75, 70, 90)
    assert (a.group_rank, a.market_group_count) == (2, 3)
    assert batch.market_context.exposure_id == exposure.id
    assert batch.pinned_run.run_id == run.id
    assert {f.feature_run_id for _, f in batch.facts} == {run.id}
    assert dict(batch.facts)["AAA"].setup_ready is True
    assert dict(batch.facts)["BBB"].setup_ready is False
    assert dict(batch.facts)["AAA"].liquidity_eligible is True
    assert reader.read("US", ("AAA", "BBB"), utc("2026-07-02T23:00:00")) == batch.inputs
    assert batch.group_context.cohort == (("Energy", 1), ("Chips", 2), ("Retail", 3))
    from app.domain.social_signals.scoring import score_confirmation
    components = dict(score_confirmation(a).components)
    assert components["rs"].value == 82  # .4 * 70 + .6 * 90
    assert components["group"].value == 50  # rank 2 of 3


@pytest.mark.parametrize("field,value", [("setup_score", True), ("setup_score", float("nan")),
                                        ("setup_ready", 1)])
def test_batch_rejects_invalid_setup_values(db_session, calendar, field, value):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    _, feature, _ = seed_batch(db_session)
    feature.details_json = {**feature.details_json, "setup_engine": {field: value}}
    reader = SocialConfirmationReader(db_session, calendar=calendar)
    assert callable(getattr(reader, "read_market", None)), "coherent batch evidence API is missing"
    batch = reader.read_market("US", ("AAA",), utc("2026-07-02T23:00:00"))
    fact = dict(batch.facts)["AAA"]
    assert fact.setup_ready is None
    assert batch.inputs[0].setup_score is None


def test_batch_stale_evidence_is_null_with_original_dates(db_session, calendar):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    _, _, _ = seed_batch(db_session)
    reader = SocialConfirmationReader(db_session, calendar=calendar)
    assert callable(getattr(reader, "read_market", None)), "coherent batch evidence API is missing"
    batch = reader.read_market("US", ("AAA",), utc("2026-07-06T23:00:00"))
    row = batch.inputs[0]
    assert (row.setup_score, row.rs_rating_1m, row.rs_rating_3m, row.group_rank) == (None,)*4
    fact = dict(batch.facts)["AAA"]
    assert (fact.setup_ready, fact.liquidity_eligible, fact.market_exposure) == (None,)*3
    assert fact.feature_freshness.actual_session == date(2026, 7, 2)
    assert batch.market_context.freshness.actual_session == date(2026, 7, 2)
    assert "stale_session" in fact.reasons


@pytest.mark.parametrize("mutation", ["row_date", "rank", "future_cohort"])
def test_batch_group_cohort_inconsistency_is_missing(db_session, calendar, mutation):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    from app.models.industry import IBDGroupRank
    _, feature, _ = seed_batch(db_session)
    if mutation == "row_date":
        feature.details_json = {**feature.details_json, "ibd_group_rank_date": "2026-07-01"}
    elif mutation == "rank":
        feature.details_json = {**feature.details_json, "ibd_group_rank": 1}
    else:
        for group in db_session.query(IBDGroupRank).filter_by(market="US", date=date(2026, 7, 2)):
            group.created_at = utc("2026-07-03T00:00:00")
    db_session.flush()
    reader = SocialConfirmationReader(db_session, calendar=calendar)
    assert callable(getattr(reader, "read_market", None)), "coherent batch evidence API is missing"
    batch = reader.read_market("US", ("AAA",), utc("2026-07-02T23:00:00"))
    assert batch.inputs[0].group_rank is None
    assert any("group" in r for r in dict(batch.facts)["AAA"].reasons)


@pytest.mark.parametrize("count,valid,available", [(2, 2, False), (4, 3, True), (5, 3, False), (10, 7, True)])
def test_batch_theme_uses_accepted_price_evidence_and_coverage(db_session, basket, count, valid, available):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    from app.models.market_exposure import MarketExposure
    from app.domain.social_signals.scoring import score_confirmation
    theme, symbols, calendar = basket(count, valid)
    now = utc("2026-07-02T23:00:00")
    db_session.add(MarketExposure(market="US", date=date(2026, 7, 2), exposure_score=60,
        stance="uptrend", benchmark_symbol="SPY", created_at=now, updated_at=now))
    db_session.flush()
    reader = SocialConfirmationReader(db_session, calendar=calendar)
    batch = reader.read_market("US", symbols, now)
    assert len(batch.theme_evidence) == 1, "accepted Theme evidence is missing"
    evidence = batch.theme_evidence[0]
    assert evidence.benchmark_symbol == batch.market_context.benchmark_symbol == "SPY"
    assert evidence.accepted_company_count == count
    assert dict(evidence.measured_company_counts)["avg_rs_rating"] == valid
    component = dict(score_confirmation(batch.inputs[0]).components)["theme"]
    assert (component.value is not None) is available
    assert {run_id for _, run_id in evidence.feature_run_ids} == {batch.pinned_run.run_id}
    # Legacy attention changes cannot alter independent saved price confirmation.
    from app.models.theme import ThemeMetrics
    theme.discovery_source = "social"
    db_session.add(ThemeMetrics(theme_cluster_id=theme.id, date=date(2026, 7, 2),
        pipeline="technical", mention_velocity=9999, momentum_score=0))
    db_session.flush()
    assert reader.read_market("US", symbols, now) == batch


def test_theme_missing_benchmark_reason_is_saved_without_fallback(db_session, basket):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    _, symbols, calendar = basket(3, 3)
    batch = SocialConfirmationReader(db_session, calendar=calendar).read_market("US", symbols, utc("2026-07-02T23:00:00"))
    assert batch.theme_reasons == (("chips", "market_benchmark_unavailable"),)
    assert not batch.theme_evidence


def test_batch_pins_before_candidate_reads_and_frozen_context_survives_edits(db_session, calendar, monkeypatch):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    from app.infra.db.models.feature_store import FeatureRunPointer
    run, _, exposure = seed_batch(db_session)
    reader = SocialConfirmationReader(db_session, calendar=calendar)
    original = reader._read
    def change_latest(*args, **kwargs):
        fact = original(*args, **kwargs)
        db_session.get(FeatureRunPointer, "latest_published_market:US").run_id = None
        exposure.exposure_score = 10
        exposure.benchmark_symbol = "^HSI"
        return fact
    monkeypatch.setattr(reader, "_read", change_latest)
    batch = reader.read_market("US", ("AAA", "BBB"), utc("2026-07-02T23:00:00"))
    assert {f.feature_run_id for _, f in batch.facts} == {run.id}
    assert {f.market_exposure for _, f in batch.facts} == {60}
    assert {f.benchmark_symbol for _, f in batch.facts} == {"SPY"}
    assert batch.market_context.exposure_score == 60


def test_staged_basket_boundary_reuses_pins_and_only_links_accepted_members(db_session, basket):
    from dataclasses import replace, asdict
    import json
    from app.services.social_confirmation_reader import SocialConfirmationReader
    from app.services.social_theme_market_service import LiveAcceptedBasketReader
    from app.models.market_exposure import MarketExposure
    from app.infra.db.models.feature_store import FeatureRunPointer
    _, symbols, calendar = basket(3, 3)
    now = utc("2026-07-02T23:00:00")
    exposure = MarketExposure(market="US", date=date(2026, 7, 2), exposure_score=60,
        stance="uptrend", benchmark_symbol="SPY", created_at=now, updated_at=now)
    db_session.add(exposure)
    db_session.flush()
    reader = SocialConfirmationReader(db_session, calendar=calendar)
    # Frozen internal accepted-basket adapter, analogous to PreparedThemeApplication.
    saved = replace(LiveAcceptedBasketReader(db_session).read("chips", "US"), theme_key="new-chips")
    class StagedMembership:
        def read(self, theme_key, market):
            assert (theme_key, market) == ("new-chips", "US")
            return saved
    batch = reader.read_market("US", symbols + ("UNLINKED",), now, theme_keys=())
    frozen = json.dumps(asdict(batch), sort_keys=True, default=str)
    db_session.get(FeatureRunPointer, "latest_published_market:US").run_id = None
    exposure.benchmark_symbol = "^HSI"
    measured = reader.with_themes(batch, theme_keys=("new-chips",), membership_reader=StagedMembership())
    assert measured.theme_evidence[0].theme_key == "new-chips"
    assert measured.theme_evidence[0].benchmark_symbol == "SPY"
    assert set(dict(measured.theme_evidence[0].feature_run_ids).values()) == {batch.pinned_run.run_id}
    assert len(measured.inputs[0].theme_confirmations) == 1
    assert measured.inputs[-1].candidate_key == "US:UNLINKED"
    assert measured.inputs[-1].theme_confirmations == ()
    assert json.dumps(asdict(batch), sort_keys=True, default=str) == frozen


def test_batch_missing_listing_liquidity_and_rs_are_explicit(db_session, calendar):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    _, feature, _ = seed_batch(db_session)
    feature.details_json = {"setup_engine": {"setup_ready": True}, "avg_dollar_volume": True}
    batch = SocialConfirmationReader(db_session, calendar=calendar).read_market("US", ("AAA", "UNKNOWN"), utc("2026-07-02T23:00:00"))
    fact = dict(batch.facts)["AAA"]
    assert fact.liquidity_eligible is None
    assert {"missing_setup_score", "missing_rs_rating_1m", "missing_rs_rating_3m", "missing_liquidity"} <= set(fact.reasons)
    assert "listing_mic_unknown" in dict(batch.facts)["UNKNOWN"].reasons
