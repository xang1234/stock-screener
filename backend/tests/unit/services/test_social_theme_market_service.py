"""Real isolated membership/identity/price rows; no provider or model calls."""
from datetime import date, datetime, timezone
import json

import pytest

from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeCluster, ThemeConstituent
from app.models.app_settings import AppSetting

NOW = datetime(2026, 7, 2, 23, tzinfo=timezone.utc)
DAY = date(2026, 7, 2)
KEYS = ("basket_rs_vs_benchmark", "avg_rs_rating", "pct_above_50ma")


@pytest.fixture
def basket(db_session, monkeypatch):
    from app.services.market_calendar_adapters import RawMarketCalendarAdapter
    from app.services.market_calendar_service import MarketCalendarService
    from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer, StockFeatureDaily
    monkeypatch.setattr(RawMarketCalendarAdapter, "_session_range_cache_client", lambda self: None)
    calendar = MarketCalendarService()
    def seed(count=4, valid=3, *, market="US", duplicate=False, unknown=0, history=50, benchmark=True):
        theme = ThemeCluster(name="Chips", display_name="Chips", canonical_key="chips", pipeline="technical", is_active=True)
        db_session.add(theme)
        db_session.flush()
        symbols = [f"AAA{chr(65 + i)}" if market == "US" else f"{i+1:04}.HK" for i in range(count)]
        if duplicate: symbols.append("DUAL")
        entries = [{"symbol": s, "company_id": f"issuer-{0 if s == 'DUAL' else i}", "verification_reference": "synthetic:review",
                    "verified_at": "2026-01-01T00:00:00+00:00"} for i, s in enumerate(symbols) if i < count - unknown or s == "DUAL"]
        db_session.add(AppSetting(key="social_company_identities", category="social", value=json.dumps({"version": 1, "policy_version": "admin-attested-company-v1", "entries": entries})))
        run = FeatureRun(as_of_date=DAY, run_type="daily_snapshot", status="published", config_json={"universe": {"market": market}},
                         completed_at=NOW, published_at=NOW)
        db_session.add(run)
        db_session.flush()
        db_session.add(FeatureRunPointer(key=f"latest_published_market:{market}", run_id=run.id))
        sessions = calendar.trading_days(market, date(2026, 1, 1), DAY)[-history:] if history else []
        for i, symbol in enumerate(symbols):
            db_session.add(StockUniverse(symbol=symbol, market=market, exchange="NYSE" if market == "US" else "HKEX", is_common_stock=True, is_active=True))
            db_session.add(ThemeConstituent(theme_cluster_id=theme.id, symbol=symbol, is_active=True))
            if i < valid or symbol == "DUAL":
                for j, day in enumerate(sessions):
                    db_session.add(StockPrice(symbol=symbol, date=day, close=100 + j, adj_close=100 + j, created_at=NOW))
                db_session.add(StockFeatureDaily(run_id=run.id, symbol=symbol, as_of_date=DAY, details_json={"rs_rating": 80}))
        if benchmark:
            for day in sessions:
                db_session.add(StockPrice(symbol="SPY" if market == "US" else "^HSI", date=day, close=100, adj_close=100, created_at=NOW))
        db_session.flush()
        return theme, tuple(symbols), calendar
    return seed


@pytest.mark.parametrize("count,valid,available", [(2, 2, False), (4, 3, True), (5, 3, False), (10, 7, True)])
def test_each_component_requires_three_companies_and_seventy_percent(db_session, basket, count, valid, available):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(count, valid)
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert evidence.accepted_company_count == count
    assert dict(evidence.measured_company_counts) == dict.fromkeys(KEYS, valid)
    assert all((value is not None) == available for _, value in evidence.components)
    if available:
        assert dict(evidence.components)["pct_above_50ma"] == 100
        assert dict(evidence.components)["avg_rs_rating"] == 80


def test_duplicate_listing_is_one_company_and_membership_is_frozen(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService, MeasurementUnavailable
    _, symbols, calendar = basket(3, 3, duplicate=True)
    service = SocialThemeMarketService(db_session, calendar=calendar)
    evidence = service.measure("chips", "US", NOW, symbols)
    assert evidence.accepted_company_count == 3
    assert set(dict(evidence.measured_company_counts).values()) == {3}
    assert len(evidence.membership) == 4
    with pytest.raises(MeasurementUnavailable, match="membership_changed"):
        service.measure("chips", "US", NOW, symbols[:3])
    setting = db_session.query(AppSetting).filter_by(key="social_company_identities").one()
    payload = json.loads(setting.value)
    payload["version"] = 2
    payload["entries"][-1]["company_id"] = "different-issuer"
    setting.value = json.dumps(payload)
    db_session.flush()
    new = service.measure("chips", "US", NOW, symbols)
    assert new.basket_version != evidence.basket_version
    assert evidence.identity_version == 1 and new.identity_version == 2


def test_unknown_accepted_stock_cannot_improve_denominator(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(5, 3, unknown=2)
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert all(value is None for _, value in evidence.components)
    assert set(dict(evidence.reasons)[key] for key in KEYS) == {"company_identity_coverage_unknown"}
    assert len(evidence.membership) == 5


@pytest.mark.parametrize("history,benchmark,want", [(0, True, (False, True, False)), (21, True, (False, True, False)),
    (22, True, (True, True, False)), (49, True, (True, True, False)), (50, False, (False, True, True))])
def test_indicator_histories_and_benchmark_are_independent(db_session, basket, history, benchmark, want):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(3, 3, history=history, benchmark=benchmark)
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert tuple(dict(evidence.components)[key] is not None for key in KEYS) == want


def test_hk_benchmark_and_other_market_membership_isolated(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    theme, symbols, calendar = basket(3, 3, market="HK")
    db_session.add(StockUniverse(symbol="USA", market="US", exchange="NYSE", is_common_stock=True, is_active=True))
    db_session.add(ThemeConstituent(theme_cluster_id=theme.id, symbol="USA", is_active=True))
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "HK", NOW, symbols)
    assert evidence.benchmark_symbol == "^HSI"
    assert evidence.accepted_company_count == 3
    assert all(value is not None for _, value in evidence.components)
    assert len(evidence.membership) == 3


def test_accepted_social_union_excludes_proposals(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    from app.infra.db.models.social_analysis import SocialThemeAssociation
    theme, symbols, calendar = basket(3, 3)
    row = db_session.query(ThemeConstituent).filter_by(symbol=symbols[-1]).one()
    db_session.delete(row)
    db_session.add(SocialThemeAssociation(theme_cluster_id=theme.id, canonical_symbol=symbols[-1], market="US", origin="social", state="accepted", policy_version="social-theme-v1", first_seen_at=NOW, updated_at=NOW))
    db_session.add(StockUniverse(symbol="PROPOSED", market="US", exchange="NYSE", is_active=True))
    db_session.add(SocialThemeAssociation(theme_cluster_id=theme.id, canonical_symbol="PROPOSED", market="US", origin="social", state="proposed", policy_version="social-theme-v1", first_seen_at=NOW, updated_at=NOW))
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert evidence.accepted_company_count == 3
    assert all(value is not None for _, value in evidence.components)
    assert evidence.membership[-1].origins == ("social",)


def test_gap_or_invalid_prices_do_not_shorten_indicator_window(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(4, 4)
    db_session.query(StockPrice).filter_by(symbol=symbols[0], date=DAY).delete()
    db_session.query(StockPrice).filter_by(symbol=symbols[1], date=DAY).update({"close": -1, "adj_close": -1})
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert dict(evidence.measured_company_counts)["pct_above_50ma"] == 2
    assert dict(evidence.components)["pct_above_50ma"] is None
    assert dict(evidence.components)["basket_rs_vs_benchmark"] is None
    assert dict(evidence.components)["avg_rs_rating"] == 80


def test_measurement_pins_feature_run_even_if_pointer_changes(db_session, basket, monkeypatch):
    from app.services.social_theme_market_service import SocialThemeMarketService
    from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer, StockFeatureDaily
    _, symbols, calendar = basket(3, 3)
    pointer = db_session.get(FeatureRunPointer, "latest_published_market:US")
    original_run_id = pointer.run_id
    run = FeatureRun(as_of_date=DAY, run_type="daily_snapshot", status="published", config_json={"universe": {"market": "US"}}, completed_at=NOW, published_at=NOW)
    db_session.add(run)
    db_session.flush()
    for symbol in symbols:
        db_session.add(StockFeatureDaily(run_id=run.id, symbol=symbol, as_of_date=DAY, details_json={"rs_rating": 20}))
    db_session.flush()
    service = SocialThemeMarketService(db_session, calendar=calendar)
    real_read = service.reader.read_facts
    def switch_pointer(*args, **kwargs):
        result = real_read(*args, **kwargs)
        pointer.run_id = run.id
        db_session.flush()
        return result
    monkeypatch.setattr(service.reader, "read_facts", switch_pointer)
    evidence = service.measure("chips", "US", NOW, symbols)
    assert dict(evidence.components)["avg_rs_rating"] == 80
    assert set(dict(evidence.feature_run_ids).values()) == {original_run_id}


def test_frozen_price_dates_and_values_survive_database_change(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(3, 3)
    service = SocialThemeMarketService(db_session, calendar=calendar)
    evidence = service.measure("chips", "US", NOW, symbols)
    assert dict(evidence.price_sessions) == dict.fromkeys(symbols, DAY)
    frozen = evidence.company_observations
    assert len(frozen) == 9
    db_session.query(StockPrice).filter_by(symbol=symbols[0], date=DAY).update({"adj_close": 1})
    db_session.flush()
    assert evidence.company_observations == frozen
    assert service.measure("chips", "US", NOW, symbols).company_observations != frozen


def test_unknown_calendar_is_optional_measurement_unavailability(db_session, basket, monkeypatch):
    from app.services.social_theme_market_service import SocialThemeMarketService, MeasurementUnavailable
    _, symbols, calendar = basket(3, 3)
    def missing(*args, **kwargs):
        raise RuntimeError("missing calendar")
    monkeypatch.setattr(calendar, "session_close", missing)
    with pytest.raises(MeasurementUnavailable, match="calendar_unavailable"):
        SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)


def test_context_does_not_make_company_denominator_unknown(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    theme, symbols, calendar = basket(3, 3)
    db_session.add(StockUniverse(symbol="SPY", market="US", exchange="NYSE", is_common_stock=False, is_active=True))
    db_session.add(ThemeConstituent(theme_cluster_id=theme.id, symbol="SPY", is_active=True))
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, (*symbols, "SPY"))
    assert evidence.accepted_company_count == 3
    assert all(value is not None for _, value in evidence.components)


def test_per_component_cohorts_are_independent(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    from app.infra.db.models.feature_store import StockFeatureDaily
    _, symbols, calendar = basket(4, 4)
    db_session.query(StockFeatureDaily).filter(StockFeatureDaily.symbol.in_(symbols[:2])).update({"details_json": {"rs_rating": None}})
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert dict(evidence.components)["avg_rs_rating"] is None
    assert dict(evidence.measured_company_counts)["avg_rs_rating"] == 2
    assert dict(evidence.components)["pct_above_50ma"] == 100


def test_missing_adjusted_history_is_not_replaced_by_raw_close(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(3, 3)
    db_session.query(StockPrice).filter(StockPrice.symbol.in_(symbols)).update({"adj_close": None})
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert dict(evidence.components)["basket_rs_vs_benchmark"] is None


def test_no_valid_inputs_never_become_zero(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(3, 0, benchmark=False)
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert all(value is None for _, value in evidence.components)


def test_basket_uses_market_excess_return_scale(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(3, 3)
    db_session.query(StockPrice).update({"adj_close": 100})
    db_session.query(StockPrice).filter(StockPrice.symbol.in_(symbols), StockPrice.date == DAY).update({"adj_close": 110})
    db_session.query(StockPrice).filter_by(symbol="SPY", date=DAY).update({"adj_close": 105})
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert dict(evidence.components)["basket_rs_vs_benchmark"] == 75
    assert float(evidence.benchmark_return_1m) == pytest.approx(0.05)


def test_actual_price_session_is_the_one_used_before_grace(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(3, 3)
    before_grace = datetime(2026, 7, 2, 21, tzinfo=timezone.utc)
    # All saved prices were available before this read, including today's bar.
    db_session.query(StockPrice).update({"created_at": before_grace})
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", before_grace, symbols)
    assert evidence.session_date == date(2026, 7, 1)
    assert set(dict(evidence.price_sessions).values()) == {date(2026, 7, 1)}


def test_explicit_run_pin_is_shared_across_measurements(db_session, basket):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    from app.services.social_theme_market_service import SocialThemeMarketService
    from app.infra.db.models.feature_store import FeatureRunPointer
    _, symbols, calendar = basket(3, 3)
    reader = SocialConfirmationReader(db_session, calendar=calendar)
    pin = reader.pin_feature_run("US")
    db_session.delete(db_session.get(FeatureRunPointer, "latest_published_market:US"))
    db_session.flush()
    service = SocialThemeMarketService(db_session, calendar=calendar, pinned_feature_run=pin)
    assert reader.read_facts(symbols[0], "US", NOW, mic="XNYS", pinned_run=pin).rs_rating == 80
    assert dict(service.measure("chips", "US", NOW, symbols).components)["avg_rs_rating"] == 80


def test_incomplete_listing_calendar_only_removes_its_price_components(db_session, basket, monkeypatch):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(4, 4)
    db_session.query(StockUniverse).filter_by(symbol=symbols[0]).update({"exchange": "NASDAQ"})
    db_session.flush()
    original = calendar.trading_days
    def unavailable_history(market, start, end, *, mic=None):
        if mic == "XNAS" and end == DAY and start == date(2025, 6, 27):
            raise RuntimeError("listing history unavailable")
        return original(market, start, end, mic=mic)
    monkeypatch.setattr(calendar, "trading_days", unavailable_history)
    evidence = SocialThemeMarketService(db_session, calendar=calendar).measure("chips", "US", NOW, symbols)
    assert dict(evidence.components)["pct_above_50ma"] == 100


def test_unknown_price_calendar_retains_independent_rs(db_session, basket, monkeypatch):
    from app.services.social_theme_market_service import SocialThemeMarketService
    _, symbols, calendar = basket(3, 3)
    service = SocialThemeMarketService(db_session, calendar=calendar)
    # Simulate a provider failing during its historical range read after it
    # already supplied the current session. Saved features remain readable.
    original = calendar.trading_days
    calls = {}
    def failing(market, start, end, *, mic=None):
        if mic == "XNYS":
            calls[mic] = calls.get(mic, 0) + 1
            if calls[mic] % 2 == 0:
                raise RuntimeError("missing historical schedule")
        return original(market, start, end, mic=mic)
    monkeypatch.setattr(calendar, "trading_days", failing)
    evidence = service.measure("chips", "US", NOW, symbols)
    assert dict(evidence.components)["avg_rs_rating"] == 80


def test_explicit_registered_fallback_benchmark_pin(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    from app.services.benchmark_registry_service import BenchmarkRegistryService
    _, symbols, calendar = basket(3, 3, market="HK")
    fallback = BenchmarkRegistryService().get_candidate_symbols("HK")[1]
    # Keep a usable primary, while requiring the exposure's actual fallback.
    prices = db_session.query(StockPrice).filter_by(symbol="^HSI").all()
    for row in prices:
        db_session.add(StockPrice(symbol=fallback, date=row.date, adj_close=100, close=100, created_at=NOW))
    db_session.flush()
    evidence = SocialThemeMarketService(db_session, calendar=calendar, benchmark_symbol=fallback).measure("chips", "HK", NOW, symbols)
    assert evidence.benchmark_symbol == fallback
    assert all(value is not None for _, value in evidence.components)
    assert evidence.benchmark_candidates == ("^HSI", fallback)


@pytest.mark.parametrize("symbol", ["^HSI", "UNREGISTERED"])
def test_wrong_market_or_unregistered_benchmark_pin_is_rejected(db_session, basket, symbol):
    from app.services.social_theme_market_service import SocialThemeMarketService, MeasurementUnavailable
    _, symbols, calendar = basket(3, 3)
    with pytest.raises(MeasurementUnavailable, match="benchmark_not_registered"):
        SocialThemeMarketService(db_session, calendar=calendar, benchmark_symbol=symbol).measure("chips", "US", NOW, symbols)


def test_missing_pinned_benchmark_does_not_choose_available_primary(db_session, basket):
    from app.services.social_theme_market_service import SocialThemeMarketService
    from app.services.benchmark_registry_service import BenchmarkRegistryService
    _, symbols, calendar = basket(3, 3, market="HK")
    fallback = BenchmarkRegistryService().get_candidate_symbols("HK")[1]
    evidence = SocialThemeMarketService(db_session, calendar=calendar, benchmark_symbol=fallback).measure("chips", "HK", NOW, symbols)
    assert evidence.benchmark_symbol == fallback
    assert dict(evidence.components)["basket_rs_vs_benchmark"] is None
    assert dict(evidence.components)["avg_rs_rating"] == 80
    assert dict(evidence.components)["pct_above_50ma"] == 100
