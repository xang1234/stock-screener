from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.validation_service as validation_service_module
from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster
from app.schemas.validation import ValidationSourceKind
from app.services.validation_service import (
    EvaluatedValidationEvent,
    FailureClusterBuilder,
    PriceOutcomeCalculator,
    RawValidationEvent,
    SCAN_PICK_TOP_N,
    ScanPickValidationSource,
    ThemeAlertValidationSource,
    ValidationService,
    mark_overlapping_windows,
)
from tests.helpers.taxonomy_authority import set_economic_authority

FIXED_TODAY = date(2026, 4, 5)
FIXED_NOW = datetime(2026, 4, 5, 14, 0, tzinfo=UTC)


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
            StockUniverse.__table__,
            ThemeCluster.__table__,
            ThemeAlert.__table__,
        ],
    )
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(
            engine,
            tables=[
                ThemeAlert.__table__,
                ThemeCluster.__table__,
                StockUniverse.__table__,
                StockFeatureDaily.__table__,
                FeatureRun.__table__,
            ],
        )


class _FakePriceCache:
    def __init__(self, histories):
        self._histories = histories

    def get_many_cached_only(self, symbols, period="2y"):
        return {symbol: self._histories.get(symbol) for symbol in symbols}


class _FixedCalendar:
    def __init__(self, last_completed: date):
        self._last_completed = last_completed

    def last_completed_trading_day(self, market):
        return self._last_completed


def _service(histories=None, *, last_completed=FIXED_TODAY):
    return ValidationService(
        outcome_calculator=PriceOutcomeCalculator(_FakePriceCache(histories or {})),
        market_calendar=_FixedCalendar(last_completed),
    )


def _history_frame(start: date, rows: list[tuple[float, float, float, float]]):
    index = pd.DatetimeIndex([start + timedelta(days=offset) for offset in range(len(rows))])
    return pd.DataFrame(
        rows,
        index=index,
        columns=["Open", "High", "Low", "Close"],
    )


def _add_run(session, as_of_date: date, *, market="US") -> FeatureRun:
    run = FeatureRun(
        as_of_date=as_of_date,
        run_type="daily_snapshot",
        status="published",
        config_json={"market": market},
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def _add_universe(session, symbol, *, market="US", industry="Semiconductors", is_common_stock=True):
    session.add(
        StockUniverse(
            symbol=symbol,
            market=market,
            industry=industry,
            is_common_stock=is_common_stock,
        )
    )


def _add_pick(session, run, symbol, composite_score, *, minervini_score=80.0, canslim_score=70.0):
    session.add(
        StockFeatureDaily(
            run_id=run.id,
            symbol=symbol,
            as_of_date=run.as_of_date,
            composite_score=composite_score,
            overall_rating=5,
            passes_count=3,
            details_json={
                "stage": 2,
                "ibd_industry_group": "Semis",
                "minervini_score": minervini_score,
                "canslim_score": canslim_score,
            },
        )
    )


def test_scan_pick_validation_source_limits_to_top_ten_and_breaks_ties_by_symbol(session):
    run = _add_run(session, FIXED_TODAY)

    symbols = ["ZZZ", "BBB", "AAA", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK"]
    for symbol in symbols:
        _add_universe(session, symbol)
        _add_pick(session, run, symbol, 90.0)
    session.commit()

    source = ScanPickValidationSource()
    events, degraded = source.collect(
        session, cutoff_date=FIXED_TODAY - timedelta(days=1), market="US"
    )

    assert degraded == []
    assert [event.symbol for event in events] == sorted(symbols)[:SCAN_PICK_TOP_N]


def test_scan_pick_top_ten_is_taken_among_rows_scored_by_both_methods_and_not_funds(session):
    run = _add_run(session, FIXED_TODAY)
    _add_universe(session, "IPOX")
    _add_pick(session, run, "IPOX", 99.0, minervini_score=None, canslim_score=None)
    _add_universe(session, "HALF")
    _add_pick(session, run, "HALF", 98.5, canslim_score=None)
    _add_universe(session, "FLAH")
    _add_pick(session, run, "FLAH", 98.4, minervini_score=None)
    _add_universe(session, "MRVU", industry="Exchange Traded Fund")
    _add_pick(session, run, "MRVU", 98.0)
    _add_universe(session, "PFD", is_common_stock=False)
    _add_pick(session, run, "PFD", 97.0)
    for offset in range(11):
        symbol = f"S{offset:02d}"
        _add_universe(session, symbol, industry=None if offset == 0 else "Semiconductors")
        _add_pick(session, run, symbol, 90.0 - offset)
    session.commit()

    events, degraded = ScanPickValidationSource().collect(
        session, cutoff_date=FIXED_TODAY - timedelta(days=1), market="US"
    )

    assert degraded == []
    assert [event.symbol for event in events] == [
        "S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
    ]


def _add_legacy_pick(session, run, symbol, composite_score, **scores):
    """A row written before 20261013_0066: scores only in details_json, typed columns NULL."""
    session.execute(
        StockFeatureDaily.__table__.insert().values(
            run_id=run.id,
            symbol=symbol,
            as_of_date=run.as_of_date,
            composite_score=composite_score,
            overall_rating=5,
            details_json={"stage": 2, **scores},
        )
    )


def test_legacy_rows_yield_no_picks_and_flag_the_window(session):
    run = _add_run(session, FIXED_TODAY)
    _add_universe(session, "OLD")
    _add_legacy_pick(session, run, "OLD", 90.0, minervini_score=80.0, canslim_score=70.0)
    session.commit()

    events, degraded = ScanPickValidationSource().collect(
        session, cutoff_date=FIXED_TODAY - timedelta(days=1), market="US"
    )

    assert (events, degraded) == ([], ["legacy_runs_without_typed_scores"])


def test_legacy_runs_outside_the_window_or_market_are_not_flagged(session):
    _add_universe(session, "OLD")
    _add_legacy_pick(session, _add_run(session, FIXED_TODAY - timedelta(days=5)), "OLD", 90.0)
    _add_legacy_pick(session, _add_run(session, FIXED_TODAY, market="HK"), "OLD", 90.0)
    _add_universe(session, "NEW")
    _add_pick(session, _add_run(session, FIXED_TODAY), "NEW", 90.0)
    session.commit()

    events, degraded = ScanPickValidationSource().collect(
        session, cutoff_date=FIXED_TODAY - timedelta(days=1), market="US"
    )

    assert ([event.symbol for event in events], degraded) == (["NEW"], [])


def test_scan_picks_are_filtered_to_the_requested_market(session):
    us_run = _add_run(session, FIXED_TODAY)
    hk_run = _add_run(session, FIXED_TODAY, market="HK")
    _add_universe(session, "AAPL")
    _add_pick(session, us_run, "AAPL", 80.0)
    _add_universe(session, "0700.HK", market="HK", industry=None)
    _add_pick(session, hk_run, "0700.HK", 95.0)
    session.commit()
    service = _service()

    us = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
        market="us",
    )
    hk = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
        market="HK",
    )

    assert (us.market, [event.attributes["symbol"] for event in us.recent_events]) == ("US", ["AAPL"])
    assert (hk.market, [event.attributes["symbol"] for event in hk.recent_events]) == ("HK", ["0700.HK"])


def test_theme_alert_tickers_are_filtered_to_the_requested_market(session):
    _add_universe(session, "NVDA")
    _add_universe(session, "0700.HK", market="HK")
    session.add(
        ThemeAlert(
            alert_type="breakout",
            title="Mixed tickers",
            severity="warning",
            related_tickers=["NVDA", "0700.HK", "NOTLISTED", "9988.HK"],
            triggered_at=FIXED_NOW,
        )
    )
    session.commit()
    source = ThemeAlertValidationSource()
    cutoff = FIXED_TODAY - timedelta(days=7)

    us_events, _ = source.collect(session, cutoff_date=cutoff, market="US")

    assert [event.symbol for event in us_events] == ["NVDA", "NOTLISTED"]


def test_theme_alerts_serve_only_the_us_market(session):
    _add_universe(session, "0700.HK", market="HK")
    session.add(
        ThemeAlert(
            alert_type="breakout",
            title="HK ticker",
            severity="warning",
            related_tickers=["0700.HK"],
            triggered_at=FIXED_NOW,
        )
    )
    session.commit()

    events, degraded = ThemeAlertValidationSource().collect(
        session, cutoff_date=FIXED_TODAY - timedelta(days=7), market="HK"
    )

    assert (events, degraded) == ([], ["theme_alerts_us_only"])


def test_freshness_is_scoped_to_the_requested_market(session):
    _add_run(session, date(2026, 4, 2))
    _add_run(session, date(2026, 4, 3), market="HK")
    session.add(
        ThemeAlert(
            alert_type="breakout",
            title="US alert",
            severity="warning",
            related_tickers=["NVDA"],
            triggered_at=FIXED_NOW,
        )
    )
    session.commit()
    service = _service()

    freshness = {
        market: service.get_overview(
            session,
            source_kind=ValidationSourceKind.SCAN_PICK,
            lookback_days=30,
            as_of_date=FIXED_TODAY,
            market=market,
        ).freshness
        for market in ("US", "HK")
    }

    assert {
        market: (item.latest_feature_as_of_date, item.latest_theme_alert_at)
        for market, item in freshness.items()
    } == {
        "US": ("2026-04-02", "2026-04-05T14:00:00+00:00"),
        "HK": ("2026-04-03", None),
    }


def test_stock_validation_uses_the_symbols_own_market(session):
    hk_run = _add_run(session, FIXED_TODAY, market="HK")
    _add_universe(session, "0700.HK", market="HK")
    _add_pick(session, hk_run, "0700.HK", 95.0)
    _add_pick(session, hk_run, "9988.HK", 90.0)
    session.commit()
    service = _service()

    refs = {
        symbol: [
            (event.source_kind, event.source_ref)
            for event in service.get_stock_validation(
                session, symbol=symbol, lookback_days=30, as_of_date=FIXED_TODAY
            ).recent_events
        ]
        for symbol in ("0700.hk", "9988.HK")
    }

    assert refs == {
        "0700.hk": [(ValidationSourceKind.SCAN_PICK, f"run:{hk_run.id}")],
        "9988.HK": [(ValidationSourceKind.SCAN_PICK, f"run:{hk_run.id}")],
    }


def test_scan_picks_follow_the_run_market_and_keep_symbols_no_longer_listed(session):
    us_run = _add_run(session, FIXED_TODAY)
    hk_run = _add_run(session, FIXED_TODAY, market="HK")
    _add_pick(session, us_run, "GONE", 90.0)
    _add_universe(session, "AAPL")
    _add_pick(session, us_run, "AAPL", 80.0)
    _add_pick(session, hk_run, "9988.HK", 95.0)
    older_runs = [
        FeatureRun(
            as_of_date=FIXED_TODAY,
            run_type="daily_snapshot",
            status="published",
            config_json=config,
        )
        for config in ({"universe": {"market": "HK"}}, None)
    ]
    session.add_all(older_runs)
    session.commit()
    _add_pick(session, older_runs[0], "0005.HK", 85.0)
    _add_pick(session, older_runs[1], "OLDUS", 85.0)
    session.commit()
    source = ScanPickValidationSource()
    cutoff = FIXED_TODAY - timedelta(days=1)

    us_events, _ = source.collect(session, cutoff_date=cutoff, market="US")
    hk_events, _ = source.collect(session, cutoff_date=cutoff, market="HK")

    assert sorted(event.symbol for event in us_events) == ["AAPL", "GONE", "OLDUS"]
    assert sorted(event.symbol for event in hk_events) == ["0005.HK", "9988.HK"]


def test_theme_alert_validation_source_filters_supported_types_and_expands_related_tickers(session):
    theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infra",
        display_name="AI Infrastructure",
        pipeline="technical",
        lifecycle_state="active",
    )
    session.add(theme)
    session.commit()
    session.refresh(theme)

    session.add_all(
        [
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="breakout",
                title="Breakout",
                severity="warning",
                related_tickers=["NVDA", "AVGO"],
                triggered_at=FIXED_NOW,
            ),
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="velocity_spike",
                title="Velocity",
                severity="critical",
                related_tickers=["MSFT"],
                triggered_at=FIXED_NOW,
            ),
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="new_theme",
                title="Ignore",
                severity="info",
                related_tickers=["AAPL"],
                triggered_at=FIXED_NOW,
            ),
        ]
    )
    session.commit()

    source = ThemeAlertValidationSource()
    events, degraded = source.collect(
        session, cutoff_date=FIXED_TODAY - timedelta(days=7), market="US"
    )

    assert degraded == []
    assert [(event.attributes["alert_type"], event.symbol) for event in events] == [
        ("velocity_spike", "MSFT"),
        ("breakout", "NVDA"),
        ("breakout", "AVGO"),
    ]


def test_theme_alert_validation_source_uses_eastern_day_bounds(session):
    theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infra",
        display_name="AI Infrastructure",
        pipeline="technical",
        lifecycle_state="active",
    )
    session.add(theme)
    session.commit()
    session.refresh(theme)

    session.add_all(
        [
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="breakout",
                title="Late evening UTC still counts for prior Eastern day",
                severity="warning",
                related_tickers=["NVDA"],
                triggered_at=datetime(2026, 4, 5, 1, 30, tzinfo=UTC),
            ),
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="breakout",
                title="Next Eastern day",
                severity="warning",
                related_tickers=["MSFT"],
                triggered_at=datetime(2026, 4, 5, 4, 30, tzinfo=UTC),
            ),
        ]
    )
    session.commit()

    source = ThemeAlertValidationSource()
    events, degraded = source.collect(
        session,
        cutoff_date=date(2026, 4, 4),
        until_date=date(2026, 4, 4),
        market="US",
    )

    assert degraded == []
    assert [event.symbol for event in events] == ["NVDA"]


def test_validation_service_freshness_uses_eastern_day_bounds_for_theme_alerts(session):
    theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infra",
        display_name="AI Infrastructure",
        pipeline="technical",
        lifecycle_state="active",
    )
    session.add(theme)
    session.commit()
    session.refresh(theme)

    session.add(
        ThemeAlert(
            theme_cluster_id=theme.id,
            alert_type="breakout",
            title="Late evening UTC still counts for prior Eastern day",
            severity="warning",
            related_tickers=["NVDA"],
            triggered_at=datetime(2026, 4, 5, 1, 30, tzinfo=UTC),
        )
    )
    session.commit()

    service = _service({"NVDA": None})

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.THEME_ALERT,
        lookback_days=30,
        as_of_date=date(2026, 4, 4),
    )

    assert payload.freshness.latest_theme_alert_at == "2026-04-05T01:30:00+00:00"
    assert len(payload.recent_events) == 1


def test_validation_service_serves_no_legacy_theme_alerts_in_economic_mode(session):
    session.add(
        ThemeAlert(
            alert_type="breakout",
            title="Legacy breakout",
            severity="warning",
            related_tickers=["NVDA"],
            triggered_at=FIXED_NOW,
        )
    )
    session.commit()
    set_economic_authority(session)
    service = _service({"NVDA": None})

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.THEME_ALERT,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert payload.recent_events == []
    assert payload.freshness.latest_theme_alert_at is None
    assert "economic_theme_alerts_unavailable" in payload.degraded_reasons


def test_validation_service_uses_eastern_now_for_default_as_of_date(session, monkeypatch):
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 4, 5, 2, 0, tzinfo=UTC)

    theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infra",
        display_name="AI Infrastructure",
        pipeline="technical",
        lifecycle_state="active",
    )
    session.add(theme)
    session.commit()
    session.refresh(theme)

    session.add(
        ThemeAlert(
            theme_cluster_id=theme.id,
            alert_type="breakout",
            title="After Eastern midnight only",
            severity="warning",
            related_tickers=["NVDA"],
            triggered_at=datetime(2026, 4, 5, 4, 30, tzinfo=UTC),
        )
    )
    session.commit()

    monkeypatch.setattr(validation_service_module, "datetime", _FixedDateTime)

    service = _service({"NVDA": None})

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.THEME_ALERT,
        lookback_days=30,
    )

    assert payload.recent_events == []
    assert payload.freshness.latest_theme_alert_at is None


def test_price_outcome_calculator_rolls_to_next_trading_session_and_computes_windows():
    saturday_event = RawValidationEvent(
        symbol="NVDA",
        source_kind=ValidationSourceKind.SCAN_PICK,
        source_ref="run:1",
        event_at=date(2026, 4, 4),
        attributes={"symbol": "NVDA"},
    )
    history = _history_frame(
        date(2026, 4, 6),
        [
            (100, 105, 99, 104),
            (104, 110, 102, 109),
            (109, 112, 103, 106),
            (106, 114, 101, 111),
            (111, 113, 98, 112),
        ],
    )
    calculator = PriceOutcomeCalculator(_FakePriceCache({"NVDA": history}))

    evaluated, _ = calculator.evaluate_many([saturday_event])

    assert len(evaluated) == 1
    event = evaluated[0]
    assert event.entry_at == date(2026, 4, 6)
    assert event.entry_price == 100.0
    assert event.return_1s_pct == pytest.approx(4.0)
    assert event.return_5s_pct == pytest.approx(12.0)
    assert event.mfe_5s_pct == pytest.approx(14.0)
    assert event.mae_5s_pct == pytest.approx(-2.0)
    assert event.missing_horizons == frozenset()


def test_price_outcome_calculator_marks_missing_five_session_history():
    event = RawValidationEvent(
        symbol="MSFT",
        source_kind=ValidationSourceKind.THEME_ALERT,
        source_ref="alert:4",
        event_at=date(2026, 4, 1),
        attributes={"symbol": "MSFT"},
    )
    history = _history_frame(
        date(2026, 4, 2),
        [
            (50, 51, 49, 50.5),
            (50.5, 52, 50, 51),
            (51, 53, 50.5, 52),
        ],
    )
    calculator = PriceOutcomeCalculator(_FakePriceCache({"MSFT": history}))

    evaluated, _ = calculator.evaluate_many([event])

    assert evaluated[0].return_1s_pct == pytest.approx(1.0)
    assert evaluated[0].return_5s_pct is None
    assert evaluated[0].missing_horizons == frozenset({5})


def test_price_outcome_calculator_degrades_zero_open_entry_prices():
    event = RawValidationEvent(
        symbol="MSFT",
        source_kind=ValidationSourceKind.THEME_ALERT,
        source_ref="alert:zero-open",
        event_at=date(2026, 4, 1),
        attributes={"symbol": "MSFT"},
    )
    history = _history_frame(
        date(2026, 4, 2),
        [
            (0, 51, 49, 50.5),
            (50.5, 52, 50, 51),
            (51, 53, 50.5, 52),
            (52, 54, 51, 53),
            (53, 55, 52, 54),
        ],
    )
    calculator = PriceOutcomeCalculator(_FakePriceCache({"MSFT": history}))

    evaluated, _ = calculator.evaluate_many([event])

    assert evaluated[0].entry_at == date(2026, 4, 2)
    assert evaluated[0].entry_price is None
    assert evaluated[0].return_1s_pct is None
    assert evaluated[0].return_5s_pct is None
    assert evaluated[0].mfe_5s_pct is None
    assert evaluated[0].mae_5s_pct is None
    assert evaluated[0].missing_horizons == frozenset({1, 5})


def test_price_outcome_calculator_uses_same_day_entry_for_premarket_alerts_only():
    history = _history_frame(
        date(2026, 4, 2),
        [
            (100, 105, 99, 104),
            (104, 110, 102, 109),
            (109, 112, 103, 106),
            (106, 114, 101, 111),
            (111, 113, 98, 112),
            (112, 116, 110, 115),
        ],
    )
    calculator = PriceOutcomeCalculator(_FakePriceCache({"NVDA": history}))
    premarket_event = RawValidationEvent(
        symbol="NVDA",
        source_kind=ValidationSourceKind.THEME_ALERT,
        source_ref="alert:premarket",
        event_at=datetime(2026, 4, 2, 8, 0, tzinfo=UTC),
        attributes={"symbol": "NVDA"},
    )
    after_open_event = RawValidationEvent(
        symbol="NVDA",
        source_kind=ValidationSourceKind.THEME_ALERT,
        source_ref="alert:after-open",
        event_at=datetime(2026, 4, 2, 15, 0, tzinfo=UTC),
        attributes={"symbol": "NVDA"},
    )

    evaluated, _ = calculator.evaluate_many([premarket_event, after_open_event])

    assert evaluated[0].entry_at == date(2026, 4, 2)
    assert evaluated[0].entry_price == 100.0
    assert evaluated[1].entry_at == date(2026, 4, 3)
    assert evaluated[1].entry_price == 104.0


def test_price_outcome_calculator_uses_eastern_calendar_day_for_late_evening_theme_alerts():
    history = _history_frame(
        date(2026, 4, 2),
        [
            (100, 105, 99, 104),
            (104, 110, 102, 109),
            (109, 112, 103, 106),
            (106, 114, 101, 111),
            (111, 113, 98, 112),
            (112, 116, 110, 115),
        ],
    )
    calculator = PriceOutcomeCalculator(_FakePriceCache({"NVDA": history}))
    late_evening_event = RawValidationEvent(
        symbol="NVDA",
        source_kind=ValidationSourceKind.THEME_ALERT,
        source_ref="alert:late-evening",
        event_at=datetime(2026, 4, 2, 0, 30, tzinfo=UTC),
        attributes={"symbol": "NVDA"},
    )

    evaluated, _ = calculator.evaluate_many([late_evening_event])

    assert evaluated[0].entry_at == date(2026, 4, 2)
    assert evaluated[0].entry_price == 100.0


def test_failure_cluster_builder_uses_source_specific_bucket_fields():
    builder = FailureClusterBuilder()
    losing_scan_event = EvaluatedValidationEvent(
        raw=RawValidationEvent(
            symbol="NVDA",
            source_kind=ValidationSourceKind.SCAN_PICK,
            source_ref="run:1",
            event_at=FIXED_TODAY,
            attributes={"rating": "Buy", "stage": 2, "ibd_industry_group": "Semis"},
        ),
        entry_at=FIXED_TODAY,
        entry_price=100.0,
        return_1s_pct=-1.0,
        return_5s_pct=-3.0,
        mfe_5s_pct=1.0,
        mae_5s_pct=-4.0,
        missing_horizons=frozenset(),
    )
    losing_theme_event = EvaluatedValidationEvent(
        raw=RawValidationEvent(
            symbol="MSFT",
            source_kind=ValidationSourceKind.THEME_ALERT,
            source_ref="alert:1",
            event_at=FIXED_NOW,
            attributes={"alert_type": "breakout", "severity": "warning", "theme": "AI"},
        ),
        entry_at=FIXED_TODAY,
        entry_price=100.0,
        return_1s_pct=-0.5,
        return_5s_pct=-2.0,
        mfe_5s_pct=1.5,
        mae_5s_pct=-2.5,
        missing_horizons=frozenset(),
    )

    clusters = builder.build([losing_scan_event, losing_theme_event])
    cluster_keys = {cluster.cluster_key for cluster in clusters}

    assert "rating:Buy" in cluster_keys
    assert "stage:2" in cluster_keys
    assert "ibd_industry_group:Semis" in cluster_keys
    assert "alert_type:breakout" in cluster_keys
    assert "severity:warning" in cluster_keys
    assert len(clusters) == 5


def test_normalize_price_history_keeps_the_exchange_date_of_tz_aware_bars():
    index = pd.DatetimeIndex(["2026-03-09", "2026-03-10"]).tz_localize("Asia/Tokyo")
    frame = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0], "Close": [1.0, 2.0]},
        index=index,
    )

    normalized = validation_service_module._normalize_price_history(frame)

    assert [d.date().isoformat() for d in normalized.index] == ["2026-03-09", "2026-03-10"]


NVDA_ROWS = [(100, 105, 99, 104), (104, 110, 102, 109), (109, 112, 103, 106), (106, 114, 101, 111), (111, 113, 98, 112)]
BENCHMARK_ROWS = [(500, 506, 499, 505), (505, 507, 503, 506), (506, 508, 504, 507), (507, 509, 505, 508), (508, 511, 507, 510)]


def test_price_outcome_calculator_measures_excess_over_the_first_cached_benchmark():
    event = RawValidationEvent(
        symbol="NVDA",
        source_kind=ValidationSourceKind.SCAN_PICK,
        source_ref="run:1",
        event_at=date(2026, 4, 4),
        attributes={"symbol": "NVDA"},
    )
    calculator = PriceOutcomeCalculator(
        _FakePriceCache(
            {
                "NVDA": _history_frame(date(2026, 4, 6), NVDA_ROWS),
                "SPY": None,
                "IVV": _history_frame(date(2026, 4, 6), BENCHMARK_ROWS),
            }
        )
    )

    evaluated, benchmark_symbol = calculator.evaluate_many(
        [event], benchmark_candidates=["SPY", "IVV"]
    )

    assert benchmark_symbol == "IVV"
    assert evaluated[0].exit_at == date(2026, 4, 10)
    assert evaluated[0].excess_return_1s_pct == pytest.approx(3.0)
    assert evaluated[0].excess_return_5s_pct == pytest.approx(10.0)


def test_price_outcome_calculator_prefers_the_benchmark_with_the_latest_bar():
    event = RawValidationEvent(
        symbol="NVDA",
        source_kind=ValidationSourceKind.SCAN_PICK,
        source_ref="run:1",
        event_at=date(2026, 4, 4),
        attributes={"symbol": "NVDA"},
    )
    stale = _history_frame(date(2026, 4, 1), BENCHMARK_ROWS)
    current = _history_frame(date(2026, 4, 6), BENCHMARK_ROWS)

    def benchmark_for(histories):
        calculator = PriceOutcomeCalculator(
            _FakePriceCache({"NVDA": _history_frame(date(2026, 4, 6), NVDA_ROWS), **histories})
        )
        return calculator.evaluate_many([event], benchmark_candidates=["SPY", "IVV"])[-1]

    assert benchmark_for({"SPY": stale, "IVV": current}) == "IVV"
    assert benchmark_for({"SPY": current, "IVV": current}) == "SPY"


def test_overview_degrades_once_when_the_benchmark_lacks_a_window_bar(session):
    run = _add_run(session, date(2026, 3, 27))
    for symbol in ("NVDA", "AMD"):
        _add_universe(session, symbol)
        _add_pick(session, run, symbol, 90.0)
    session.commit()
    start = date(2026, 3, 28)
    service = _service(
        {
            "NVDA": _history_frame(start, NVDA_ROWS),
            "AMD": _history_frame(start, NVDA_ROWS),
            "SPY": _history_frame(start, BENCHMARK_ROWS[:4]),
        }
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert (payload.benchmark_symbol, payload.degraded_reasons) == (
        "SPY",
        ["missing_benchmark_prices"],
    )
    assert [
        (event.excess_return_1s_pct, event.excess_return_5s_pct) for event in payload.recent_events
    ] == [(3.0, None), (3.0, None)]


def test_overview_reports_benchmark_relative_returns(session):
    run = _add_run(session, date(2026, 3, 27))
    for symbol, score in (("NVDA", 95.0), ("AMD", 90.0), ("MU", 85.0)):
        _add_universe(session, symbol)
        _add_pick(session, run, symbol, score)
    session.commit()
    start = date(2026, 3, 28)
    service = _service(
        {
            "NVDA": _history_frame(start, NVDA_ROWS),
            "AMD": _history_frame(start, [(50, 51, 49, 50)] * 4 + [(50, 51, 48, 49)]),
            "MU": _history_frame(start, [(200, 211, 199, 210)] + [(210, 211, 205, 206)] * 4),
            "SPY": _history_frame(start, BENCHMARK_ROWS),
        }
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert (payload.benchmark_symbol, payload.degraded_reasons) == ("SPY", [])
    assert [
        (h.horizon_sessions, h.avg_excess_return_pct, h.median_excess_return_pct)
        for h in payload.horizons
    ] == [(1, 2.0, 3.0), (5, 2.3333, 1.0)]
    assert {
        event.attributes["symbol"]: (event.excess_return_1s_pct, event.excess_return_5s_pct)
        for event in payload.recent_events
    } == {"NVDA": (3.0, 10.0), "AMD": (-1.0, -4.0), "MU": (4.0, 1.0)}


def test_overview_without_a_cached_benchmark_reports_returns_but_no_excess(session):
    run = _add_run(session, date(2026, 3, 27))
    _add_universe(session, "NVDA")
    _add_pick(session, run, "NVDA", 90.0)
    session.commit()
    service = _service({"NVDA": _history_frame(date(2026, 3, 28), NVDA_ROWS)})

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert (payload.benchmark_symbol, payload.degraded_reasons) == (
        None,
        ["missing_benchmark_prices"],
    )
    assert [
        (h.sample_size, h.avg_return_pct, h.avg_excess_return_pct, h.excess_sample_size)
        for h in payload.horizons
    ] == [(1, 4.0, None, 0), (1, 12.0, None, 0)]


def test_overview_reports_how_many_events_the_excess_figures_cover(session):
    _add_universe(session, "NVDA")
    _add_universe(session, "AMD")
    _add_pick(session, _add_run(session, date(2026, 3, 27)), "NVDA", 90.0)
    _add_pick(session, _add_run(session, date(2026, 3, 29)), "AMD", 90.0)
    session.commit()
    service = _service(
        {
            "NVDA": _history_frame(date(2026, 3, 28), NVDA_ROWS),
            "AMD": _history_frame(date(2026, 3, 30), NVDA_ROWS),
            "SPY": _history_frame(date(2026, 3, 28), BENCHMARK_ROWS).drop(
                pd.Timestamp(2026, 3, 30)
            ),
        }
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.excess_sample_size) for h in payload.horizons
    ] == [(1, 2, 1), (5, 2, 1)]


def test_overview_ignores_bars_after_the_last_completed_session(session):
    run = _add_run(session, date(2026, 3, 27))
    _add_universe(session, "NVDA")
    _add_pick(session, run, "NVDA", 95.0)
    session.commit()
    start = date(2026, 3, 28)
    service = _service(
        {
            "NVDA": _history_frame(start, NVDA_ROWS),
            "SPY": _history_frame(start, BENCHMARK_ROWS[:4]),
            "IVV": _history_frame(start, BENCHMARK_ROWS),
        },
        last_completed=date(2026, 3, 31),
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert payload.benchmark_symbol == "SPY"
    assert [(h.horizon_sessions, h.sample_size, h.skipped_missing_history) for h in payload.horizons] == [
        (1, 1, 0),
        (5, 0, 1),
    ]
    assert payload.recent_events[0].return_5s_pct is None


def test_overview_counts_a_repeated_pick_once_per_five_session_window(session):
    _add_universe(session, "NVDA")
    for as_of in (date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4), date(2026, 3, 6), date(2026, 3, 7)):
        _add_pick(session, _add_run(session, as_of), "NVDA", 90.0)
    session.commit()
    service = _service({"NVDA": _history_frame(date(2026, 3, 3), [(10, 11, 9, 10.5)] * 12)})

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=60,
        as_of_date=FIXED_TODAY,
    )

    assert [(event.attributes["run_as_of_date"], event.counted) for event in payload.recent_events] == [
        ("2026-03-07", True),
        ("2026-03-06", False),
        ("2026-03-04", False),
        ("2026-03-03", False),
        ("2026-03-02", True),
    ]
    assert [
        (h.horizon_sessions, h.sample_size, h.suppressed_overlapping) for h in payload.horizons
    ] == [(1, 5, 0), (5, 2, 3)]


def test_a_suppressed_repeat_does_not_degrade_the_benchmark_comparison(session):
    _add_universe(session, "NVDA")
    for as_of in (date(2026, 3, 27), date(2026, 3, 28)):
        _add_pick(session, _add_run(session, as_of), "NVDA", 90.0)
    session.commit()
    start = date(2026, 3, 28)
    service = _service(
        {
            "NVDA": _history_frame(start, NVDA_ROWS + [(112, 113, 111, 112)]),
            "SPY": _history_frame(start, BENCHMARK_ROWS),
        }
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert [(event.event_at, event.counted) for event in payload.recent_events] == [
        ("2026-03-28", False),
        ("2026-03-27", True),
    ]
    assert payload.degraded_reasons == []


def test_a_pick_just_before_the_lookback_still_blocks_repeats_after_it(session):
    _add_universe(session, "NVDA")
    before_cutoff = _add_run(session, date(2026, 3, 5))
    _add_pick(session, before_cutoff, "NVDA", 90.0)
    _add_pick(session, before_cutoff, "GONE", 80.0)
    _add_pick(session, _add_run(session, date(2026, 3, 7)), "NVDA", 90.0)
    session.commit()
    bars = _history_frame(date(2026, 3, 6), [(10, 11, 9, 10.5)] * 12)
    service = _service({"NVDA": bars, "SPY": bars})

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert [(event.event_at, event.counted) for event in payload.recent_events] == [
        ("2026-03-07", False),
    ]
    assert [
        (h.horizon_sessions, h.sample_size, h.suppressed_overlapping) for h in payload.horizons
    ] == [(1, 1, 0), (5, 0, 1)]
    assert payload.degraded_reasons == []


def _evaluated(source_kind, symbol, source_ref, entry_at, exit_at=None, *, entry_price=10.0):
    return EvaluatedValidationEvent(
        raw=RawValidationEvent(
            symbol=symbol,
            source_kind=source_kind,
            source_ref=source_ref,
            event_at=date(2026, 3, 20),
            attributes={},
        ),
        entry_at=entry_at,
        entry_price=entry_price if entry_at else None,
        return_1s_pct=1.0 if entry_at else None,
        return_5s_pct=None,
        mfe_5s_pct=None,
        mae_5s_pct=None,
        missing_horizons=frozenset({5}) if entry_at else frozenset({1, 5}),
        exit_at=exit_at,
    )


def test_an_incomplete_window_blocks_later_events_for_the_same_source_and_symbol():
    scan, alert = ValidationSourceKind.SCAN_PICK, ValidationSourceKind.THEME_ALERT
    events = [
        _evaluated(scan, "AAA", "run:2", date(2026, 3, 30)),
        _evaluated(scan, "AAA", "run:1", date(2026, 3, 25)),
        _evaluated(scan, "AAA", "run:3", None),
        _evaluated(scan, "BBB", "run:4", date(2026, 3, 30)),
        _evaluated(alert, "AAA", "alert:1", date(2026, 3, 30)),
    ]

    marked = mark_overlapping_windows(events)

    assert [(event.raw.source_ref, event.counted) for event in marked] == [
        ("run:2", False),
        ("run:1", True),
        ("run:3", True),
        ("run:4", True),
        ("alert:1", True),
    ]


def test_an_entry_without_a_price_does_not_block_a_window():
    scan = ValidationSourceKind.SCAN_PICK
    events = [
        _evaluated(scan, "AAA", "run:1", date(2026, 3, 25), date(2026, 3, 31), entry_price=None),
        _evaluated(scan, "AAA", "run:2", date(2026, 3, 26)),
    ]

    marked = mark_overlapping_windows(events)

    assert [(event.raw.source_ref, event.counted) for event in marked] == [
        ("run:1", True),
        ("run:2", True),
    ]
