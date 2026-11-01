from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.validation_service as validation_service_module
from app.database import Base
from app.infra.db.models.backtest_replay import DEFINITION_VERSION, BacktestReplayPick
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster
from app.schemas.validation import ValidationSourceKind
from app.services.market_calendar_service import CalendarCoverageExpired
from app.services.validation_service import (
    EvaluatedValidationEvent,
    FailureClusterBuilder,
    HorizonOutcome,
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
            BacktestReplayPick.__table__,
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
                BacktestReplayPick.__table__,
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
    """Every day is a session, as in ``_history_frame``, unless ``sessions`` lists them."""

    def __init__(self, last_completed: date, sessions=None):
        self._last_completed = last_completed
        self._sessions = sessions

    def last_completed_trading_day(self, market):
        return self._last_completed

    def trading_days(self, market, start, end):
        if self._sessions is None:
            return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
        return [day for day in self._sessions if start <= day <= end]


def _service(histories=None, *, last_completed=FIXED_TODAY, sessions=None):
    return ValidationService(
        outcome_calculator=PriceOutcomeCalculator(_FakePriceCache(histories or {})),
        market_calendar=_FixedCalendar(last_completed, sessions),
    )


def _sessions_of(frame):
    return list(frame.index.date)


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

    evaluated, _ = calculator.evaluate_many([saturday_event], sessions=_sessions_of(history))

    assert len(evaluated) == 1
    event = evaluated[0]
    assert event.entry_at == date(2026, 4, 6)
    assert event.entry_price == 100.0
    assert set(event.outcomes) == {1, 5}
    one, five = event.outcomes[1], event.outcomes[5]
    assert (one.return_pct, one.mfe_pct, one.mae_pct) == pytest.approx((4.0, 5.0, -1.0))
    assert (five.return_pct, five.mfe_pct, five.mae_pct) == pytest.approx((12.0, 14.0, -2.0))


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
    sessions = [date(2026, 4, 2) + timedelta(days=offset) for offset in range(9)]

    evaluated, _ = calculator.evaluate_many([event], sessions=sessions)

    assert evaluated[0].outcomes[1].return_pct == pytest.approx(1.0)
    assert set(evaluated[0].outcomes) == {1}


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

    evaluated, _ = calculator.evaluate_many([event], sessions=_sessions_of(history))

    assert (evaluated[0].entry_at, evaluated[0].entry_price, evaluated[0].outcomes) == (
        date(2026, 4, 2),
        None,
        {},
    )


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

    evaluated, _ = calculator.evaluate_many(
        [premarket_event, after_open_event], sessions=_sessions_of(history)
    )

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

    evaluated, _ = calculator.evaluate_many([late_evening_event], sessions=_sessions_of(history))

    assert evaluated[0].entry_at == date(2026, 4, 2)
    assert evaluated[0].entry_price == 100.0


def _outcome(return_pct=0.0):
    return HorizonOutcome(
        return_pct=return_pct,
        mfe_pct=0.0,
        mae_pct=0.0,
        stopped_return_pct=return_pct,
        stopped_out=False,
    )


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
        outcomes={1: _outcome(-1.0), 5: _outcome(-3.0)},
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
        outcomes={1: _outcome(-0.5), 5: _outcome(-2.0)},
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
        [event], sessions=_sessions_of(_history_frame(date(2026, 4, 6), NVDA_ROWS)),
        benchmark_candidates=["SPY", "IVV"],
    )

    assert benchmark_symbol == "IVV"
    assert evaluated[0].exits[5] == date(2026, 4, 10)
    assert evaluated[0].outcomes[1].excess_return_pct == pytest.approx(3.0)
    assert evaluated[0].outcomes[5].excess_return_pct == pytest.approx(10.0)


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
        return calculator.evaluate_many(
            [event], sessions=_sessions_of(current), benchmark_candidates=["SPY", "IVV"]
        )[-1]

    assert benchmark_for({"SPY": stale, "IVV": current}) == "IVV"
    assert benchmark_for({"SPY": current, "IVV": current}) == "SPY"


# An entry of 64 keeps every percentage exact in binary; its 7% stop sits at 59.52.
FLAT_ROW = (64, 66, 62, 68)
GAP_ROWS = [(64, 66, 62, 64), (48, 50, 46, 49), (49, 52, 48, 50), (50, 56, 50, 56), (56, 64, 56, 64)]
ENTRY_DAY_STOP_ROWS = [(64, 66, 58, 62)] + [(62, 70, 62, 68)] * 4


def _scan_event(symbol, event_at=date(2026, 1, 1)):
    return RawValidationEvent(
        symbol=symbol,
        source_kind=ValidationSourceKind.SCAN_PICK,
        source_ref="run:1",
        event_at=event_at,
        attributes={"symbol": symbol},
    )


def test_price_outcome_calculator_reports_twenty_and_sixty_session_outcomes():
    rows = [(64, 64, 62, 64)] * 60
    rows[19] = (64, 96, 62, 80)
    rows[59] = (64, 64, 60, 72)
    calculator = PriceOutcomeCalculator(
        _FakePriceCache(
            {
                "NVDA": _history_frame(date(2026, 1, 2), rows),
                "AMD": _history_frame(date(2026, 1, 2), rows[:25]),
            }
        )
    )

    nvda, amd = calculator.evaluate_many(
        [_scan_event("NVDA"), _scan_event("AMD")],
        sessions=_sessions_of(_history_frame(date(2026, 1, 2), rows)),
    )[0]

    def unstopped(return_pct, mfe_pct, mae_pct):
        return HorizonOutcome(
            return_pct=return_pct,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            stopped_return_pct=return_pct,
            stopped_out=False,
        )

    assert nvda.exits == {
        1: date(2026, 1, 2), 5: date(2026, 1, 6), 20: date(2026, 1, 21), 60: date(2026, 3, 2),
    }
    assert nvda.outcomes == {
        1: unstopped(0.0, 0.0, -3.125),
        5: unstopped(0.0, 0.0, -3.125),
        20: unstopped(25.0, 50.0, -3.125),
        60: unstopped(12.5, 50.0, -6.25),
    }
    assert set(amd.outcomes) == {1, 5, 20}


def test_price_outcome_calculator_exits_at_the_stop_or_the_open_that_gaps_through_it():
    intraday_rows = [(64, 66, 62, 62), (62, 63, 61, 62), (62, 63, 58, 60), (60, 66, 60, 64), (64, 70, 63, 68)]
    calculator = PriceOutcomeCalculator(
        _FakePriceCache(
            {
                symbol: _history_frame(date(2026, 1, 2), rows)
                for symbol, rows in (
                    ("INTRADAY", intraday_rows),
                    ("GAP", GAP_ROWS),
                    ("ENTRYDAY", ENTRY_DAY_STOP_ROWS),
                )
            }
        )
    )

    evaluated, _ = calculator.evaluate_many(
        [_scan_event(symbol) for symbol in ("INTRADAY", "GAP", "ENTRYDAY")],
        sessions=_sessions_of(_history_frame(date(2026, 1, 2), GAP_ROWS)),
    )

    assert {
        event.raw.symbol: [
            (horizon, outcome.return_pct, outcome.stopped_return_pct, outcome.stopped_out)
            for horizon, outcome in event.outcomes.items()
        ]
        for event in evaluated
    } == {
        "INTRADAY": [(1, -3.125, -3.125, False), (5, 6.25, -7.0, True)],
        "GAP": [(1, 0.0, 0.0, False), (5, 0.0, -25.0, True)],
        "ENTRYDAY": [(1, -3.125, -7.0, True), (5, 6.25, -7.0, True)],
    }


def _evaluate_rows(rows):
    history = _history_frame(date(2026, 1, 2), rows)
    calculator = PriceOutcomeCalculator(_FakePriceCache({"NVDA": history}))
    return calculator.evaluate_many([_scan_event("NVDA")], sessions=_sessions_of(history))[0][0]


def _stop_figures(event):
    return [
        (horizon, outcome.stopped_return_pct, outcome.stopped_out, outcome.mfe_pct, outcome.mae_pct)
        for horizon, outcome in event.outcomes.items()
    ]


def test_a_zero_open_or_low_mid_window_is_a_bad_tick_not_a_stop():
    flat = (64, 66, 62, 64)

    zero_open = _evaluate_rows([flat, flat, (0, 66, 62, 64), flat, flat])
    zero_low = _evaluate_rows([flat, flat, (64, 66, 0, 64), flat, flat])

    assert _stop_figures(zero_open) == [(1, 0.0, False, 3.125, -3.125), (5, 0.0, False, 3.125, -3.125)]
    assert _stop_figures(zero_low) == [(1, 0.0, False, 3.125, -3.125), (5, 0.0, False, 3.125, -3.125)]


def test_a_zero_high_or_low_on_the_entry_bar_measures_from_the_entry_price():
    zero_high = _evaluate_rows([(64, 0, 62, 64)] + [FLAT_ROW] * 4)
    zero_low = _evaluate_rows([(64, 66, 0, 64)] + [FLAT_ROW] * 4)

    assert _stop_figures(zero_high) == [(1, 0.0, False, 0.0, -3.125), (5, 6.25, False, 3.125, -3.125)]
    assert _stop_figures(zero_low) == [(1, 0.0, False, 3.125, 0.0), (5, 6.25, False, 3.125, -3.125)]


def test_a_zero_or_missing_close_leaves_its_horizon_unscored():
    zero_close = _evaluate_rows([(64, 66, 62, 0)] + [FLAT_ROW] * 4)
    missing_close = _evaluate_rows([FLAT_ROW] * 4 + [(64, 66, 62, float("nan"))])

    assert [(h, o.return_pct) for h, o in zero_close.outcomes.items()] == [(5, 6.25)]
    assert [(h, o.return_pct) for h, o in missing_close.outcomes.items()] == [(1, 6.25)]


def test_a_zero_or_missing_benchmark_price_gives_no_excess_return():
    def excess(benchmark_rows):
        calculator = PriceOutcomeCalculator(
            _FakePriceCache(
                {
                    "NVDA": _history_frame(date(2026, 4, 6), NVDA_ROWS),
                    "SPY": _history_frame(date(2026, 4, 6), benchmark_rows),
                }
            )
        )
        evaluated, _ = calculator.evaluate_many(
            [_scan_event("NVDA", event_at=date(2026, 4, 4))],
            sessions=_sessions_of(_history_frame(date(2026, 4, 6), NVDA_ROWS)),
            benchmark_candidates=["SPY"],
        )
        return [(h, o.excess_return_pct) for h, o in evaluated[0].outcomes.items()]

    zero_close = [(500, 506, 499, 0)] + BENCHMARK_ROWS[1:]
    missing_open = [(float("nan"), 506, 499, 505)] + BENCHMARK_ROWS[1:]

    assert excess(zero_close) == [(1, None), (5, pytest.approx(10.0))]
    assert excess(missing_open) == [(1, None), (5, None)]


def test_a_missing_entry_open_scores_nothing():
    event = _evaluate_rows([(float("nan"), 66, 62, 64)] + [FLAT_ROW] * 4)

    assert (event.entry_at, event.entry_price, event.outcomes) == (date(2026, 1, 2), None, {})


def test_a_missing_low_does_not_stop_and_a_later_low_below_the_stop_does():
    rows = [FLAT_ROW] * 20
    rows[3] = (64, 66, float("nan"), 64)
    rows[5] = (62, 63, 58, 60)

    event = _evaluate_rows(rows)

    assert [
        (horizon, outcome.stopped_return_pct, outcome.stopped_out)
        for horizon, outcome in event.outcomes.items()
    ] == [(1, 6.25, False), (5, 6.25, False), (20, -7.0, True)]


def test_a_missing_high_or_low_on_the_last_bar_keeps_the_earlier_extremes():
    rows = [(64, 66, 62, 64), (64, 72, 60, 64), (64, 66, 62, 64), (64, 66, 62, 64)]

    event = _evaluate_rows(rows + [(64, float("nan"), float("nan"), 64)])

    assert _stop_figures(event) == [(1, 0.0, False, 3.125, -3.125), (5, 0.0, False, 12.5, -6.25)]


def test_the_first_stop_after_the_fifth_session_sets_the_longer_horizons():
    def stops_with(**hits):
        rows = [(64, 64, 62, 64)] * 60
        for session_number, row in hits.items():
            rows[int(session_number[1:]) - 1] = row
        return [
            (horizon, outcome.stopped_return_pct, outcome.stopped_out)
            for horizon, outcome in _evaluate_rows(rows).outcomes.items()
        ]

    # Session 6's intraday hit comes before session 30's worse gap, so it sets the exit.
    assert stops_with(s6=(62, 63, 58, 60), s30=(40, 42, 38, 40)) == [
        (1, 0.0, False), (5, 0.0, False), (20, -7.0, True), (60, -7.0, True),
    ]
    assert stops_with(s20=(62, 63, 59.52, 60)) == [
        (1, 0.0, False), (5, 0.0, False), (20, -7.0, True), (60, -7.0, True),
    ]
    assert stops_with(s21=(48, 50, 46, 49)) == [
        (1, 0.0, False), (5, 0.0, False), (20, 0.0, False), (60, -25.0, True),
    ]
    assert stops_with(s60=(56, 60, 55, 58)) == [
        (1, 0.0, False), (5, 0.0, False), (20, 0.0, False), (60, -12.5, True),
    ]


# US sessions from the Friday after Thanksgiving 2025 to 2026-01-30.
WINTER_SESSIONS = [
    day.date()
    for day in pd.bdate_range("2025-11-28", "2026-01-30")
    if day.date() not in (date(2025, 12, 25), date(2026, 1, 1), date(2026, 1, 19))
]
STOP_DAY = date(2025, 12, 11)


def _winter_bars(*missing):
    """Flat bars at 100 from the 2025-12-01 entry, with a low of 90 on 2025-12-11."""
    days = [day for day in WINTER_SESSIONS if day >= date(2025, 12, 1) and day not in missing]
    rows = [(100, 101, 90, 95) if day == STOP_DAY else (100, 101, 99, 100) for day in days]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(days), columns=["Open", "High", "Low", "Close"])


def test_the_calendar_dates_each_window_and_a_missing_session_leaves_it_unscored():
    def evaluate(bars):
        calculator = PriceOutcomeCalculator(_FakePriceCache({"NVDA": bars}))
        event = _scan_event("NVDA", event_at=date(2025, 11, 28))
        return calculator.evaluate_many([event], sessions=WINTER_SESSIONS)[0][0]

    complete = evaluate(_winter_bars())
    without_stop_day = evaluate(_winter_bars(STOP_DAY))
    without_entry_day = evaluate(_winter_bars(date(2025, 12, 1)))

    assert (complete.exits, complete.outcomes[20].stopped_return_pct) == (
        {1: date(2025, 12, 1), 5: date(2025, 12, 5), 20: date(2025, 12, 29)},
        -7.0,
    )
    assert (without_stop_day.exits, sorted(without_stop_day.outcomes)) == (complete.exits, [1, 5])
    assert (without_entry_day.entry_at, without_entry_day.entry_price, without_entry_day.outcomes) == (
        date(2025, 12, 1),
        None,
        {},
    )


def _winter_overview(session, picks, histories):
    for symbol in histories:
        _add_universe(session, symbol)
    for symbol, as_of in picks:
        _add_pick(session, _add_run(session, as_of), symbol, 90.0)
    session.commit()
    service = _service(
        {"SPY": _winter_bars(), **histories},
        last_completed=date(2026, 1, 30),
        sessions=WINTER_SESSIONS,
    )
    return service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=90,
        as_of_date=date(2026, 1, 30),
    )


def test_a_session_missing_from_the_cache_skips_the_horizons_it_falls_in(session):
    payload = _winter_overview(
        session,
        [("NVDA", date(2025, 11, 28)), ("AMD", date(2025, 11, 28))],
        {"NVDA": _winter_bars(STOP_DAY), "AMD": _winter_bars()},
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.avg_stopped_return_pct)
        for h in payload.horizons
    ] == [(1, 2, 0, 0.0), (5, 2, 0, 0.0), (20, 1, 1, -7.0), (60, 0, 0, None)]
    assert payload.degraded_reasons == ["incomplete_price_history"]


def test_a_zero_close_skips_its_horizon_as_incomplete_history(session):
    bars = _winter_bars()
    bars.loc[pd.Timestamp(2025, 12, 1), "Close"] = 0

    payload = _winter_overview(session, [("NVDA", date(2025, 11, 28))], {"NVDA": bars})

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.avg_return_pct)
        for h in payload.horizons
    ] == [(1, 0, 1, None), (5, 1, 0, 0.0), (20, 1, 0, 0.0), (60, 0, 0, None)]
    assert payload.degraded_reasons == ["incomplete_price_history"]


def test_a_window_still_open_is_pending_whether_or_not_its_history_is_cached(session):
    payload = _winter_overview(
        session,
        [("NVDA", date(2025, 11, 28)), ("GONE", date(2025, 11, 28))],
        {"NVDA": _winter_bars(), "GONE": None},
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.pending_windows)
        for h in payload.horizons
    ] == [(1, 1, 1, 0), (5, 1, 1, 0), (20, 1, 1, 0), (60, 0, 0, 2)]


def test_a_session_completed_before_the_refresh_caches_it_is_pending(session):
    # The calendar counts 2026-01-30 completed; the newest cached bar is 2026-01-29.
    payload = _winter_overview(
        session,
        [("NVDA", date(2026, 1, 29))],
        {"NVDA": _winter_bars()[:-1], "SPY": _winter_bars()[:-1]},
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.pending_windows)
        for h in payload.horizons
    ] == [(1, 0, 0, 1), (5, 0, 0, 1), (20, 0, 0, 1), (60, 0, 0, 1)]
    assert payload.degraded_reasons == []


def test_a_gap_before_the_newest_cached_bar_still_degrades(session):
    payload = _winter_overview(
        session,
        [("NVDA", date(2026, 1, 29)), ("AMD", date(2025, 11, 28))],
        {
            "NVDA": _winter_bars()[:-1],
            "AMD": _winter_bars(STOP_DAY)[:-1],
            "SPY": _winter_bars()[:-1],
        },
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.pending_windows)
        for h in payload.horizons
    ] == [(1, 1, 0, 1), (5, 1, 0, 1), (20, 0, 1, 1), (60, 0, 0, 2)]
    assert payload.degraded_reasons == ["incomplete_price_history"]


def test_a_symbol_whose_batch_has_not_refreshed_yet_is_pending_once_spy_has(session):
    # The refresh writes SPY's 2026-01-30 bar before NVDA's batch runs.
    payload = _winter_overview(
        session, [("NVDA", date(2026, 1, 29))], {"NVDA": _winter_bars()[:-1]}
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.pending_windows)
        for h in payload.horizons
    ] == [(1, 0, 0, 1), (5, 0, 0, 1), (20, 0, 0, 1), (60, 0, 0, 1)]
    assert payload.degraded_reasons == []


def test_a_pre_open_alert_on_the_last_completed_session_is_pending_before_its_refresh(session):
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
    # Before the open of 2026-01-30, the last completed session; every cached bar ends 2026-01-29.
    session.add(
        ThemeAlert(
            theme_cluster_id=theme.id,
            alert_type="breakout",
            title="Pre-open alert",
            severity="warning",
            related_tickers=["NVDA"],
            triggered_at=datetime(2026, 1, 30, 8, 0, tzinfo=UTC),
        )
    )
    session.commit()
    service = _service(
        {"NVDA": _winter_bars()[:-1], "SPY": _winter_bars()[:-1]},
        last_completed=date(2026, 1, 30),
        sessions=WINTER_SESSIONS,
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.THEME_ALERT,
        lookback_days=30,
        as_of_date=date(2026, 1, 30),
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.pending_windows)
        for h in payload.horizons
    ] == [(1, 0, 0, 1), (5, 0, 0, 1), (20, 0, 0, 1), (60, 0, 0, 1)]
    assert payload.degraded_reasons == []


def test_histories_stale_for_many_sessions_skip_windows_that_closed(session):
    # Every history ends 2026-01-15, ten sessions before the last completed one.
    payload = _winter_overview(
        session,
        [("NVDA", date(2026, 1, 12))],
        {"NVDA": _winter_bars()[:-10], "SPY": _winter_bars()[:-10]},
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.pending_windows)
        for h in payload.horizons
    ] == [(1, 1, 0, 0), (5, 0, 1, 0), (20, 0, 0, 1), (60, 0, 0, 1)]
    assert payload.degraded_reasons == ["incomplete_price_history"]


def test_a_window_missing_a_session_blocks_repeats_only_until_its_calendar_exit(session):
    payload = _winter_overview(
        session,
        [("NVDA", date(2025, 11, 28)), ("NVDA", date(2025, 12, 29))],
        {"NVDA": _winter_bars(STOP_DAY)},
    )

    assert [
        (event.event_at, event.return_20s_pct) for event in payload.recent_events
    ] == [("2025-12-29", 0.0), ("2025-11-28", None)]
    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.suppressed_overlapping)
        for h in payload.horizons
    ][2] == (20, 1, 1, 0)


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
    ] == [(1, 2.0, 3.0), (5, 2.3333, 1.0), (20, None, None), (60, None, None)]
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
    ] == [(1, 4.0, None, 0), (1, 12.0, None, 0), (0, None, None, 0), (0, None, None, 0)]


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
    ] == [(1, 2, 1), (5, 2, 1), (20, 0, 0), (60, 0, 0)]


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
    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.pending_windows)
        for h in payload.horizons
    ] == [(1, 1, 0, 0), (5, 0, 0, 1), (20, 0, 0, 1), (60, 0, 0, 1)]
    assert payload.recent_events[0].return_5s_pct is None


class _ExpiredCalendar(_FixedCalendar):
    def __init__(self):
        super().__init__(date(2026, 3, 31))

    def _expired(self, market, requested_date):
        return CalendarCoverageExpired(
            market=market,
            requested_date=requested_date,
            verified_through=date(2026, 3, 31),
            source_url="https://example.test/holidays",
            operations_doc="docs/OPERATIONS.md",
        )

    def last_completed_trading_day(self, market):
        raise self._expired(market, FIXED_TODAY)

    def trading_days(self, market, start, end):
        if end > date(2026, 3, 31):
            raise self._expired(market, end)
        return super().trading_days(market, start, end)


def test_overview_caps_prices_at_verified_coverage_when_the_calendar_expires(session):
    run = _add_run(session, date(2026, 3, 27))
    _add_universe(session, "NVDA")
    _add_pick(session, run, "NVDA", 95.0)
    session.commit()
    start = date(2026, 3, 28)
    service = ValidationService(
        outcome_calculator=PriceOutcomeCalculator(
            _FakePriceCache(
                {
                    "NVDA": _history_frame(start, NVDA_ROWS),
                    "SPY": _history_frame(start, BENCHMARK_ROWS),
                }
            )
        ),
        market_calendar=_ExpiredCalendar(),
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.skipped_missing_history, h.pending_windows)
        for h in payload.horizons
    ] == [(1, 1, 0, 0), (5, 0, 0, 1), (20, 0, 0, 1), (60, 0, 0, 1)]
    assert payload.degraded_reasons == ["calendar_coverage_expired"]


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
    ] == [(1, 5, 0), (5, 2, 3), (20, 0, 4), (60, 0, 4)]


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
    ] == [(1, 1, 0), (5, 0, 1), (20, 0, 1), (60, 0, 1)]
    assert payload.degraded_reasons == []


def test_a_pick_before_a_long_holiday_still_blocks_repeats_after_the_lookback_start(session):
    _add_universe(session, "NVDA")
    _add_pick(session, _add_run(session, date(2026, 2, 6)), "NVDA", 90.0)
    _add_pick(session, _add_run(session, date(2026, 2, 23)), "NVDA", 90.0)
    session.commit()
    sessions = [date(2026, 2, 9), date(2026, 2, 10), date(2026, 2, 11)] + [
        date(2026, 2, 23) + timedelta(days=offset) for offset in range(8)
    ]
    bars = pd.DataFrame(
        [(10, 11, 9, 10.5)] * len(sessions),
        index=pd.DatetimeIndex(sessions),
        columns=["Open", "High", "Low", "Close"],
    )
    service = _service({"NVDA": bars, "SPY": bars}, sessions=sessions)

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=date(2026, 3, 24),
    )

    assert [(event.event_at, event.counted) for event in payload.recent_events] == [
        ("2026-02-23", False),
    ]
    assert [
        (h.horizon_sessions, h.sample_size, h.suppressed_overlapping) for h in payload.horizons
    ] == [(1, 1, 0), (5, 0, 1), (20, 0, 1), (60, 0, 1)]


def test_overview_summarizes_returns_with_the_stop(session):
    run = _add_run(session, date(2026, 3, 27))
    for symbol, score in (("NOSTOP", 95.0), ("GAP", 90.0), ("ENTRYDAY", 85.0)):
        _add_universe(session, symbol)
        _add_pick(session, run, symbol, score)
    session.commit()
    start = date(2026, 3, 28)
    service = _service(
        {
            "NOSTOP": _history_frame(start, [(64, 66, 62, 64)] * 4 + [(64, 80, 62, 72)]),
            "GAP": _history_frame(start, GAP_ROWS),
            "ENTRYDAY": _history_frame(start, ENTRY_DAY_STOP_ROWS),
        }
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert payload.stop_loss_pct == 7.0
    assert [
        (
            h.horizon_sessions,
            h.avg_stopped_return_pct,
            h.median_stopped_return_pct,
            h.stopped_out_rate,
        )
        for h in payload.horizons
    ] == [
        (1, -2.3333, 0.0, 0.3333),
        (5, -6.5, -7.0, 0.6667),
        (20, None, None, None),
        (60, None, None, None),
    ]


def test_a_repeat_ten_sessions_later_counts_at_five_sessions_but_not_at_twenty(session):
    _add_universe(session, "NVDA")
    for as_of in (date(2026, 2, 1), date(2026, 2, 11)):
        _add_pick(session, _add_run(session, as_of), "NVDA", 90.0)
    session.commit()
    service = _service({"NVDA": _history_frame(date(2026, 2, 2), [FLAT_ROW] * 63)})

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=90,
        as_of_date=FIXED_TODAY,
    )

    assert [
        (h.horizon_sessions, h.sample_size, h.suppressed_overlapping) for h in payload.horizons
    ] == [(1, 2, 0), (5, 2, 0), (20, 1, 1), (60, 1, 1)]
    assert [
        (event.event_at, event.counted, event.return_20s_pct, event.return_60s_pct)
        for event in payload.recent_events
    ] == [("2026-02-11", True, 6.25, None), ("2026-02-01", True, 6.25, 6.25)]


def test_recent_events_name_the_horizons_each_event_counts_at(session):
    _add_universe(session, "NVDA")
    for as_of in (date(2026, 2, 1), date(2026, 2, 18), date(2026, 2, 21)):
        _add_pick(session, _add_run(session, as_of), "NVDA", 90.0)
    session.commit()
    service = _service({"NVDA": _history_frame(date(2026, 2, 2), [FLAT_ROW] * 63)})

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=90,
        as_of_date=FIXED_TODAY,
    )

    # The 2026-02-21 pick repeats inside the 2026-02-18 pick's 5 sessions, but the
    # 2026-02-18 pick is itself a 20-session repeat, so it blocks nothing at 20.
    assert [
        (event.event_at, event.counted, event.counted_horizons) for event in payload.recent_events
    ] == [
        ("2026-02-21", False, [1, 20]),
        ("2026-02-18", True, [1, 5]),
        ("2026-02-01", True, [1, 5, 20, 60]),
    ]


def test_a_pick_seventy_days_before_the_cutoff_blocks_repeats_inside_its_sixty_sessions(session):
    _add_universe(session, "NVDA")
    _add_pick(session, _add_run(session, date(2025, 12, 26)), "NVDA", 90.0)
    _add_pick(session, _add_run(session, date(2026, 3, 10)), "NVDA", 90.0)
    session.commit()
    sessions = pd.bdate_range(date(2025, 12, 29), date(2026, 4, 3))
    bars = pd.DataFrame(
        [FLAT_ROW] * len(sessions), index=sessions, columns=["Open", "High", "Low", "Close"]
    )
    service = _service({"NVDA": bars}, sessions=list(sessions.date))

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.SCAN_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert [event.event_at for event in payload.recent_events] == ["2026-03-10"]
    assert [
        (
            h.horizon_sessions,
            h.sample_size,
            h.skipped_missing_history,
            h.pending_windows,
            h.suppressed_overlapping,
        )
        for h in payload.horizons
    ] == [(1, 1, 0, 0, 0), (5, 1, 0, 0, 0), (20, 0, 0, 1, 0), (60, 0, 0, 0, 1)]


def _evaluated(source_kind, symbol, source_ref, entry_at, exit_at=None, *, entry_price=10.0):
    """An event whose 5-session window ends at ``exit_at``, or is still open without it."""
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
        exits={5: exit_at} if exit_at else {},
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

    assert [(event.raw.source_ref, event.suppressed) for event in marked] == [
        ("run:2", frozenset({5, 20, 60})),
        ("run:1", frozenset()),
        ("run:3", frozenset()),
        ("run:4", frozenset()),
        ("alert:1", frozenset()),
    ]


def test_an_entry_without_a_price_does_not_block_a_window():
    scan = ValidationSourceKind.SCAN_PICK
    events = [
        _evaluated(scan, "AAA", "run:1", date(2026, 3, 25), date(2026, 3, 31), entry_price=None),
        _evaluated(scan, "AAA", "run:2", date(2026, 3, 26)),
    ]

    marked = mark_overlapping_windows(events)

    assert [(event.raw.source_ref, event.suppressed) for event in marked] == [
        ("run:1", frozenset()),
        ("run:2", frozenset()),
    ]


def _add_replay_pick(session, as_of_date, rank, symbol, *, score, rs, market="US", version=DEFINITION_VERSION):
    session.add(
        BacktestReplayPick(
            market=market,
            as_of_date=as_of_date,
            rank=rank,
            symbol=symbol,
            minervini_score=score,
            rs_rating=rs,
            stage=2,
            definition_version=version,
            market_rs_run_id=41,
        )
    )


def test_replay_picks_of_the_current_definition_are_scored_as_their_own_source(session):
    replayed = date(2026, 3, 27)
    _add_replay_pick(session, replayed, 1, "NVDA", score=91.5, rs=95)
    _add_replay_pick(session, replayed, 2, "AMD", score=88.25, rs=82)
    _add_replay_pick(session, replayed, 1, "MU", score=99.0, rs=99, market="HK")
    _add_replay_pick(session, replayed, 1, "MU", score=99.0, rs=99, version="minervini-top10-v0")
    session.commit()
    start = date(2026, 3, 28)
    service = _service(
        {
            "NVDA": _history_frame(start, NVDA_ROWS),
            "AMD": _history_frame(start, [(50, 51, 49, 50)] * 4 + [(50, 51, 48, 49)]),
            "MU": _history_frame(start, [(200, 211, 199, 210)] * 5),
            "SPY": _history_frame(start, BENCHMARK_ROWS),
        }
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.REPLAY_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert (payload.source_kind, payload.benchmark_symbol, payload.degraded_reasons) == (
        ValidationSourceKind.REPLAY_PICK,
        "SPY",
        [],
    )
    assert [
        (event.source_ref, event.attributes, event.return_5s_pct, event.excess_return_5s_pct)
        for event in payload.recent_events
    ] == [
        (
            "replay:US:2026-03-27",
            {"symbol": "NVDA", "rank": 1, "minervini_score": 91.5, "rs_rating": 95, "rs_band": "90-99", "stage": 2},
            12.0,
            10.0,
        ),
        (
            "replay:US:2026-03-27",
            {"symbol": "AMD", "rank": 2, "minervini_score": 88.25, "rs_rating": 82, "rs_band": "80-89", "stage": 2},
            -2.0,
            -4.0,
        ),
    ]
    assert [h.sample_size for h in payload.horizons] == [2, 2, 0, 0]
    assert [(c.cluster_key, c.label, c.sample_size) for c in payload.failure_clusters] == [
        ("rs_band:80-89", "RS Band: 80-89", 1),
        ("stage:2", "Stage: 2", 1),
    ]


def test_replay_overview_without_stored_picks_reports_none_match(session):
    _add_replay_pick(session, date(2026, 3, 27), 1, "NVDA", score=91.5, rs=95, market="HK")
    session.commit()

    payload = _service().get_overview(
        session,
        source_kind=ValidationSourceKind.REPLAY_PICK,
        lookback_days=30,
        as_of_date=FIXED_TODAY,
    )

    assert (payload.recent_events, payload.degraded_reasons) == ([], ["no_matching_replay_picks"])


def test_replay_overlap_counting_runs_over_every_stored_pick_before_the_lookback(session):
    sessions = pd.bdate_range(date(2026, 4, 6), date(2026, 10, 30))
    for as_of in sessions[:-1].date:
        _add_replay_pick(session, as_of, 1, "NVDA", score=91.5, rs=95)
    session.commit()
    bars = pd.DataFrame(
        [FLAT_ROW] * len(sessions), index=sessions, columns=["Open", "High", "Low", "Close"]
    )
    service = _service(
        {"NVDA": bars, "SPY": bars}, last_completed=date(2026, 10, 30), sessions=list(sessions.date)
    )

    payload = service.get_overview(
        session,
        source_kind=ValidationSourceKind.REPLAY_PICK,
        lookback_days=30,
        as_of_date=date(2026, 10, 30),
    )

    assert [event.event_at for event in payload.recent_events if event.counted] == [
        "2026-10-26",
        "2026-10-19",
        "2026-10-12",
        "2026-10-05",
    ]
    assert [
        (h.horizon_sessions, h.sample_size, h.pending_windows, h.suppressed_overlapping)
        for h in payload.horizons
    ] == [(1, 22, 0, 0), (5, 3, 1, 18), (20, 0, 1, 21), (60, 0, 0, 22)]


def test_freshness_reports_the_latest_replayed_session_of_the_market_and_definition(session):
    _add_replay_pick(session, date(2026, 3, 27), 1, "NVDA", score=91.5, rs=95)
    _add_replay_pick(session, date(2026, 3, 26), 1, "NVDA", score=91.5, rs=95)
    _add_replay_pick(session, date(2026, 4, 10), 1, "NVDA", score=91.5, rs=95)
    _add_replay_pick(session, date(2026, 4, 1), 1, "MU", score=99.0, rs=99, market="HK")
    _add_replay_pick(session, date(2026, 4, 2), 1, "MU", score=99.0, rs=99, version="minervini-top10-v0")
    session.commit()

    freshness = [
        _service().get_overview(
            session,
            source_kind=ValidationSourceKind.SCAN_PICK,
            lookback_days=30,
            as_of_date=FIXED_TODAY,
            market=market,
        ).freshness.latest_replay_as_of_date
        for market in ("US", "HK", "JP")
    ]

    assert freshness == ["2026-03-27", "2026-04-01", None]
