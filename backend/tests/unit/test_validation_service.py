from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.theme import ThemeAlert, ThemeCluster
from app.schemas.validation import ValidationSourceKind
from app.services.validation_service import (
    EvaluatedValidationEvent,
    FailureClusterBuilder,
    PriceOutcomeCalculator,
    RawValidationEvent,
    ScanPickValidationSource,
    ThemeAlertValidationSource,
)


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
                StockFeatureDaily.__table__,
                FeatureRun.__table__,
            ],
        )


class _FakePriceCache:
    def __init__(self, histories):
        self._histories = histories

    def get_many_cached_only(self, symbols, period="2y"):
        return {symbol: self._histories.get(symbol) for symbol in symbols}


def _history_frame(start: date, rows: list[tuple[float, float, float, float]]):
    index = pd.DatetimeIndex([start + timedelta(days=offset) for offset in range(len(rows))])
    return pd.DataFrame(
        rows,
        index=index,
        columns=["Open", "High", "Low", "Close"],
    )


def test_scan_pick_validation_source_limits_to_top_ten_and_breaks_ties_by_symbol(session):
    run = FeatureRun(
        as_of_date=date.today(),
        run_type="daily_snapshot",
        status="published",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    symbols = ["ZZZ", "BBB", "AAA", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK"]
    for symbol in symbols:
        session.add(
            StockFeatureDaily(
                run_id=run.id,
                symbol=symbol,
                as_of_date=run.as_of_date,
                composite_score=90.0,
                overall_rating=5,
                passes_count=3,
                details_json={"stage": 2, "ibd_industry_group": "Semis"},
            )
        )
    session.commit()

    source = ScanPickValidationSource()
    events, degraded = source.collect(session, cutoff_date=date.today() - timedelta(days=1))

    assert degraded == []
    assert [event.symbol for event in events] == sorted(symbols)[:10]


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
                triggered_at=datetime.now(UTC),
            ),
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="velocity_spike",
                title="Velocity",
                severity="critical",
                related_tickers=["MSFT"],
                triggered_at=datetime.now(UTC),
            ),
            ThemeAlert(
                theme_cluster_id=theme.id,
                alert_type="new_theme",
                title="Ignore",
                severity="info",
                related_tickers=["AAPL"],
                triggered_at=datetime.now(UTC),
            ),
        ]
    )
    session.commit()

    source = ThemeAlertValidationSource()
    events, degraded = source.collect(session, cutoff_date=date.today() - timedelta(days=7))

    assert degraded == []
    assert [(event.attributes["alert_type"], event.symbol) for event in events] == [
        ("velocity_spike", "MSFT"),
        ("breakout", "NVDA"),
        ("breakout", "AVGO"),
    ]


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

    evaluated, degraded = calculator.evaluate_many([saturday_event])

    assert degraded == []
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


def test_failure_cluster_builder_uses_source_specific_bucket_fields():
    builder = FailureClusterBuilder()
    losing_scan_event = EvaluatedValidationEvent(
        raw=RawValidationEvent(
            symbol="NVDA",
            source_kind=ValidationSourceKind.SCAN_PICK,
            source_ref="run:1",
            event_at=date.today(),
            attributes={"rating": "Buy", "stage": 2, "ibd_industry_group": "Semis"},
        ),
        entry_at=date.today(),
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
            event_at=datetime.now(UTC),
            attributes={"alert_type": "breakout", "severity": "warning", "theme": "AI"},
        ),
        entry_at=date.today(),
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
