from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import httpx
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.stocks import _get_validation_service as _get_stock_validation_service
from app.api.v1.validation import _get_validation_service
from app.database import Base, get_db
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.main import app
from app.models.theme import ThemeAlert, ThemeCluster
from app.services import server_auth
from app.services.validation_service import ValidationService

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _disable_server_auth():
    server_auth.settings.server_auth_enabled = False
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


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


def _override_db(session):
    def _yield_db():
        try:
            yield session
        finally:
            pass

    return _yield_db


def _service_with_histories(histories):
    class _TestValidationService(ValidationService):
        def __init__(self):
            super().__init__()
            self._outcome_calculator._price_cache = _FakePriceCache(histories)

    return _TestValidationService()


def _history_frame(start: date, closes: list[tuple[float, float, float, float]]):
    index = pd.DatetimeIndex([start + timedelta(days=offset) for offset in range(len(closes))])
    return pd.DataFrame(closes, index=index, columns=["Open", "High", "Low", "Close"])


def _seed_validation_data(session):
    today = datetime.now(UTC).date()
    run = FeatureRun(as_of_date=today - timedelta(days=3), run_type="daily_snapshot", status="published")
    session.add(run)
    theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infra",
        display_name="AI Infrastructure",
        pipeline="technical",
        lifecycle_state="active",
    )
    session.add(theme)
    session.commit()
    session.refresh(run)
    session.refresh(theme)

    session.add_all(
        [
            StockFeatureDaily(
                run_id=run.id,
                symbol="NVDA",
                as_of_date=run.as_of_date,
                composite_score=95.0,
                overall_rating=5,
                passes_count=3,
                details_json={"stage": 2, "ibd_industry_group": "Semiconductors"},
            ),
            StockFeatureDaily(
                run_id=run.id,
                symbol="MSFT",
                as_of_date=run.as_of_date,
                composite_score=90.0,
                overall_rating=4,
                passes_count=3,
                details_json={"stage": 2, "ibd_industry_group": "Software"},
            ),
        ]
    )
    session.add(
        ThemeAlert(
            theme_cluster_id=theme.id,
            alert_type="breakout",
            title="AI breakout",
            severity="warning",
            related_tickers=["NVDA", "AVGO"],
            triggered_at=datetime.now(UTC) - timedelta(days=2),
        )
    )
    session.commit()
    return today


@pytest.mark.asyncio
async def test_validation_overview_endpoint_returns_scan_pick_metrics(client, session):
    today = _seed_validation_data(session)
    histories = {
        "NVDA": _history_frame(
            today - timedelta(days=2),
            [
                (100, 104, 99, 103),
                (103, 105, 101, 104),
                (104, 108, 103, 107),
                (107, 109, 104, 108),
                (108, 111, 107, 110),
            ],
        ),
        "MSFT": None,
    }

    app.dependency_overrides[get_db] = _override_db(session)
    app.dependency_overrides[_get_validation_service] = lambda: _service_with_histories(histories)
    app.dependency_overrides[_get_stock_validation_service] = lambda: _service_with_histories(histories)

    response = await client.get("/api/v1/validation/overview", params={"source_kind": "scan_pick", "lookback_days": 90})

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_kind"] == "scan_pick"
    assert payload["horizons"][0]["sample_size"] == 1
    assert payload["horizons"][1]["skipped_missing_history"] == 1
    assert payload["recent_events"][0]["attributes"]["symbol"] == "NVDA"
    assert "missing_price_cache" in payload["degraded_reasons"]


@pytest.mark.asyncio
async def test_validation_overview_endpoint_returns_theme_alert_metrics(client, session):
    today = _seed_validation_data(session)
    histories = {
        "NVDA": _history_frame(
            today - timedelta(days=1),
            [
                (50, 52, 49, 51),
                (51, 53, 50, 52),
                (52, 54, 51, 53),
                (53, 55, 52, 54),
                (54, 56, 53, 55),
            ],
        ),
        "AVGO": _history_frame(
            today - timedelta(days=1),
            [
                (80, 81, 78, 79),
                (79, 82, 77, 78),
                (78, 80, 76, 77),
                (77, 79, 75, 76),
                (76, 78, 74, 75),
            ],
        ),
    }

    app.dependency_overrides[get_db] = _override_db(session)
    app.dependency_overrides[_get_validation_service] = lambda: _service_with_histories(histories)
    app.dependency_overrides[_get_stock_validation_service] = lambda: _service_with_histories(histories)

    response = await client.get("/api/v1/validation/overview", params={"source_kind": "theme_alert", "lookback_days": 90})

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_kind"] == "theme_alert"
    assert len(payload["recent_events"]) == 2
    assert payload["failure_clusters"]
    assert payload["recent_events"][0]["attributes"]["alert_type"] == "breakout"


@pytest.mark.asyncio
async def test_stock_validation_endpoint_returns_both_source_breakdowns(client, session):
    today = _seed_validation_data(session)
    histories = {
        "NVDA": _history_frame(
            today - timedelta(days=2),
            [
                (100, 104, 99, 103),
                (103, 105, 101, 104),
                (104, 108, 103, 107),
                (107, 109, 104, 108),
                (108, 111, 107, 110),
            ],
        ),
    }

    app.dependency_overrides[get_db] = _override_db(session)
    app.dependency_overrides[_get_validation_service] = lambda: _service_with_histories(histories)
    app.dependency_overrides[_get_stock_validation_service] = lambda: _service_with_histories(histories)

    response = await client.get("/api/v1/stocks/NVDA/validation", params={"lookback_days": 365})

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "NVDA"
    assert [item["source_kind"] for item in payload["source_breakdown"]] == [
        "scan_pick",
        "theme_alert",
    ]
    assert len(payload["recent_events"]) == 2
    assert payload["freshness"]["latest_feature_as_of_date"] is not None
