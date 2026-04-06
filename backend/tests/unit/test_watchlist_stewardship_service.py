from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1 import user_watchlists as watchlists_module
from app.database import Base, get_db
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.main import app
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.services import server_auth
from app.services.watchlist_stewardship_service import WatchlistStewardshipService

pytestmark = pytest.mark.integration


class _FakeEventContextService:
    def __init__(self, earnings_map: dict[str, tuple[date | None, int | None]] | None = None):
        self._earnings_map = earnings_map or {}

    def get_next_earnings_summary(self, symbol: str, *, as_of_date=None):  # noqa: ANN001
        return self._earnings_map.get(symbol.upper(), (None, None))


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _disable_server_auth():
    original_enabled = server_auth.settings.server_auth_enabled
    server_auth.settings.server_auth_enabled = False
    yield
    server_auth.settings.server_auth_enabled = original_enabled
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
            UserWatchlist.__table__,
            WatchlistItem.__table__,
            MarketBreadth.__table__,
            StockUniverse.__table__,
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
                StockUniverse.__table__,
                MarketBreadth.__table__,
                WatchlistItem.__table__,
                UserWatchlist.__table__,
                StockFeatureDaily.__table__,
                FeatureRun.__table__,
            ],
        )


def _override_db(session):
    def _yield_db():
        try:
            yield session
        finally:
            pass

    return _yield_db


def _feature_row(*, run_id: int, as_of_date: date, symbol: str, score: float, rating: int, rs: float, stage: int) -> StockFeatureDaily:
    return StockFeatureDaily(
        run_id=run_id,
        symbol=symbol,
        as_of_date=as_of_date,
        composite_score=score,
        overall_rating=rating,
        passes_count=1,
        details_json={
            "company_name": f"{symbol} Corp",
            "rs_rating": rs,
            "stage": stage,
        },
    )


def _seed_watchlist_stewardship_data(session):
    latest_run = FeatureRun(
        id=10,
        as_of_date=date(2026, 4, 4),
        run_type="daily_snapshot",
        status="published",
        completed_at=datetime(2026, 4, 4, 20, 0, tzinfo=UTC),
        published_at=datetime(2026, 4, 4, 20, 5, tzinfo=UTC),
    )
    previous_run = FeatureRun(
        id=9,
        as_of_date=date(2026, 4, 3),
        run_type="daily_snapshot",
        status="published",
        completed_at=datetime(2026, 4, 3, 20, 0, tzinfo=UTC),
        published_at=datetime(2026, 4, 3, 20, 5, tzinfo=UTC),
    )
    session.add_all([latest_run, previous_run])
    session.add(
        MarketBreadth(
            date=date(2026, 4, 4),
            stocks_up_4pct=120,
            stocks_down_4pct=90,
            ratio_5day=0.8,
            ratio_10day=0.7,
            total_stocks_scanned=4200,
        )
    )
    session.add_all([
        StockUniverse(symbol="AAPL", name="Apple", is_active=True, status="active"),
        StockUniverse(symbol="MSFT", name="Microsoft", is_active=True, status="active"),
        StockUniverse(symbol="NVDA", name="NVIDIA", is_active=True, status="active"),
        StockUniverse(symbol="TSLA", name="Tesla", is_active=True, status="active"),
        StockUniverse(symbol="AMZN", name="Amazon", is_active=True, status="active"),
    ])
    watchlist = UserWatchlist(name="Core", position=0)
    session.add(watchlist)
    session.commit()
    session.refresh(watchlist)
    session.add_all([
        WatchlistItem(watchlist_id=watchlist.id, symbol="AAPL", position=0),
        WatchlistItem(watchlist_id=watchlist.id, symbol="MSFT", position=1),
        WatchlistItem(watchlist_id=watchlist.id, symbol="NVDA", position=2),
        WatchlistItem(watchlist_id=watchlist.id, symbol="TSLA", position=3),
        WatchlistItem(watchlist_id=watchlist.id, symbol="AMZN", position=4),
    ])
    session.add_all([
        _feature_row(run_id=9, as_of_date=date(2026, 4, 3), symbol="AAPL", score=70, rating=4, rs=80, stage=2),
        _feature_row(run_id=10, as_of_date=date(2026, 4, 4), symbol="AAPL", score=75, rating=4, rs=80, stage=2),
        _feature_row(run_id=9, as_of_date=date(2026, 4, 3), symbol="MSFT", score=75, rating=4, rs=82, stage=2),
        _feature_row(run_id=10, as_of_date=date(2026, 4, 4), symbol="MSFT", score=76, rating=4, rs=83, stage=2),
        _feature_row(run_id=9, as_of_date=date(2026, 4, 3), symbol="NVDA", score=88, rating=5, rs=95, stage=2),
        _feature_row(run_id=10, as_of_date=date(2026, 4, 4), symbol="NVDA", score=82, rating=4, rs=84, stage=2),
        _feature_row(run_id=9, as_of_date=date(2026, 4, 3), symbol="TSLA", score=62, rating=3, rs=76, stage=2),
        _feature_row(run_id=10, as_of_date=date(2026, 4, 4), symbol="TSLA", score=50, rating=2, rs=70, stage=1),
    ])
    session.add(
        ThemeAlert(
            alert_type="breakout",
            title="Older NVDA support",
            severity="info",
            related_tickers=["NVDA"],
            is_dismissed=False,
            triggered_at=datetime(2026, 3, 26, 15, 0, tzinfo=UTC),
        )
    )
    session.commit()
    return watchlist


def test_watchlist_stewardship_service_classifies_statuses_and_sorts_default_profile(session):
    watchlist = _seed_watchlist_stewardship_data(session)
    service = WatchlistStewardshipService(
        event_context_service=_FakeEventContextService(
            {
                "TSLA": (date(2026, 4, 6), 2),
                "AAPL": (date(2026, 5, 1), 27),
            }
        )
    )

    payload = service.get_watchlist_stewardship(
        session,
        watchlist_id=watchlist.id,
        as_of_date=date(2026, 4, 4),
        profile="default",
    )

    statuses = {item.symbol: item.status for item in payload.items}
    assert statuses == {
        "TSLA": "exit_risk",
        "NVDA": "deteriorating",
        "AAPL": "strengthening",
        "MSFT": "unchanged",
        "AMZN": "missing_from_run",
    }
    assert payload.summary_counts.exit_risk == 1
    assert payload.summary_counts.deteriorating == 1
    assert payload.summary_counts.strengthening == 1
    assert payload.summary_counts.unchanged == 1
    assert payload.summary_counts.missing_from_run == 1
    assert payload.items[0].symbol == "TSLA"
    assert payload.items[1].symbol == "NVDA"


def test_watchlist_stewardship_service_applies_risk_off_thresholds(session):
    watchlist = _seed_watchlist_stewardship_data(session)
    service = WatchlistStewardshipService(event_context_service=_FakeEventContextService())

    payload = service.get_watchlist_stewardship(
        session,
        watchlist_id=watchlist.id,
        as_of_date=date(2026, 4, 4),
        profile="risk_off",
    )

    statuses = {item.symbol: item.status for item in payload.items}
    assert statuses["AAPL"] == "unchanged"


@pytest.mark.asyncio
async def test_watchlist_stewardship_endpoint_returns_profile_aware_payload(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    watchlist = _seed_watchlist_stewardship_data(session)
    service = WatchlistStewardshipService(event_context_service=_FakeEventContextService())
    app.dependency_overrides[watchlists_module._get_watchlist_stewardship_service] = lambda: service

    response = await client.get(
        f"/api/v1/user-watchlists/{watchlist.id}/stewardship",
        params={"profile": "risk_off", "as_of_date": "2026-04-04"},
    )

    assert response.status_code == 200
    payload = response.json()
    statuses = {item["symbol"]: item["status"] for item in payload["items"]}
    assert statuses["AAPL"] == "unchanged"
    assert payload["summary_counts"]["all"] == 5
