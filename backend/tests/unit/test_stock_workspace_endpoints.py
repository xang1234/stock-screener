from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1 import stocks as stocks_module
from app.api.v1 import user_watchlists as watchlists_module
from app.database import Base, get_db
from app.domain.feature_store.models import FeatureRow, FeatureRunDomain, RunStatus, RunType
from app.domain.scanning.models import ScanResultItemDomain
from app.main import app
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeCluster, ThemeConstituent, ThemeMetrics
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.services import server_auth
from app.wiring.bootstrap import get_uow


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _disable_server_auth(monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
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
            StockUniverse.__table__,
            UserWatchlist.__table__,
            WatchlistItem.__table__,
            MarketBreadth.__table__,
            ThemeCluster.__table__,
            ThemeConstituent.__table__,
            ThemeMetrics.__table__,
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
                ThemeMetrics.__table__,
                ThemeConstituent.__table__,
                ThemeCluster.__table__,
                MarketBreadth.__table__,
                WatchlistItem.__table__,
                UserWatchlist.__table__,
                StockUniverse.__table__,
            ],
        )


class _FakeFeatureStore:
    def __init__(self, item, row, peers):
        self._item = item
        self._row = row
        self._peers = peers

    def get_by_symbol_for_run(self, run_id, symbol, include_sparklines=False, include_setup_payload=False):
        return self._item if symbol == self._item.symbol else None

    def get_row_by_symbol(self, run_id, symbol):
        return self._row if symbol == self._row.symbol else None

    def get_peers_by_industry_for_run(self, run_id, industry_group):
        return tuple(self._peers)


class _FakeFeatureRuns:
    def __init__(self, latest_run):
        self._latest_run = latest_run

    def get_latest_published(self):
        return self._latest_run


class _FakeUow:
    def __init__(self, latest_run, item=None, row=None, peers=()):
        self.feature_runs = _FakeFeatureRuns(latest_run)
        self.feature_store = _FakeFeatureStore(item, row, peers) if latest_run and item and row else None
        self._item = item
        self._row = row
        self._peers = peers

    def __enter__(self):
        if self.feature_store is None and self.feature_runs.get_latest_published() is not None:
            self.feature_store = _FakeFeatureStore(self._item, self._row, self._peers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _override_db(session):
    def _yield_db():
        try:
            yield session
        finally:
            pass

    return _yield_db


def _seed_search_and_import_data(session):
    session.add_all(
        [
            StockUniverse(symbol="AA", name="Alcoa", sector="Materials", industry="Metals", is_active=True, status="active"),
            StockUniverse(symbol="AAPL", name="Apple Inc.", sector="Technology", industry="Consumer Electronics", is_active=True, status="active"),
            StockUniverse(symbol="MAA", name="Mid-America Apartment", sector="Real Estate", industry="REIT", is_active=True, status="active"),
            StockUniverse(symbol="MSFT", name="Microsoft", sector="Technology", industry="Software", is_active=True, status="active"),
            StockUniverse(symbol="NVDA", name="NVIDIA", sector="Technology", industry="Semiconductors", is_active=True, status="active"),
        ]
    )
    watchlist = UserWatchlist(name="Leaders", position=0)
    session.add(watchlist)
    session.commit()
    session.refresh(watchlist)
    session.add(WatchlistItem(watchlist_id=watchlist.id, symbol="AAPL", position=0))
    session.commit()
    return watchlist


def _seed_dashboard_data(session):
    session.add(
        MarketBreadth(
            date=date(2026, 4, 2),
            stocks_up_4pct=120,
            stocks_down_4pct=35,
            ratio_5day=1.6,
            ratio_10day=1.3,
            total_stocks_scanned=4800,
        )
    )
    theme = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai-infra",
        display_name="AI Infrastructure",
        pipeline="technical",
        category="technology",
        is_emerging=False,
        is_active=True,
        is_validated=True,
        lifecycle_state="active",
    )
    session.add(theme)
    session.commit()
    session.refresh(theme)
    session.add(
        ThemeConstituent(
            theme_cluster_id=theme.id,
            symbol="NVDA",
            confidence=0.91,
            mention_count=12,
            correlation_to_theme=0.77,
            is_active=True,
        )
    )
    session.add(
        ThemeMetrics(
            theme_cluster_id=theme.id,
            date=date(2026, 4, 2),
            pipeline="technical",
            mention_velocity=1.8,
            basket_return_1m=12.4,
            momentum_score=81.2,
            status="trending",
        )
    )
    session.commit()


def _make_feature_context():
    latest_run = FeatureRunDomain(
        id=77,
        as_of_date=date(2026, 4, 2),
        run_type=RunType.DAILY_SNAPSHOT,
        status=RunStatus.PUBLISHED,
        created_at=datetime(2026, 4, 2, 20, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 2, 20, 5, tzinfo=UTC),
        published_at=datetime(2026, 4, 2, 20, 6, tzinfo=UTC),
    )
    feature_item = ScanResultItemDomain(
        symbol="NVDA",
        composite_score=88.5,
        rating="Strong Buy",
        current_price=921.45,
        screener_outputs={},
        screeners_run=["minervini", "canslim"],
        composite_method="weighted_average",
        screeners_passed=2,
        screeners_total=2,
        extended_fields={
            "company_name": "NVIDIA Corp",
            "gics_sector": "Technology",
            "gics_industry": "Semiconductors",
            "ibd_industry_group": "Semiconductors",
            "ibd_group_rank": 3,
            "rs_rating": 97,
            "rs_rating_1m": 95,
            "rs_rating_3m": 98,
            "rs_rating_12m": 99,
            "stage": 2,
            "adr_percent": 4.2,
            "eps_rating": 96,
            "minervini_score": 86,
            "vcp_detected": True,
            "vcp_score": 78,
            "vcp_pivot": 940,
            "vcp_ready_for_breakout": True,
            "ma_alignment": True,
            "passes_template": True,
            "eps_growth_qq": 54.2,
            "sales_growth_qq": 42.1,
        },
    )
    feature_row = FeatureRow(
        run_id=77,
        symbol="NVDA",
        as_of_date=date(2026, 4, 2),
        composite_score=88.5,
        overall_rating=5,
        passes_count=2,
        details={
            "current_price": 921.45,
            "screeners_run": ["minervini", "canslim"],
            "composite_method": "weighted_average",
            "screeners_passed": 2,
            "screeners_total": 2,
            "details": {
                "screeners": {
                    "minervini": {
                        "score": 86.0,
                        "passes": True,
                        "rating": "Strong Buy",
                        "breakdown": {
                            "rs_rating": {"points": 18.0, "max_points": 20, "passes": True},
                            "stage": {"points": 20.0, "max_points": 20, "passes": True},
                            "vcp": {"points": 0.0, "max_points": 20, "passes": False},
                        },
                    },
                    "canslim": {
                        "score": 82.0,
                        "passes": True,
                        "rating": "Strong Buy",
                        "breakdown": {
                            "current_earnings": 18.0,
                            "annual_earnings": 15.0,
                        },
                    },
                }
            },
        },
    )
    peer = ScanResultItemDomain(
        symbol="AVGO",
        composite_score=82.1,
        rating="Buy",
        current_price=1440.2,
        screener_outputs={},
        screeners_run=["minervini"],
        composite_method="weighted_average",
        screeners_passed=1,
        screeners_total=1,
        extended_fields={
            "company_name": "Broadcom",
            "rs_rating": 93,
            "stage": 2,
        },
    )
    return latest_run, feature_item, feature_row, [peer]


@pytest.mark.asyncio
async def test_search_stocks_endpoint_ranks_exact_and_prefix_matches(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    _seed_search_and_import_data(session)

    response = await client.get("/api/v1/stocks/search", params={"q": "aa", "limit": 3})

    assert response.status_code == 200
    assert [row["symbol"] for row in response.json()] == ["AA", "AAPL", "MAA"]


@pytest.mark.asyncio
async def test_decision_dashboard_endpoint_returns_normalized_payload(client, session, monkeypatch):
    app.dependency_overrides[get_db] = _override_db(session)
    latest_run, feature_item, feature_row, peers = _make_feature_context()
    app.dependency_overrides[get_uow] = lambda: _FakeUow(latest_run, feature_item, feature_row, peers)
    _seed_dashboard_data(session)

    monkeypatch.setattr(stocks_module, "_get_stock_info_or_404", lambda symbol: {
        "symbol": symbol,
        "name": "NVIDIA Corp",
        "sector": "Technology",
        "industry": "Semiconductors",
        "current_price": 921.45,
        "market_cap": 2.3e12,
    })
    monkeypatch.setattr(stocks_module, "_get_stock_fundamentals_payload", lambda symbol, force_refresh=False: {
        "symbol": symbol,
        "market_cap": 2.3e12,
        "eps_growth_quarterly": 54.2,
        "revenue_growth": 42.1,
    })
    monkeypatch.setattr(stocks_module, "_get_stock_technicals_payload", lambda symbol, db, force_refresh=False: {
        "symbol": symbol,
        "current_price": 921.45,
        "high_52w": 974.0,
        "low_52w": 410.0,
    })
    monkeypatch.setattr(stocks_module, "_load_price_history", lambda symbol, period="6mo": [
        {"date": "2026-04-01", "open": 900.0, "high": 930.0, "low": 890.0, "close": 921.45, "volume": 1000000}
    ])

    response = await client.get("/api/v1/stocks/NVDA/decision-dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "NVDA"
    assert payload["decision_summary"]["composite_score"] == 88.5
    assert payload["decision_summary"]["top_strengths"]
    assert payload["peers"][0]["symbol"] == "AVGO"
    assert payload["themes"][0]["display_name"] == "AI Infrastructure"
    assert payload["regime"]["label"] == "offense"
    assert payload["degraded_reasons"] == []


@pytest.mark.asyncio
async def test_decision_dashboard_endpoint_reports_degraded_mode_without_feature_run(client, session, monkeypatch):
    app.dependency_overrides[get_db] = _override_db(session)
    app.dependency_overrides[get_uow] = lambda: _FakeUow(None)

    monkeypatch.setattr(stocks_module, "_get_stock_info_or_404", lambda symbol: {
        "symbol": symbol,
        "name": "NVIDIA Corp",
    })
    monkeypatch.setattr(stocks_module, "_get_stock_fundamentals_payload", lambda symbol, force_refresh=False: None)
    monkeypatch.setattr(stocks_module, "_get_stock_technicals_payload", lambda symbol, db, force_refresh=False: None)
    monkeypatch.setattr(stocks_module, "_load_price_history", lambda symbol, period="6mo": [])

    response = await client.get("/api/v1/stocks/NVDA/decision-dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert "missing_feature_run" in payload["degraded_reasons"]
    assert "missing_explanation" in payload["degraded_reasons"]
    assert payload["decision_summary"]["composite_score"] is None
    assert payload["peers"] == []


@pytest.mark.asyncio
async def test_watchlist_import_endpoint_returns_partial_success(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    watchlist = _seed_search_and_import_data(session)

    response = await client.post(
        f"/api/v1/user-watchlists/{watchlist.id}/items/import",
        json={"content": "symbol\nAAPL\nMSFT\nBAD$\nNVDA", "format": "csv"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested_count"] == 4
    assert payload["added"] == ["MSFT", "NVDA"]
    assert payload["skipped_existing"] == ["AAPL"]
    assert payload["invalid_symbols"] == ["BAD$"]
