from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.api.v1.user_watchlists as user_watchlists_module
from app.database import Base, get_db
from app.main import app
from app.models.industry import IBDIndustryGroup
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.services import server_auth


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
            StockPrice.__table__,
            IBDIndustryGroup.__table__,
            UserWatchlist.__table__,
            WatchlistItem.__table__,
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
                WatchlistItem.__table__,
                UserWatchlist.__table__,
                IBDIndustryGroup.__table__,
                StockPrice.__table__,
                StockUniverse.__table__,
            ],
        )


def _override_db(db):
    def _get_db():
        try:
            yield db
        finally:
            pass

    return _get_db


def _seed_price_history(session, symbol: str, start_close: float, end_close: float, *, days: int = 280) -> None:
    start_date = date.today() - timedelta(days=days - 1)
    price_step = (end_close - start_close) / float(max(days - 1, 1))
    for offset in range(days):
        close = round(start_close + (price_step * offset), 4)
        session.add(
            StockPrice(
                symbol=symbol,
                date=start_date + timedelta(days=offset),
                open=close * 0.99,
                high=close * 1.01,
                low=close * 0.98,
                close=close,
                adj_close=close,
                volume=1_000_000 + offset,
            )
        )


@pytest.mark.asyncio
async def test_watchlist_data_endpoint_uses_db_price_history_without_price_cache(client, session, monkeypatch):
    app.dependency_overrides[get_db] = _override_db(session)

    watchlist = UserWatchlist(name="Leaders", position=0)
    session.add_all(
        [
            watchlist,
            StockUniverse(symbol="AAPL", name="Apple Inc."),
            StockUniverse(symbol="SPY", name="SPDR S&P 500 ETF"),
            IBDIndustryGroup(symbol="AAPL", industry_group="Computer-Hardware/Peripherals"),
        ]
    )
    session.flush()
    session.add(WatchlistItem(watchlist_id=watchlist.id, symbol="AAPL", position=0))
    _seed_price_history(session, "AAPL", 150.0, 220.0)
    _seed_price_history(session, "SPY", 400.0, 520.0)
    session.commit()

    load_calls: list[list[str]] = []
    real_loader = user_watchlists_module._load_price_frames_from_db

    def _spy_load_price_frames_from_db(symbols, db):
        load_calls.append(list(symbols))
        return real_loader(symbols, db)

    monkeypatch.setattr(
        user_watchlists_module,
        "_load_price_frames_from_db",
        _spy_load_price_frames_from_db,
    )

    response = await client.get(f"/api/v1/user-watchlists/{watchlist.id}/data")

    assert response.status_code == 200
    assert load_calls == [["AAPL", "SPY"]]
    payload = response.json()
    assert payload["name"] == "Leaders"
    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["symbol"] == "AAPL"
    assert item["company_name"] == "Apple Inc."
    assert item["ibd_industry"] == "Computer-Hardware/Peripherals"
    assert len(item["price_data"]) == 30
    assert len(item["rs_data"]) == 30
    assert item["change_12m"] is not None


# ── Reorder loads the rows once (#432) ─────────────────────────────────


def _count_selects(session):
    from sqlalchemy import event

    selects: list[str] = []

    def before_execute(conn, cursor, statement, *args):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    event.listen(session.get_bind(), "before_cursor_execute", before_execute)
    return selects


@pytest.mark.asyncio
async def test_reorder_watchlists_sets_request_positions_with_one_select(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    first, second, third = (UserWatchlist(name=n, position=i) for i, n in enumerate("ABC"))
    session.add_all([first, second, third])
    session.commit()
    # Unknown ids are skipped but still take their slot, as before.
    ids = [third.id, first.id, 9999, second.id]  # read before counting
    selects = _count_selects(session)

    response = await client.put("/api/v1/user-watchlists/reorder", json={"watchlist_ids": ids})
    request_selects = len(selects)

    assert response.status_code == 200
    assert request_selects == 1
    session.expire_all()
    assert (third.position, first.position, second.position) == (0, 1, 3)


@pytest.mark.asyncio
async def test_reorder_items_only_touches_the_watchlist_with_one_select(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    mine, other = UserWatchlist(name="Mine", position=0), UserWatchlist(name="Other", position=1)
    session.add_all([mine, other])
    session.flush()
    a, b = WatchlistItem(watchlist_id=mine.id, symbol="AAPL", position=0), WatchlistItem(
        watchlist_id=mine.id, symbol="MSFT", position=1
    )
    foreign = WatchlistItem(watchlist_id=other.id, symbol="NVDA", position=5)
    session.add_all([a, b, foreign])
    session.commit()
    url = f"/api/v1/user-watchlists/{mine.id}/items/reorder"
    ids = [b.id, foreign.id, a.id]  # read before counting
    selects = _count_selects(session)

    response = await client.put(url, json={"item_ids": ids})
    request_selects = len(selects)

    assert response.status_code == 200
    assert request_selects == 1
    session.expire_all()
    assert (b.position, a.position, foreign.position) == (0, 2, 5)


def _seed_watchlists(session):
    leaders = UserWatchlist(name="Leaders", position=0)
    empty = UserWatchlist(name="Empty", position=1)
    session.add_all([leaders, empty])
    session.commit()
    session.add_all([
        WatchlistItem(watchlist_id=leaders.id, symbol="ANET", position=1),
        WatchlistItem(watchlist_id=leaders.id, symbol="NVDA", position=0),
        WatchlistItem(watchlist_id=leaders.id, symbol="0700.HK", position=2),
    ])
    session.commit()
    return leaders.id, empty.id


def test_watchlist_symbols_reads_ordered_symbols_in_one_query(session):
    from sqlalchemy import event

    from app.services.watchlist_symbols import watchlist_symbols

    leaders_id, empty_id = _seed_watchlists(session)
    statements: list[str] = []
    event.listen(
        session.get_bind(),
        "before_cursor_execute",
        lambda *args: statements.append(args[2]),
    )

    assert watchlist_symbols(session, leaders_id) == ["NVDA", "ANET", "0700.HK"]
    assert len(statements) == 1
    assert watchlist_symbols(session, empty_id) == []
    assert watchlist_symbols(session, 999) is None


@pytest.mark.asyncio
async def test_watchlist_symbols_endpoint(client, session):
    app.dependency_overrides[get_db] = _override_db(session)
    leaders_id, empty_id = _seed_watchlists(session)

    leaders = await client.get(f"/api/v1/user-watchlists/{leaders_id}/symbols")
    empty = await client.get(f"/api/v1/user-watchlists/{empty_id}/symbols")
    missing = await client.get("/api/v1/user-watchlists/999/symbols")

    assert leaders.status_code == 200
    assert leaders.json() == {"symbols": ["NVDA", "ANET", "0700.HK"]}
    assert empty.json() == {"symbols": []}
    assert missing.status_code == 404
