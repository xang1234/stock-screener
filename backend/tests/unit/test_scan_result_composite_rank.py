from __future__ import annotations

import importlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.domain.common.query import SortOrder, SortSpec
from app.domain.scanning.filter_expression_model import QuerySpec
from app.infra.db.repositories.scan_result_repo import SqlScanResultRepository
from app.models.industry import IBDGroupPeerCache
from app.models.scan_result import Scan, ScanResult
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse

# (symbol, composite, minervini, canslim). IPO has only the IPO-uplifted
# composite; BEST and CANSLIM were scored by at least one core screener.
ROWS = (
    ("IPO", 72.0, None, None),
    ("BEST", 65.0, 65.0, 62.0),
    ("CANSLIM", 60.0, None, 60.0),
    ("UNSCORED", None, None, None),
)
RANKED = ["BEST", "CANSLIM", "IPO", "UNSCORED"]


@pytest.fixture
def session():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(
        engine,
        tables=[
            Scan.__table__,
            ScanResult.__table__,
            StockUniverse.__table__,
            StockFundamental.__table__,
            IBDGroupPeerCache.__table__,
        ],
    )
    db = sessionmaker(bind=engine)()
    db.add(Scan(scan_id="scan-1", status="completed"))
    db.add_all(
        ScanResult(
            scan_id="scan-1",
            symbol=symbol,
            composite_score=composite,
            minervini_score=minervini,
            canslim_score=canslim,
            ibd_industry_group="Software",
            gics_sector="Technology",
        )
        for symbol, composite, minervini, canslim in ROWS
    )
    db.commit()
    yield db
    db.close()
    engine.dispose()


def _page_symbols(session, sort: SortSpec) -> list[str]:
    page = SqlScanResultRepository(session).query("scan-1", QuerySpec(sort=sort))
    return [item.symbol for item in page.items]


def test_scan_page_composite_sort_ranks_fully_scored_rows_first(session):
    assert _page_symbols(session, SortSpec()) == RANKED


def test_scan_page_ascending_composite_sort_follows_the_score_alone(session):
    assert _page_symbols(
        session, SortSpec(field="composite_score", order=SortOrder.ASC)
    ) == ["CANSLIM", "BEST", "IPO", "UNSCORED"]


def test_peers_rank_fully_scored_rows_first(session):
    repo = SqlScanResultRepository(session)

    industry = repo.get_peers_by_industry("scan-1", "Software")
    sector = repo.get_peers_by_sector("scan-1", "Technology")

    assert [peer.symbol for peer in industry] == RANKED
    assert [peer.symbol for peer in sector] == RANKED


@pytest.mark.parametrize("module_name", ["app.services.scan_execution", "app.tasks.scan_tasks"])
def test_peer_cache_top_symbol_is_the_best_fully_scored_row(session, module_name):
    importlib.import_module(module_name).compute_industry_peer_metrics(session, "scan-1")

    cache = session.query(IBDGroupPeerCache).one()
    assert (cache.top_symbol, cache.top_score) == ("BEST", 65.0)


def test_chart_prewarm_warms_the_top_fully_scored_rows(session, monkeypatch):
    import app.tasks.cache_tasks as module
    from app.wiring import bootstrap

    checked: list[str] = []

    class _PriceCache:
        def get_cache_stats(self, symbol):
            checked.append(symbol)
            return {"redis_cached": True}

    monkeypatch.setattr(module, "SessionLocal", lambda: session)
    monkeypatch.setattr(bootstrap, "get_price_cache", _PriceCache)
    monkeypatch.setattr(module.prewarm_chart_cache_for_scan, "update_state", lambda **_kw: None)

    result = module.prewarm_chart_cache_for_scan.run("scan-1", top_n=2)

    assert checked == ["BEST", "CANSLIM"]
    assert result["already_cached"] == 2
