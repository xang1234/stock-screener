"""An explicit candidate market must never resolve to another market's run."""

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import delete, event

from app.infra.db.models.feature_store import (
    FeatureRun,
    FeatureRunPointer,
    StockFeatureDaily,
)
from app.interfaces.mcp.market_copilot import MarketCopilotService
from app.models.stock_universe import StockUniverse
from tests.helpers.mcp_fixture import (
    _details,
    create_mcp_test_session_factory,
    seed_market_copilot_data,
)


@pytest.fixture()
def markets():
    factory, engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    with factory() as db:
        for run in db.query(FeatureRun).all():
            run.config_json = {
                "market": "US", "universe_type": "market", "universe": {"market": "US"},
            }
        # The later German publication reproduces the real global-selector bug.
        db.add(FeatureRun(
            id=3, as_of_date=date(2026, 3, 29), run_type="daily_snapshot", status="published",
            created_at=datetime(2026, 3, 29, 22, tzinfo=UTC),
            completed_at=datetime(2026, 3, 29, 22, 5, tzinfo=UTC),
            published_at=datetime(2026, 3, 29, 22, 6, tzinfo=UTC),
            config_json={"market": "DE", "universe_type": "market", "universe": {"market": "DE"}},
        ))
        db.add(StockUniverse(
            symbol="SIE.DE", name="German test fixture", market="DE", currency="EUR",
            timezone="Europe/Berlin", exchange="XETRA", is_active=True,
            status="active", source="fixture",
        ))
        db.add(StockFeatureDaily(
            run_id=3, symbol="SIE.DE", as_of_date=date(2026, 3, 29),
            composite_score=90, overall_rating=4, passes_count=2,
            details_json=_details(
                "SIE.DE", 90, "Buy", 2, 92, "Industrials", 180,
                5_000_000, 100_000_000_000, 15, 10,
            ),
        ))
        db.add_all([
            FeatureRunPointer(key="latest_published_market:US", run_id=2),
            FeatureRunPointer(key="latest_published_market:DE", run_id=3),
        ])
        db.commit()
    service = MarketCopilotService(factory, SimpleNamespace(
        mcp_watchlist_writes_enabled=False, mcp_server_name="market-fixture",
    ))
    yield service, factory, engine
    engine.dispose()


def candidates(service, market, **kwargs):
    return service.call_tool("find_candidates", {"market": market, "limit": 100, **kwargs})


@pytest.mark.parametrize("market,run_id", [("US", 2), ("DE", 3), (" us ", 2), ("de", 3)])
def test_explicit_market_selects_its_own_publication(markets, market, run_id):
    service, _, _ = markets
    result = candidates(service, market)
    assert result.get("isError") is not True
    payload = result["structuredContent"]
    canonical = market.strip().upper()
    assert payload["run"]["id"] == run_id
    assert payload["run"]["market"] == canonical
    assert payload["run"]["universe_type"] == "market"
    assert payload["run"]["pointer_key"] == f"latest_published_market:{canonical}"
    assert payload["scope"]["market"] == canonical
    assert payload["symbols"]
    assert all(row["symbol"].endswith(".DE") == (canonical == "DE") for row in payload["symbols"])


def test_legacy_omitted_market_keeps_global_behavior(markets):
    service, _, _ = markets
    payload = service.call_tool("find_candidates", {"limit": 100})["structuredContent"]
    assert payload["run"]["id"] == 3
    assert "market" not in payload["run"]
    assert payload["scope"] == {"type": "published_run", "value": "latest published run"}


def test_market_mode_does_not_need_a_global_pointer(markets):
    service, factory, _ = markets
    with factory() as db:
        db.execute(delete(FeatureRunPointer).where(FeatureRunPointer.key == "latest_published"))
        db.commit()
    assert candidates(service, "US")["structuredContent"]["run"]["id"] == 2
    assert candidates(service, "DE")["structuredContent"]["run"]["id"] == 3


@pytest.mark.parametrize("market", ["", " ", "XX", "all", 7, True, ["US", "DE"]])
def test_invalid_market_fails_closed(markets, market):
    service, _, _ = markets
    result = candidates(service, market)
    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "invalid_arguments"


@pytest.mark.parametrize("failure", ["missing", "dangling", "wrong_market", "unpublished"])
def test_broken_pointer_never_uses_global_or_other_market(markets, failure):
    service, factory, _ = markets
    with factory() as db:
        pointer = db.get(FeatureRunPointer, "latest_published_market:US")
        if failure == "missing":
            db.delete(pointer)
        elif failure == "dangling":
            pointer.run_id = 99999
        elif failure == "wrong_market":
            pointer.run_id = 3
        else:
            db.get(FeatureRun, 2).status = "completed"
        db.commit()
    result = candidates(service, "US")
    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "market_run_unavailable"
    assert "run" not in result["structuredContent"]
    assert "symbols" not in result["structuredContent"]


@pytest.mark.parametrize("config", [
    {},
    {"market": "US"},
    {"market": "US", "universe_type": "custom"},
    {"universe_type": "market"},
    {"market": 1, "universe_type": "market"},
    {"market": "US", "universe_type": "market", "universe": "invalid"},
    {"market": "US", "universe_type": "market", "universe": {"market": "DE"}},
])
def test_missing_or_conflicting_market_evidence_fails_closed(markets, config):
    service, factory, _ = markets
    with factory() as db:
        db.get(FeatureRun, 2).config_json = config
        db.commit()
    result = candidates(service, "US")
    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "market_run_unavailable"


def test_empty_valid_market_result_keeps_provenance(markets):
    service, _, _ = markets
    payload = candidates(service, "DE", filters={"min_rs_rating": 100})["structuredContent"]
    assert payload["symbols"] == []
    assert payload["run"]["id"] == 3
    assert payload["run"]["market"] == "DE"


def test_watchlist_scope_remains_independent_of_market(markets):
    service, _, _ = markets
    payload = candidates(service, "US", universe="Leaders")["structuredContent"]
    assert payload["run"]["market"] == "US"
    assert payload["scope"] == {"type": "watchlist", "value": "Leaders", "market": "US"}
    # The watchlist still filters: Leaders holds NVDA and AAPL, the US run has
    # no AAPL row, so only NVDA can survive both restrictions.
    assert [row["symbol"] for row in payload["symbols"]] == ["NVDA"]


@pytest.mark.parametrize("universe", ["all", "latest_published", "ALL"])
def test_the_all_scope_stays_all_in_market_mode(markets, universe):
    """'all' scopes the run, it is never resolved as a watchlist name."""
    service, _, _ = markets
    payload = candidates(service, "US", universe=universe)["structuredContent"]
    assert payload["run"]["id"] == 2
    assert payload["scope"] == {
        "type": "published_run",
        "value": "published US market run",
        "market": "US",
    }
    assert {row["symbol"] for row in payload["symbols"]} == {
        "NVDA", "MSFT", "AVGO", "PANW", "SNOW"
    }


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param({"status": "published_typo"}, id="status"),
        pytest.param({"run_type": "snapshot_typo"}, id="run-type"),
    ],
)
def test_a_stored_run_with_an_unknown_enum_reports_stored_data(markets, corruption):
    """A broken enum is damaged storage, not a bad request."""
    service, factory, _ = markets
    with factory() as db:
        db.query(FeatureRun).filter(FeatureRun.id == 2).update(corruption)
        db.commit()

    result = candidates(service, "US")

    assert result.get("isError") is True
    assert result["structuredContent"]["error"]["code"] == "invalid_stored_data"


def test_requested_market_is_advertised_in_schema(markets):
    service, _, _ = markets
    descriptor = next(t for t in service.list_tools() if t["name"] == "find_candidates")
    assert "market" in descriptor["inputSchema"]["properties"]


def test_market_candidate_lookup_has_no_database_writes(markets):
    service, _, engine = markets
    statements = []

    def record(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lstrip().split(None, 1)[0].upper())

    event.listen(engine, "before_cursor_execute", record)
    try:
        assert candidates(service, "US").get("isError") is not True
        assert candidates(service, "DE").get("isError") is not True
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert statements
    assert set(statements) == {"SELECT"}
