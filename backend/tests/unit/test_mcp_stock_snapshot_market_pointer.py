"""Regression tests: ``stock_snapshot``/``stock_lookup`` must read the per-market pointer.

Publication writes ``latest_published_market:{market}`` and no longer maintains the global
``latest_published`` pointer. Both tools read the globally newest published run instead, so
on an installation with US and DE every US symbol was resolved against the smaller DE run
that finished last: ``technicals`` came back ``null`` although the figures existed, and a
consumer reading ``technicals.stage`` failed on it.

Each test drives the tool through the public ``call_tool`` entry point, so it fails if the
pointer resolution regresses — not merely if a helper changes shape.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

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

#: The seeded runs. Run 1 is the older US run, run 2 the newer one; both published.
US_RUN = 1
NEWEST_US_RUN = 2

#: A published run of the smaller DE market: newest of all, and it holds no US symbol.
DE_RUN = 3
DE_RUN_DATE = date(2026, 3, 30)


@pytest.fixture()
def session_factory():
    """In-memory database carrying the shared MCP seed data."""
    factory, _engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    return factory


@pytest.fixture()
def service(session_factory):
    """Read-only copilot service over the seeded session factory."""
    return MarketCopilotService(
        session_factory,
        SimpleNamespace(
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
    )


def _set_market(session_factory, symbol: str, market: str) -> None:
    """Move a seeded symbol into *market*, as a per-market universe would carry it."""
    session = session_factory()
    try:
        (
            session.query(StockUniverse)
            .filter(StockUniverse.symbol == symbol)
            .update({StockUniverse.market: market})
        )
        session.commit()
    finally:
        session.close()


def _point(session_factory, key: str, run_id: int) -> None:
    """Create or move a pointer, mirroring what the publish path does."""
    session = session_factory()
    try:
        pointer = session.get(FeatureRunPointer, key)
        if pointer is None:
            session.add(FeatureRunPointer(key=key, run_id=run_id))
        else:
            pointer.run_id = run_id
        session.commit()
    finally:
        session.close()


def _drop_pointer(session_factory, key: str) -> None:
    """Delete a pointer, reproducing an install that never had it."""
    session = session_factory()
    try:
        session.query(FeatureRunPointer).filter(FeatureRunPointer.key == key).delete()
        session.commit()
    finally:
        session.close()


def _seed_de_run(session_factory) -> int:
    """Add the published DE run the deployment publishes after the US one.

    It is the newest published run and carries not a single US symbol — the exact state
    that made the tools answer ``technicals: null`` for AAPL.
    """
    session = session_factory()
    try:
        session.add(
            FeatureRun(
                id=DE_RUN,
                as_of_date=DE_RUN_DATE,
                run_type="daily_snapshot",
                status="published",
                created_at=datetime(2026, 3, 30, 18, 0, tzinfo=UTC),
                completed_at=datetime(2026, 3, 30, 19, 0, tzinfo=UTC),
                published_at=datetime(2026, 3, 30, 19, 30, tzinfo=UTC),
            )
        )
        session.add(
            StockUniverse(
                symbol="SAP.DE",
                name="SAP SE",
                sector="Information Technology",
                industry="Software",
                market_cap=200_000_000_000,
                exchange="XETRA",
                market="DE",
                is_active=True,
                status="active",
                source="fixture",
            )
        )
        session.add(
            StockFeatureDaily(
                run_id=DE_RUN,
                symbol="SAP.DE",
                as_of_date=DE_RUN_DATE,
                composite_score=71.0,
                overall_rating=4,
                passes_count=2,
                details_json=_details(
                    "SAP.DE",
                    71.0,
                    "Buy",
                    2,
                    78,
                    "Software",
                    180.0,
                    4_000_000,
                    200_000_000_000,
                    12.0,
                    9.0,
                ),
            )
        )
        session.commit()
    finally:
        session.close()
    return DE_RUN


def _seed_failed_run_with_rows(session_factory, run_id: int = 4) -> int:
    """A run that failed after its feature rows were written.

    A pointer can be left behind on it, and the rows are real - which is exactly why the
    status has to be checked instead of the rows.
    """
    session = session_factory()
    try:
        session.add(
            FeatureRun(
                id=run_id,
                as_of_date=date(2026, 3, 31),
                run_type="daily_snapshot",
                status="failed",
                created_at=datetime(2026, 3, 31, 18, 0, tzinfo=UTC),
                completed_at=None,
                published_at=None,
            )
        )
        session.add(
            StockFeatureDaily(
                run_id=run_id,
                symbol="NVDA",
                as_of_date=date(2026, 3, 31),
                composite_score=11.0,
                overall_rating=1,
                passes_count=0,
                details_json=_details(
                    "NVDA",
                    11.0,
                    "Avoid",
                    4,
                    12,
                    "Semiconductors",
                    100.0,
                    1_000_000,
                    3_000_000_000_000,
                    0.0,
                    0.0,
                ),
            )
        )
        session.commit()
    finally:
        session.close()
    return run_id


def _newest_published_run_id(session_factory) -> int | None:
    """The run ``_latest_published_run`` would have answered before the fix.

    Queried here so a test can prove the DE run really is the newest one; without that
    precondition the test would pass for the wrong reason.
    """
    session = session_factory()
    try:
        row = (
            session.query(FeatureRun)
            .filter(FeatureRun.status == "published")
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.id.desc())
            .first()
        )
        return row.id if row is not None else None
    finally:
        session.close()


def _run_holds(session_factory, run_id: int, symbol: str) -> bool:
    """Whether a published run carries a feature row for *symbol*."""
    session = session_factory()
    try:
        return (
            session.query(StockFeatureDaily)
            .filter(
                StockFeatureDaily.run_id == run_id,
                StockFeatureDaily.symbol == symbol,
            )
            .first()
            is not None
        )
    finally:
        session.close()


def _snapshot(service, symbol: str) -> dict:
    """Call ``stock_snapshot`` and return its structured payload."""
    result = service.call_tool("stock_snapshot", {"symbol": symbol})
    assert result.get("isError") is not True
    return result["structuredContent"]


def _lookup(service, symbol: str, *, include_technicals: bool = True) -> dict:
    """Call ``stock_lookup`` and return its structured payload."""
    result = service.call_tool(
        "stock_lookup",
        {"symbol": symbol, "include_technicals": include_technicals},
    )
    assert result.get("isError") is not True
    return result["structuredContent"]


def test_stock_snapshot_resolves_a_us_symbol_against_its_own_market_run(
    service, session_factory
):
    """The reported bug, end to end.

    Two markets published, the DE run finished last, the global pointer is gone — as on
    the reference installation. Before the fix this answered ``technicals: null`` for AAPL
    although run 1 carried it.
    """
    _seed_de_run(session_factory)
    _set_market(session_factory, "AAPL", "US")
    _point(session_factory, "latest_published_market:US", US_RUN)
    _point(session_factory, "latest_published_market:DE", DE_RUN)
    _drop_pointer(session_factory, "latest_published")

    # Preconditions: the DE run really is the newest published run, and really has no AAPL.
    assert _newest_published_run_id(session_factory) == DE_RUN
    assert _run_holds(session_factory, US_RUN, "AAPL") is True
    assert _run_holds(session_factory, DE_RUN, "AAPL") is False

    payload = _snapshot(service, "AAPL")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == US_RUN
    assert payload["technicals"]["stage"] == 2
    assert payload["technicals"]["rs_rating"] == 85


def test_stock_lookup_resolves_a_us_symbol_against_its_own_market_run(
    service, session_factory
):
    """The same resolution for ``stock_lookup`` with ``include_technicals``."""
    _seed_de_run(session_factory)
    _set_market(session_factory, "AAPL", "US")
    _point(session_factory, "latest_published_market:US", US_RUN)
    _point(session_factory, "latest_published_market:DE", DE_RUN)
    _drop_pointer(session_factory, "latest_published")

    assert _newest_published_run_id(session_factory) == DE_RUN

    payload = _lookup(service, "AAPL")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == US_RUN
    assert payload["technicals"]["stage"] == 2


def test_stock_snapshot_keeps_the_global_pointer_for_a_single_market_install(
    service, session_factory
):
    """No market pointer, global pointer present: the previous result, unchanged."""
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published_market:US")
    _point(session_factory, "latest_published", NEWEST_US_RUN)

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == NEWEST_US_RUN
    assert payload["technicals"]["composite_score"] == 92.0


def test_stock_snapshot_leaves_a_symbol_outside_the_universe_unchanged(
    service, session_factory
):
    """An unknown symbol has no market to resolve: no exception, no technicals."""
    _seed_de_run(session_factory)
    _point(session_factory, "latest_published_market:DE", DE_RUN)

    payload = _snapshot(service, "NOSUCH")

    assert payload["stock"] is None
    assert payload["technicals"] is None
    assert "NOSUCH" in payload["summary"]


def test_stock_snapshot_falls_back_when_the_market_pointer_target_is_not_published(
    service, session_factory
):
    """A pointer left on a failed run must not answer: the global pointer takes over.

    ``get_latest_published()`` follows the pointer without checking the target's status,
    and a failed run can still carry rows written before it failed. Resolving the pointer
    blindly would serve those rows as if they were published.
    """
    failed_run = _seed_failed_run_with_rows(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _point(session_factory, "latest_published_market:US", failed_run)
    _point(session_factory, "latest_published", NEWEST_US_RUN)

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == NEWEST_US_RUN
    assert payload["technicals"]["composite_score"] == 92.0


def test_stock_snapshot_reports_no_run_when_no_pointer_target_is_published(
    service, session_factory
):
    """Neither the market pointer nor a global pointer leads to a published run."""
    failed_run = _seed_failed_run_with_rows(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _point(session_factory, "latest_published_market:US", failed_run)
    _drop_pointer(session_factory, "latest_published")

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is None


def test_stock_snapshot_does_not_follow_the_pointer_of_another_market(
    service, session_factory
):
    """A foreign market pointer is not a fallback: only the symbol's own market counts."""
    _seed_de_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published_market:US")
    _point(session_factory, "latest_published_market:DE", DE_RUN)
    _point(session_factory, "latest_published", NEWEST_US_RUN)

    assert _newest_published_run_id(session_factory) == DE_RUN

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == NEWEST_US_RUN
