"""Regression tests: ``explain_symbol`` must read the per-market publish pointer.

Publication writes ``latest_published_market:{market}`` and the global
``latest_published`` pointer is no longer maintained. ``explain_symbol`` read only the
global pointer, so it reported "No published feature run is available" while the run
was published and ``find_candidates`` was still listing it.

Each test drives the tool through the public ``call_tool`` entry point, so it fails if
the pointer resolution regresses — not merely if a helper changes shape.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.interfaces.mcp.market_copilot import MarketCopilotService
from app.models.stock_universe import StockUniverse
from tests.helpers.mcp_fixture import (
    create_mcp_test_session_factory,
    seed_market_copilot_data,
)

NO_RUN_MESSAGE = "No published feature run is available for symbol explanation."


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


def _delete_symbol(session_factory, symbol: str) -> None:
    """Remove a symbol from the universe entirely.

    ``market`` is NOT NULL with a ``"US"`` default, so "no market" cannot be expressed
    by nulling the column — a symbol outside the universe is the only way to exercise
    the no-market branch. ``_set_market`` defaults were applied at insert, which is why
    the pre-existing tests in this module see market ``"US"`` and never noticed the bug.
    """
    session = session_factory()
    try:
        (
            session.query(StockUniverse)
            .filter(StockUniverse.symbol == symbol)
            .delete()
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


def _payload(service, symbol: str) -> dict:
    """Call ``explain_symbol`` and return its structured payload."""
    result = service.call_tool("explain_symbol", {"symbol": symbol, "depth": "brief"})
    assert result.get("isError") is not True
    return result["structuredContent"]


def _published_run_ids(session_factory) -> set[int]:
    """Ids of every published run, so a test can assert the run really exists."""
    session = session_factory()
    try:
        rows = (
            session.query(FeatureRun)
            .filter(FeatureRun.status == "published")
            .all()
        )
        return {row.id for row in rows}
    finally:
        session.close()


def test_explain_symbol_resolves_run_via_market_pointer_when_global_pointer_is_gone(
    service, session_factory
):
    """The reported bug, end to end.

    Global pointer absent (as in the live system), the market pointer present, the
    symbol in market US. Before the fix this returned ``NO_RUN_MESSAGE``.
    """
    _set_market(session_factory, "NVDA", "US")
    _point(session_factory, "latest_published_market:US", 2)
    _drop_pointer(session_factory, "latest_published")

    # Preconditions: the run really is published, and the global pointer really is gone.
    assert 2 in _published_run_ids(session_factory)
    session = session_factory()
    try:
        assert session.get(FeatureRunPointer, "latest_published") is None
    finally:
        session.close()

    payload = _payload(service, "NVDA")

    assert payload["result"]["rating"] == "Strong Buy"
    assert payload["explanation"]["symbol"] == "NVDA"
    assert NO_RUN_MESSAGE not in str(payload["facts"])


def test_explain_symbol_ignores_a_market_pointer_pointing_at_another_run(
    service, session_factory
):
    """The market pointer is authoritative: run 1, not the newest published run 2."""
    _set_market(session_factory, "NVDA", "US")
    _point(session_factory, "latest_published_market:US", 1)
    _drop_pointer(session_factory, "latest_published")

    session = session_factory()
    try:
        # Run 1 carries NVDA too, so this asserts pointer choice, not run choice.
        assert session.get(FeatureRunPointer, "latest_published_market:US").run_id == 1
    finally:
        session.close()

    payload = _payload(service, "NVDA")

    assert payload["result"]["rating"] == "Strong Buy"
    assert payload["citations"][0]["as_of"] == "2026-03-28"


def test_explain_symbol_falls_back_to_global_pointer_without_a_market_pointer(
    service, session_factory
):
    """Legacy single-market install: market known, no market pointer -> global pointer."""
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published_market:US")
    _point(session_factory, "latest_published", 2)

    payload = _payload(service, "NVDA")

    assert payload["result"]["rating"] == "Strong Buy"
    assert payload["citations"][0]["as_of"] == "2026-03-29"


def test_explain_symbol_falls_back_to_global_pointer_without_a_market(
    service, session_factory
):
    """A symbol outside the universe has no market; the global pointer still applies.

    Note this is the *only* way to reach the no-market branch: ``market`` carries a
    ``"US"`` default, so every seeded row has one. That is exactly why the pre-existing
    tests in this module kept passing while the live system failed — the fixture never
    needed the market pointer, the real deployment does.
    """
    _delete_symbol(session_factory, "NVDA")
    _point(session_factory, "latest_published_market:US", 1)
    _point(session_factory, "latest_published", 2)

    payload = _payload(service, "NVDA")

    assert payload["result"]["rating"] == "Strong Buy"
    assert payload["citations"][0]["as_of"] == "2026-03-29"


def test_explain_symbol_does_not_manufacture_a_global_pointer(
    service, session_factory
):
    """Reading must not repair the state it reads.

    The fix deliberately does not create ``latest_published`` as a side effect:
    publication owns that key, and a read path inventing it would hide a genuinely
    missing publication.
    """
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")

    payload = _payload(service, "NVDA")
    assert NO_RUN_MESSAGE in payload["summary"]

    session = session_factory()
    try:
        assert session.get(FeatureRunPointer, "latest_published") is None
        assert session.get(FeatureRunPointer, "latest_published_market:US") is None
    finally:
        session.close()
