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
                config_json={"universe": {"market": "DE"}},
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


def _corrupt_run(session_factory, run_id: int, **values) -> None:
    """Store a run row the enum rejects, the way a damaged database carries it."""
    session = session_factory()
    try:
        session.query(FeatureRun).filter(FeatureRun.id == run_id).update(values)
        session.commit()
    finally:
        session.close()


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
    assert payload["technicals"]["run_source"] == "market_pointer"
    assert payload["technicals"]["rs_rating"] == 85
    assert payload["technicals"]["run_source"] == "market_pointer"


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
    assert payload["technicals"]["run_source"] == "market_pointer"


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
    assert payload["technicals"]["run_source"] == "global_pointer"


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
    assert payload["technicals"]["run_source"] == "global_pointer"


def test_stock_snapshot_reports_no_run_when_no_pointer_target_is_published(
    service, session_factory
):
    """No pointer and no published run at all: no run, no exception."""
    failed_run = _seed_failed_run_with_rows(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _point(session_factory, "latest_published_market:US", failed_run)
    _drop_pointer(session_factory, "latest_published")
    session = session_factory()
    try:
        session.query(FeatureRun).update({FeatureRun.status: "failed"})
        session.commit()
    finally:
        session.close()

    assert _newest_published_run_id(session_factory) is None

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


def test_stock_snapshot_reads_the_market_run_when_no_pointer_was_ever_written(
    service, session_factory
):
    """The regression the pointer-only resolution would have caused.

    Published rows exist, no pointer does - a legacy database, a cleaned-up pointer, or runs
    inserted without one. Resolving through pointers alone answers ``technicals: null`` for
    every symbol there, although the figures are present.
    """
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == NEWEST_US_RUN
    assert payload["technicals"]["run_source"] == "published_run"


def test_stock_snapshot_does_not_answer_from_another_markets_published_run(
    service, session_factory
):
    """The fallback scans for the symbol's own market, never for whatever runs exist.

    The DE run carries a row for the US symbol on purpose: a run whose universe marker does
    not match its rows is exactly the state in which "take any published run" answers with
    another market's figures.
    """
    _seed_de_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")
    session = session_factory()
    try:
        session.add(
            StockFeatureDaily(
                run_id=DE_RUN,
                symbol="NVDA",
                as_of_date=DE_RUN_DATE,
                composite_score=55.0,
                overall_rating=3,
                passes_count=1,
                details_json=_details(
                    "NVDA",
                    55.0,
                    "Watch",
                    3,
                    55,
                    "Semiconductors",
                    120.0,
                    5_000_000,
                    3_000_000_000_000,
                    5.0,
                    4.0,
                ),
            )
        )
        # Only the DE run stays published: nothing in the table applies to a US symbol.
        session.query(FeatureRun).filter(FeatureRun.id.in_((US_RUN, NEWEST_US_RUN))).update(
            {FeatureRun.status: "failed"}, synchronize_session=False
        )
        session.commit()
    finally:
        session.close()

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is None


def test_stock_snapshot_reports_a_damaged_run_row_as_stored_data(service, session_factory):
    """A stored status outside the enum is damaged data, not a bad argument.

    ``get_latest_published`` maps the row to the domain object, which raises ``ValueError``
    for an unknown status. Unwrapped, ``call_tool``'s ValueError branch reported the caller's
    arguments as invalid - the request was fine, the row was not.
    """
    _set_market(session_factory, "NVDA", "US")
    _corrupt_run(session_factory, NEWEST_US_RUN, status="published_typo")
    _point(session_factory, "latest_published_market:US", NEWEST_US_RUN)

    result = service.call_tool("stock_snapshot", {"symbol": "NVDA"})

    assert result.get("isError") is True
    assert result["structuredContent"]["error"]["code"] == "invalid_stored_data"


def test_stock_snapshot_answers_an_unknown_symbol_even_when_a_run_row_is_damaged(
    service, session_factory
):
    """The symbol is looked up first: a damaged run row cannot break the not-found answer."""
    _corrupt_run(session_factory, NEWEST_US_RUN, status="published_typo")
    _point(session_factory, "latest_published_market:US", NEWEST_US_RUN)

    payload = _snapshot(service, "NOSUCH")

    assert payload["stock"] is None
    assert payload["technicals"] is None
    assert "NOSUCH" in payload["summary"]


def test_stock_snapshot_dates_breadth_by_the_us_run_for_a_symbol_outside_the_us(
    service, session_factory
):
    """Breadth is a US series, so its date comes from the US run, not from the German one.

    The German run is moved to 2026-03-27, where the US breadth row is 1.55; the US run sits on
    2026-03-29 with 1.82. Answering with the German run's date would hand a reader the older US
    row and call it this symbol's context.
    """
    _seed_de_run(session_factory)
    session = session_factory()
    try:
        session.query(FeatureRun).filter(FeatureRun.id == DE_RUN).update(
            {FeatureRun.as_of_date: date(2026, 3, 27)}, synchronize_session=False
        )
        session.query(StockFeatureDaily).filter(StockFeatureDaily.run_id == DE_RUN).update(
            {StockFeatureDaily.as_of_date: date(2026, 3, 27)}, synchronize_session=False
        )
        session.commit()
    finally:
        session.close()
    _point(session_factory, "latest_published_market:DE", DE_RUN)

    payload = _snapshot(service, "SAP.DE")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == DE_RUN
    assert payload["breadth"] is not None
    assert payload["breadth"]["date"] == "2026-03-29"


def _seed_rollout_run(session_factory, run_id: int = 5) -> int:
    """A staged Market RS rollout run: published, but never activated.

    ``build_balanced_feature_snapshot`` publishes it under
    ``rollout_rs:{formula}:{market}``, and ``MarketRsActivator`` repoints
    ``latest_published_market:{market}`` only after the run passes validation. Until then it
    carries ``status = published`` and real rows, without ever having gone live.
    """
    session = session_factory()
    try:
        session.add(
            FeatureRun(
                id=run_id,
                as_of_date=date(2026, 3, 31),
                run_type="daily_snapshot",
                status="published",
                config_json={"universe": {"market": "US"}},
                created_at=datetime(2026, 3, 31, 18, 0, tzinfo=UTC),
                completed_at=datetime(2026, 3, 31, 19, 0, tzinfo=UTC),
                published_at=datetime(2026, 3, 31, 19, 30, tzinfo=UTC),
            )
        )
        session.add(
            StockFeatureDaily(
                run_id=run_id,
                symbol="NVDA",
                as_of_date=date(2026, 3, 31),
                composite_score=97.0,
                overall_rating=5,
                passes_count=2,
                details_json=_details(
                    "NVDA",
                    97.0,
                    "Strong Buy",
                    2,
                    99,
                    "Semiconductors",
                    150.0,
                    30_000_000,
                    3_000_000_000_000,
                    40.0,
                    33.0,
                ),
            )
        )
        session.add(
            FeatureRunPointer(key="rollout_rs:balanced_v3:US", run_id=run_id)
        )
        session.commit()
    finally:
        session.close()
    return run_id


def test_stock_snapshot_does_not_serve_a_staged_rollout_run(service, session_factory):
    """A run staged by the rollout builder is published, but it never went live.

    Without a live pointer the scan is the only thing that could answer, and it must not
    answer with a run that failed (or has not yet passed) market activation.
    """
    rollout_run = _seed_rollout_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")
    session = session_factory()
    try:
        session.query(FeatureRun).filter(FeatureRun.id.in_((US_RUN, NEWEST_US_RUN))).update(
            {FeatureRun.status: "failed"}, synchronize_session=False
        )
        session.commit()
    finally:
        session.close()

    assert _newest_published_run_id(session_factory) == rollout_run

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is None


def test_stock_snapshot_serves_a_rollout_run_that_was_activated(service, session_factory):
    """Once activation repoints the market pointer, the same run is the market's run."""
    rollout_run = _seed_rollout_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _point(session_factory, "latest_published_market:US", rollout_run)

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == rollout_run
    assert payload["technicals"]["run_source"] == "market_pointer"


def test_stock_snapshot_ignores_a_market_pointer_that_names_another_markets_run(
    service, session_factory
):
    """A hand-set pointer can disagree with the run behind it; the chain falls through.

    ``publish_pointer_key`` is passed into the build unchanged and nothing compares it with
    the build's market, so the check belongs on the target, as it already was for the global
    pointer.
    """
    _seed_de_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _point(session_factory, "latest_published_market:US", DE_RUN)  # run declares market DE
    _point(session_factory, "latest_published", NEWEST_US_RUN)

    payload = _snapshot(service, "NVDA")

    assert payload["technicals"] is not None
    assert payload["technicals"]["run_id"] == NEWEST_US_RUN
    assert payload["technicals"]["run_source"] == "global_pointer"
