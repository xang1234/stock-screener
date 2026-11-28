"""Regression tests: the stock tools must read the run of the symbol's own market.

Publication writes ``latest_published_market:{market}`` and no longer maintains the global
``latest_published`` pointer, so reading the globally newest published run resolved every US
symbol against whichever run finished last: ``technicals`` came back ``null`` although the
figures existed. Every case drives the tool through ``call_tool``.
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

#: The seeded runs: 1 is the older US run, 2 the newer one; both published.
US_RUN = 1
NEWEST_US_RUN = 2

#: A published run of the smaller DE market: newest of all, holds no US symbol.
DE_RUN = 3
DE_RUN_DATE = date(2026, 3, 30)

#: A Market RS rollout run: published before validation, live once its market pointer moves.
ROLLOUT_RUN = 5


@pytest.fixture()
def session_factory():
    factory, _engine = create_mcp_test_session_factory()
    seed_market_copilot_data(factory)
    return factory


@pytest.fixture()
def service(session_factory):
    return MarketCopilotService(
        session_factory,
        SimpleNamespace(
            mcp_watchlist_writes_enabled=False,
            mcp_server_name="stockscreen-market-copilot",
        ),
    )


def _update(session_factory, model, where: dict, values: dict) -> None:
    session = session_factory()
    try:
        session.query(model).filter(
            *[c.in_(v) if isinstance(v, tuple) else c == v for c, v in where.items()]
        ).update(values, synchronize_session=False)
        session.commit()
    finally:
        session.close()


def _set_market(session_factory, symbol: str, market: str) -> None:
    """Move a seeded symbol into *market*, as a per-market universe would carry it."""
    _update(session_factory, StockUniverse, {StockUniverse.symbol: symbol},
            {StockUniverse.market: market})


def _point(session_factory, key: str, run_id: int) -> None:
    """Create or move a pointer, as the publish path does."""
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


def _feature(run_id: int, symbol: str, day: date, *, score: float, stage: int, rs: int,
             rating: str = "Buy", industry: str = "Semiconductors") -> StockFeatureDaily:
    return StockFeatureDaily(
        run_id=run_id,
        symbol=symbol,
        as_of_date=day,
        composite_score=score,
        overall_rating=stage,
        passes_count=2,
        details_json=_details(
            symbol, score, rating, stage, rs, industry,
            150.0, 5_000_000, 3_000_000_000_000, 20.0, 15.0,
        ),
    )


def _seed_de_run(session_factory) -> int:
    """The smaller DE run published after the US one, carrying not a single US symbol."""
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
                published_at=datetime(2026, 3, 30, 19, 30, tzinfo=UTC),
            )
        )
        session.add(
            StockUniverse(
                symbol="SAP.DE", name="SAP SE", sector="Information Technology",
                industry="Software", market_cap=200_000_000_000, exchange="XETRA",
                market="DE", is_active=True, status="active", source="fixture",
            )
        )
        session.add(
            _feature(DE_RUN, "SAP.DE", DE_RUN_DATE, score=71.0, stage=2, rs=78,
                     industry="Software")
        )
        session.commit()
    finally:
        session.close()
    return DE_RUN


def _seed_rollout_run(session_factory, *, with_pointer: bool = True) -> int:
    """A staged rollout run: published, live only once activation repoints the market pointer.

    ``with_pointer=False`` reproduces the state after a retry: ``publish_atomically`` moved the
    rollout pointer to a newer run, leaving this one behind no pointer at all. Its own config
    still records the key it was published to.
    """
    session = session_factory()
    try:
        session.add(
            FeatureRun(
                id=ROLLOUT_RUN,
                as_of_date=date(2026, 3, 31),
                run_type="daily_snapshot",
                status="published",
                config_json={
                    "universe": {"market": "US"},
                    "publish_pointer_key": "rollout_rs:balanced_v3:US",
                },
                created_at=datetime(2026, 3, 31, 18, 0, tzinfo=UTC),
                published_at=datetime(2026, 3, 31, 19, 30, tzinfo=UTC),
            )
        )
        session.add(
            _feature(ROLLOUT_RUN, "NVDA", date(2026, 3, 31), score=97.0, stage=2, rs=99,
                     rating="Strong Buy")
        )
        if with_pointer:
            session.add(FeatureRunPointer(key="rollout_rs:balanced_v3:US", run_id=ROLLOUT_RUN))
        session.commit()
    finally:
        session.close()
    return ROLLOUT_RUN


def _fail_runs(session_factory, *run_ids: int) -> None:
    _update(session_factory, FeatureRun, {FeatureRun.id: tuple(run_ids)},
            {FeatureRun.status: "failed"})


def _corrupt_run(session_factory, run_id: int, **values) -> None:
    """Store a run row the enum rejects, the way a damaged database carries it."""
    _update(session_factory, FeatureRun, {FeatureRun.id: run_id}, values)


def _newest_published_run_id(session_factory) -> int | None:
    """The run the previous code would have answered, so a test cannot pass by accident."""
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


def _snapshot(service, symbol: str) -> dict:
    result = service.call_tool("stock_snapshot", {"symbol": symbol})
    assert result.get("isError") is not True
    return result["structuredContent"]


def _lookup(service, symbol: str) -> dict:
    result = service.call_tool("stock_lookup", {"symbol": symbol, "include_technicals": True})
    assert result.get("isError") is not True
    return result["structuredContent"]


def _two_markets(session_factory) -> None:
    """The reported bug: DE finished last, the global pointer is gone."""
    _seed_de_run(session_factory)
    _set_market(session_factory, "AAPL", "US")
    _point(session_factory, "latest_published_market:US", US_RUN)
    _point(session_factory, "latest_published_market:DE", DE_RUN)
    _drop_pointer(session_factory, "latest_published")
    assert _newest_published_run_id(session_factory) == DE_RUN


def _global_pointer_only(session_factory) -> None:
    """No market pointer, global pointer present: the previous answer, unchanged."""
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published_market:US")


def _pointer_on_a_failed_run(session_factory) -> None:
    """A failed run keeps the rows it wrote: the status has to decide, not the rows."""
    _update(session_factory, FeatureRun, {FeatureRun.id: NEWEST_US_RUN},
            {FeatureRun.status: "failed"})
    _set_market(session_factory, "NVDA", "US")
    _point(session_factory, "latest_published_market:US", NEWEST_US_RUN)
    _point(session_factory, "latest_published", US_RUN)


def _pointers_dropped(session_factory) -> None:
    """Published rows, no pointer: a legacy database or a cleaned-up pointer."""
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")


def _foreign_run_only(session_factory) -> None:
    """Only the DE run stays published, and it even carries a row for the US symbol."""
    _seed_de_run(session_factory)
    session = session_factory()
    try:
        session.add(_feature(DE_RUN, "NVDA", DE_RUN_DATE, score=55.0, stage=3, rs=55,
                             rating="Watch"))
        session.commit()
    finally:
        session.close()
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")
    _fail_runs(session_factory, US_RUN, NEWEST_US_RUN)


def _foreign_pointer(session_factory) -> None:
    """A pointer of another market is not a fallback for this symbol."""
    _seed_de_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published_market:US")
    _point(session_factory, "latest_published_market:DE", DE_RUN)


def _staged_rollout(session_factory) -> None:
    _seed_rollout_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")
    _fail_runs(session_factory, US_RUN, NEWEST_US_RUN)
    assert _newest_published_run_id(session_factory) == ROLLOUT_RUN


def _superseded_rollout(session_factory) -> None:
    _seed_rollout_run(session_factory, with_pointer=False)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")
    _fail_runs(session_factory, US_RUN, NEWEST_US_RUN)


def _rollout_named_by_a_foreign_pointer(session_factory) -> None:
    _seed_rollout_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _drop_pointer(session_factory, "latest_published_market:US")
    _point(session_factory, "latest_published_market:DE", ROLLOUT_RUN)
    _fail_runs(session_factory, US_RUN, NEWEST_US_RUN)


def _activated_rollout(session_factory) -> None:
    _seed_rollout_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _drop_pointer(session_factory, "latest_published")
    _point(session_factory, "latest_published_market:US", ROLLOUT_RUN)


def _market_pointer_naming_a_foreign_run(session_factory) -> None:
    """A hand-set ``publish_pointer_key`` can disagree with the build's market."""
    _seed_de_run(session_factory)
    _set_market(session_factory, "NVDA", "US")
    _point(session_factory, "latest_published_market:US", DE_RUN)


@pytest.mark.parametrize(
    ("setup", "symbol", "expected_run", "expected_source"),
    [
        (_two_markets, "AAPL", US_RUN, "market_pointer"),
        (_global_pointer_only, "NVDA", NEWEST_US_RUN, "global_pointer"),
        (_pointer_on_a_failed_run, "NVDA", US_RUN, "global_pointer"),
        (_pointers_dropped, "NVDA", NEWEST_US_RUN, "published_run"),
        (_foreign_run_only, "NVDA", None, None),
        (_foreign_pointer, "NVDA", NEWEST_US_RUN, "global_pointer"),
        (_staged_rollout, "NVDA", None, None),
        (_superseded_rollout, "NVDA", None, None),
        (_rollout_named_by_a_foreign_pointer, "NVDA", None, None),
        (_activated_rollout, "NVDA", ROLLOUT_RUN, "market_pointer"),
        (_market_pointer_naming_a_foreign_run, "NVDA", NEWEST_US_RUN, "global_pointer"),
    ],
    ids=[
        "own market pointer",
        "global pointer of a single-market install",
        "pointer on a failed run does not answer",
        "published rows without any pointer",
        "only another market's run published",
        "foreign pointer is not a fallback",
        "staged rollout is not served",
        "superseded rollout is not served",
        "foreign pointer does not activate a rollout",
        "activated rollout is served",
        "market pointer naming a foreign run falls through",
    ],
)
def test_the_run_that_applies(service, session_factory, setup, symbol,
                              expected_run, expected_source):
    setup(session_factory)

    technicals = _snapshot(service, symbol)["technicals"]

    if expected_run is None:
        assert technicals is None
        return
    assert technicals["run_id"] == expected_run
    assert technicals["run_source"] == expected_source


def test_stock_lookup_resolves_the_symbols_market_run(service, session_factory):
    """``stock_lookup`` resolves the same way, not only ``stock_snapshot``."""
    _two_markets(session_factory)

    technicals = _lookup(service, "AAPL")["technicals"]

    assert technicals["run_id"] == US_RUN
    assert technicals["stage"] == 2
    assert technicals["run_source"] == "market_pointer"


def test_a_damaged_run_row_is_stored_data_not_a_bad_argument(service, session_factory):
    """Mapping a row with an unknown status raises ``ValueError``; unwrapped, ``call_tool``
    reported the caller's arguments as invalid - the request was fine, the row was not."""
    _set_market(session_factory, "NVDA", "US")
    _corrupt_run(session_factory, NEWEST_US_RUN, status="published_typo")
    _point(session_factory, "latest_published_market:US", NEWEST_US_RUN)

    result = service.call_tool("stock_snapshot", {"symbol": "NVDA"})

    assert result.get("isError") is True
    assert result["structuredContent"]["error"]["code"] == "invalid_stored_data"


def test_an_unknown_symbol_stays_not_found_even_when_a_run_row_is_damaged(
    service, session_factory
):
    """The symbol is looked up first, so a damaged row cannot break the not-found answer."""
    _seed_de_run(session_factory)
    _corrupt_run(session_factory, NEWEST_US_RUN, status="published_typo")
    _point(session_factory, "latest_published_market:US", NEWEST_US_RUN)
    _point(session_factory, "latest_published_market:DE", DE_RUN)

    payload = _snapshot(service, "NOSUCH")

    assert payload["stock"] is None
    assert payload["technicals"] is None


def _move_de_run_to(session_factory, day: date) -> None:
    _update(session_factory, FeatureRun, {FeatureRun.id: DE_RUN}, {FeatureRun.as_of_date: day})
    _update(session_factory, StockFeatureDaily, {StockFeatureDaily.run_id: DE_RUN},
            {StockFeatureDaily.as_of_date: day})


@pytest.mark.parametrize("foreign_us_pointer", [False, True], ids=["plain", "us pointer foreign"])
def test_breadth_is_dated_by_the_us_run(service, session_factory, foreign_us_pointer):
    """Breadth is a US series: its date comes from the US run, not from the symbol's market run.

    The German run sits on 2026-03-27, where the US breadth row is 1.55; the US run sits on
    2026-03-29 with 1.82. The German run's date would hand a reader the older row and call it
    this symbol's context - whether the US pointer is correct or names the German run.
    """
    _seed_de_run(session_factory)
    _move_de_run_to(session_factory, date(2026, 3, 27))
    _point(session_factory, "latest_published_market:DE", DE_RUN)
    if foreign_us_pointer:
        _point(session_factory, "latest_published_market:US", DE_RUN)
        _drop_pointer(session_factory, "latest_published")

    payload = _snapshot(service, "SAP.DE")

    assert payload["technicals"]["run_id"] == DE_RUN
    assert payload["breadth"]["date"] == "2026-03-29"


def test_a_damaged_us_run_does_not_fail_a_foreign_symbol(service, session_factory):
    """Breadth is context for a symbol that lives elsewhere: an unreadable US publication may
    cost the date, not the snapshot."""
    _seed_de_run(session_factory)
    _corrupt_run(session_factory, NEWEST_US_RUN, status="published_typo")
    _point(session_factory, "latest_published_market:DE", DE_RUN)
    _point(session_factory, "latest_published_market:US", NEWEST_US_RUN)

    payload = _snapshot(service, "SAP.DE")

    assert payload.get("error") is None
    assert payload["technicals"]["run_id"] == DE_RUN
    assert payload["breadth"] is not None
