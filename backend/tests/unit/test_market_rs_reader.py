from datetime import date, datetime, timezone

import pytest
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from app.domain.relative_strength import (
    BALANCED_RS_PRICE_BASIS,
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.domain.scanning.ports import (
    CanonicalMarketRsSource,
    LegacyMarketRsSource,
)
from app.infra.db.models.relative_strength import (
    MarketRsFormulaPointer,
    MarketRsRun,
    StockRsSnapshot,
)
from app.services.market_rs_reader import (
    CanonicalMarketRsUnavailable,
    SqlMarketRsReader,
)


AS_OF = date(2026, 4, 10)


def _snapshot(symbol, *, overall, one_month, three_month, twelve_month):
    return StockRsSnapshot(
        symbol=symbol,
        overall_rs=overall,
        rs_1m=one_month,
        rs_3m=three_month,
        rs_6m=50,
        rs_9m=50,
        rs_12m=twelve_month,
        weighted_composite=float(overall),
        excess_return_1m=0.1,
        excess_return_3m=0.1,
        excess_return_6m=0.1,
        excess_return_9m=0.1,
        excess_return_12m=0.1,
    )


def _seed_balanced_run(db_session):
    run = MarketRsRun(
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
        status="completed",
        benchmark_symbol="SPY",
        benchmark_as_of_date=AS_OF,
        universe_hash="a" * 64,
        expected_symbol_count=3,
        eligible_symbol_count=3,
        excluded_symbol_count=0,
        diagnostics_json={"price_basis": BALANCED_RS_PRICE_BASIS},
        completed_at=datetime.now(timezone.utc),
        rows=[
            _snapshot(
                "AAA", overall=99, one_month=50, three_month=75, twelve_month=99
            ),
            _snapshot(
                "BBB", overall=50, one_month=25, three_month=50, twelve_month=75
            ),
            _snapshot(
                "CCC", overall=1, one_month=1, three_month=1, twelve_month=1
            ),
        ],
    )
    db_session.add(run)
    db_session.commit()
    return run


def _reader(db_session):
    return SqlMarketRsReader(sessionmaker(bind=db_session.get_bind()))


def test_reader_returns_exact_market_snapshot_without_repercentiling_requested_set(
    db_session,
):
    run = _seed_balanced_run(db_session)
    db_session.add(
        MarketRsFormulaPointer(
            market="US", formula_version=BALANCED_RS_FORMULA_VERSION
        )
    )
    db_session.commit()
    reader = _reader(db_session)

    resolution = reader.get(
        market="US",
        symbols=("AAA", "CCC"),
        as_of_date=AS_OF,
    )

    assert resolution.formula_version == BALANCED_RS_FORMULA_VERSION
    assert isinstance(resolution.source, CanonicalMarketRsSource)
    assert resolution.run_id == run.id
    assert resolution.universe_size == 3
    assert resolution.ratings_by_symbol["AAA"] == {
        "rs_rating": 99,
        "rs_rating_1m": 50,
        "rs_rating_3m": 75,
        "rs_rating_12m": 99,
    }
    assert "BBB" not in resolution.ratings_by_symbol

    watchlist_resolution = reader.get(
        market="US",
        symbols=("AAA",),
        as_of_date=AS_OF,
    )
    assert (
        watchlist_resolution.ratings_by_symbol["AAA"]
        == resolution.ratings_by_symbol["AAA"]
    )


def test_reader_fails_closed_when_balanced_exact_run_is_missing(db_session):
    db_session.add(
        MarketRsFormulaPointer(
            market="US", formula_version=BALANCED_RS_FORMULA_VERSION
        )
    )
    db_session.commit()

    with pytest.raises(CanonicalMarketRsUnavailable, match="2026-04-10"):
        _reader(db_session).get(
            market="US", symbols=("AAA",), as_of_date=AS_OF
        )


def test_reader_legacy_mode_does_not_query_stock_snapshot_rows(db_session):
    db_session.add(
        MarketRsFormulaPointer(
            market="US", formula_version=LEGACY_RS_FORMULA_VERSION
        )
    )
    db_session.commit()
    statements = []

    def _capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lower())

    event.listen(db_session.get_bind(), "before_cursor_execute", _capture)
    try:
        resolution = _reader(db_session).get(
            market="US", symbols=("AAA",), as_of_date=AS_OF
        )
    finally:
        event.remove(db_session.get_bind(), "before_cursor_execute", _capture)

    assert isinstance(resolution.source, LegacyMarketRsSource)
    assert resolution.ratings_by_symbol == {}
    assert not any("stock_rs_snapshots" in statement for statement in statements)


def test_explicit_balanced_override_bypasses_legacy_pointer(db_session):
    run = _seed_balanced_run(db_session)
    db_session.add(
        MarketRsFormulaPointer(
            market="US", formula_version=LEGACY_RS_FORMULA_VERSION
        )
    )
    db_session.commit()

    resolution = _reader(db_session).get(
        market="US",
        symbols=("AAA",),
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert resolution.run_id == run.id
    assert isinstance(resolution.source, CanonicalMarketRsSource)
