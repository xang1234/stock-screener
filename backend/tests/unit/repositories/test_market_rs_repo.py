from datetime import date

import pytest
from sqlalchemy.orm import sessionmaker

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    StockRsScore,
)
from app.infra.db.models.relative_strength import (
    MarketRsFormulaPointer,
    MarketRsRun,
    StockRsSnapshot,
)
from app.infra.db.repositories.market_rs_repo import (
    MarketRsFormulaNotConfigured,
    MarketRsFormulaUnsupported,
    MarketRsRunRepository,
)


AS_OF = date(2026, 4, 10)


def _run_kwargs(**overrides):
    values = {
        "market": "US",
        "as_of_date": AS_OF,
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "benchmark_symbol": "SPY",
        "benchmark_as_of_date": AS_OF,
        "universe_hash": "a" * 64,
        "expected_symbol_count": 2,
    }
    values.update(overrides)
    return values


def _score(symbol, value):
    return StockRsScore(
        symbol=symbol,
        overall_rs=value,
        rs_1m=value,
        rs_3m=value,
        rs_6m=value,
        rs_9m=value,
        rs_12m=value,
        weighted_composite=float(value),
        excess_return_1m=0.1,
        excess_return_3m=0.1,
        excess_return_6m=0.1,
        excess_return_9m=0.1,
        excess_return_12m=0.1,
    )


def test_completed_run_is_invisible_until_rows_and_status_commit(db_session):
    repo = MarketRsRunRepository()
    repo.start_or_restart(db_session, **_run_kwargs())
    db_session.commit()

    assert repo.get_completed_exact(
        db_session,
        market="US",
        as_of_date=AS_OF,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    ) is None


def test_failed_restart_clears_partial_rows_but_completed_run_is_idempotent(
    db_session,
):
    repo = MarketRsRunRepository()
    run = MarketRsRun(
        **_run_kwargs(),
        status="failed",
        eligible_symbol_count=1,
        excluded_symbol_count=1,
        diagnostics_json={"error": "old"},
    )
    run.rows.append(
        StockRsSnapshot(
            symbol="PARTIAL",
            overall_rs=50,
            rs_1m=50,
            rs_3m=50,
            rs_6m=50,
            rs_9m=50,
            rs_12m=50,
            weighted_composite=50.0,
            excess_return_1m=0.0,
            excess_return_3m=0.0,
            excess_return_6m=0.0,
            excess_return_9m=0.0,
            excess_return_12m=0.0,
        )
    )
    db_session.add(run)
    db_session.commit()

    restarted = repo.start_or_restart(db_session, **_run_kwargs())

    assert restarted.id == run.id
    assert restarted.status == "running"
    assert restarted.rows == []

    repo.replace_rows(
        db_session,
        restarted,
        {"AAA": _score("AAA", 1), "BBB": _score("BBB", 99)},
    )
    repo.mark_completed(restarted, excluded_symbol_count=0, diagnostics={})
    db_session.commit()

    completed = repo.start_or_restart(db_session, **_run_kwargs(universe_hash="b" * 64))
    assert completed.id == restarted.id
    assert completed.status == "completed"
    assert completed.universe_hash == "a" * 64
    assert len(completed.rows) == 2


def test_formula_activation_is_market_scoped_and_validated(db_session):
    repo = MarketRsRunRepository()
    db_session.add_all(
        [
            MarketRsFormulaPointer(
                market="US", formula_version=LEGACY_RS_FORMULA_VERSION
            ),
            MarketRsFormulaPointer(
                market="HK", formula_version=LEGACY_RS_FORMULA_VERSION
            ),
        ]
    )
    db_session.commit()

    with pytest.raises(ValueError, match="completed"):
        repo.activate_formula(
            db_session,
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
        )

    db_session.add(
        MarketRsRun(
            **_run_kwargs(),
            status="completed",
            eligible_symbol_count=0,
            excluded_symbol_count=2,
            diagnostics_json={},
        )
    )
    db_session.commit()
    repo.activate_formula(
        db_session,
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )

    assert repo.active_formula(db_session, market="US") == BALANCED_RS_FORMULA_VERSION
    assert repo.active_formula(db_session, market="HK") == LEGACY_RS_FORMULA_VERSION

    repo.activate_formula(
        db_session,
        market="US",
        formula_version=LEGACY_RS_FORMULA_VERSION,
    )
    assert repo.active_formula(db_session, market="US") == LEGACY_RS_FORMULA_VERSION

    with pytest.raises(MarketRsFormulaUnsupported):
        repo.activate_formula(db_session, market="US", formula_version="unknown")


def test_active_formula_rejects_an_unconfigured_market(db_session):
    with pytest.raises(MarketRsFormulaNotConfigured):
        MarketRsRunRepository().active_formula(db_session, market="US")


def test_two_sessions_share_one_completed_winner_and_never_read_partial_rows(
    db_session,
):
    repo = MarketRsRunRepository()
    Session = sessionmaker(bind=db_session.get_bind())
    first = Session()
    second = Session()
    try:
        run = repo.start_or_restart(first, **_run_kwargs())
        repo.replace_rows(
            first,
            run,
            {"AAA": _score("AAA", 1), "BBB": _score("BBB", 99)},
        )
        assert repo.get_completed_exact(
            second,
            market="US",
            as_of_date=AS_OF,
            formula_version=BALANCED_RS_FORMULA_VERSION,
        ) is None
        repo.mark_completed(run, excluded_symbol_count=0, diagnostics={})
        first.commit()

        winner = repo.start_or_restart(second, **_run_kwargs(universe_hash="z" * 64))
        second.commit()

        assert winner.id == run.id
        assert winner.status == "completed"
        assert winner.universe_hash == "a" * 64
        assert second.query(MarketRsRun).count() == 1
        assert len(winner.rows) == 2
    finally:
        first.close()
        second.close()
