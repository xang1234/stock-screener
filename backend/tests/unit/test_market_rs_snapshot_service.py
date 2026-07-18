from datetime import date

import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.infra.db.models.relative_strength import MarketRsRun
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.services.market_rs_inputs import MarketRsInputs, MarketRsInputUnavailable
from app.services.market_rs_snapshot_service import MarketRsSnapshotService


AS_OF = date(2026, 4, 10)


def _complete_inputs():
    return MarketRsInputs(
        market="US",
        as_of_date=AS_OF,
        benchmark_symbol="SPY",
        benchmark_as_of_date=AS_OF,
        universe_hash="u" * 64,
        expected_symbols=("AAA", "BBB", "CCC"),
        excess_returns_by_symbol={
            "AAA": {"1m": 0.3, "3m": 0.3, "6m": 0.3, "9m": 0.3, "12m": 0.3},
            "BBB": {"1m": 0.2, "3m": 0.2, "6m": 0.2, "9m": 0.2, "12m": 0.2},
            "CCC": {"1m": 0.1, "3m": 0.1, "6m": 0.1, "9m": 0.1, "12m": 0.1},
        },
        exclusions={},
        current_price_coverage=1.0,
    )


class _FakeInputLoader:
    def __init__(self, inputs=None, error=None):
        self.inputs = inputs
        self.error = error
        self.calls = []

    def load(self, _db, *, market, as_of_date):
        self.calls.append((market, as_of_date))
        if self.error:
            raise self.error
        return self.inputs


def test_snapshot_service_publishes_all_rows_and_run_atomically(db_session):
    service = MarketRsSnapshotService(
        input_loader=_FakeInputLoader(_complete_inputs()),
        repository=MarketRsRunRepository(),
    )

    run = service.calculate(db_session, market="US", as_of_date=AS_OF)

    assert run.status == "completed"
    assert run.eligible_symbol_count == 3
    assert len(run.rows) == 3
    assert all(1 <= row.overall_rs <= 99 for row in run.rows)
    assert (
        db_session.query(MarketRsRun)
        .filter(MarketRsRun.status == "completed")
        .count()
        == 1
    )


def test_snapshot_input_failure_keeps_previous_completed_date_readable(db_session):
    repository = MarketRsRunRepository()
    previous_service = MarketRsSnapshotService(
        input_loader=_FakeInputLoader(_complete_inputs()),
        repository=repository,
    )
    previous = previous_service.calculate(
        db_session, market="US", as_of_date=date(2026, 4, 9)
    )
    error = MarketRsInputUnavailable(
        "benchmark missing",
        reason_code="benchmark_anchor_missing",
        diagnostics={"missing_offsets": [252]},
        benchmark_symbol="SPY",
        universe_hash="v" * 64,
        expected_symbol_count=3,
    )
    service = MarketRsSnapshotService(
        input_loader=_FakeInputLoader(error=error),
        repository=repository,
    )

    with pytest.raises(MarketRsInputUnavailable):
        service.calculate(db_session, market="US", as_of_date=AS_OF)

    latest = repository.get_latest_completed(
        db_session,
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    assert latest.id == previous.id
    failed = (
        db_session.query(MarketRsRun)
        .filter(MarketRsRun.as_of_date == AS_OF)
        .one()
    )
    assert failed.status == "failed"
    assert failed.diagnostics_json["reason_code"] == "benchmark_anchor_missing"


def test_snapshot_service_rejects_legacy_without_loading_or_writing(db_session):
    loader = _FakeInputLoader(_complete_inputs())
    service = MarketRsSnapshotService(
        input_loader=loader,
        repository=MarketRsRunRepository(),
    )

    with pytest.raises(ValueError, match="balanced-horizon-percentile-v2"):
        service.calculate(
            db_session,
            market="US",
            as_of_date=AS_OF,
            formula_version=LEGACY_RS_FORMULA_VERSION,
        )

    assert loader.calls == []
    assert db_session.query(MarketRsRun).count() == 0
