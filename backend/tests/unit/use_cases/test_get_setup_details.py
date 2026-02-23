"""Unit tests for GetSetupDetailsUseCase."""

from __future__ import annotations

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.use_cases.scanning.get_setup_details import (
    GetSetupDetailsQuery,
    GetSetupDetailsUseCase,
)
from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeScanResultRepository,
    FakeUnitOfWork,
    make_domain_item,
)


AS_OF = date(2026, 2, 17)


def _make_feature_row(symbol: str) -> FeatureRow:
    return FeatureRow(
        run_id=11,
        symbol=symbol,
        as_of_date=AS_OF,
        composite_score=88.0,
        overall_rating=5,
        passes_count=1,
        details={
            "setup_engine": {
                "explain": {"summary": "feature-store payload"},
                "candidates": [{"pattern": "cup_with_handle"}],
            }
        },
    )


class TestUnboundScanFallback:
    def test_reads_setup_payload_from_scan_results_for_unbound_scan(self):
        scan_results = FakeScanResultRepository(
            items=[
                make_domain_item(
                    "AAPL",
                    se_explain={"summary": "legacy payload"},
                    se_candidates=[{"pattern": "vcp"}],
                )
            ]
        )
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = GetSetupDetailsUseCase()

        result = uc.execute(
            uow,
            GetSetupDetailsQuery(scan_id="scan-legacy", symbol="aapl"),
        )

        assert result.symbol == "AAPL"
        assert result.se_explain == {"summary": "legacy payload"}
        assert result.se_candidates == [{"pattern": "vcp"}]
        assert scan_results.last_get_setup_payload_args is not None
        assert scan_results.last_get_setup_payload_args["symbol"] == "AAPL"


class TestFeatureStoreRouting:
    def test_reads_setup_payload_from_feature_store_for_bound_scan(self):
        feature_store = FakeFeatureStoreRepository()
        feature_store.upsert_snapshot_rows(11, [_make_feature_row("MSFT")])
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-fs", status="completed", feature_run_id=11)
        uc = GetSetupDetailsUseCase()

        result = uc.execute(
            uow,
            GetSetupDetailsQuery(scan_id="scan-fs", symbol="msft"),
        )

        assert result.symbol == "MSFT"
        assert result.se_explain == {"summary": "feature-store payload"}
        assert result.se_candidates == [{"pattern": "cup_with_handle"}]
        assert feature_store.last_get_setup_payload_for_run_args is not None
        assert feature_store.last_get_setup_payload_for_run_args["run_id"] == 11
        assert feature_store.last_get_setup_payload_for_run_args["symbol"] == "MSFT"


class TestErrors:
    def test_missing_symbol_raises_not_found(self):
        uow = FakeUnitOfWork(scan_results=FakeScanResultRepository())
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = GetSetupDetailsUseCase()

        with pytest.raises(EntityNotFoundError, match="ScanResult.*TSLA"):
            uc.execute(
                uow,
                GetSetupDetailsQuery(scan_id="scan-legacy", symbol="tsla"),
            )

    def test_missing_scan_raises_not_found(self):
        uc = GetSetupDetailsUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*missing-scan"):
            uc.execute(
                FakeUnitOfWork(),
                GetSetupDetailsQuery(scan_id="missing-scan", symbol="AAPL"),
            )
