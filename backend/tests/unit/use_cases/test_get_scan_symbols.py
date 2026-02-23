"""Unit tests for GetScanSymbolsUseCase."""

from __future__ import annotations

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.domain.scanning.filter_spec import FilterSpec, PageSpec, SortSpec
from app.use_cases.scanning.get_scan_symbols import (
    GetScanSymbolsQuery,
    GetScanSymbolsUseCase,
)
from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeScanResultRepository,
    FakeUnitOfWork,
    make_domain_item,
)


AS_OF = date(2026, 2, 17)


def _make_feature_row(symbol: str, score: float = 80.0) -> FeatureRow:
    return FeatureRow(
        run_id=1,
        symbol=symbol,
        as_of_date=AS_OF,
        composite_score=score,
        overall_rating=4,
        passes_count=1,
        details={"rating": "Buy", "current_price": 100.0},
    )


class TestUnboundScanRouting:
    def test_unbound_scan_queries_scan_results_symbols(self):
        scan_results = FakeScanResultRepository(
            items=[
                make_domain_item("AAPL"),
                make_domain_item("MSFT"),
                make_domain_item("NVDA"),
            ]
        )
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = GetScanSymbolsUseCase()

        result = uc.execute(
            uow,
            GetScanSymbolsQuery(scan_id="scan-legacy"),
        )

        assert result.symbols == ("AAPL", "MSFT", "NVDA")
        assert result.total == 3
        assert scan_results.last_query_symbols_args is not None
        assert scan_results.last_query_symbols_args["scan_id"] == "scan-legacy"

    def test_passes_only_adds_rating_filter_without_mutating_input_filters(self):
        scan_results = FakeScanResultRepository(items=[make_domain_item("AAPL")])
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = GetScanSymbolsUseCase()

        original_filters = FilterSpec()
        original_filters.add_categorical("gics_sector", ("Technology",))

        uc.execute(
            uow,
            GetScanSymbolsQuery(
                scan_id="scan-legacy",
                filters=original_filters,
                passes_only=True,
            ),
        )

        forwarded = scan_results.last_query_symbols_args["filters"]
        assert any(
            cf.field == "rating" and tuple(cf.values) == ("Strong Buy", "Buy")
            for cf in forwarded.categorical_filters
        )
        assert not any(
            cf.field == "rating" for cf in original_filters.categorical_filters
        )


class TestFeatureStoreRouting:
    def test_bound_scan_queries_feature_store_symbols(self):
        feature_store = FakeFeatureStoreRepository()
        feature_store.upsert_snapshot_rows(
            7,
            [
                _make_feature_row("AAPL", 85.0),
                _make_feature_row("MSFT", 75.0),
            ],
        )
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-fs", status="completed", feature_run_id=7)
        uc = GetScanSymbolsUseCase()

        result = uc.execute(
            uow,
            GetScanSymbolsQuery(
                scan_id="scan-fs",
                sort=SortSpec(field="symbol"),
                page=PageSpec(page=1, per_page=1),
            ),
        )

        assert result.symbols == ("AAPL",)
        assert result.total == 2
        assert result.page == 1
        assert result.per_page == 1
        assert feature_store.last_query_run_symbols_args is not None
        assert feature_store.last_query_run_symbols_args["run_id"] == 7
        assert feature_store.last_query_run_symbols_args["page"].page == 1

    def test_missing_scan_raises_not_found(self):
        uc = GetScanSymbolsUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*missing"):
            uc.execute(
                FakeUnitOfWork(),
                GetScanSymbolsQuery(scan_id="missing"),
            )
