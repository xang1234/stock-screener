"""Unit tests for GetScanResultsUseCase — pure in-memory, no infrastructure."""

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.domain.scanning.filter_spec import (
    PageSpec,
    QuerySpec,
    SortOrder,
    SortSpec,
)
from app.domain.scanning.models import ResultPage
from app.use_cases.scanning.get_scan_results import (
    GetScanResultsQuery,
    GetScanResultsResult,
    GetScanResultsUseCase,
)

from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeScanResultRepository,
    FakeUnitOfWork,
    make_domain_item,
)


# ── Helpers ──────────────────────────────────────────────────────────────


AS_OF = date(2026, 2, 17)


def _make_query(**overrides) -> GetScanResultsQuery:
    defaults = dict(scan_id="scan-123")
    defaults.update(overrides)
    return GetScanResultsQuery(**defaults)


def _make_feature_row(symbol: str, score: float = 85.0) -> FeatureRow:
    """Build a FeatureRow with minimal details for bridge method tests."""
    return FeatureRow(
        run_id=1,
        symbol=symbol,
        as_of_date=AS_OF,
        composite_score=score,
        overall_rating=4,  # Buy
        passes_count=1,
        details={
            "composite_score": score,
            "rating": "Buy",
            "current_price": 150.0,
            "screeners_run": ["minervini"],
            "composite_method": "weighted_average",
            "screeners_passed": 1,
            "screeners_total": 1,
        },
    )


def _setup_bound_scan(uow, feature_store, scan_id="scan-123", run_id=1, rows=None):
    """Create a scan bound to a feature run, with rows in the feature store."""
    uow.scans.create(scan_id=scan_id, status="completed", feature_run_id=run_id)
    if rows is None:
        rows = [_make_feature_row("AAPL"), _make_feature_row("MSFT", score=70.0)]
    feature_store.upsert_snapshot_rows(run_id, rows)


# ── Tests ────────────────────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for retrieving scan results."""

    def test_returns_result_page(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = GetScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, GetScanResultsResult)
        assert isinstance(result.page, ResultPage)
        assert result.page.total == 2
        assert len(result.page.items) == 2

    def test_passes_query_spec(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = GetScanResultsUseCase()

        spec = QuerySpec(
            sort=SortSpec(field="rs_rating", order=SortOrder.ASC),
            page=PageSpec(page=1, per_page=10),
        )
        result = uc.execute(uow, _make_query(query_spec=spec))

        assert result.page.per_page == 10

    def test_include_sparklines_flag(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = GetScanResultsUseCase()

        # Just verify it doesn't crash
        result = uc.execute(uow, _make_query(include_sparklines=False))
        assert result.page.total == 2

    def test_empty_results_returns_empty_page(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[])
        uc = GetScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert result.page.total == 0
        assert len(result.page.items) == 0

    def test_pagination_metadata(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        rows = [_make_feature_row(f"SYM{i}") for i in range(75)]
        _setup_bound_scan(uow, feature_store, rows=rows)
        uc = GetScanResultsUseCase()

        spec = QuerySpec(page=PageSpec(page=2, per_page=25))
        result = uc.execute(uow, _make_query(query_spec=spec))

        assert result.page.total == 75
        assert result.page.page == 2
        assert result.page.per_page == 25
        assert result.page.total_pages == 3
        assert len(result.page.items) == 25


class TestScanNotFound:
    """Use case raises EntityNotFoundError for missing scans."""

    def test_nonexistent_scan_raises_not_found(self):
        uow = FakeUnitOfWork()
        uc = GetScanResultsUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*not-a-scan"):
            uc.execute(uow, _make_query(scan_id="not-a-scan"))

    def test_not_found_error_has_entity_and_identifier(self):
        uow = FakeUnitOfWork()
        uc = GetScanResultsUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(scan_id="missing"))

        assert exc_info.value.entity == "Scan"
        assert exc_info.value.identifier == "missing"


class TestDefaultQuerySpec:
    """Default query spec uses sensible defaults."""

    def test_default_query_spec_applied(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = GetScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert result.page.page == 1
        assert result.page.per_page == 50

    def test_default_include_sparklines_is_true(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = GetScanResultsUseCase()

        # No crash = sparklines enabled by default
        result = uc.execute(uow, _make_query())
        assert result.page.total == 2


class TestUnboundScanFallback:
    """Scans without a feature run fall back to scan_results."""

    def test_unbound_scan_returns_results_from_scan_results(self):
        """Scan without feature_run_id reads from scan_results table."""
        items = [make_domain_item("AAPL"), make_domain_item("MSFT", score=70.0)]
        scan_results = FakeScanResultRepository(items=items)
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = GetScanResultsUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-legacy"))

        assert isinstance(result.page, ResultPage)
        assert result.page.total == 2
        assert {i.symbol for i in result.page.items} == {"AAPL", "MSFT"}

    def test_unbound_scan_passes_query_args_to_scan_results(self):
        """Verify scan_id and spec are forwarded to scan_results.query()."""
        items = [make_domain_item("AAPL")]
        scan_results = FakeScanResultRepository(items=items)
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = GetScanResultsUseCase()

        uc.execute(uow, _make_query(scan_id="scan-legacy"))

        assert scan_results.last_query_args is not None
        assert scan_results.last_query_args["scan_id"] == "scan-legacy"

    def test_unbound_scan_forwards_setup_payload_flag(self):
        items = [make_domain_item("AAPL")]
        scan_results = FakeScanResultRepository(items=items)
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = GetScanResultsUseCase()

        uc.execute(
            uow,
            _make_query(
                scan_id="scan-legacy",
                include_setup_payload=True,
            ),
        )

        assert scan_results.last_query_args is not None
        assert scan_results.last_query_args["include_setup_payload"] is True


class TestFeatureStoreRouting:
    """Verify the use case queries the feature store correctly."""

    def test_bound_scan_queries_feature_store(self):
        """Scan with feature_run_id routes to feature store."""
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, scan_id="scan-bound")
        uc = GetScanResultsUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-bound"))

        assert result.page.total == 2
        assert result.page.items[0].symbol in ("AAPL", "MSFT")

    def test_bound_scan_returns_result_page(self):
        """Feature store path returns a proper ResultPage."""
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, scan_id="scan-bound")
        uc = GetScanResultsUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-bound"))

        assert isinstance(result.page, ResultPage)
        assert result.page.page == 1
        assert result.page.per_page == 50

    def test_missing_feature_run_raises_not_found(self):
        """If feature_run_id points to a deleted run, raise EntityNotFoundError."""
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        # Bind to run_id=999 which doesn't exist in feature store
        uow.scans.create(
            scan_id="scan-orphan", status="completed", feature_run_id=999
        )
        uc = GetScanResultsUseCase()

        with pytest.raises(EntityNotFoundError, match="FeatureRun"):
            uc.execute(uow, _make_query(scan_id="scan-orphan"))

    def test_bound_scan_forwards_setup_payload_flag(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, scan_id="scan-bound")
        uc = GetScanResultsUseCase()

        uc.execute(
            uow,
            _make_query(
                scan_id="scan-bound",
                include_setup_payload=True,
            ),
        )

        assert feature_store.last_query_run_as_scan_results_args is not None
        assert (
            feature_store.last_query_run_as_scan_results_args["include_setup_payload"]
            is True
        )
