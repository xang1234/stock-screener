"""Unit tests for ExportScanResultsUseCase — pure in-memory, no infrastructure."""

import csv
import io
from dataclasses import replace
from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.domain.scanning.models import ExportFormat
from app.use_cases.scanning.export_scan_results import (
    ExportScanResultsQuery,
    ExportScanResultsResult,
    ExportScanResultsUseCase,
)

from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeScanResultRepository,
    FakeUnitOfWork,
    make_domain_item,
)


# ── Helpers ──────────────────────────────────────────────────────────────


AS_OF = date(2026, 2, 17)


def _make_query(**overrides) -> ExportScanResultsQuery:
    defaults = dict(scan_id="scan-123")
    defaults.update(overrides)
    return ExportScanResultsQuery(**defaults)


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
    """Core business logic for exporting scan results."""

    def test_returns_csv_bytes(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, ExportScanResultsResult)
        assert result.media_type == "text/csv"
        assert result.filename.startswith("scan_")
        assert result.filename.endswith(".csv")
        # CSV content starts with UTF-8 BOM
        assert result.content.startswith(b"\xef\xbb\xbf")

    def test_csv_contains_header_and_data_rows(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[_make_feature_row("AAPL")])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        # Decode and check
        csv_text = result.content.decode("utf-8-sig")
        lines = csv_text.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "Symbol" in lines[0]
        assert "AAPL" in lines[1]

    def test_empty_scan_exports_header_only(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        csv_text = result.content.decode("utf-8-sig")
        lines = csv_text.strip().split("\n")
        assert len(lines) == 1  # header only


class TestScanNotFound:
    """Use case raises EntityNotFoundError for missing scans."""

    def test_nonexistent_scan_raises_not_found(self):
        uow = FakeUnitOfWork()
        uc = ExportScanResultsUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*not-a-scan"):
            uc.execute(uow, _make_query(scan_id="not-a-scan"))

    def test_not_found_error_has_entity_and_identifier(self):
        uow = FakeUnitOfWork()
        uc = ExportScanResultsUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(scan_id="missing"))

        assert exc_info.value.entity == "Scan"
        assert exc_info.value.identifier == "missing"


class TestUnboundScanFallback:
    """Scans without a feature run fall back to scan_results."""

    def test_unbound_scan_exports_from_scan_results(self):
        """Scan without feature_run_id exports from scan_results table."""
        items = [make_domain_item("AAPL"), make_domain_item("MSFT", score=70.0)]
        scan_results = FakeScanResultRepository(items=items)
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-legacy"))

        csv_text = result.content.decode("utf-8-sig")
        lines = csv_text.strip().split("\n")
        # Header + 2 data rows from scan_results
        assert len(lines) == 3
        assert "AAPL" in csv_text
        assert "MSFT" in csv_text

    def test_unbound_scan_exports_young_ipo_partial_metrics(self):
        item = make_domain_item(
            "NEWIPO",
            score=None,
            data_status="insufficient_history",
            scan_mode="listing_only",
            is_scannable=False,
            history_bars=30,
            rs_rating=None,
            rs_rating_1m=50.0,
            rs_rating_3m=None,
            rs_rating_12m=None,
            adr_percent=3.5,
            price_change_1d=2.1,
        )
        item = replace(
            item,
            rating="Insufficient Data",
            screeners_run=[],
            screeners_passed=0,
            screeners_total=0,
        )
        scan_results = FakeScanResultRepository(items=[item])
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-legacy"))

        csv_text = result.content.decode("utf-8-sig")
        row = next(csv.DictReader(io.StringIO(csv_text)))
        assert row["Symbol"] == "NEWIPO"
        assert row["Composite Score"] == ""
        assert row["Rating"] == "Insufficient Data"
        assert row["RS Rating"] == ""
        assert row["RS 1M"] == "50.0"
        assert row["RS 3M"] == ""
        assert row["RS 12M"] == ""
        assert row["ADR %"] == "3.5"


class TestFeatureStoreRouting:
    """Verify the use case queries the feature store correctly."""

    def test_bound_scan_queries_feature_store(self):
        """Scan with feature_run_id routes to feature store for export."""
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, scan_id="scan-bound")
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-bound"))

        csv_text = result.content.decode("utf-8-sig")
        lines = csv_text.strip().split("\n")
        # Header + 2 data rows from feature store
        assert len(lines) == 3
        assert "AAPL" in csv_text
        assert "MSFT" in csv_text

    def test_missing_feature_run_raises_not_found(self):
        """If feature_run_id points to a deleted run, raise EntityNotFoundError."""
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(
            scan_id="scan-orphan", status="completed", feature_run_id=999
        )
        uc = ExportScanResultsUseCase()

        with pytest.raises(EntityNotFoundError, match="FeatureRun"):
            uc.execute(uow, _make_query(scan_id="scan-orphan"))
