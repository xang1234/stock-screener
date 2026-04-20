"""Unit tests for ExportScanResultsUseCase — pure in-memory, no infrastructure."""

import csv
import io
from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.domain.scanning.filter_spec import FilterSpec, SortOrder, SortSpec
from app.domain.scanning.models import ExportFormat
from app.use_cases.scanning.export_scan_results import (
    ExportScanResultsQuery,
    ExportScanResultsResult,
    ExportScanResultsUseCase,
    _CSV_COLUMNS,
    _format_value,
)

from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeScanResultRepository,
    FakeUnitOfWork,
    make_domain_item,
)


# ── Constants ────────────────────────────────────────────────────────────

AS_OF = date(2026, 2, 17)


# ── Specialised fake ────────────────────────────────────────────────────


class TrackingFeatureStoreRepo(FakeFeatureStoreRepository):
    """Extends FakeFeatureStoreRepository to record query_all_as_scan_results args."""

    def __init__(self):
        super().__init__()
        self.last_query_all_args: dict | None = None

    def query_all_as_scan_results(self, run_id, filters, sort, *, include_sparklines=False):
        self.last_query_all_args = {
            "run_id": run_id,
            "filters": filters,
            "sort": sort,
            "include_sparklines": include_sparklines,
        }
        return super().query_all_as_scan_results(
            run_id, filters, sort, include_sparklines=include_sparklines
        )


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_feature_row(symbol, score=85.0, **details_extra):
    details = {
        "composite_score": score,
        "rating": "Buy",
        "current_price": 150.0,
        "screeners_run": ["minervini"],
        "composite_method": "weighted_average",
        "screeners_passed": 1,
        "screeners_total": 1,
    }
    details.update(details_extra)
    return FeatureRow(
        run_id=1, symbol=symbol, as_of_date=AS_OF,
        composite_score=score, overall_rating=4,
        passes_count=1, details=details,
    )


def _setup_bound_scan(uow, feature_store, scan_id="scan-123", run_id=1, rows=None):
    """Create a scan bound to a feature run, with rows in the feature store."""
    uow.scans.create(scan_id=scan_id, status="completed", feature_run_id=run_id)
    feature_store.upsert_snapshot_rows(run_id, rows or [])


def _make_query(**overrides) -> ExportScanResultsQuery:
    defaults = dict(scan_id="scan-123")
    defaults.update(overrides)
    return ExportScanResultsQuery(**defaults)


def _parse_csv_bytes(content: bytes) -> list[list[str]]:
    """Parse CSV bytes (skipping BOM) into rows."""
    text = content.lstrip(b"\xef\xbb\xbf").decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    return list(reader)


# ── Tests ────────────────────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for exporting scan results."""

    def test_returns_export_result(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL"), _make_feature_row("MSFT"),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, ExportScanResultsResult)
        assert isinstance(result.content, bytes)
        assert result.media_type == "text/csv"
        assert result.filename.endswith(".csv")

    def test_csv_has_correct_row_count(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL"), _make_feature_row("MSFT"),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        # 1 header + 2 data rows
        assert len(rows) == 3

    def test_csv_header_matches_data_column_count(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", rs_rating=88.5),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        assert len(header) == len(data_row)

    def test_csv_contains_symbol_data(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("NVDA", score=92.0),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        symbol_idx = header.index("Symbol")
        assert data_row[symbol_idx] == "NVDA"

    def test_csv_contains_company_name(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", company_name="Apple Inc."),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        name_idx = header.index("Company Name")
        assert data_row[name_idx] == "Apple Inc."

    def test_csv_coerces_scalar_market_themes_without_character_splitting(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", market_themes="AI Infrastructure"),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        themes_idx = header.index("Market Themes")
        assert data_row[themes_idx] == "AI Infrastructure"


class TestScanNotFound:
    """Use case raises EntityNotFoundError for missing scans."""

    def test_nonexistent_scan_raises_not_found(self):
        uow = FakeUnitOfWork()
        uc = ExportScanResultsUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*not-a-scan"):
            uc.execute(uow, _make_query(scan_id="not-a-scan"))


class TestUnboundScanFallsBackToLegacy:
    """Scan exists but has no feature_run_id — export uses legacy scan_results."""

    def test_unbound_scan_exports_from_legacy_scan_results(self):
        shared_fields = {
            "company_name": "Apple Inc.",
            "market": "US",
            "exchange": "NASDAQ",
            "currency": "USD",
            "market_themes": ["AI Infrastructure", "Cloud Platforms"],
        }
        legacy_uow = FakeUnitOfWork(
            scan_results=FakeScanResultRepository(items=[make_domain_item("AAPL", **shared_fields)])
        )
        legacy_uow.scans.create(scan_id="scan-123", status="completed")  # no feature_run_id

        bound_feature_store = FakeFeatureStoreRepository()
        bound_uow = FakeUnitOfWork(feature_store=bound_feature_store)
        _setup_bound_scan(bound_uow, bound_feature_store, rows=[
            _make_feature_row("AAPL", **shared_fields),
        ])

        uc = ExportScanResultsUseCase()

        legacy_rows = _parse_csv_bytes(uc.execute(legacy_uow, _make_query()).content)
        bound_rows = _parse_csv_bytes(uc.execute(bound_uow, _make_query()).content)
        header = legacy_rows[0]
        data_row = legacy_rows[1]

        assert header == [column for column, _ in _CSV_COLUMNS]
        assert header == bound_rows[0]
        assert data_row == bound_rows[1]
        assert data_row[header.index("Symbol")] == "AAPL"
        assert data_row[header.index("Company Name")] == "Apple Inc."
        assert data_row[header.index("Market")] == "US"
        assert data_row[header.index("Market Themes")] == "AI Infrastructure | Cloud Platforms"


class TestEmptyResults:
    """Empty scan still produces a valid CSV with headers only."""

    def test_empty_results_produces_header_only_csv(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        assert len(rows) == 1  # header only
        assert rows[0][0] == "Symbol"


class TestFilterAndSortPassthrough:
    """Verify filters and sort are passed to the feature store repository."""

    def test_passes_filters_to_repository(self):
        feature_store = TrackingFeatureStoreRepo()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = ExportScanResultsUseCase()

        filters = FilterSpec()
        filters.add_range("rs_rating", 70, None)
        uc.execute(uow, _make_query(filters=filters))

        # Use equality check (not identity) because use case copies filters
        passed_filters = feature_store.last_query_all_args["filters"]
        assert passed_filters.range_filters == filters.range_filters

    def test_passes_sort_to_repository(self):
        feature_store = TrackingFeatureStoreRepo()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = ExportScanResultsUseCase()

        sort = SortSpec(field="rs_rating", order=SortOrder.ASC)
        uc.execute(uow, _make_query(sort=sort))

        assert feature_store.last_query_all_args["sort"] is sort

    def test_passes_run_id_to_repository(self):
        feature_store = TrackingFeatureStoreRepo()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, scan_id="scan-xyz")
        uc = ExportScanResultsUseCase()

        uc.execute(uow, _make_query(scan_id="scan-xyz"))

        assert feature_store.last_query_all_args["run_id"] == 1


class TestPassesOnlyFilter:
    """passes_only=True adds a categorical filter for rating."""

    def test_passes_only_adds_rating_filter(self):
        feature_store = TrackingFeatureStoreRepo()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store)
        uc = ExportScanResultsUseCase()

        uc.execute(uow, _make_query(passes_only=True))

        filters = feature_store.last_query_all_args["filters"]
        cat_filters = filters.categorical_filters
        assert len(cat_filters) == 1
        assert cat_filters[0].field == "rating"
        assert set(cat_filters[0].values) == {"Strong Buy", "Buy"}


class TestUTF8BOM:
    """CSV content starts with UTF-8 BOM for Excel compatibility."""

    def test_content_starts_with_bom(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert result.content[:3] == b"\xef\xbb\xbf"


class TestValueFormatting:
    """Edge-case-safe value formatting for CSV cells."""

    def test_none_becomes_empty_string(self):
        assert _format_value(None) == ""

    def test_bool_true_becomes_yes(self):
        assert _format_value(True) == "Yes"

    def test_bool_false_becomes_no(self):
        assert _format_value(False) == "No"

    def test_float_rounded_to_2dp(self):
        assert _format_value(85.6789) == "85.68"

    def test_nan_becomes_empty_string(self):
        assert _format_value(float("nan")) == ""

    def test_inf_becomes_empty_string(self):
        assert _format_value(float("inf")) == ""

    def test_negative_inf_becomes_empty_string(self):
        assert _format_value(float("-inf")) == ""

    def test_list_joined_with_comma(self):
        assert _format_value(["minervini", "canslim"]) == "minervini, canslim"

    def test_integer_becomes_string(self):
        assert _format_value(42) == "42"

    def test_string_passthrough(self):
        assert _format_value("Technology") == "Technology"


class TestNaNHandlingInCSV:
    """Domain items with NaN scores produce empty cells."""

    def test_nan_score_in_csv(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", rs_rating=float("nan")),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        rs_idx = header.index("RS Rating")
        assert data_row[rs_idx] == ""


class TestBooleanFormattingInCSV:
    """Boolean fields display as Yes/No in CSV."""

    def test_passes_template_true_shows_yes(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", passes_template=True),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        pt_idx = header.index("Passes Template")
        assert data_row[pt_idx] == "Yes"

    def test_passes_template_false_shows_no(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", passes_template=False),
        ])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        pt_idx = header.index("Passes Template")
        assert data_row[pt_idx] == "No"


class TestFilename:
    """Filename contains scan metadata and timestamp."""

    def test_filename_contains_scan_id_prefix(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert result.filename.startswith("scan_")
        assert result.filename.endswith(".csv")

    def test_filename_truncates_long_scan_id(self):
        long_id = "abcdefghijklmnopqrstuvwxyz-1234567890"
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, scan_id=long_id, rows=[])
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query(scan_id=long_id))

        # scan_ + 12 chars + _ + timestamp + .csv
        assert "abcdefghijkl" in result.filename
        assert long_id not in result.filename


class TestDefaultQueryValues:
    """Default query uses sensible defaults."""

    def test_default_sort_is_composite_score_desc(self):
        query = ExportScanResultsQuery(scan_id="scan-123")
        assert query.sort.field == "composite_score"
        assert query.sort.order == SortOrder.DESC

    def test_default_format_is_csv(self):
        query = ExportScanResultsQuery(scan_id="scan-123")
        assert query.export_format == ExportFormat.CSV

    def test_default_passes_only_is_false(self):
        query = ExportScanResultsQuery(scan_id="scan-123")
        assert query.passes_only is False
