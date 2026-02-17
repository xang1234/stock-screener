"""Unit tests for ExportScanResultsUseCase — pure in-memory, no infrastructure."""

import csv
import io
import math

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.scanning.filter_spec import FilterSpec, SortOrder, SortSpec
from app.domain.scanning.models import ExportFormat
from app.use_cases.scanning.export_scan_results import (
    ExportScanResultsQuery,
    ExportScanResultsResult,
    ExportScanResultsUseCase,
    _format_value,
)

from tests.unit.scanning_fakes import (
    FakeScanResultRepository,
    FakeUnitOfWork,
    make_domain_item,
    setup_scan,
)


# ── Specialised fake ────────────────────────────────────────────────────


class ExportableScanResultRepo(FakeScanResultRepository):
    """Fake that stores items and returns them from query_all()."""

    def __init__(self, items=None):
        self._items = tuple(items or [])
        self.last_query_all_args: dict | None = None

    def query_all(self, scan_id, filters, sort, *, include_sparklines=False):
        self.last_query_all_args = {
            "scan_id": scan_id,
            "filters": filters,
            "sort": sort,
            "include_sparklines": include_sparklines,
        }
        return self._items


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_query(**overrides) -> ExportScanResultsQuery:
    defaults = dict(scan_id="scan-123")
    defaults.update(overrides)
    return ExportScanResultsQuery(**defaults)


def _parse_csv_bytes(content: bytes) -> list[list[str]]:
    """Parse CSV bytes (skipping BOM) into rows."""
    # Strip UTF-8 BOM
    text = content.lstrip(b"\xef\xbb\xbf").decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    return list(reader)


# ── Tests ────────────────────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for exporting scan results."""

    def test_returns_export_result(self):
        items = [make_domain_item("AAPL"), make_domain_item("MSFT")]
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo(items))
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, ExportScanResultsResult)
        assert isinstance(result.content, bytes)
        assert result.media_type == "text/csv"
        assert result.filename.endswith(".csv")

    def test_csv_has_correct_row_count(self):
        items = [make_domain_item("AAPL"), make_domain_item("MSFT")]
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo(items))
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        # 1 header + 2 data rows
        assert len(rows) == 3

    def test_csv_header_matches_data_column_count(self):
        items = [make_domain_item("AAPL", score=90.0, rs_rating=88.5)]
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo(items))
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        assert len(header) == len(data_row)

    def test_csv_contains_symbol_data(self):
        items = [make_domain_item("NVDA", score=92.0)]
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo(items))
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        symbol_idx = header.index("Symbol")
        assert data_row[symbol_idx] == "NVDA"

    def test_csv_contains_company_name(self):
        items = [make_domain_item("AAPL", company_name="Apple Inc.")]
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo(items))
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        name_idx = header.index("Company Name")
        assert data_row[name_idx] == "Apple Inc."


class TestScanNotFound:
    """Use case raises EntityNotFoundError for missing scans."""

    def test_nonexistent_scan_raises_not_found(self):
        uow = FakeUnitOfWork()
        uc = ExportScanResultsUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*not-a-scan"):
            uc.execute(uow, _make_query(scan_id="not-a-scan"))


class TestEmptyResults:
    """Empty scan still produces a valid CSV with headers only."""

    def test_empty_results_produces_header_only_csv(self):
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo([]))
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        assert len(rows) == 1  # header only
        assert rows[0][0] == "Symbol"


class TestFilterAndSortPassthrough:
    """Verify filters and sort are passed to the repository."""

    def test_passes_filters_to_repository(self):
        repo = ExportableScanResultRepo()
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        filters = FilterSpec()
        filters.add_range("rs_rating", 70, None)
        uc.execute(uow, _make_query(filters=filters))

        assert repo.last_query_all_args["filters"] is filters

    def test_passes_sort_to_repository(self):
        repo = ExportableScanResultRepo()
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        sort = SortSpec(field="rs_rating", order=SortOrder.ASC)
        uc.execute(uow, _make_query(sort=sort))

        assert repo.last_query_all_args["sort"] is sort

    def test_passes_scan_id_to_repository(self):
        repo = ExportableScanResultRepo()
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow, "scan-xyz")
        uc = ExportScanResultsUseCase()

        uc.execute(uow, _make_query(scan_id="scan-xyz"))

        assert repo.last_query_all_args["scan_id"] == "scan-xyz"


class TestPassesOnlyFilter:
    """passes_only=True adds a categorical filter for rating."""

    def test_passes_only_adds_rating_filter(self):
        repo = ExportableScanResultRepo()
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        uc.execute(uow, _make_query(passes_only=True))

        filters = repo.last_query_all_args["filters"]
        cat_filters = filters.categorical_filters
        assert len(cat_filters) == 1
        assert cat_filters[0].field == "rating"
        assert set(cat_filters[0].values) == {"Strong Buy", "Buy"}


class TestUTF8BOM:
    """CSV content starts with UTF-8 BOM for Excel compatibility."""

    def test_content_starts_with_bom(self):
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo([]))
        setup_scan(uow)
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
        items = [make_domain_item("AAPL", score=85.0, rs_rating=float("nan"))]
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo(items))
        setup_scan(uow)
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
        items = [make_domain_item("AAPL", passes_template=True)]
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo(items))
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())
        rows = _parse_csv_bytes(result.content)

        header = rows[0]
        data_row = rows[1]
        pt_idx = header.index("Passes Template")
        assert data_row[pt_idx] == "Yes"

    def test_passes_template_false_shows_no(self):
        items = [make_domain_item("AAPL", passes_template=False)]
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo(items))
        setup_scan(uow)
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
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo([]))
        setup_scan(uow)
        uc = ExportScanResultsUseCase()

        result = uc.execute(uow, _make_query())

        assert result.filename.startswith("scan_")
        assert result.filename.endswith(".csv")

    def test_filename_truncates_long_scan_id(self):
        long_id = "abcdefghijklmnopqrstuvwxyz-1234567890"
        uow = FakeUnitOfWork(scan_results=ExportableScanResultRepo([]))
        setup_scan(uow, long_id)
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
