"""Unit tests for GetFilterOptionsUseCase — pure in-memory, no infrastructure."""

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.scanning.models import FilterOptions
from app.domain.scanning.ports import ScanResultRepository
from app.use_cases.scanning.get_filter_options import (
    GetFilterOptionsQuery,
    GetFilterOptionsResult,
    GetFilterOptionsUseCase,
)

from tests.unit.scanning_fakes import (
    FakeScanResultRepository,
    FakeUnitOfWork,
    setup_scan,
)


# ── Specialised fake ────────────────────────────────────────────────────


class FilterOptionsScanResultRepo(FakeScanResultRepository):
    """Fake that returns canned FilterOptions."""

    def __init__(self, options: FilterOptions | None = None):
        self._options = options or FilterOptions(
            ibd_industries=(), gics_sectors=(), ratings=()
        )
        self.last_filter_scan_id: str | None = None

    def get_filter_options(self, scan_id: str) -> FilterOptions:
        self.last_filter_scan_id = scan_id
        return self._options


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_query(**overrides) -> GetFilterOptionsQuery:
    defaults = dict(scan_id="scan-123")
    defaults.update(overrides)
    return GetFilterOptionsQuery(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for retrieving filter options."""

    def test_returns_filter_options(self):
        options = FilterOptions(
            ibd_industries=("Electronics", "Software"),
            gics_sectors=("Information Technology", "Healthcare"),
            ratings=("Buy", "Strong Buy"),
        )
        uow = FakeUnitOfWork(scan_results=FilterOptionsScanResultRepo(options))
        setup_scan(uow)
        uc = GetFilterOptionsUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, GetFilterOptionsResult)
        assert result.options is options
        assert result.options.ibd_industries == ("Electronics", "Software")
        assert result.options.gics_sectors == ("Information Technology", "Healthcare")
        assert result.options.ratings == ("Buy", "Strong Buy")

    def test_passes_scan_id_to_repository(self):
        repo = FilterOptionsScanResultRepo()
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow, "scan-xyz")
        uc = GetFilterOptionsUseCase()

        uc.execute(uow, _make_query(scan_id="scan-xyz"))

        assert repo.last_filter_scan_id == "scan-xyz"

    def test_empty_options(self):
        options = FilterOptions(
            ibd_industries=(), gics_sectors=(), ratings=()
        )
        uow = FakeUnitOfWork(scan_results=FilterOptionsScanResultRepo(options))
        setup_scan(uow)
        uc = GetFilterOptionsUseCase()

        result = uc.execute(uow, _make_query())

        assert result.options.ibd_industries == ()
        assert result.options.gics_sectors == ()
        assert result.options.ratings == ()


class TestScanNotFound:
    """Use case raises EntityNotFoundError for missing scans."""

    def test_nonexistent_scan_raises_not_found(self):
        uow = FakeUnitOfWork()
        uc = GetFilterOptionsUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*not-a-scan"):
            uc.execute(uow, _make_query(scan_id="not-a-scan"))

    def test_not_found_error_has_entity_and_identifier(self):
        uow = FakeUnitOfWork()
        uc = GetFilterOptionsUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(scan_id="missing"))

        assert exc_info.value.entity == "Scan"
        assert exc_info.value.identifier == "missing"
