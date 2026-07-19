"""Unit tests for GetFilterOptionsUseCase — pure in-memory, no infrastructure."""

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.domain.scanning.models import FilterOptions
from app.use_cases.scanning.get_filter_options import (
    GetFilterOptionsQuery,
    GetFilterOptionsResult,
    GetFilterOptionsUseCase,
)

from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeUnitOfWork,
)


# ── Constants ────────────────────────────────────────────────────────────

AS_OF = date(2026, 2, 17)


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


def _make_query(**overrides) -> GetFilterOptionsQuery:
    defaults = dict(scan_id="scan-123")
    defaults.update(overrides)
    return GetFilterOptionsQuery(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for retrieving filter options."""

    def test_returns_filter_options(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row(
                "AAPL",
                ibd_industry_group="Electronics",
                gics_sector="Information Technology",
            ),
            _make_feature_row(
                "MSFT",
                ibd_industry_group="Software",
                gics_sector="Information Technology",
            ),
            _make_feature_row(
                "JNJ",
                ibd_industry_group="Electronics",
                gics_sector="Healthcare",
            ),
        ])
        uc = GetFilterOptionsUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, GetFilterOptionsResult)
        assert set(result.options.ibd_industries) == {"Electronics", "Software"}
        assert set(result.options.gics_sectors) == {"Information Technology", "Healthcare"}

    def test_passes_scan_id_correctly(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-xyz", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [
            _make_feature_row("AAPL", ibd_industry_group="Electronics"),
        ])
        uc = GetFilterOptionsUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-xyz"))

        assert "Electronics" in result.options.ibd_industries

    def test_empty_options(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        # Insert rows with no classification fields
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("AAPL")])
        uc = GetFilterOptionsUseCase()

        result = uc.execute(uow, _make_query())

        assert result.options.ibd_industries == ()
        assert result.options.gics_sectors == ()


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
