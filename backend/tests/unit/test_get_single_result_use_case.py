"""Unit tests for GetSingleResultUseCase and _domain_to_response mapper.

Pure in-memory tests — no infrastructure.
"""

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.domain.scanning.models import ScanResultItemDomain
from app.use_cases.scanning.get_single_result import (
    GetSingleResultQuery,
    GetSingleResultResult,
    GetSingleResultUseCase,
)

from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeUnitOfWork,
    make_domain_item,
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


def _setup_bound_scan(uow, feature_store, scan_id="scan-123", symbols=None):
    """Create a scan bound to feature_run_id=1 and populate feature store."""
    uow.scans.create(scan_id=scan_id, status="completed", feature_run_id=1)
    if symbols:
        rows = [_make_feature_row(s) for s in symbols]
        feature_store.upsert_snapshot_rows(1, rows)


def _make_query(**overrides) -> GetSingleResultQuery:
    defaults = dict(scan_id="scan-123", symbol="AAPL")
    defaults.update(overrides)
    return GetSingleResultQuery(**defaults)


# ── Tests: Use Case ─────────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for single-result lookup."""

    def test_returns_correct_item(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("AAPL", score=92.0)])
        uc = GetSingleResultUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, GetSingleResultResult)
        assert result.item.symbol == "AAPL"
        assert result.item.composite_score == 92.0

    def test_passes_correct_scan_id(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-xyz", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("TSLA")])
        uc = GetSingleResultUseCase()

        result = uc.execute(uow, _make_query(scan_id="scan-xyz", symbol="TSLA"))

        assert result.item.symbol == "TSLA"


class TestScanNotFound:
    """Use case raises EntityNotFoundError for missing scans."""

    def test_nonexistent_scan_raises_not_found(self):
        uow = FakeUnitOfWork()
        uc = GetSingleResultUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*not-a-scan"):
            uc.execute(uow, _make_query(scan_id="not-a-scan"))

    def test_not_found_error_has_entity_and_identifier(self):
        uow = FakeUnitOfWork()
        uc = GetSingleResultUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(scan_id="missing"))

        assert exc_info.value.entity == "Scan"
        assert exc_info.value.identifier == "missing"


class TestUnboundScanFallsBackToScanResults:
    """Scan exists but has no feature_run_id — falls back to scan_results lookup."""

    def test_unbound_scan_without_row_raises_scan_result_not_found(self):
        uow = FakeUnitOfWork()
        uow.scans.create(scan_id="scan-123", status="completed")  # no feature_run_id
        uc = GetSingleResultUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query())

        assert exc_info.value.entity == "ScanResult"


class TestResultNotFound:
    """Use case raises EntityNotFoundError when symbol is not in the scan."""

    def test_missing_symbol_raises_not_found(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        # Feature store has run_id=1 but no AAPL row
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("MSFT")])
        uc = GetSingleResultUseCase()

        with pytest.raises(EntityNotFoundError, match="ScanResult.*AAPL"):
            uc.execute(uow, _make_query(symbol="AAPL"))

    def test_not_found_error_has_scan_result_entity(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("MSFT")])
        uc = GetSingleResultUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(symbol="NOPE"))

        assert exc_info.value.entity == "ScanResult"
        assert exc_info.value.identifier == "NOPE"


class TestSymbolNormalization:
    """Lowercase input is normalised to uppercase before querying."""

    def test_lowercase_symbol_normalised(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("AAPL")])
        uc = GetSingleResultUseCase()

        result = uc.execute(uow, _make_query(symbol="aapl"))

        assert result.item.symbol == "AAPL"

    def test_mixed_case_symbol_normalised(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        uow.scans.create(scan_id="scan-123", status="completed", feature_run_id=1)
        feature_store.upsert_snapshot_rows(1, [_make_feature_row("MSFT")])
        uc = GetSingleResultUseCase()

        result = uc.execute(uow, _make_query(symbol="MsFt"))

        assert result.item.symbol == "MSFT"


# ── Tests: _domain_to_response mapper ───────────────────────────────────


class TestDomainToResponse:
    """Verify the domain->HTTP mapper unpacks all extended_fields correctly."""

    def test_all_fields_mapped(self):
        from app.schemas.scanning import ScanResultItem

        item = ScanResultItemDomain(
            symbol="TEST",
            composite_score=88.5,
            rating="Strong Buy",
            current_price=250.0,
            screener_outputs={},
            screeners_run=["minervini", "canslim"],
            composite_method="weighted_average",
            screeners_passed=2,
            screeners_total=2,
            extended_fields={
                "company_name": "Test Corp",
                "minervini_score": 90.0,
                "canslim_score": 85.0,
                "ipo_score": 70.0,
                "custom_score": 60.0,
                "volume_breakthrough_score": 55.0,
                "rs_rating": 95.0,
                "rs_rating_1m": 92.0,
                "rs_rating_3m": 88.0,
                "rs_rating_12m": 80.0,
                "stage": 2,
                "stage_name": "Advancing",
                "volume": 1_000_000.0,
                "market_cap": 50_000_000_000.0,
                "ma_alignment": True,
                "vcp_detected": True,
                "vcp_score": 8.5,
                "vcp_pivot": 255.0,
                "vcp_ready_for_breakout": False,
                "vcp_contraction_ratio": 0.35,
                "vcp_atr_score": 7.2,
                "passes_template": True,
                "adr_percent": 3.5,
                "eps_growth_qq": 25.0,
                "sales_growth_qq": 30.0,
                "eps_growth_yy": 20.0,
                "sales_growth_yy": 18.0,
                "peg_ratio": 1.2,
                "eps_rating": 95,
                "ibd_industry_group": "Semiconductor",
                "ibd_group_rank": 5,
                "market_themes": ["AI Infrastructure", "Foundry"],
                "gics_sector": "Information Technology",
                "gics_industry": "Semiconductors",
                "rs_sparkline_data": [1.0, 1.1, 1.2],
                "rs_trend": 1,
                "price_sparkline_data": [100.0, 105.0, 110.0],
                "price_change_1d": 2.5,
                "price_trend": 1,
                "ipo_date": "2015-06-15",
                "beta": 1.3,
                "beta_adj_rs": 90.0,
                "beta_adj_rs_1m": 88.0,
                "beta_adj_rs_3m": 85.0,
                "beta_adj_rs_12m": 82.0,
            },
        )

        resp = ScanResultItem.from_domain(item)

        # Core fields from domain item directly
        assert resp.symbol == "TEST"
        assert resp.composite_score == 88.5
        assert resp.rating == "Strong Buy"
        assert resp.current_price == 250.0
        assert resp.screeners_run == ["minervini", "canslim"]

        # All extended_fields should be non-None
        assert resp.company_name == "Test Corp"
        assert resp.minervini_score == 90.0
        assert resp.canslim_score == 85.0
        assert resp.ipo_score == 70.0
        assert resp.custom_score == 60.0
        assert resp.volume_breakthrough_score == 55.0
        assert resp.rs_rating == 95.0
        assert resp.rs_rating_1m == 92.0
        assert resp.rs_rating_3m == 88.0
        assert resp.rs_rating_12m == 80.0
        assert resp.stage == 2
        assert resp.stage_name == "Advancing"
        assert resp.volume == 1_000_000.0
        assert resp.market_cap == 50_000_000_000.0
        assert resp.ma_alignment is True
        assert resp.vcp_detected is True
        assert resp.vcp_score == 8.5
        assert resp.vcp_pivot == 255.0
        assert resp.vcp_ready_for_breakout is False
        assert resp.vcp_contraction_ratio == 0.35
        assert resp.vcp_atr_score == 7.2
        assert resp.passes_template is True
        assert resp.adr_percent == 3.5
        assert resp.eps_growth_qq == 25.0
        assert resp.sales_growth_qq == 30.0
        assert resp.eps_growth_yy == 20.0
        assert resp.sales_growth_yy == 18.0
        assert resp.peg_ratio == 1.2
        assert resp.eps_rating == 95
        assert resp.ibd_industry_group == "Semiconductor"
        assert resp.ibd_group_rank == 5
        assert resp.market_themes == ["AI Infrastructure", "Foundry"]
        assert resp.gics_sector == "Information Technology"
        assert resp.gics_industry == "Semiconductors"
        assert resp.rs_sparkline_data == [1.0, 1.1, 1.2]
        assert resp.rs_trend == 1
        assert resp.price_sparkline_data == [100.0, 105.0, 110.0]
        assert resp.price_change_1d == 2.5
        assert resp.price_trend == 1
        assert resp.ipo_date == "2015-06-15"
        assert resp.beta == 1.3
        assert resp.beta_adj_rs == 90.0
        assert resp.beta_adj_rs_1m == 88.0
        assert resp.beta_adj_rs_3m == 85.0
        assert resp.beta_adj_rs_12m == 82.0

    def test_missing_extended_fields_default_to_none(self):
        from app.schemas.scanning import ScanResultItem

        item = make_domain_item("BARE")
        resp = ScanResultItem.from_domain(item)

        assert resp.symbol == "BARE"
        assert resp.company_name == "BARE Inc"
        # All screener-specific fields should be None
        assert resp.minervini_score is None
        assert resp.canslim_score is None
        assert resp.rs_sparkline_data is None
        assert resp.beta is None
