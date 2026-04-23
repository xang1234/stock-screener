"""Tests for ExplainStockUseCase — feature store and scan_results fallback."""

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.use_cases.scanning.explain_stock import (
    ExplainStockQuery,
    ExplainStockUseCase,
)
from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeScanResultRepository,
    FakeUnitOfWork,
    make_domain_item,
)


class TestExplainStockHappyPath:
    """Tests for the feature-store-backed explain path."""

    def test_feature_store_path_extracts_screeners(self):
        """When scan has feature_run_id, explain reads from feature store."""
        store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=store)

        # Create scan with feature_run_id
        uow.scans.create(scan_id="scan-fs", status="completed", feature_run_id=42)

        # Populate feature store with screener outputs in details
        store._rows[42] = [
            FeatureRow(
                run_id=42,
                symbol="NVDA",
                as_of_date=date(2026, 2, 18),
                composite_score=82.0,
                overall_rating=4,
                passes_count=2,
                details={
                    "screeners_run": ["minervini", "canslim"],
                    "composite_method": "weighted_average",
                    "screeners_passed": 2,
                    "screeners_total": 2,
                    "current_price": 850.0,
                    "details": {
                        "screeners": {
                            "minervini": {
                                "score": 85.0,
                                "passes": True,
                                "rating": "Strong Buy",
                                "breakdown": {"rs_rating": 20, "stage": 20},
                                "details": {},
                            },
                            "canslim": {
                                "score": 70.0,
                                "passes": True,
                                "rating": "Buy",
                                "breakdown": {"current_earnings": 18},
                                "details": {},
                            },
                        }
                    },
                },
            ),
        ]

        uc = ExplainStockUseCase()
        result = uc.execute(uow, ExplainStockQuery(scan_id="scan-fs", symbol="NVDA"))

        assert result.explanation.symbol == "NVDA"
        assert result.explanation.composite_score == 82.0
        assert len(result.explanation.screener_explanations) == 2

        names = {se.screener_name for se in result.explanation.screener_explanations}
        assert names == {"minervini", "canslim"}

    def test_case_insensitive_symbol_lookup(self):
        """Symbol matching should be case-insensitive."""
        store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=store)
        uow.scans.create(scan_id="scan-ci", status="completed", feature_run_id=42)

        store._rows[42] = [
            FeatureRow(
                run_id=42, symbol="AAPL", as_of_date=date(2026, 2, 18),
                composite_score=75.0, overall_rating=4, passes_count=1,
                details={
                    "screeners_run": ["minervini"],
                    "composite_method": "weighted_average",
                    "screeners_passed": 1,
                    "screeners_total": 1,
                    "details": {"screeners": {}},
                },
            ),
        ]

        uc = ExplainStockUseCase()
        # Query with lowercase
        result = uc.execute(uow, ExplainStockQuery(scan_id="scan-ci", symbol="aapl"))
        assert result.explanation.symbol == "AAPL"

    def test_feature_store_path_preserves_unscored_rows(self):
        store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=store)
        uow.scans.create(scan_id="scan-fs", status="completed", feature_run_id=42)

        store._rows[42] = [
            FeatureRow(
                run_id=42,
                symbol="MINI",
                as_of_date=date(2026, 4, 23),
                composite_score=None,
                overall_rating=None,
                passes_count=0,
                details={
                    "rating": "Insufficient Data",
                    "screeners_run": [],
                    "composite_method": "weighted_average",
                    "screeners_passed": 0,
                    "screeners_total": 0,
                    "current_price": 18.5,
                    "details": {"screeners": {}},
                },
            ),
        ]

        uc = ExplainStockUseCase()
        result = uc.execute(uow, ExplainStockQuery(scan_id="scan-fs", symbol="MINI"))

        assert result.explanation.symbol == "MINI"
        assert result.explanation.composite_score is None
        assert result.explanation.rating == "Insufficient Data"


class TestExplainStockErrors:
    """Error conditions for ExplainStockUseCase."""

    def test_scan_not_found_raises(self):
        uow = FakeUnitOfWork()
        uc = ExplainStockUseCase()
        with pytest.raises(EntityNotFoundError, match="Scan"):
            uc.execute(uow, ExplainStockQuery(scan_id="nope", symbol="AAPL"))

    def test_unbound_scan_falls_back_to_scan_results(self):
        """Scan without feature_run_id reads details from scan_results."""
        items = [make_domain_item("NVDA", score=82.0)]
        scan_results = FakeScanResultRepository(items=items)
        # Set a details map with screener breakdowns for the explain path
        scan_results._details_map = {
            "NVDA": {
                "composite_score": 82.0,
                "rating": "Buy",
                "current_price": 850.0,
                "screeners_run": ["minervini", "canslim"],
                "composite_method": "weighted_average",
                "screeners_passed": 2,
                "screeners_total": 2,
                "details": {
                    "screeners": {
                        "minervini": {
                            "score": 85.0,
                            "passes": True,
                            "rating": "Strong Buy",
                            "breakdown": {"rs_rating": 20, "stage": 20},
                            "details": {},
                        },
                        "canslim": {
                            "score": 70.0,
                            "passes": True,
                            "rating": "Buy",
                            "breakdown": {"current_earnings": 18},
                            "details": {},
                        },
                    }
                },
            },
        }
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = ExplainStockUseCase()

        result = uc.execute(uow, ExplainStockQuery(scan_id="scan-legacy", symbol="NVDA"))

        assert result.explanation.symbol == "NVDA"
        assert result.explanation.composite_score == 82.0
        assert len(result.explanation.screener_explanations) == 2
        names = {se.screener_name for se in result.explanation.screener_explanations}
        assert names == {"minervini", "canslim"}

    def test_unbound_scan_symbol_not_found_raises_error(self):
        """Fallback path raises EntityNotFoundError for missing symbol."""
        scan_results = FakeScanResultRepository(items=[])
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")
        uc = ExplainStockUseCase()

        with pytest.raises(EntityNotFoundError, match="ScanResult"):
            uc.execute(uow, ExplainStockQuery(scan_id="scan-legacy", symbol="GHOST"))

    def test_unbound_scan_preserves_unscored_details_blob(self):
        item = make_domain_item("MINI", score=None)
        scan_results = FakeScanResultRepository(items=[item])
        scan_results._details_map = {
            "MINI": {
                "composite_score": None,
                "rating": "Insufficient Data",
                "current_price": 12.0,
                "screeners_run": [],
                "composite_method": "weighted_average",
                "screeners_passed": 0,
                "screeners_total": 0,
                "details": {"screeners": {}},
            },
        }
        uow = FakeUnitOfWork(scan_results=scan_results)
        uow.scans.create(scan_id="scan-legacy", status="completed")

        uc = ExplainStockUseCase()
        result = uc.execute(uow, ExplainStockQuery(scan_id="scan-legacy", symbol="MINI"))

        assert result.explanation.composite_score is None
        assert result.explanation.rating == "Insufficient Data"

    def test_symbol_not_found_raises(self):
        store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=store)
        uow.scans.create(scan_id="scan-nf", status="completed", feature_run_id=42)
        store._rows[42] = []

        uc = ExplainStockUseCase()
        with pytest.raises(EntityNotFoundError, match="ScanResult"):
            uc.execute(uow, ExplainStockQuery(scan_id="scan-nf", symbol="GHOST"))
