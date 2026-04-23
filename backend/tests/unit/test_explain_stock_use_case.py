"""Unit tests for ExplainStockUseCase.

Pure in-memory tests — no infrastructure.
"""

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.feature_store.models import FeatureRow
from app.domain.scanning.models import StockExplanation
from app.domain.scanning.scoring import (
    BUY_THRESHOLD,
    STRONG_BUY_THRESHOLD,
    WATCH_THRESHOLD,
)
from app.use_cases.scanning.explain_stock import (
    ExplainStockQuery,
    ExplainStockResult,
    ExplainStockUseCase,
    _normalize_composite_score,
)

from tests.unit.use_cases.conftest import (
    FakeFeatureStoreRepository,
    FakeUnitOfWork,
)


# ── Constants ────────────────────────────────────────────────────────────

AS_OF = date(2026, 2, 17)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_minervini_screener_data(score=75.0):
    """Build a Minervini screener data dict for FeatureRow details."""
    return {
        "score": score,
        "passes": score >= 60,
        "rating": "Buy",
        "breakdown": {
            "rs_rating": {"points": 18.0, "max_points": 20, "value": 88, "passes": True},
            "stage": {"points": 20.0, "max_points": 20, "value": 2, "passes": True},
            "ma_alignment": {"points": 12.0, "max_points": 15, "value": 80, "passes": True},
            "position_52w": {"points": 10.0, "max_points": 15, "passes": False},
            "vcp": {"points": 0.0, "max_points": 20, "value": 0, "passes": False},
        },
        "details": {"rs_value": 88},
    }


def _make_canslim_screener_data(score=80.0):
    """Build a CANSLIM screener data dict for FeatureRow details."""
    return {
        "score": score,
        "passes": True,
        "rating": "Strong Buy",
        "breakdown": {
            "current_earnings": 20.0,
            "annual_earnings": 15.0,
            "new_highs": 10.0,
            "supply_demand": 12.0,
            "leader": 18.0,
            "institutional": 5.0,
        },
        "details": {},
    }


def _make_feature_row(
    symbol="AAPL",
    composite_score=85.0,
    screeners=None,
):
    """Build a FeatureRow with screener data embedded in details."""
    screeners = screeners or {}
    details = {
        "screeners_run": list(screeners.keys()),
        "composite_method": "weighted_average",
        "screeners_passed": sum(1 for s in screeners.values() if s.get("passes", False)),
        "screeners_total": len(screeners),
        "current_price": 150.0,
        "details": {"screeners": screeners},
    }
    # 5 = Strong Buy, 4 = Buy
    overall_rating = 5 if composite_score >= 80 else 4
    return FeatureRow(
        run_id=1,
        symbol=symbol,
        as_of_date=AS_OF,
        composite_score=composite_score,
        overall_rating=overall_rating,
        passes_count=sum(1 for s in screeners.values() if s.get("passes", False)),
        details=details,
    )


def _make_query(**overrides) -> ExplainStockQuery:
    defaults = dict(scan_id="scan-123", symbol="AAPL")
    defaults.update(overrides)
    return ExplainStockQuery(**defaults)


def _setup_bound_scan(uow, feature_store, scan_id="scan-123", run_id=1, rows=None):
    """Create a scan bound to a feature run, with rows in the feature store."""
    uow.scans.create(scan_id=scan_id, status="completed", feature_run_id=run_id)
    if rows:
        feature_store.upsert_snapshot_rows(run_id, rows)
    else:
        feature_store.upsert_snapshot_rows(run_id, [])


# ── Tests: Happy Path ──────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for explaining a stock's score."""

    def test_returns_explanation_with_correct_structure(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 85.0, {"minervini": _make_minervini_screener_data()}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        assert isinstance(result, ExplainStockResult)
        assert isinstance(result.explanation, StockExplanation)
        assert result.explanation.symbol == "AAPL"
        assert result.explanation.composite_score == 85.0
        assert result.explanation.rating == "Strong Buy"
        assert result.explanation.composite_method == "weighted_average"
        assert result.explanation.screeners_passed == 1
        assert result.explanation.screeners_total == 1

    def test_criteria_count_matches_breakdown(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 85.0, {"minervini": _make_minervini_screener_data()}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        screener_exp = result.explanation.screener_explanations[0]
        assert len(screener_exp.criteria) == 5  # minervini has 5 breakdown keys

    def test_max_scores_from_nested_breakdown(self):
        """Minervini nested dicts contain max_points directly."""
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 85.0, {"minervini": _make_minervini_screener_data()}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        criteria_by_name = {
            c.name: c for c in result.explanation.screener_explanations[0].criteria
        }
        assert criteria_by_name["rs_rating"].max_score == 20
        assert criteria_by_name["stage"].max_score == 20
        assert criteria_by_name["ma_alignment"].max_score == 15
        assert criteria_by_name["position_52w"].max_score == 15
        assert criteria_by_name["vcp"].max_score == 20

    def test_max_scores_from_flat_breakdown_use_lookup(self):
        """CANSLIM uses flat breakdowns, so max_score comes from _MAX_SCORES lookup."""
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 80.0, {"canslim": _make_canslim_screener_data()}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        criteria_by_name = {
            c.name: c for c in result.explanation.screener_explanations[0].criteria
        }
        assert criteria_by_name["current_earnings"].max_score == 20
        assert criteria_by_name["annual_earnings"].max_score == 15
        assert criteria_by_name["leader"].max_score == 20
        assert criteria_by_name["institutional"].max_score == 15

    def test_passed_flag_from_nested_breakdown(self):
        """Minervini uses nested dicts with explicit 'passes' booleans."""
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 85.0, {"minervini": _make_minervini_screener_data()}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        criteria_by_name = {
            c.name: c for c in result.explanation.screener_explanations[0].criteria
        }
        # rs_rating passes=True in breakdown, vcp passes=False in breakdown
        assert criteria_by_name["rs_rating"].passed is True
        assert criteria_by_name["vcp"].passed is False
        # position_52w has points=10.0 (>0) but passes=False in breakdown
        assert criteria_by_name["position_52w"].passed is False

    def test_rating_thresholds_included(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 85.0, {"minervini": _make_minervini_screener_data()}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        thresholds = result.explanation.rating_thresholds
        assert thresholds["Strong Buy"] == STRONG_BUY_THRESHOLD
        assert thresholds["Buy"] == BUY_THRESHOLD
        assert thresholds["Watch"] == WATCH_THRESHOLD
        assert thresholds["Pass"] == 0.0


class TestCompositeScoreNormalization:
    def test_nan_is_treated_as_unscored(self):
        assert _normalize_composite_score(float("nan")) is None

    def test_infinity_is_treated_as_unscored(self):
        assert _normalize_composite_score(float("inf")) is None


# ── Tests: Scan Not Found ──────────────────────────────────────────────


class TestScanNotFound:
    """Use case raises EntityNotFoundError for missing scans."""

    def test_nonexistent_scan_raises_not_found(self):
        uow = FakeUnitOfWork()
        uc = ExplainStockUseCase()

        with pytest.raises(EntityNotFoundError, match="Scan.*not-a-scan"):
            uc.execute(uow, _make_query(scan_id="not-a-scan"))

    def test_not_found_error_has_entity_and_identifier(self):
        uow = FakeUnitOfWork()
        uc = ExplainStockUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(scan_id="missing"))

        assert exc_info.value.entity == "Scan"
        assert exc_info.value.identifier == "missing"


# ── Tests: Unbound Scan ───────────────────────────────────────────────


class TestUnboundScanRaises:
    """Scan exists but has no feature_run_id — falls back to legacy scan_results.

    When the legacy path also has no data for the symbol, an
    EntityNotFoundError("ScanResult", ...) is raised.
    """

    def test_unbound_scan_falls_back_to_legacy_and_raises_scan_result_not_found(self):
        uow = FakeUnitOfWork()
        uow.scans.create(scan_id="scan-123", status="completed")  # no feature_run_id
        uc = ExplainStockUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query())

        # Use case falls back to scan_results (dual-source), symbol not found there either
        assert exc_info.value.entity == "ScanResult"
        assert exc_info.value.identifier == "AAPL"


# ── Tests: Result Not Found ────────────────────────────────────────────


class TestResultNotFound:
    """Use case raises EntityNotFoundError when symbol is not in the scan."""

    def test_missing_symbol_raises_not_found(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[])
        uc = ExplainStockUseCase()

        with pytest.raises(EntityNotFoundError, match="ScanResult.*AAPL"):
            uc.execute(uow, _make_query(symbol="AAPL"))

    def test_not_found_error_has_scan_result_entity(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[])
        uc = ExplainStockUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(symbol="NOPE"))

        assert exc_info.value.entity == "ScanResult"
        assert exc_info.value.identifier == "NOPE"


# ── Tests: Symbol Normalization ────────────────────────────────────────


class TestSymbolNormalization:
    """Lowercase input is normalised to uppercase before querying."""

    def test_lowercase_symbol_normalised(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 85.0, {"minervini": _make_minervini_screener_data()}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query(symbol="aapl"))

        assert result.explanation.symbol == "AAPL"

    def test_mixed_case_symbol_normalised(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("MSFT", 75.0, {"minervini": _make_minervini_screener_data()}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query(symbol="MsFt"))

        assert result.explanation.symbol == "MSFT"


# ── Tests: No Screener Outputs ─────────────────────────────────────────


class TestNoScreenerOutputs:
    """Stock with no screener outputs yields empty explanation tuple."""

    def test_empty_screener_outputs_gives_empty_explanations(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 0.0, screeners={}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        assert result.explanation.screener_explanations == ()
        assert result.explanation.screeners_total == 0


# ── Tests: Unknown Screener ────────────────────────────────────────────


class TestUnknownScreener:
    """Screener not in _MAX_SCORES gets max_score=0.0 for all criteria."""

    def test_unknown_screener_criteria_have_zero_max_score(self):
        custom_screener_data = {
            "score": 65.0,
            "passes": True,
            "rating": "Buy",
            "breakdown": {"pe_ratio": 10.0, "market_cap": 5.0},
            "details": {},
        }
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 65.0, {"custom": custom_screener_data}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        screener_exp = result.explanation.screener_explanations[0]
        assert screener_exp.screener_name == "custom"
        for criterion in screener_exp.criteria:
            assert criterion.max_score == 0.0

    def test_volume_breakthrough_modifier_keys_have_zero_max_for_unknown(self):
        """Non-criteria keys like bonus_points get max_score=0.0."""
        vb_screener_data = {
            "score": 70.0,
            "passes": True,
            "rating": "Buy",
            "breakdown": {
                "five_year_high": 40.0,
                "one_year_high": 30.0,
                "since_ipo_high": 20.0,
                "bonus_points": 5.0,
                "decay_multiplier": 0.8,
            },
            "details": {},
        }
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 70.0, {"volume_breakthrough": vb_screener_data}),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        criteria_by_name = {
            c.name: c for c in result.explanation.screener_explanations[0].criteria
        }
        # Known criteria get their max_scores
        assert criteria_by_name["five_year_high"].max_score == 43
        assert criteria_by_name["one_year_high"].max_score == 43
        # Modifier keys not in _MAX_SCORES get 0.0
        assert criteria_by_name["bonus_points"].max_score == 0.0
        assert criteria_by_name["decay_multiplier"].max_score == 0.0


# ── Tests: Multiple Screeners ──────────────────────────────────────────


class TestMultipleScreeners:
    """Verify multiple screener explanations are produced correctly."""

    def test_two_screeners_produce_two_explanations(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 82.5, {
                "minervini": _make_minervini_screener_data(75.0),
                "canslim": _make_canslim_screener_data(80.0),
            }),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        assert len(result.explanation.screener_explanations) == 2
        names = {se.screener_name for se in result.explanation.screener_explanations}
        assert names == {"minervini", "canslim"}

    def test_each_screener_has_own_score_and_criteria(self):
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(feature_store=feature_store)
        _setup_bound_scan(uow, feature_store, rows=[
            _make_feature_row("AAPL", 82.5, {
                "minervini": _make_minervini_screener_data(75.0),
                "canslim": _make_canslim_screener_data(80.0),
            }),
        ])
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        by_name = {
            se.screener_name: se for se in result.explanation.screener_explanations
        }
        assert by_name["minervini"].score == 75.0
        assert by_name["canslim"].score == 80.0
        assert len(by_name["minervini"].criteria) == 5
        assert len(by_name["canslim"].criteria) == 6
