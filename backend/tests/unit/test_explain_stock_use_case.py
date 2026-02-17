"""Unit tests for ExplainStockUseCase.

Pure in-memory tests — no infrastructure.
"""

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.scanning.models import (
    CriterionResult,
    ScreenerExplanation,
    ScreenerOutputDomain,
    ScanResultItemDomain,
    StockExplanation,
)
from app.domain.scanning.scoring import (
    BUY_THRESHOLD,
    STRONG_BUY_THRESHOLD,
    WATCH_THRESHOLD,
)
from app.use_cases.scanning.explain_stock import (
    ExplainStockQuery,
    ExplainStockResult,
    ExplainStockUseCase,
)

from tests.unit.scanning_fakes import (
    FakeScanResultRepository,
    FakeUnitOfWork,
    setup_scan,
)


# ── Specialised fake ────────────────────────────────────────────────────


class ExplainableResultRepo(FakeScanResultRepository):
    """Fake that supports get_by_symbol() with a dict of items."""

    def __init__(self, items_by_symbol: dict[str, ScanResultItemDomain] | None = None):
        self._items = items_by_symbol or {}

    def get_by_symbol(self, scan_id: str, symbol: str) -> ScanResultItemDomain | None:
        return self._items.get(symbol)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_query(**overrides) -> ExplainStockQuery:
    defaults = dict(scan_id="scan-123", symbol="AAPL")
    defaults.update(overrides)
    return ExplainStockQuery(**defaults)


def _make_item_with_outputs(
    symbol: str = "AAPL",
    composite_score: float = 85.0,
    screener_outputs: dict[str, ScreenerOutputDomain] | None = None,
) -> ScanResultItemDomain:
    """Build a ScanResultItemDomain with explicit screener_outputs."""
    outputs = screener_outputs or {}
    return ScanResultItemDomain(
        symbol=symbol,
        composite_score=composite_score,
        rating="Strong Buy" if composite_score >= 80 else "Buy",
        current_price=150.0,
        screener_outputs=outputs,
        screeners_run=list(outputs.keys()),
        composite_method="weighted_average",
        screeners_passed=sum(1 for o in outputs.values() if o.passes),
        screeners_total=len(outputs),
        extended_fields={"company_name": f"{symbol} Inc"},
    )


def _make_minervini_output(score: float = 75.0) -> ScreenerOutputDomain:
    return ScreenerOutputDomain(
        screener_name="minervini",
        score=score,
        passes=score >= 60,
        rating="Buy",
        breakdown={
            "rs_rating": 18.0,
            "stage": 20.0,
            "ma_alignment": 12.0,
            "position_52w": 10.0,
            "vcp": 0.0,
        },
        details={"rs_value": 88},
    )


def _make_canslim_output(score: float = 80.0) -> ScreenerOutputDomain:
    return ScreenerOutputDomain(
        screener_name="canslim",
        score=score,
        passes=True,
        rating="Strong Buy",
        breakdown={
            "current_earnings": 20.0,
            "annual_earnings": 15.0,
            "new_highs": 10.0,
            "supply_demand": 12.0,
            "leader": 18.0,
            "institutional": 5.0,
        },
        details={},
    )


# ── Tests: Happy Path ──────────────────────────────────────────────────


class TestHappyPath:
    """Core business logic for explaining a stock's score."""

    def test_returns_explanation_with_correct_structure(self):
        item = _make_item_with_outputs(
            "AAPL", 85.0, {"minervini": _make_minervini_output()}
        )
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
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
        item = _make_item_with_outputs(
            "AAPL", 85.0, {"minervini": _make_minervini_output()}
        )
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        screener_exp = result.explanation.screener_explanations[0]
        assert len(screener_exp.criteria) == 5  # minervini has 5 breakdown keys

    def test_max_scores_populated_for_known_screener(self):
        item = _make_item_with_outputs(
            "AAPL", 85.0, {"minervini": _make_minervini_output()}
        )
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
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

    def test_passed_flag_reflects_positive_score(self):
        item = _make_item_with_outputs(
            "AAPL", 85.0, {"minervini": _make_minervini_output()}
        )
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        criteria_by_name = {
            c.name: c for c in result.explanation.screener_explanations[0].criteria
        }
        # rs_rating=18.0 → passed=True, vcp=0.0 → passed=False
        assert criteria_by_name["rs_rating"].passed is True
        assert criteria_by_name["vcp"].passed is False

    def test_rating_thresholds_included(self):
        item = _make_item_with_outputs(
            "AAPL", 85.0, {"minervini": _make_minervini_output()}
        )
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        thresholds = result.explanation.rating_thresholds
        assert thresholds["Strong Buy"] == STRONG_BUY_THRESHOLD
        assert thresholds["Buy"] == BUY_THRESHOLD
        assert thresholds["Watch"] == WATCH_THRESHOLD
        assert thresholds["Pass"] == 0.0


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


# ── Tests: Result Not Found ────────────────────────────────────────────


class TestResultNotFound:
    """Use case raises EntityNotFoundError when symbol is not in the scan."""

    def test_missing_symbol_raises_not_found(self):
        repo = ExplainableResultRepo()  # empty — no items
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        with pytest.raises(EntityNotFoundError, match="ScanResult.*AAPL"):
            uc.execute(uow, _make_query(symbol="AAPL"))

    def test_not_found_error_has_scan_result_entity(self):
        repo = ExplainableResultRepo()
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        with pytest.raises(EntityNotFoundError) as exc_info:
            uc.execute(uow, _make_query(symbol="NOPE"))

        assert exc_info.value.entity == "ScanResult"
        assert exc_info.value.identifier == "NOPE"


# ── Tests: Symbol Normalization ────────────────────────────────────────


class TestSymbolNormalization:
    """Lowercase input is normalised to uppercase before querying."""

    def test_lowercase_symbol_normalised(self):
        item = _make_item_with_outputs("AAPL", 85.0, {"minervini": _make_minervini_output()})
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query(symbol="aapl"))

        assert result.explanation.symbol == "AAPL"

    def test_mixed_case_symbol_normalised(self):
        item = _make_item_with_outputs("MSFT", 75.0, {"minervini": _make_minervini_output()})
        repo = ExplainableResultRepo(items_by_symbol={"MSFT": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query(symbol="MsFt"))

        assert result.explanation.symbol == "MSFT"


# ── Tests: No Screener Outputs ─────────────────────────────────────────


class TestNoScreenerOutputs:
    """Stock with no screener outputs yields empty explanation tuple."""

    def test_empty_screener_outputs_gives_empty_explanations(self):
        item = _make_item_with_outputs("AAPL", 0.0, screener_outputs={})
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        assert result.explanation.screener_explanations == ()
        assert result.explanation.screeners_total == 0


# ── Tests: Unknown Screener ────────────────────────────────────────────


class TestUnknownScreener:
    """Screener not in _MAX_SCORES gets max_score=0.0 for all criteria."""

    def test_unknown_screener_criteria_have_zero_max_score(self):
        custom_output = ScreenerOutputDomain(
            screener_name="custom",
            score=65.0,
            passes=True,
            rating="Buy",
            breakdown={"pe_ratio": 10.0, "market_cap": 5.0},
            details={},
        )
        item = _make_item_with_outputs("AAPL", 65.0, {"custom": custom_output})
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        screener_exp = result.explanation.screener_explanations[0]
        assert screener_exp.screener_name == "custom"
        for criterion in screener_exp.criteria:
            assert criterion.max_score == 0.0

    def test_volume_breakthrough_modifier_keys_have_zero_max_for_unknown(self):
        """Non-criteria keys like bonus_points get max_score=0.0."""
        vb_output = ScreenerOutputDomain(
            screener_name="volume_breakthrough",
            score=70.0,
            passes=True,
            rating="Buy",
            breakdown={
                "five_year_high": 40.0,
                "one_year_high": 30.0,
                "since_ipo_high": 20.0,
                "bonus_points": 5.0,
                "decay_multiplier": 0.8,
            },
            details={},
        )
        item = _make_item_with_outputs("AAPL", 70.0, {"volume_breakthrough": vb_output})
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
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
        item = _make_item_with_outputs(
            "AAPL",
            82.5,
            {
                "minervini": _make_minervini_output(75.0),
                "canslim": _make_canslim_output(80.0),
            },
        )
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        assert len(result.explanation.screener_explanations) == 2
        names = {se.screener_name for se in result.explanation.screener_explanations}
        assert names == {"minervini", "canslim"}

    def test_each_screener_has_own_score_and_criteria(self):
        item = _make_item_with_outputs(
            "AAPL",
            82.5,
            {
                "minervini": _make_minervini_output(75.0),
                "canslim": _make_canslim_output(80.0),
            },
        )
        repo = ExplainableResultRepo(items_by_symbol={"AAPL": item})
        uow = FakeUnitOfWork(scan_results=repo)
        setup_scan(uow)
        uc = ExplainStockUseCase()

        result = uc.execute(uow, _make_query())

        by_name = {
            se.screener_name: se for se in result.explanation.screener_explanations
        }
        assert by_name["minervini"].score == 75.0
        assert by_name["canslim"].score == 80.0
        assert len(by_name["minervini"].criteria) == 5
        assert len(by_name["canslim"].criteria) == 6
