"""Boundary tests for domain scanning types.

Verifies:
- All types instantiate correctly with valid data
- Frozen dataclasses reject attribute mutation
- Score validation rejects out-of-range values
- PageSpec validation rejects bad inputs
- ResultPage.total_pages computes correctly
- FilterSpec builder pattern chains correctly
- No IO imports leak into domain modules
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from app.domain.scanning.models import (
    CompositeMethod,
    RatingCategory,
    ResultPage,
    ScanConfig,
    ScanResultItemDomain,
    ScanStatus,
    ScreenerName,
    ScreenerOutputDomain,
    UniverseSpec,
)
from app.domain.scanning.filter_spec import (
    BooleanFilter,
    CategoricalFilter,
    FilterMode,
    FilterSpec,
    PageSpec,
    QuerySpec,
    RangeFilter,
    SortOrder,
    SortSpec,
    TextSearchFilter,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_screener_output(**overrides) -> ScreenerOutputDomain:
    defaults = dict(
        screener_name="minervini",
        score=85.0,
        passes=True,
        rating="Strong Buy",
        breakdown={"ma_alignment": True},
        details={"rs_rating": 92},
    )
    defaults.update(overrides)
    return ScreenerOutputDomain(**defaults)


def _make_scan_result_item(**overrides) -> ScanResultItemDomain:
    output = _make_screener_output()
    defaults = dict(
        symbol="AAPL",
        composite_score=85.0,
        rating="Strong Buy",
        current_price=150.0,
        screener_outputs={"minervini": output},
        screeners_run=["minervini"],
        composite_method="weighted_average",
        screeners_passed=1,
        screeners_total=1,
    )
    defaults.update(overrides)
    return ScanResultItemDomain(**defaults)


# ── Enum Tests ───────────────────────────────────────────────────────


class TestEnums:
    def test_screener_name_values(self):
        assert ScreenerName.MINERVINI == "minervini"
        assert ScreenerName.CANSLIM == "canslim"
        assert ScreenerName.IPO == "ipo"
        assert ScreenerName.CUSTOM == "custom"
        assert ScreenerName.VOLUME_BREAKTHROUGH == "volume_breakthrough"

    def test_composite_method_values(self):
        assert CompositeMethod.WEIGHTED_AVERAGE == "weighted_average"
        assert CompositeMethod.MAXIMUM == "maximum"
        assert CompositeMethod.MINIMUM == "minimum"

    def test_rating_category_values(self):
        assert RatingCategory.STRONG_BUY == "Strong Buy"
        assert RatingCategory.BUY == "Buy"
        assert RatingCategory.WATCH == "Watch"
        assert RatingCategory.PASS == "Pass"
        assert RatingCategory.ERROR == "Error"

    def test_scan_status_values(self):
        assert ScanStatus.RUNNING == "running"
        assert ScanStatus.COMPLETED == "completed"
        assert ScanStatus.FAILED == "failed"
        assert ScanStatus.CANCELLED == "cancelled"

    def test_str_enum_is_string(self):
        """str,Enum members compare equal to their string values."""
        assert isinstance(ScreenerName.MINERVINI, str)
        assert ScreenerName.CANSLIM == "canslim"
        assert ScreenerName.CANSLIM.value == "canslim"


# ── UniverseSpec Tests ───────────────────────────────────────────────


class TestUniverseSpec:
    def test_basic_construction(self):
        spec = UniverseSpec(type="all")
        assert spec.type == "all"
        assert spec.exchange is None
        assert spec.symbols is None

    def test_with_symbols_tuple(self):
        spec = UniverseSpec(type="custom", symbols=("AAPL", "MSFT"))
        assert spec.symbols == ("AAPL", "MSFT")

    def test_frozen(self):
        spec = UniverseSpec(type="all")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.type = "exchange"  # type: ignore[misc]

    def test_hashable(self):
        """Frozen dataclass with tuple symbols is hashable."""
        a = UniverseSpec(type="custom", symbols=("AAPL",))
        b = UniverseSpec(type="custom", symbols=("AAPL",))
        assert hash(a) == hash(b)
        assert a == b


# ── ScreenerOutputDomain Tests ───────────────────────────────────────


class TestScreenerOutputDomain:
    def test_valid_construction(self):
        output = _make_screener_output()
        assert output.screener_name == "minervini"
        assert output.score == 85.0
        assert output.passes is True

    def test_score_boundary_zero(self):
        output = _make_screener_output(score=0.0)
        assert output.score == 0.0

    def test_score_boundary_hundred(self):
        output = _make_screener_output(score=100.0)
        assert output.score == 100.0

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="Score must be 0-100"):
            _make_screener_output(score=-1.0)

    def test_score_above_hundred_raises(self):
        with pytest.raises(ValueError, match="Score must be 0-100"):
            _make_screener_output(score=100.1)

    def test_frozen(self):
        output = _make_screener_output()
        with pytest.raises(dataclasses.FrozenInstanceError):
            output.score = 50.0  # type: ignore[misc]


# ── ScanResultItemDomain Tests ───────────────────────────────────────


class TestScanResultItemDomain:
    def test_valid_construction(self):
        item = _make_scan_result_item()
        assert item.symbol == "AAPL"
        assert item.composite_score == 85.0
        assert item.screeners_passed == 1

    def test_composite_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="composite_score must be 0-100"):
            _make_scan_result_item(composite_score=-5.0)

    def test_composite_score_above_hundred_raises(self):
        with pytest.raises(ValueError, match="composite_score must be 0-100"):
            _make_scan_result_item(composite_score=101.0)

    def test_extended_fields_default_empty(self):
        item = _make_scan_result_item()
        assert item.extended_fields == {}

    def test_data_errors_default_none(self):
        item = _make_scan_result_item()
        assert item.data_errors is None

    def test_frozen(self):
        item = _make_scan_result_item()
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.symbol = "MSFT"  # type: ignore[misc]


# ── ScanConfig Tests ─────────────────────────────────────────────────


class TestScanConfig:
    def test_valid_construction(self):
        cfg = ScanConfig(
            screeners=[ScreenerName.MINERVINI, ScreenerName.CANSLIM],
            composite_method=CompositeMethod.WEIGHTED_AVERAGE,
            criteria={"min_rs": 70},
            universe=UniverseSpec(type="all"),
        )
        assert len(cfg.screeners) == 2
        assert cfg.composite_method == CompositeMethod.WEIGHTED_AVERAGE
        assert cfg.weights is None

    def test_with_typed_weights(self):
        cfg = ScanConfig(
            screeners=[ScreenerName.MINERVINI],
            composite_method=CompositeMethod.WEIGHTED_AVERAGE,
            criteria={},
            universe=UniverseSpec(type="all"),
            weights={ScreenerName.MINERVINI: 1.0},
        )
        assert cfg.weights[ScreenerName.MINERVINI] == 1.0

    def test_frozen(self):
        cfg = ScanConfig(
            screeners=[ScreenerName.MINERVINI],
            composite_method=CompositeMethod.MAXIMUM,
            criteria={},
            universe=UniverseSpec(type="all"),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.composite_method = CompositeMethod.MINIMUM  # type: ignore[misc]


# ── ResultPage Tests ─────────────────────────────────────────────────


class TestResultPage:
    def test_total_pages_exact_division(self):
        page = ResultPage(items=(), total=100, page=1, per_page=50)
        assert page.total_pages == 2

    def test_total_pages_with_remainder(self):
        page = ResultPage(items=(), total=101, page=1, per_page=50)
        assert page.total_pages == 3

    def test_total_pages_single_page(self):
        page = ResultPage(items=(), total=5, page=1, per_page=50)
        assert page.total_pages == 1

    def test_total_pages_zero_items(self):
        page = ResultPage(items=(), total=0, page=1, per_page=50)
        assert page.total_pages == 0

    def test_total_pages_zero_per_page(self):
        page = ResultPage(items=(), total=10, page=1, per_page=0)
        assert page.total_pages == 0

    def test_frozen(self):
        page = ResultPage(items=(), total=0, page=1, per_page=50)
        with pytest.raises(dataclasses.FrozenInstanceError):
            page.total = 99  # type: ignore[misc]


# ── Filter Types Tests ───────────────────────────────────────────────


class TestFilterTypes:
    def test_range_filter_is_empty(self):
        assert RangeFilter(field="score").is_empty() is True
        assert RangeFilter(field="score", min_value=50).is_empty() is False
        assert RangeFilter(field="score", max_value=90).is_empty() is False

    def test_categorical_filter_is_empty(self):
        assert CategoricalFilter(field="sector", values=()).is_empty() is True
        assert CategoricalFilter(field="sector", values=("Tech",)).is_empty() is False

    def test_boolean_filter_is_never_empty(self):
        assert BooleanFilter(field="passes", value=True).is_empty() is False
        assert BooleanFilter(field="passes", value=False).is_empty() is False

    def test_text_search_filter_is_empty(self):
        assert TextSearchFilter(field="symbol", pattern="").is_empty() is True
        assert TextSearchFilter(field="symbol", pattern="AA").is_empty() is False

    def test_categorical_filter_mode_default(self):
        f = CategoricalFilter(field="sector", values=("Tech",))
        assert f.mode == FilterMode.INCLUDE

    def test_frozen_individual_filters(self):
        r = RangeFilter(field="score", min_value=50)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.min_value = 60  # type: ignore[misc]

        c = CategoricalFilter(field="sector", values=("Tech",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.field = "industry"  # type: ignore[misc]


# ── FilterSpec Builder Tests ─────────────────────────────────────────


class TestFilterSpecBuilder:
    def test_empty_by_default(self):
        fs = FilterSpec()
        assert fs.range_filters == []
        assert fs.categorical_filters == []
        assert fs.boolean_filters == []
        assert fs.text_searches == []

    def test_fluent_chaining(self):
        fs = (
            FilterSpec()
            .add_range("score", min_value=50)
            .add_categorical("sector", ("Tech", "Health"))
            .add_boolean("passes", True)
            .add_text_search("symbol", "AA")
        )
        assert len(fs.range_filters) == 1
        assert len(fs.categorical_filters) == 1
        assert len(fs.boolean_filters) == 1
        assert len(fs.text_searches) == 1

    def test_skips_empty_range(self):
        fs = FilterSpec().add_range("score")  # no min or max
        assert len(fs.range_filters) == 0

    def test_skips_empty_categorical(self):
        fs = FilterSpec().add_categorical("sector", ())
        assert len(fs.categorical_filters) == 0

    def test_skips_empty_text_search(self):
        fs = FilterSpec().add_text_search("symbol", "")
        assert len(fs.text_searches) == 0

    def test_accepts_list_for_categorical(self):
        fs = FilterSpec().add_categorical("sector", ["Tech", "Health"])
        assert fs.categorical_filters[0].values == ("Tech", "Health")


# ── PageSpec Tests ───────────────────────────────────────────────────


class TestPageSpec:
    def test_defaults(self):
        ps = PageSpec()
        assert ps.page == 1
        assert ps.per_page == 50

    def test_offset_and_limit(self):
        ps = PageSpec(page=3, per_page=20)
        assert ps.offset == 40
        assert ps.limit == 20

    def test_page_zero_raises(self):
        with pytest.raises(ValueError, match="page must be >= 1"):
            PageSpec(page=0)

    def test_negative_page_raises(self):
        with pytest.raises(ValueError, match="page must be >= 1"):
            PageSpec(page=-1)

    def test_per_page_zero_raises(self):
        with pytest.raises(ValueError, match="per_page must be 1-100"):
            PageSpec(per_page=0)

    def test_per_page_over_100_raises(self):
        with pytest.raises(ValueError, match="per_page must be 1-100"):
            PageSpec(per_page=101)

    def test_per_page_boundary_1(self):
        ps = PageSpec(per_page=1)
        assert ps.per_page == 1

    def test_per_page_boundary_100(self):
        ps = PageSpec(per_page=100)
        assert ps.per_page == 100


# ── SortSpec / QuerySpec Tests ───────────────────────────────────────


class TestSortSpec:
    def test_defaults(self):
        ss = SortSpec()
        assert ss.field == "composite_score"
        assert ss.order == SortOrder.DESC

    def test_frozen(self):
        ss = SortSpec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ss.field = "symbol"  # type: ignore[misc]


class TestQuerySpec:
    def test_default_construction(self):
        qs = QuerySpec()
        assert isinstance(qs.filters, FilterSpec)
        assert isinstance(qs.sort, SortSpec)
        assert isinstance(qs.page, PageSpec)


# ── No IO Imports Test ───────────────────────────────────────────────


DOMAIN_DIR = Path(__file__).resolve().parents[2] / "app" / "domain" / "scanning"

FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(import|from)\s+(sqlalchemy|fastapi|pydantic|redis|celery)\b",
    re.MULTILINE,
)


class TestNoIOLeakage:
    @pytest.mark.parametrize(
        "filename", ["models.py", "filter_spec.py"],
    )
    def test_no_forbidden_imports(self, filename: str):
        source = (DOMAIN_DIR / filename).read_text()
        matches = FORBIDDEN_IMPORTS.findall(source)
        assert matches == [], (
            f"{filename} contains forbidden IO imports: {matches}"
        )

    def test_models_module_has_all(self):
        from app.domain.scanning import models
        assert hasattr(models, "__all__")
        assert len(models.__all__) > 0

    def test_filter_spec_module_has_all(self):
        from app.domain.scanning import filter_spec
        assert hasattr(filter_spec, "__all__")
        assert len(filter_spec.__all__) > 0
