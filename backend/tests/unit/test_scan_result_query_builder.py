"""Unit tests for the scan result query builder — filter/sort translation logic.

These tests verify the query builder's public API (apply_filters,
apply_sort_and_paginate) by checking the behavior of the _COLUMN_MAP,
_JSON_FIELD_MAP, and _PYTHON_SORT_FIELDS constants and helper functions.
"""

import pytest

from app.domain.scanning.filter_spec import (
    FilterMode,
    FilterSpec,
    PageSpec,
    SortOrder,
    SortSpec,
)
from app.infra.query.scan_result_query import (
    _COLUMN_MAP,
    _JSON_FIELD_MAP,
    _PYTHON_SORT_FIELDS,
)


class TestColumnMapCoverage:
    """Verify the column map covers all expected fields."""

    @pytest.mark.parametrize("field", [
        "symbol", "composite_score", "minervini_score", "canslim_score",
        "ipo_score", "custom_score", "volume_breakthrough_score",
        "price", "current_price", "volume", "market_cap",
        "stage", "rating",
        "rs_rating", "rs_rating_1m", "rs_rating_3m", "rs_rating_12m",
        "eps_growth_qq", "sales_growth_qq", "eps_growth_yy", "sales_growth_yy",
        "peg_ratio", "peg", "adr_percent", "eps_rating",
        "ibd_industry_group", "ibd_group_rank", "gics_sector",
        "rs_trend", "price_change_1d",
        "perf_week", "perf_month", "perf_3m", "perf_6m",
        "gap_percent", "volume_surge",
        "ema_10_distance", "ema_20_distance", "ema_50_distance",
        "week_52_high_distance", "week_52_low_distance",
        "ipo_date", "beta", "beta_adj_rs",
        "beta_adj_rs_1m", "beta_adj_rs_3m", "beta_adj_rs_12m",
    ])
    def test_sql_column_field_is_mapped(self, field):
        assert field in _COLUMN_MAP, f"{field} should be in _COLUMN_MAP"

    @pytest.mark.parametrize("field", [
        "vcp_score", "vcp_pivot", "vcp_detected",
        "vcp_ready_for_breakout", "ma_alignment",
    ])
    def test_json_field_is_mapped(self, field):
        assert field in _JSON_FIELD_MAP, f"{field} should be in _JSON_FIELD_MAP"

    @pytest.mark.parametrize("field", [
        "stage_name", "ma_alignment", "vcp_detected", "passes_template",
    ])
    def test_python_sort_fields(self, field):
        assert field in _PYTHON_SORT_FIELDS, f"{field} should be in _PYTHON_SORT_FIELDS"


class TestFilterSpecBuilder:
    """Test that FilterSpec builder methods work correctly."""

    def test_add_range_skips_none_values(self):
        spec = FilterSpec()
        spec.add_range("price", None, None)
        assert len(spec.range_filters) == 0

    def test_add_range_with_min_only(self):
        spec = FilterSpec()
        spec.add_range("price", min_value=10.0)
        assert len(spec.range_filters) == 1
        assert spec.range_filters[0].field == "price"
        assert spec.range_filters[0].min_value == 10.0
        assert spec.range_filters[0].max_value is None

    def test_add_range_with_both(self):
        spec = FilterSpec()
        spec.add_range("rs_rating", min_value=50.0, max_value=100.0)
        rf = spec.range_filters[0]
        assert rf.min_value == 50.0
        assert rf.max_value == 100.0

    def test_add_categorical_skips_empty(self):
        spec = FilterSpec()
        spec.add_categorical("rating", [])
        assert len(spec.categorical_filters) == 0

    def test_add_categorical_with_exclude_mode(self):
        spec = FilterSpec()
        spec.add_categorical("gics_sector", ["Tech", "Energy"], FilterMode.EXCLUDE)
        cf = spec.categorical_filters[0]
        assert cf.mode == FilterMode.EXCLUDE
        assert cf.values == ("Tech", "Energy")

    def test_add_boolean(self):
        spec = FilterSpec()
        spec.add_boolean("vcp_detected", True)
        assert len(spec.boolean_filters) == 1
        assert spec.boolean_filters[0].value is True

    def test_add_text_search_skips_empty(self):
        spec = FilterSpec()
        spec.add_text_search("symbol", "")
        assert len(spec.text_searches) == 0

    def test_fluent_chaining(self):
        spec = (
            FilterSpec()
            .add_range("price", min_value=10.0)
            .add_categorical("rating", ["Buy"])
            .add_boolean("ma_alignment", True)
            .add_text_search("symbol", "AA")
        )
        assert len(spec.range_filters) == 1
        assert len(spec.categorical_filters) == 1
        assert len(spec.boolean_filters) == 1
        assert len(spec.text_searches) == 1


class TestPageSpec:
    """Test PageSpec validation and computed properties."""

    def test_offset_calculation(self):
        p = PageSpec(page=3, per_page=25)
        assert p.offset == 50
        assert p.limit == 25

    def test_page_1_offset_is_zero(self):
        p = PageSpec(page=1, per_page=50)
        assert p.offset == 0

    def test_invalid_page_raises(self):
        with pytest.raises(ValueError, match="page must be >= 1"):
            PageSpec(page=0, per_page=50)

    def test_per_page_too_high_raises(self):
        with pytest.raises(ValueError, match="per_page must be 1-100"):
            PageSpec(page=1, per_page=200)

    def test_per_page_zero_raises(self):
        with pytest.raises(ValueError, match="per_page must be 1-100"):
            PageSpec(page=1, per_page=0)


class TestSortSpec:
    """Test SortSpec defaults."""

    def test_default_sort(self):
        s = SortSpec()
        assert s.field == "composite_score"
        assert s.order == SortOrder.DESC

    def test_custom_sort(self):
        s = SortSpec(field="rs_rating", order=SortOrder.ASC)
        assert s.field == "rs_rating"
        assert s.order == SortOrder.ASC


class TestAliases:
    """Test that field aliases resolve correctly."""

    def test_current_price_maps_to_price(self):
        assert _COLUMN_MAP["current_price"] is _COLUMN_MAP["price"]

    def test_peg_maps_to_peg_ratio(self):
        assert _COLUMN_MAP["peg"] is _COLUMN_MAP["peg_ratio"]
