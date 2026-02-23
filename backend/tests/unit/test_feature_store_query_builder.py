"""Unit tests for the feature store query builder — field coverage and parity.

Verifies that _JSON_FIELD_MAP, _JSON_SORT_NUMERIC, and _COLUMN_MAP
have the expected entries, and that se_* fields are consistent
between scan_result_query and feature_store_query.
"""

import pytest

from app.infra.query.feature_store_query import (
    _COLUMN_MAP,
    _JSON_FIELD_MAP,
    _JSON_SORT_NUMERIC,
)
from app.infra.query import scan_result_query as srq


class TestSetupEngineFieldCoverage:
    """Verify all setup_engine query fields are registered."""

    SE_NUMERIC_FIELDS = [
        "se_setup_score", "se_quality_score", "se_readiness_score",
        "se_pattern_confidence", "se_pivot_price", "se_distance_to_pivot_pct",
        "se_base_length_weeks", "se_base_depth_pct", "se_support_tests_count",
        "se_tight_closes_count",
        "se_atr14_pct", "se_atr14_pct_trend", "se_bb_width_pct",
        "se_bb_width_pctile_252", "se_volume_vs_50d",
        "se_up_down_volume_ratio_10d", "se_quiet_days_10d", "se_rs",
        "se_rs_vs_spy_65d", "se_rs_vs_spy_trend_20d",
    ]

    SE_BOOLEAN_FIELDS = [
        "se_setup_ready",
        "se_rs_line_new_high",
        "se_in_early_zone",
        "se_extended_from_pivot",
        "se_bb_squeeze",
    ]

    SE_STRING_FIELDS = ["se_pattern_primary", "se_pivot_type"]

    SE_ALL_FIELDS = SE_NUMERIC_FIELDS + SE_BOOLEAN_FIELDS + SE_STRING_FIELDS

    @pytest.mark.parametrize("field", SE_ALL_FIELDS)
    def test_se_field_in_json_field_map(self, field):
        assert field in _JSON_FIELD_MAP, f"{field} should be in _JSON_FIELD_MAP"

    @pytest.mark.parametrize("field", SE_ALL_FIELDS)
    def test_se_field_has_setup_engine_prefix_in_path(self, field):
        path = _JSON_FIELD_MAP[field]
        assert path.startswith("$.setup_engine."), (
            f"{field} path should start with $.setup_engine., got {path}"
        )

    def test_se_field_count(self):
        se_fields = [k for k in _JSON_FIELD_MAP if k.startswith("se_")]
        assert len(se_fields) == 27

    @pytest.mark.parametrize("field", SE_NUMERIC_FIELDS)
    def test_numeric_se_field_in_sort_numeric(self, field):
        assert field in _JSON_SORT_NUMERIC, f"{field} should be in _JSON_SORT_NUMERIC"

    @pytest.mark.parametrize("field", SE_BOOLEAN_FIELDS + SE_STRING_FIELDS)
    def test_non_numeric_se_field_not_in_sort_numeric(self, field):
        assert field not in _JSON_SORT_NUMERIC, (
            f"{field} should NOT be in _JSON_SORT_NUMERIC"
        )


class TestJsonSortNumericConsistency:
    """Verify _JSON_SORT_NUMERIC is consistent with _JSON_FIELD_MAP."""

    def test_sort_numeric_is_subset_of_json_field_map(self):
        assert _JSON_SORT_NUMERIC <= _JSON_FIELD_MAP.keys(), (
            f"Fields in _JSON_SORT_NUMERIC but not in _JSON_FIELD_MAP: "
            f"{_JSON_SORT_NUMERIC - _JSON_FIELD_MAP.keys()}"
        )

    def test_vcp_numeric_fields_in_sort_numeric(self):
        assert "vcp_score" in _JSON_SORT_NUMERIC
        assert "vcp_pivot" in _JSON_SORT_NUMERIC


class TestParityWithScanResultQuery:
    """Verify se_* fields are identical between both query builders."""

    def test_se_fields_have_same_json_paths(self):
        """Every se_* field must map to the same JSON path in both modules."""
        se_fields_fs = {
            k: v for k, v in _JSON_FIELD_MAP.items() if k.startswith("se_")
        }
        se_fields_sr = {
            k: v for k, v in srq._JSON_FIELD_MAP.items() if k.startswith("se_")
        }
        assert se_fields_fs == se_fields_sr

    def test_se_sort_numeric_parity(self):
        """se_* fields in _JSON_SORT_NUMERIC must be the same in both modules."""
        se_numeric_fs = {f for f in _JSON_SORT_NUMERIC if f.startswith("se_")}
        se_numeric_sr = {f for f in srq._JSON_SORT_NUMERIC if f.startswith("se_")}
        assert se_numeric_fs == se_numeric_sr

    def test_vcp_sort_numeric_parity(self):
        """vcp_* fields in _JSON_SORT_NUMERIC must be the same in both modules."""
        vcp_numeric_fs = {f for f in _JSON_SORT_NUMERIC if f.startswith("vcp_")}
        vcp_numeric_sr = {f for f in srq._JSON_SORT_NUMERIC if f.startswith("vcp_")}
        assert vcp_numeric_fs == vcp_numeric_sr


class TestColumnMapBasics:
    """Verify the column map covers expected indexed fields."""

    @pytest.mark.parametrize("field", [
        "symbol", "composite_score", "overall_rating",
        "passes_count", "as_of_date",
    ])
    def test_indexed_column_is_mapped(self, field):
        assert field in _COLUMN_MAP, f"{field} should be in _COLUMN_MAP"
