"""Unit tests for the custom-criteria → feature-store FilterSpec compiler."""

from __future__ import annotations

import pytest

from app.domain.common.query import FilterMode
from app.domain.scanning.custom_criteria_compiler import (
    CompiledCustomCriteria,
    compile_custom_criteria,
)


def _range(spec, field):
    return next((r for r in spec.range_filters if r.field == field), None)


def _categorical(spec, field):
    return next(
        (c for c in spec.categorical_filters if c.field == field), None
    )


def _boolean(spec, field):
    return next((b for b in spec.boolean_filters if b.field == field), None)


class TestEmptyCriteria:
    def test_none_returns_no_constraints(self):
        result = compile_custom_criteria(None, screeners=["custom"])

        assert result.is_fully_representable
        # Custom-only with no explicit min_score still applies the
        # CustomScanner default (70). That alone means has_constraints==True.
        assert result.score_field == "custom_score"
        assert result.min_score == 70.0
        assert result.has_constraints
        assert result.filter_spec.range_filters == []
        assert result.unrepresentable_keys == ()

    def test_empty_dict_treated_like_none(self):
        result = compile_custom_criteria({}, screeners=["custom"])

        assert result.is_fully_representable
        assert result.filter_spec.range_filters == []

    def test_no_screeners_yields_no_score_field(self):
        result = compile_custom_criteria(None, screeners=[])

        assert result.score_field is None
        assert result.min_score is None
        assert not result.has_constraints


class TestRangeFilters:
    def test_price_range_compiles_to_current_price(self):
        result = compile_custom_criteria(
            {"custom_filters": {"price_min": 20, "price_max": 500}},
            screeners=["custom"],
        )

        rf = _range(result.filter_spec, "current_price")
        assert rf is not None
        assert rf.min_value == 20
        assert rf.max_value == 500
        assert "price_min" in result.representable_keys
        assert "price_max" in result.representable_keys

    def test_rs_rating_min_compiles_to_rs_rating(self):
        result = compile_custom_criteria(
            {"custom_filters": {"rs_rating_min": 80}},
            screeners=["custom"],
        )

        rf = _range(result.filter_spec, "rs_rating")
        assert rf is not None
        assert rf.min_value == 80
        assert rf.max_value is None

    def test_eps_growth_compiles_to_eps_growth_qq(self):
        result = compile_custom_criteria(
            {"custom_filters": {"eps_growth_min": 25}},
            screeners=["custom"],
        )

        rf = _range(result.filter_spec, "eps_growth_qq")
        assert rf is not None
        assert rf.min_value == 25

    def test_near_52w_high_compiles_to_distance_max(self):
        result = compile_custom_criteria(
            {"custom_filters": {"near_52w_high": 15}},
            screeners=["custom"],
        )

        rf = _range(result.filter_spec, "week_52_high_distance")
        assert rf is not None
        assert rf.max_value == 15
        assert rf.min_value is None


class TestUsdUnitCompatibility:
    def test_volume_min_compiles_for_mixed_market(self):
        result = compile_custom_criteria(
            {"custom_filters": {"volume_min": 1_000_000}},
            screeners=["custom"],
            universe_market=None,
        )

        rf = _range(result.filter_spec, "adv_usd")
        assert rf is not None
        assert rf.min_value == 1_000_000
        assert "volume_min" in result.representable_keys
        assert "volume_min" not in result.unrepresentable_keys

    def test_volume_min_compiles_for_us_market(self):
        result = compile_custom_criteria(
            {"custom_filters": {"volume_min": 1_000_000}},
            screeners=["custom"],
            universe_market="US",
        )

        rf = _range(result.filter_spec, "adv_usd")
        assert rf is not None

    def test_volume_min_unrepresentable_for_hk_market(self):
        result = compile_custom_criteria(
            {"custom_filters": {"volume_min": 1_000_000}},
            screeners=["custom"],
            universe_market="HK",
        )

        assert _range(result.filter_spec, "adv_usd") is None
        assert "volume_min" in result.unrepresentable_keys
        assert not result.is_fully_representable

    def test_volume_min_zero_is_noop(self):
        """CustomScanner treats volume_min<=0 as a disabled filter."""
        result = compile_custom_criteria(
            {"custom_filters": {"volume_min": 0}},
            screeners=["custom"],
            universe_market="HK",  # would otherwise be unrepresentable
        )

        assert "volume_min" not in result.unrepresentable_keys
        assert _range(result.filter_spec, "adv_usd") is None

    def test_market_cap_unrepresentable_for_non_usd_market(self):
        result = compile_custom_criteria(
            {"custom_filters": {"market_cap_min": 1e9, "market_cap_max": 5e10}},
            screeners=["custom"],
            universe_market="JP",
        )

        assert _range(result.filter_spec, "market_cap_usd") is None
        assert "market_cap_min" in result.unrepresentable_keys
        assert "market_cap_max" in result.unrepresentable_keys


class TestBooleanAndCategorical:
    def test_ma_alignment_true_adds_boolean_filter(self):
        result = compile_custom_criteria(
            {"custom_filters": {"ma_alignment": True}},
            screeners=["custom"],
        )

        bf = _boolean(result.filter_spec, "ma_alignment")
        assert bf is not None
        assert bf.value is True

    def test_ma_alignment_false_is_noop(self):
        result = compile_custom_criteria(
            {"custom_filters": {"ma_alignment": False}},
            screeners=["custom"],
        )

        assert _boolean(result.filter_spec, "ma_alignment") is None
        assert "ma_alignment" not in result.unrepresentable_keys

    def test_sectors_compile_to_categorical_include(self):
        result = compile_custom_criteria(
            {"custom_filters": {"sectors": ["Technology", "Healthcare"]}},
            screeners=["custom"],
        )

        cf = _categorical(result.filter_spec, "gics_sector")
        assert cf is not None
        assert cf.values == ("Technology", "Healthcare")
        assert cf.mode == FilterMode.INCLUDE

    def test_exclude_industries_compile_to_categorical_exclude(self):
        result = compile_custom_criteria(
            {"custom_filters": {"exclude_industries": ["Tobacco", "Gambling"]}},
            screeners=["custom"],
        )

        cf = _categorical(result.filter_spec, "gics_industry")
        assert cf is not None
        assert cf.values == ("Tobacco", "Gambling")
        assert cf.mode == FilterMode.EXCLUDE


class TestUnrepresentable:
    def test_debt_to_equity_marked_unrepresentable(self):
        result = compile_custom_criteria(
            {"custom_filters": {"debt_to_equity_max": 0.5}},
            screeners=["custom"],
        )

        assert "debt_to_equity_max" in result.unrepresentable_keys
        assert not result.is_fully_representable

    def test_unknown_filter_marked_unrepresentable(self):
        result = compile_custom_criteria(
            {"custom_filters": {"some_made_up_filter": 42}},
            screeners=["custom"],
        )

        assert "some_made_up_filter" in result.unrepresentable_keys

    def test_fully_unrepresentable_with_known_and_unknown(self):
        result = compile_custom_criteria(
            {
                "custom_filters": {
                    "price_min": 10,  # representable
                    "debt_to_equity_max": 0.5,  # not representable
                },
            },
            screeners=["custom"],
        )

        assert "price_min" in result.representable_keys
        assert "debt_to_equity_max" in result.unrepresentable_keys
        assert not result.is_fully_representable


class TestScoreGate:
    def test_custom_only_default_min_score_70(self):
        result = compile_custom_criteria(
            {"custom_filters": {"price_min": 10}},
            screeners=["custom"],
        )

        assert result.score_field == "custom_score"
        assert result.min_score == 70.0

    def test_explicit_min_score_overrides_default(self):
        result = compile_custom_criteria(
            {"custom_filters": {"price_min": 10}, "min_score": 85},
            screeners=["custom"],
        )

        assert result.min_score == 85.0

    def test_multi_screener_no_score_gate(self):
        """Multi-screener composites need their own logic; we don't gate."""
        result = compile_custom_criteria(
            {"custom_filters": {"price_min": 10}},
            screeners=["custom", "minervini"],
        )

        assert result.score_field is None
        assert result.min_score is None

    def test_minervini_only_no_score_gate(self):
        result = compile_custom_criteria(
            {"custom_filters": {"price_min": 10}},
            screeners=["minervini"],
        )

        assert result.score_field is None


class TestTopLevelLegacyForm:
    """CustomScanner accepts both nested custom_filters and top-level keys."""

    def test_top_level_keys_compile(self):
        result = compile_custom_criteria(
            {"price_min": 20, "rs_rating_min": 80, "min_score": 75},
            screeners=["custom"],
        )

        assert _range(result.filter_spec, "current_price") is not None
        assert _range(result.filter_spec, "rs_rating") is not None
        assert result.min_score == 75.0

    def test_nested_overrides_top_level(self):
        """Mirrors CustomScanner._get_filters_config precedence."""
        result = compile_custom_criteria(
            {
                "price_min": 5,
                "custom_filters": {"price_min": 50},
            },
            screeners=["custom"],
        )

        rf = _range(result.filter_spec, "current_price")
        assert rf.min_value == 50
