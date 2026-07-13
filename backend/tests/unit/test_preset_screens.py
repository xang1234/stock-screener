from datetime import date

from app.domain.scanning.legacy_filter_expression import legacy_filters_to_expression
from app.services.preset_screens import (
    PRESET_SCREENS,
    _matches_preset_filters,
    get_preset_chart_symbols,
    resolve_preset_screens_for_defaults,
)


def _leaders_screen():
    return next(
        screen
        for screen in PRESET_SCREENS
        if screen["id"] == "leaders_in_leading_groups"
    )


def test_leaders_in_leading_groups_preset_filters_for_v1_contract():
    screen = _leaders_screen()

    matching = {
        "symbol": "LEAD",
        "ibd_group_rank": 40,
        "rs_rating": 80,
        "composite_score": 64.23,
        "volume": 1_500_000,
    }

    assert screen["name"] == "Leaders in Leading Groups"
    assert screen["sort_by"] == "composite_score"
    assert screen["sort_order"] == "desc"
    assert "apply_default_filters" not in screen
    assert "compositeScore" not in screen["filters"]
    assert "minVolume" not in screen["filters"]
    assert _matches_preset_filters(matching, screen["filters"]) is True
    assert _matches_preset_filters(
        {**matching, "ibd_group_rank": 41},
        screen["filters"],
    ) is False
    assert _matches_preset_filters(
        {**matching, "rs_rating": 79},
        screen["filters"],
    ) is False


def test_resolved_leaders_filters_materialize_market_defaults():
    screen = _leaders_screen()

    [resolved_screen] = resolve_preset_screens_for_defaults(
        [screen],
        {"minVolume": 1_300_000},
    )

    assert resolved_screen["filters"] == {
        "minVolume": 1_300_000,
        "ibdGroupRank": {"min": None, "max": 40},
        "rsRating": {"min": 80, "max": None},
    }
    assert resolved_screen["filter_schema_version"] == 2
    assert resolved_screen["filter_expression"]["expression_version"] == 1
    assert resolved_screen["filter_expression"]["required"]["conditions"] == [
        {"kind": "range", "field": "rs_rating", "min": 80, "max": None},
        {"kind": "range", "field": "ibd_group_rank", "min": None, "max": 40},
        {"kind": "range", "field": "volume", "min": 1_300_000, "max": None},
    ]
    assert "minVolume" not in screen["filters"]


def test_preset_chart_symbols_use_resolved_market_defaults():
    screen = _leaders_screen()
    [resolved_screen] = resolve_preset_screens_for_defaults(
        [screen],
        {"minVolume": 1_300_000},
    )
    rows = [
        {
            "symbol": "LIQUID",
            "ibd_group_rank": 10,
            "rs_rating": 90,
            "volume": 2_000_000,
            "composite_score": 64.0,
        },
        {
            "symbol": "THIN",
            "ibd_group_rank": 5,
            "rs_rating": 99,
            "volume": 900_000,
            "composite_score": 99.0,
        },
    ]

    assert get_preset_chart_symbols(
        rows,
        presets=[resolved_screen],
        top_n=5,
    ) == {"LIQUID"}


def test_noop_scalar_and_range_filters_match_like_frontend():
    row = {"symbol": "ROW", "volume": None, "composite_score": None}

    assert _matches_preset_filters(row, {"minVolume": None}) is True
    assert _matches_preset_filters(
        row,
        {"compositeScore": {"min": None, "max": None}},
    ) is True


def test_chart_symbols_prefer_grouped_expression_over_legacy_filters():
    preset = {
        "filters": {"compositeScore": {"min": 101, "max": None}},
        "filter_expression": {
            "expression_version": 1,
            "required": {
                "id": "required",
                "name": "Always require",
                "match": "all",
                "enabled": True,
                "conditions": [],
            },
            "group_join": "any",
            "groups": [
                {
                    "id": "growth",
                    "name": "Growth",
                    "match": "any",
                    "enabled": True,
                    "conditions": [
                        {
                            "kind": "range",
                            "field": "eps_growth_qq",
                            "min": 30,
                            "max": None,
                        },
                        {
                            "kind": "range",
                            "field": "sales_growth_qq",
                            "min": 30,
                            "max": None,
                        },
                    ],
                }
            ],
        },
        "sort_by": "composite_score",
        "sort_order": "desc",
    }
    rows = [
        {"symbol": "EPS", "eps_growth_qq": 35, "composite_score": 80},
        {"symbol": "SALES", "sales_growth_qq": 40, "composite_score": 70},
        {"symbol": "SLOW", "eps_growth_qq": 10, "composite_score": 99},
    ]

    assert get_preset_chart_symbols(rows, presets=[preset]) == {"EPS", "SALES"}


def test_legacy_ipo_preset_matches_browser_month_rollover():
    expression = legacy_filters_to_expression(
        {"ipoAfter": "6m"},
        today=date(2026, 3, 31),
    )

    assert expression.required.conditions[0].min_value == "2025-10-01"
