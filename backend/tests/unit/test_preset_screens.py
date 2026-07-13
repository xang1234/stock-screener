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
    assert "filter_schema_version" not in resolved_screen
    assert "filter_expression" not in resolved_screen
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
