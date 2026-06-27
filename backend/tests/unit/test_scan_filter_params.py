from inspect import signature

from app.api.v1.scan_filter_params import parse_scan_filters


def test_parse_scan_filters_maps_ibd_group_rank_range():
    kwargs = {name: None for name in signature(parse_scan_filters).parameters}
    filters = parse_scan_filters(
        **{
            **kwargs,
            "min_ibd_group_rank": 1,
            "max_ibd_group_rank": 40,
        }
    )

    rank_filter = next(
        item for item in filters.range_filters if item.field == "ibd_group_rank"
    )
    assert rank_filter.min_value == 1
    assert rank_filter.max_value == 40


def test_parse_scan_filters_maps_rs_line_leadership_booleans():
    kwargs = {name: None for name in signature(parse_scan_filters).parameters}
    filters = parse_scan_filters(
        **{
            **kwargs,
            "rs_line_new_high": True,
            "rs_line_new_high_before_price": False,
            "rs_line_blue_dot_recent": True,
        }
    )

    by_field = {item.field: item.value for item in filters.boolean_filters}
    assert by_field["rs_line_new_high"] is True
    assert by_field["rs_line_new_high_before_price"] is False
    assert by_field["rs_line_blue_dot_recent"] is True
