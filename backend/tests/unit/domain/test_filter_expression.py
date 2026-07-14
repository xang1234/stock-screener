import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterMode,
    RangeFilter,
)
from app.domain.scanning.filter_expression_evaluator import (
    annotate_matched_groups,
    evaluate_condition,
    evaluate_expression,
    matched_setup_groups,
)
from app.domain.scanning.filter_expression_serialization import expression_fingerprint
from app.domain.scanning.legacy_filter_expression import legacy_filters_to_expression
from app.domain.scanning.filter_expression_model import (
    FilterExpression,
    FilterGroup,
    ListingDiscoveryFilter,
    MatchOperator,
)
from app.domain.scanning.models import MatchedGroupDomain, ScanResultItemDomain
from app.schemas.filter_expression import ScanQueryRequest
from app.schemas.filter_expression_payload import expression_from_payload


def _expression(group_join: MatchOperator = MatchOperator.ANY) -> FilterExpression:
    return FilterExpression(
        required=FilterGroup(
            id="required",
            name="Always require",
            conditions=(RangeFilter("price", min_value=10),),
        ),
        group_join=group_join,
        groups=(
            FilterGroup(
                id="breakout",
                name="Breakout ready",
                match=MatchOperator.ALL,
                conditions=(
                    RangeFilter("rs_rating", min_value=90),
                    BooleanFilter("vcp_ready_for_breakout", True),
                ),
            ),
            FilterGroup(
                id="growth",
                name="Growth leader",
                match=MatchOperator.ANY,
                conditions=(
                    RangeFilter("eps_growth_qq", min_value=30),
                    RangeFilter("sales_growth_qq", min_value=30),
                ),
            ),
        ),
    )


def test_required_and_any_setup_semantics_with_explanations():
    expression = _expression()
    row = {
        "price": 25,
        "rs_rating": 95,
        "vcp_ready_for_breakout": True,
        "eps_growth_qq": 10,
        "sales_growth_qq": 12,
    }

    assert evaluate_expression(row, expression) is True
    assert matched_setup_groups(row, expression) == (
        MatchedGroupDomain(id="breakout", name="Breakout ready"),
    )


def test_all_setup_join_requires_every_enabled_group():
    row = {
        "price": 25,
        "rs_rating": 95,
        "vcp_ready_for_breakout": True,
        "eps_growth_qq": 10,
        "sales_growth_qq": 12,
    }
    assert evaluate_expression(row, _expression(MatchOperator.ALL)) is False


def test_disabled_groups_do_not_participate():
    expression = _expression(MatchOperator.ALL)
    expression = FilterExpression(
        required=expression.required,
        group_join=expression.group_join,
        groups=(
            expression.groups[0],
            FilterGroup(**{**expression.groups[1].__dict__, "enabled": False}),
        ),
    )
    row = {"price": 25, "rs_rating": 95, "vcp_ready_for_breakout": True}
    assert evaluate_expression(row, expression) is True


def test_missing_values_follow_explicit_policy():
    assert evaluate_condition({}, BooleanFilter("ma_alignment", False)) is False
    assert (
        evaluate_condition(
            {},
            CategoricalFilter("rating", ("Pass",), mode=FilterMode.EXCLUDE),
        )
        is True
    )
    assert evaluate_condition({}, RangeFilter("rs_rating", min_value=80)) is False


def test_listing_discovery_preserves_liquidity_for_normal_rows():
    condition = ListingDiscoveryFilter(min_volume=1_000_000)

    assert (
        evaluate_condition({"scan_mode": "listing_only", "volume": None}, condition)
        is True
    )
    assert evaluate_condition({"scan_mode": "full", "volume": 100}, condition) is False
    assert (
        evaluate_condition({"scan_mode": "full", "volume": 2_000_000}, condition)
        is True
    )


@pytest.mark.parametrize(
    ("field", "minimum", "maximum"),
    [
        ("price", "not-a-number", None),
        ("price", float("inf"), None),
        ("price", True, None),
        ("ipo_date", 20260713, None),
        ("ipo_date", "2026-99-99", None),
    ],
)
def test_range_request_rejects_values_outside_the_field_type(field, minimum, maximum):
    with pytest.raises(ValidationError):
        ScanQueryRequest.model_validate(
            {
                "required": {
                    "id": "required",
                    "name": "Always require",
                    "conditions": [
                        {
                            "kind": "range",
                            "field": field,
                            "min": minimum,
                            "max": maximum,
                        }
                    ],
                }
            }
        )


def test_range_request_normalizes_numeric_strings_and_iso_dates():
    request = ScanQueryRequest.model_validate(
        {
            "required": {
                "id": "required",
                "name": "Always require",
                "conditions": [
                    {"kind": "range", "field": "price", "min": "10.5", "max": 20},
                    {
                        "kind": "range",
                        "field": "ipo_date",
                        "min": "2026-01-01",
                        "max": "2026-07-13",
                    },
                ],
            }
        }
    )
    numeric, ipo_date = request.to_expression().required.conditions

    assert numeric.min_value == 10.5
    assert numeric.max_value == 20
    assert ipo_date.min_value == "2026-01-01"


def test_match_annotations_keep_missing_boolean_distinct_from_false():
    item = ScanResultItemDomain(
        symbol="MISS",
        composite_score=80,
        rating="Buy",
        current_price=25,
        screener_outputs={},
        screeners_run=[],
        composite_method="weighted_average",
        screeners_passed=0,
        screeners_total=0,
        extended_fields={"passes_template": None},
    )
    expression = FilterExpression(
        group_join=MatchOperator.ANY,
        groups=(
            FilterGroup(
                id="not-passing",
                name="Not passing",
                conditions=(BooleanFilter("passes_template", False),),
            ),
            FilterGroup(
                id="score",
                name="Score",
                conditions=(RangeFilter("composite_score", min_value=70),),
            ),
        ),
    )

    annotated = annotate_matched_groups((item,), expression)[0]

    assert annotated.matched_groups == (MatchedGroupDomain(id="score", name="Score"),)
    assert "matched_groups" not in annotated.extended_fields


def test_request_contract_builds_domain_and_rejects_empty_enabled_group():
    request = ScanQueryRequest.model_validate(
        {
            "expression_version": 1,
            "required": {
                "id": "required",
                "name": "Always require",
                "match": "all",
                "conditions": [
                    {"kind": "range", "field": "price", "min": 10, "max": 50}
                ],
            },
            "groups": [
                {
                    "id": "breakout",
                    "name": "Breakout ready",
                    "match": "any",
                    "conditions": [
                        {"kind": "boolean", "field": "vcp_detected", "value": True}
                    ],
                }
            ],
        }
    )
    assert request.to_expression().groups[0].name == "Breakout ready"

    with pytest.raises(ValidationError, match="Enabled setup groups cannot be empty"):
        ScanQueryRequest.model_validate(
            {"groups": [{"id": "empty", "name": "Empty", "conditions": []}]}
        )


def test_request_contract_rejects_unknown_sort_and_accepts_listing_discovery():
    with pytest.raises(ValidationError, match="Unsupported sort field"):
        ScanQueryRequest.model_validate({"sort": {"field": "not_a_real_field"}})

    request = ScanQueryRequest.model_validate(
        {
            "required": {
                "id": "required",
                "name": "Always require",
                "conditions": [{"kind": "listing_discovery", "min_volume": 1_000_000}],
            }
        }
    )
    assert request.to_expression().required.conditions == (
        ListingDiscoveryFilter(min_volume=1_000_000),
    )


def test_shared_browser_and_backend_truth_table():
    fixture_path = (
        Path(__file__).resolve().parents[4]
        / "contracts"
        / "scan_filter_truth_table.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expression = expression_from_payload(fixture["expression"])

    for case in fixture["rows"]:
        assert evaluate_expression(case["row"], expression) is case["matches"]
        if case["matches"]:
            assert [
                group.id for group in matched_setup_groups(case["row"], expression)
            ] == case["matched_groups"]


def test_fingerprint_is_stable_and_changes_with_logic():
    expression = _expression()
    assert expression_fingerprint(expression) == expression_fingerprint(expression)
    assert expression_fingerprint(expression) != expression_fingerprint(
        _expression(MatchOperator.ALL)
    )


def test_domain_expression_rejects_invalid_structure_and_ranges():
    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        FilterExpression(
            required=FilterGroup(
                id="required",
                name="Always require",
                conditions=(RangeFilter("price", min_value=100, max_value=10),),
            )
        )

    with pytest.raises(ValueError, match="Enabled setup groups cannot be empty"):
        FilterExpression(groups=(FilterGroup(id="empty", name="Empty", conditions=()),))

    with pytest.raises(ValueError, match="Group joins must be MatchOperator"):
        FilterExpression(group_join="some")

    with pytest.raises(ValueError, match="Numeric range bounds must be numbers"):
        FilterExpression(
            required=FilterGroup(
                id="required",
                name="Always require",
                conditions=(RangeFilter("price", min_value="10"),),
            )
        )


def test_payload_codec_rejects_string_booleans_instead_of_inverting_them():
    payload = {
        "expression_version": 1,
        "required": {
            "id": "required",
            "name": "Always require",
            "match": "all",
            "enabled": True,
            "conditions": [],
        },
        "groups": [
            {
                "id": "boolean",
                "name": "Boolean",
                "match": "all",
                "enabled": True,
                "conditions": [
                    {
                        "kind": "boolean",
                        "field": "vcp_detected",
                        "value": "false",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValidationError, match="valid boolean"):
        expression_from_payload(payload)

    with pytest.raises(ValidationError, match="valid boolean"):
        ScanQueryRequest.model_validate(payload)


def test_payload_codecs_reject_values_they_cannot_preserve():
    payload = {
        "required": {
            "id": "required",
            "name": "Always require",
            "conditions": [
                {
                    "kind": "categorical",
                    "field": "rating",
                    "values": ["Buy", 1],
                }
            ],
        }
    }

    with pytest.raises(ValidationError, match="valid string"):
        expression_from_payload(payload)

    with pytest.raises(ValueError, match="must be a boolean"):
        legacy_filters_to_expression({"maAlignment": "false"})


def test_static_decoder_accepts_legacy_aliases_that_the_live_api_rejects():
    payload = {
        "required": {
            "id": "required",
            "name": "Always require",
            "conditions": [
                {"kind": "range", "field": "pct_day", "min": 5, "max": None}
            ],
        }
    }

    expression = expression_from_payload(payload)
    assert expression.required.conditions == (RangeFilter("pct_day", min_value=5),)

    with pytest.raises(ValidationError, match="Unsupported range field: pct_day"):
        ScanQueryRequest.model_validate(payload)

    with pytest.raises(ValidationError, match="Input should be 1"):
        expression_from_payload({"expression_version": "1"})

    with pytest.raises(ValidationError, match="valid string"):
        expression_from_payload(
            {
                "required": {
                    "id": "required",
                    "name": "Always require",
                    "match": "all",
                    "conditions": [{"kind": "range", "field": 1, "min": 10}],
                },
                "group_join": "any",
            }
        )
