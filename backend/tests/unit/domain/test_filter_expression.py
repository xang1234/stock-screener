import pytest
from pydantic import ValidationError

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterExpression,
    FilterGroup,
    FilterMode,
    MatchOperator,
    RangeFilter,
)
from app.domain.scanning.filter_expression import (
    evaluate_condition,
    evaluate_expression,
    expression_fingerprint,
    matched_setup_groups,
)
from app.schemas.filter_expression import ScanQueryRequest


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
        {"id": "breakout", "name": "Breakout ready"},
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
        groups=(expression.groups[0], FilterGroup(**{**expression.groups[1].__dict__, "enabled": False})),
    )
    row = {"price": 25, "rs_rating": 95, "vcp_ready_for_breakout": True}
    assert evaluate_expression(row, expression) is True


def test_missing_values_follow_explicit_policy():
    assert evaluate_condition({}, BooleanFilter("ma_alignment", False)) is False
    assert evaluate_condition(
        {},
        CategoricalFilter(
            "rating", ("Pass",), mode=FilterMode.EXCLUDE
        ),
    ) is True
    assert evaluate_condition({}, RangeFilter("rs_rating", min_value=80)) is False


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
            {
                "groups": [
                    {"id": "empty", "name": "Empty", "conditions": []}
                ]
            }
        )


def test_fingerprint_is_stable_and_changes_with_logic():
    expression = _expression()
    assert expression_fingerprint(expression) == expression_fingerprint(expression)
    assert expression_fingerprint(expression) != expression_fingerprint(
        _expression(MatchOperator.ALL)
    )
