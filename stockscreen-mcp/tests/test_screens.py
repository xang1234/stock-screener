"""The evaluator mirrors frontend/src/features/scan/filterExpressionEvaluator.js."""

import json

import pytest

from stockscreen_mcp.screens import (
    evaluate_condition,
    evaluate_expression,
    explain_condition,
    explain_expression,
    filters_to_conditions,
    preset_requires_opportunity_state,
    sort_rows,
)

from conftest import PRESETS, REPO, condition


def _contract(name):
    # Shared with frontend/src/features/scan/filterExpression.test.js. Outside the
    # repository checkout (an unpacked sdist) the cases are absent and these skip;
    # inside it a missing file fails collection.
    if not (REPO / "backend").exists():
        return {}
    return json.loads((REPO / "contracts" / name).read_text())


CASES = _contract("scan_filter_condition_cases.json")
TRUTH_TABLE = _contract("scan_filter_truth_table.json")


@pytest.mark.parametrize("case", CASES.get("conditions", []))
def test_condition_matches_the_frontend(case):
    assert evaluate_condition(case["row"], case["condition"]) is case["matches"]


@pytest.mark.parametrize("case", CASES.get("expression_rows", []))
def test_expression_matches_the_frontend(case):
    expression = CASES["expression"] | {"group_join": case["group_join"]}
    assert evaluate_expression(case["row"], expression) is case["matches"]


@pytest.mark.parametrize("case", TRUTH_TABLE.get("rows", []))
def test_expression_matches_the_browser_and_backend_truth_table(case):
    assert evaluate_expression(case["row"], TRUTH_TABLE["expression"]) is case["matches"]


def test_presets_using_opportunity_state_fields_are_flagged():
    flagged = {p["id"] for p in PRESETS if preset_requires_opportunity_state(p)}
    assert flagged == {"correction_survivors"}
    assert preset_requires_opportunity_state(PRESETS[0] | {"sort_by": "resilience_score"})


def test_filters_become_frontend_conditions():
    conditions = filters_to_conditions([
        {"field": "rs_rating", "min": 90},
        {"field": "stage", "min": 2, "max": 2},
        {"field": "ma_alignment", "equals": True},
        {"field": "se_pattern_primary", "equals": "cup_handle"},
    ])
    assert conditions == [
        {"field": "rs_rating", "kind": "range", "min": 90, "max": None},
        {"field": "stage", "kind": "range", "min": 2, "max": 2},
        {"field": "ma_alignment", "kind": "boolean", "value": True},
        {"field": "se_pattern_primary", "kind": "categorical", "values": ["cup_handle"], "mode": "include"},
    ]


ROWS = [
    {"symbol": "D", "v": 2, "composite_score": 2},
    {"symbol": "A", "v": None, "composite_score": None},
    {"symbol": "C", "v": 2, "composite_score": 2},
    {"symbol": "B", "v": 5, "composite_score": 5},
    {"symbol": "E"},
]


def symbols(rows):
    return [r["symbol"] for r in rows]


def test_site_order_breaks_ties_by_symbol_in_both_directions():
    # sortStaticScanRows: the symbol tiebreak is not reversed for descending sorts.
    assert symbols(sort_rows(ROWS, "v", descending=True))[-2:] == ["C", "D"]
    assert symbols(sort_rows(ROWS, "v", descending=False))[:2] == ["C", "D"]


def test_site_order_puts_nulls_first_when_descending_except_for_score_fields():
    assert symbols(sort_rows(ROWS, "v", descending=True)) == ["A", "E", "B", "C", "D"]
    assert symbols(sort_rows(ROWS, "v", descending=False)) == ["C", "D", "B", "A", "E"]
    assert symbols(sort_rows(ROWS, "composite_score", descending=True)) == ["B", "C", "D", "A", "E"]
    assert symbols(sort_rows(ROWS, "composite_score", descending=False)) == ["C", "D", "B", "A", "E"]


@pytest.mark.parametrize("site_order", [True, False])
def test_rating_sorts_by_rank_not_alphabetically(site_order):
    # scanClient.js getSortValue maps rating labels to ranks before comparing.
    rows = [{"symbol": s, "rating": r} for s, r in
            [("A", "Watch"), ("B", "Strong Buy"), ("C", "Pass"), ("D", "Buy"), ("E", "Insufficient Data")]]
    ordered = sort_rows(rows, "rating", descending=True, site_order=site_order)
    assert [r["rating"] for r in ordered] == ["Strong Buy", "Buy", "Watch", "Pass", "Insufficient Data"]


def test_caller_sorts_keep_rows_without_the_field_last():
    assert symbols(sort_rows(ROWS, "v", descending=True, site_order=False)) == ["B", "C", "D", "A", "E"]


@pytest.mark.parametrize(
    ("row", "cond", "expected"),
    [
        ({"rs_rating": 60}, condition("rs_rating", "range", min=70, max=None),
         {"field": "rs_rating", "value": 60, "needs": {"min": 70}, "passes": False, "short_by": 10.0}),
        ({"extended": 30}, condition("extended", "range", min=None, max=25),
         {"field": "extended", "value": 30, "needs": {"max": 25}, "passes": False, "short_by": 5.0}),
        ({"stage": 3}, condition("stage", "range", min=2, max=2),
         {"field": "stage", "value": 3, "needs": {"min": 2, "max": 2}, "passes": False, "short_by": 1.0}),
        ({"stage": 1}, condition("stage", "range", min=2, max=2),
         {"field": "stage", "value": 1, "needs": {"min": 2, "max": 2}, "passes": False, "short_by": 1.0}),
        ({"rs_rating": 80}, condition("rs_rating", "range", min=70, max=None),
         {"field": "rs_rating", "value": 80, "needs": {"min": 70}, "passes": True}),
        ({"canslim_score": 50.52}, condition("canslim_score", "range", min=70, max=None),
         {"field": "canslim_score", "value": 50.52, "needs": {"min": 70}, "passes": False, "short_by": 19.48}),
        ({"atr": 1.4999999}, condition("atr", "range", min=1.5, max=None),
         {"field": "atr", "value": 1.4999999, "needs": {"min": 1.5}, "passes": False, "short_by": 1.00000000058e-07}),
        ({"atr": 1.5000001}, condition("atr", "range", min=None, max=1.5),
         {"field": "atr", "value": 1.5000001, "needs": {"max": 1.5}, "passes": False, "short_by": 1.00000000058e-07}),
        ({"rs_rating": None}, condition("rs_rating", "range", min=70, max=None),
         {"field": "rs_rating", "value": None, "needs": {"min": 70}, "passes": False}),
        ({}, condition("rs_rating", "range", min=70, max=None),
         {"field": "rs_rating", "value": None, "needs": {"min": 70}, "passes": False}),
        ({"x": -1e308}, condition("x", "range", min=1e308, max=None),
         {"field": "x", "value": -1e308, "needs": {"min": 1e308}, "passes": False}),
        ({"rs_rating": "50"}, condition("rs_rating", "range", min=70, max=None),
         {"field": "rs_rating", "value": "50", "needs": {"min": 70}, "passes": False, "short_by": 20.0}),
        ({"current_price": 4}, condition("price", "range", min=10, max=None),
         {"field": "price", "value": 4, "needs": {"min": 10}, "passes": False, "short_by": 6.0}),
        ({"scan_mode": "listing_only"}, condition("listing_aware_volume", "range", min=100, max=None),
         {"field": "listing_aware_volume", "value": None, "needs": {"min": 100}, "passes": True}),
        ({"scan_mode": "listing_only"}, condition("listing_aware_volume", "range", min=None, max=100),
         {"field": "listing_aware_volume", "value": None, "needs": {"max": 100}, "passes": False}),
        ({"ma_alignment": "true"}, condition("ma_alignment", "boolean", value=True),
         {"field": "ma_alignment", "value": "true", "needs": {"is": True}, "passes": False}),
        ({"p": "cup"}, condition("p", "categorical", values=["cup"], mode="include"),
         {"field": "p", "value": "cup", "needs": {"in": ["cup"]}, "passes": True}),
        ({"p": "cup"}, condition("p", "categorical", values=["cup", "vcp"], mode="exclude"),
         {"field": "p", "value": "cup", "needs": {"not_in": ["cup", "vcp"]}, "passes": False}),
        ({"group": "Semiconductors"}, condition("group", "text", pattern="semi"),
         {"field": "group", "value": "Semiconductors", "needs": {"contains": "semi"}, "passes": True}),
    ],
)
def test_explain_condition_reports_the_requirement_and_shortfall(row, cond, expected):
    assert explain_condition(row, cond) == expected


RS_RULE = condition("rs_rating", "range", min=70, max=None)
STAGE_RULE = condition("stage", "range", min=2, max=2)
MET_GROUP = {"enabled": True, "match": "all", "conditions": [STAGE_RULE]}
UNMET_GROUP = {"enabled": True, "match": "all", "conditions": [condition("sector", "categorical", values=["Tech"])]}
RS_FAILS = {"field": "rs_rating", "value": 60, "needs": {"min": 70}, "short_by": 10.0}
EPS_RULE = condition("eps", "boolean", value=True)
EPS_FAILS = {"field": "eps", "value": None, "needs": {"is": True}}
GROUP_NOTE = "fails an optional rule group"
DISABLED_NOTE = "required rule group is disabled"


@pytest.mark.parametrize(
    ("required", "groups", "group_join", "expected"),
    [
        ({"match": "all", "conditions": [STAGE_RULE]}, [], "any",
         {"passes": True, "failed": [], "near_miss": False}),
        ({"match": "all", "conditions": [RS_RULE, STAGE_RULE]}, [], "any",
         {"passes": False, "failed": [RS_FAILS], "near_miss": True}),
        ({"match": "all", "conditions": [RS_RULE, EPS_RULE]}, [], "any",
         {"passes": False, "failed": [RS_FAILS, EPS_FAILS], "near_miss": False}),
        ({"match": "all", "conditions": [RS_RULE]}, [], "any",
         {"passes": False, "failed": [RS_FAILS], "near_miss": False}),
        ({"match": "any", "conditions": [RS_RULE, STAGE_RULE]}, [], "any",
         {"passes": True, "failed": [], "near_miss": False}),
        ({"match": "any", "conditions": [RS_RULE, EPS_RULE]}, [], "any",
         {"passes": False, "failed": [RS_FAILS, EPS_FAILS], "near_miss": False}),
        ({"enabled": False, "match": "all", "conditions": [STAGE_RULE]}, [], "any",
         {"passes": False, "failed": [], "near_miss": False, "note": DISABLED_NOTE}),
        ({"enabled": False, "match": "all", "conditions": [RS_RULE, STAGE_RULE]}, [UNMET_GROUP], "any",
         {"passes": False, "failed": [RS_FAILS], "near_miss": False, "note": DISABLED_NOTE}),
        # Fixing the one rule would not make the stock pass while the groups fail.
        ({"match": "all", "conditions": [RS_RULE, STAGE_RULE]}, [UNMET_GROUP], "any",
         {"passes": False, "failed": [RS_FAILS], "near_miss": False, "note": GROUP_NOTE}),
        ({"match": "all", "conditions": [STAGE_RULE]}, [UNMET_GROUP], "any",
         {"passes": False, "failed": [], "near_miss": False, "note": GROUP_NOTE}),
        ({"match": "all", "conditions": [RS_RULE, STAGE_RULE]}, [MET_GROUP, UNMET_GROUP], "any",
         {"passes": False, "failed": [RS_FAILS], "near_miss": True}),
        ({"match": "all", "conditions": [RS_RULE, STAGE_RULE]}, [MET_GROUP, UNMET_GROUP], "all",
         {"passes": False, "failed": [RS_FAILS], "near_miss": False, "note": GROUP_NOTE}),
        ({"match": "all", "conditions": [RS_RULE, STAGE_RULE]}, [UNMET_GROUP | {"enabled": False}], "any",
         {"passes": False, "failed": [RS_FAILS], "near_miss": True}),
    ],
)
def test_explain_expression_keeps_the_verdict_and_its_reasons_together(required, groups, group_join, expected):
    row = {"rs_rating": 60, "stage": 2}
    expression = {"required": required, "groups": groups, "group_join": group_join}

    assert explain_expression(row, expression) == expected
