"""Screen evaluation, ported from frontend/src/features/scan/filterExpressionEvaluator.js.

Kept line-for-line equivalent so a preset matches exactly the rows the static
site shows for it.
"""

from __future__ import annotations

import functools
import math
from typing import Any

# frontend/src/features/scan/scanFilterFields.js: OPPORTUNITY_STATE_FIELDS
OPPORTUNITY_STATE_FIELDS = frozenset({"correction_survivor", "resilience_score", "action_state"})


def _first_present(row: dict[str, Any], field: str, fallback: str) -> Any:
    value = row.get(field)
    return row.get(fallback) if value is None else value


def row_value(row: dict[str, Any], field: str) -> Any:
    if field == "price":
        return _first_present(row, "price", "current_price")
    if field == "listing_search":
        return f"{row.get('symbol') or ''} {row.get('company_name') or ''}".strip()
    if field == "listing_aware_volume":
        return math.inf if row.get("scan_mode") == "listing_only" else row.get("volume")
    if field == "price_change_1d":
        return _first_present(row, "price_change_1d", "pct_day")
    if field == "perf_week":
        return _first_present(row, "perf_week", "pct_week")
    if field == "perf_month":
        return _first_present(row, "perf_month", "pct_month")
    if field == "market_cap_usd":
        if row.get("market_cap_usd") is not None:
            return row["market_cap_usd"]
        usd = str(row.get("currency") or "").upper() == "USD" or row.get("market") == "US"
        return row.get("market_cap") if usd else None
    return row.get(field)


def _js_number(value: Any) -> float:
    """JS Number(): booleans are 0/1, numeric strings convert, anything else is NaN."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return math.nan
    return math.nan


def _less(a: Any, b: Any) -> bool:
    """JS `a < b`: string vs string compares text, otherwise numbers (NaN is never less)."""
    if isinstance(a, str) and isinstance(b, str):
        return a < b
    return _js_number(a) < _js_number(b)


def _js_equal(a: Any, b: Any) -> bool:
    """JS `===` / includes(): true is not 1, 1 is 1.0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _range(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    value = row_value(row, condition["field"])
    if value is None:
        return False
    if condition.get("min") is not None and _less(value, condition["min"]):
        return False
    if condition.get("max") is not None and _less(condition["max"], value):
        return False
    return True


def _categorical(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    value = row_value(row, condition["field"])
    included = value is not None and any(_js_equal(value, v) for v in condition.get("values", []))
    if condition.get("mode") == "exclude":
        return value is None or not included
    return included


def _boolean(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    value = row_value(row, condition["field"])
    return isinstance(value, bool) and _js_equal(value, condition.get("value"))


def _text(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    value = row_value(row, condition["field"])
    return value is not None and str(condition.get("pattern", "")).lower() in str(value).lower()


_EVALUATORS = {"range": _range, "categorical": _categorical, "boolean": _boolean, "text": _text}


def evaluate_condition(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    evaluator = _EVALUATORS.get(condition.get("kind"))
    return bool(evaluator and evaluator(row, condition))


_NEEDS = {
    "range": lambda c: {bound: c[bound] for bound in ("min", "max") if c.get(bound) is not None},
    "categorical": lambda c: {"not_in" if c.get("mode") == "exclude" else "in": c.get("values", [])},
    "boolean": lambda c: {"is": c.get("value")},
    "text": lambda c: {"contains": c.get("pattern", "")},
}


def explain_condition(row: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    """One condition's verdict for a row: its value, the requirement, and any numeric shortfall."""
    kind = condition.get("kind")
    value = row_value(row, condition["field"])
    if isinstance(value, float) and not math.isfinite(value):
        value = None  # inf/NaN (e.g. listing_aware_volume) are not valid JSON
    passes = evaluate_condition(row, condition)
    explanation = {
        "field": condition["field"],
        "value": value,
        "needs": _NEEDS[kind](condition) if kind in _NEEDS else {},
        "passes": passes,
    }
    number = _js_number(value)
    if kind == "range" and not passes and math.isfinite(number):
        # Compares the number, not the raw value, so text bounds can't yield a negative or NaN gap.
        # Rounded because 70 - 50.52 is 19.479999999999997, and callers repeat this number to people.
        if _less(number, condition.get("min")):
            explanation["short_by"] = round(_js_number(condition["min"]) - number, 6)
        elif _less(condition.get("max"), number):
            explanation["short_by"] = round(number - _js_number(condition["max"]), 6)
    return explanation


def evaluate_group(row: dict[str, Any], group: dict[str, Any]) -> bool:
    if group.get("enabled") is False:
        return False
    matches = [evaluate_condition(row, c) for c in group.get("conditions") or []]
    return any(matches) if group.get("match") == "any" else all(matches)


def evaluate_expression(row: dict[str, Any], expression: dict[str, Any]) -> bool:
    if not evaluate_group(row, expression["required"]):
        return False
    groups = [g for g in expression.get("groups") or [] if g.get("enabled") is not False]
    if not groups:
        return True
    matches = [evaluate_group(row, g) for g in groups]
    return all(matches) if expression.get("group_join") == "all" else any(matches)


def expression_fields(expression: dict[str, Any]) -> list[str]:
    groups = [expression["required"], *(expression.get("groups") or [])]
    return [c["field"] for g in groups for c in g.get("conditions") or []]


def preset_requires_opportunity_state(preset: dict[str, Any]) -> bool:
    """opportunityCapabilityPolicy.js: hidden where the market lacks opportunity_state."""
    fields = {*expression_fields(preset["filter_expression"]), preset.get("sort_by")}
    return bool(fields & OPPORTUNITY_STATE_FIELDS)


def filters_to_conditions(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool-facing {field, min, max, equals} filters as frontend expression conditions."""
    conditions = []
    for f in filters:
        equals = f.get("equals")
        if isinstance(equals, bool):
            conditions.append({"field": f["field"], "kind": "boolean", "value": equals})
        elif isinstance(equals, str):
            conditions.append({"field": f["field"], "kind": "categorical", "values": [equals], "mode": "include"})
        elif equals is not None:
            conditions.append({"field": f["field"], "kind": "range", "min": equals, "max": equals})
        else:
            conditions.append({"field": f["field"], "kind": "range", "min": f.get("min"), "max": f.get("max")})
    return conditions


# frontend/src/static/scanClient.js: SCORE_SORT_FIELDS keep nulls last in both directions,
# and getSortValue ranks rating labels instead of comparing them as text.
SCORE_SORT_FIELDS = frozenset({"composite_score", "resilience_score"})
RATING_SORT_ORDER = {"Strong Buy": 5, "Buy": 4, "Watch": 3, "Pass": 2, "Error": 1, "Insufficient Data": 0}


def _sort_value(row: dict[str, Any], field: str) -> Any:
    if field == "rating":
        return RATING_SORT_ORDER.get(row.get("rating"), 0)
    return row.get(field)


def _compare_values(left: Any, right: Any) -> int:
    """scanClient.js compareValues: nulls last (before the direction is applied)."""
    if left is None and right is None:
        return 0
    if left is None:
        return 1
    if right is None:
        return -1
    if isinstance(left, str) or isinstance(right, str):
        # ponytail: approximates localeCompare's collation (case-insensitive first) without ICU.
        lkey, rkey = (str(left).casefold(), str(left)), (str(right).casefold(), str(right))
        return (lkey > rkey) - (lkey < rkey)
    diff = _js_number(left) - _js_number(right)
    return 0 if math.isnan(diff) else (diff > 0) - (diff < 0)


def sort_rows(
    rows: list[dict[str, Any]], field: str, *, descending: bool, site_order: bool = True
) -> list[dict[str, Any]]:
    """Sort like the site's sortStaticScanRows: ties by symbol, ascending in both directions.

    With site_order (presets, default order) nulls follow the site: last for score fields,
    first when descending otherwise. A caller-chosen sort keeps rows without the field last.
    """
    direction = -1 if descending else 1
    nulls_last = field in SCORE_SORT_FIELDS or not site_order

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        left_value, right_value = _sort_value(left, field), _sort_value(right, field)
        if nulls_last:
            if left_value is None and right_value is None:
                return _compare_values(left.get("symbol"), right.get("symbol"))
            if left_value is None:
                return 1
            if right_value is None:
                return -1
        comparison = _compare_values(left_value, right_value)
        if comparison:
            return comparison * direction
        return _compare_values(left.get("symbol"), right.get("symbol"))

    return sorted(rows, key=functools.cmp_to_key(compare))
