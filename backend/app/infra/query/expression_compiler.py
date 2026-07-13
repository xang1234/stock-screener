"""Storage-agnostic traversal for bounded scan filter expressions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import and_, false, or_, true

from app.domain.scanning.filter_expression_model import (
    FilterCondition,
    FilterExpression,
    FilterGroup,
    MatchOperator,
)


LeafCompiler = Callable[[FilterCondition], Any]


def compile_expression(
    expression: FilterExpression,
    compile_leaf: LeafCompiler,
):
    """Compile expression structure while delegating storage-specific leaves."""

    required = _compile_group(expression.required, compile_leaf)
    groups = expression.enabled_groups
    if not groups:
        return required

    predicates = [_compile_group(group, compile_leaf) for group in groups]
    joined = (
        and_(*predicates)
        if expression.group_join == MatchOperator.ALL
        else or_(*predicates)
    )
    return and_(required, joined)


def _compile_group(group: FilterGroup, compile_leaf: LeafCompiler):
    predicates = [compile_leaf(condition) for condition in group.conditions]
    if not predicates:
        return true() if group.match == MatchOperator.ALL else false()
    return and_(*predicates) if group.match == MatchOperator.ALL else or_(*predicates)


__all__ = ["compile_expression"]
