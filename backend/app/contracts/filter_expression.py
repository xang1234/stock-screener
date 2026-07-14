"""Typed external codec shared by HTTP and persisted scan-filter payloads."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterMode,
    RangeFilter,
    TextSearchFilter,
)
from app.domain.scanning.filter_capabilities import FIELD_CAPABILITIES
from app.domain.scanning.filter_expression_model import (
    GROUP_ID_PATTERN,
    MAX_CATEGORICAL_VALUES,
    MAX_EXPRESSION_GROUPS,
    MAX_GROUP_CONDITIONS,
    MAX_GROUP_ID_LENGTH,
    MAX_GROUP_NAME_LENGTH,
    MAX_TEXT_PATTERN_LENGTH,
    FilterCondition,
    FilterExpression,
    FilterGroup,
    MatchOperator,
)
from app.domain.scanning.filter_values import normalize_range_bound


class FilterExpressionFieldPolicy(str, Enum):
    """Select which registered fields an external boundary accepts."""

    API = "api"
    STATIC = "static"


class ExpressionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _filter_field(
    field: str,
    kind: str,
    policy: FilterExpressionFieldPolicy,
) -> str:
    capability = FIELD_CAPABILITIES.get(field)
    if capability is None or not capability.supports(kind):
        raise ValueError(f"Unsupported {kind} field: {field}")
    if policy == FilterExpressionFieldPolicy.API and not capability.supports_api(kind):
        raise ValueError(f"Unsupported {kind} field: {field}")
    return field


class RangeConditionPayload(ExpressionPayloadModel):
    kind: Literal["range"]
    field: str
    min: int | float | str | None = None
    max: int | float | str | None = None

    def to_domain(self, policy: FilterExpressionFieldPolicy) -> FilterCondition:
        field = _filter_field(self.field, self.kind, policy)
        return RangeFilter(
            field=field,
            min_value=normalize_range_bound(field, self.min),
            max_value=normalize_range_bound(field, self.max),
        )


class CategoricalConditionPayload(ExpressionPayloadModel):
    kind: Literal["categorical"]
    field: str
    values: list[str] = Field(min_length=1, max_length=MAX_CATEGORICAL_VALUES)
    mode: Literal["include", "exclude"] = "include"

    def to_domain(self, policy: FilterExpressionFieldPolicy) -> FilterCondition:
        field = _filter_field(self.field, self.kind, policy)
        values = tuple(dict.fromkeys(value.strip() for value in self.values if value.strip()))
        return CategoricalFilter(field=field, values=values, mode=FilterMode(self.mode))


class BooleanConditionPayload(ExpressionPayloadModel):
    kind: Literal["boolean"]
    field: str
    value: bool

    def to_domain(self, policy: FilterExpressionFieldPolicy) -> FilterCondition:
        return BooleanFilter(
            field=_filter_field(self.field, self.kind, policy),
            value=self.value,
        )


class TextConditionPayload(ExpressionPayloadModel):
    kind: Literal["text"]
    field: str
    pattern: str = Field(min_length=1, max_length=MAX_TEXT_PATTERN_LENGTH)

    def to_domain(self, policy: FilterExpressionFieldPolicy) -> FilterCondition:
        return TextSearchFilter(
            field=_filter_field(self.field, self.kind, policy),
            pattern=self.pattern.strip(),
        )


FilterConditionPayload = Annotated[
    RangeConditionPayload
    | CategoricalConditionPayload
    | BooleanConditionPayload
    | TextConditionPayload,
    Field(discriminator="kind"),
]


class FilterGroupPayload(ExpressionPayloadModel):
    id: str = Field(
        min_length=1,
        max_length=MAX_GROUP_ID_LENGTH,
        pattern=GROUP_ID_PATTERN,
    )
    name: str = Field(min_length=1, max_length=MAX_GROUP_NAME_LENGTH)
    match: Literal["all", "any"] = "all"
    conditions: list[FilterConditionPayload] = Field(
        default_factory=list,
        max_length=MAX_GROUP_CONDITIONS,
    )
    enabled: bool = True

    def to_domain(self, policy: FilterExpressionFieldPolicy) -> FilterGroup:
        return FilterGroup(
            id=self.id,
            name=self.name.strip(),
            match=MatchOperator(self.match),
            enabled=self.enabled,
            conditions=tuple(condition.to_domain(policy) for condition in self.conditions),
        )


class FilterExpressionPayload(ExpressionPayloadModel):
    expression_version: Literal[1] = 1
    required: FilterGroupPayload = Field(
        default_factory=lambda: FilterGroupPayload(
            id="required",
            name="Always require",
            match="all",
        )
    )
    group_join: Literal["all", "any"] = "any"
    groups: list[FilterGroupPayload] = Field(
        default_factory=list,
        max_length=MAX_EXPRESSION_GROUPS,
    )

    def to_domain_expression(
        self,
        policy: FilterExpressionFieldPolicy,
    ) -> FilterExpression:
        return FilterExpression(
            required=self.required.to_domain(policy),
            group_join=MatchOperator(self.group_join),
            groups=tuple(group.to_domain(policy) for group in self.groups),
            version=self.expression_version,
        )


def expression_from_payload(payload: Mapping[str, object]) -> FilterExpression:
    """Validate and decode a persisted/static expression through the typed codec."""

    return FilterExpressionPayload.model_validate(payload).to_domain_expression(
        FilterExpressionFieldPolicy.STATIC
    )


__all__ = [
    "ExpressionPayloadModel",
    "FilterConditionPayload",
    "FilterExpressionFieldPolicy",
    "FilterExpressionPayload",
    "FilterGroupPayload",
    "expression_from_payload",
]
