"""Versioned HTTP contract for guided scan-result filter expressions."""

from __future__ import annotations

from datetime import date
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterCondition,
    FilterExpression,
    FilterGroup,
    FilterMode,
    ListingDiscoveryFilter,
    MatchOperator,
    PageSpec,
    RangeFilter,
    SortOrder,
    SortSpec,
    TextSearchFilter,
)
from app.domain.scanning.filter_capabilities import (
    BOOLEAN_FIELDS,
    CATEGORICAL_FIELDS,
    RANGE_FIELDS,
    SORT_FIELDS,
    TEXT_FIELDS,
)


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RangeConditionRequest(_ContractModel):
    kind: Literal["range"]
    field: str
    min: int | float | str | None = None
    max: int | float | str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_bounds(cls, value):
        if isinstance(value, dict) and any(
            isinstance(value.get(bound), bool) for bound in ("min", "max")
        ):
            raise ValueError("Numeric range bounds cannot be booleans")
        return value

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in RANGE_FIELDS:
            raise ValueError(f"Unsupported range field: {value}")
        return value

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min is None and self.max is None:
            raise ValueError("Range conditions require a minimum or maximum")

        if self.field == "ipo_date":
            self.min = _normalize_iso_date(self.min)
            self.max = _normalize_iso_date(self.max)
        else:
            self.min = _normalize_finite_number(self.min)
            self.max = _normalize_finite_number(self.max)

        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Range minimum cannot exceed maximum")
        return self

    def to_domain(self) -> RangeFilter:
        return RangeFilter(field=self.field, min_value=self.min, max_value=self.max)


def _normalize_iso_date(value: int | float | str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("IPO date bounds must use ISO YYYY-MM-DD strings")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError("IPO date bounds must use ISO YYYY-MM-DD strings") from exc


def _normalize_finite_number(value: int | float | str | None) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Numeric range bounds cannot be booleans")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Numeric range bounds must be finite numbers") from exc
    if not math.isfinite(normalized):
        raise ValueError("Numeric range bounds must be finite numbers")
    return int(normalized) if normalized.is_integer() else normalized


class CategoricalConditionRequest(_ContractModel):
    kind: Literal["categorical"]
    field: str
    values: list[str] = Field(min_length=1, max_length=100)
    mode: Literal["include", "exclude"] = "include"

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in CATEGORICAL_FIELDS:
            raise ValueError(f"Unsupported categorical field: {value}")
        return value

    @field_validator("values")
    @classmethod
    def normalize_values(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not normalized:
            raise ValueError("Categorical conditions require at least one value")
        return normalized

    def to_domain(self) -> CategoricalFilter:
        return CategoricalFilter(
            field=self.field,
            values=tuple(self.values),
            mode=FilterMode(self.mode),
        )


class BooleanConditionRequest(_ContractModel):
    kind: Literal["boolean"]
    field: str
    value: bool

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in BOOLEAN_FIELDS:
            raise ValueError(f"Unsupported boolean field: {value}")
        return value

    def to_domain(self) -> BooleanFilter:
        return BooleanFilter(field=self.field, value=self.value)


class TextConditionRequest(_ContractModel):
    kind: Literal["text"]
    field: str
    pattern: str = Field(min_length=1, max_length=100)

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in TEXT_FIELDS:
            raise ValueError(f"Unsupported text field: {value}")
        return value

    @field_validator("pattern")
    @classmethod
    def normalize_pattern(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Text patterns cannot be blank")
        return normalized

    def to_domain(self) -> TextSearchFilter:
        return TextSearchFilter(field=self.field, pattern=self.pattern)


class ListingDiscoveryConditionRequest(_ContractModel):
    kind: Literal["listing_discovery"]
    min_volume: float = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_volume(cls, value):
        if isinstance(value, dict) and isinstance(value.get("min_volume"), bool):
            raise ValueError("Listing-discovery volume must be a positive number")
        return value

    def to_domain(self) -> ListingDiscoveryFilter:
        return ListingDiscoveryFilter(min_volume=self.min_volume)


FilterConditionRequest = Annotated[
    RangeConditionRequest
    | CategoricalConditionRequest
    | BooleanConditionRequest
    | TextConditionRequest
    | ListingDiscoveryConditionRequest,
    Field(discriminator="kind"),
]


class FilterGroupRequest(_ContractModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=60)
    match: Literal["all", "any"] = "all"
    conditions: list[FilterConditionRequest] = Field(default_factory=list, max_length=20)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Group names cannot be blank")
        return normalized

    def to_domain(self) -> FilterGroup:
        conditions: tuple[FilterCondition, ...] = tuple(
            condition.to_domain() for condition in self.conditions
        )
        return FilterGroup(
            id=self.id,
            name=self.name,
            match=MatchOperator(self.match),
            conditions=conditions,
            enabled=self.enabled,
        )


class SortRequest(_ContractModel):
    field: str = Field(default="composite_score", min_length=1, max_length=80)
    order: Literal["asc", "desc"] = "desc"

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in SORT_FIELDS:
            raise ValueError(f"Unsupported sort field: {value}")
        return value

    def to_domain(self) -> SortSpec:
        return SortSpec(field=self.field, order=SortOrder(self.order))


class PageRequest(_ContractModel):
    number: int = Field(default=1, ge=1)
    size: int = Field(default=50, ge=1, le=100)

    def to_domain(self) -> PageSpec:
        return PageSpec(page=self.number, per_page=self.size)


class QueryOptionsRequest(_ContractModel):
    detail_level: Literal["table", "full"] = "table"
    include_sparklines: bool = True


class ScanQueryRequest(_ContractModel):
    expression_version: Literal[1] = 1
    required: FilterGroupRequest = Field(
        default_factory=lambda: FilterGroupRequest(
            id="required", name="Always require", match="all"
        )
    )
    group_join: Literal["all", "any"] = "any"
    groups: list[FilterGroupRequest] = Field(default_factory=list, max_length=8)
    sort: SortRequest = Field(default_factory=SortRequest)
    page: PageRequest | None = None
    options: QueryOptionsRequest = Field(default_factory=QueryOptionsRequest)
    passes_only: bool = False

    @model_validator(mode="after")
    def validate_expression(self):
        if self.required.id != "required" or self.required.match != "all":
            raise ValueError("The required group must use id='required' and match='all'")
        if not self.required.enabled:
            raise ValueError("The required group cannot be disabled")
        enabled_groups = [group for group in self.groups if group.enabled]
        if any(not group.conditions for group in enabled_groups):
            raise ValueError("Enabled setup groups cannot be empty")
        ids = [group.id for group in self.groups]
        if len(ids) != len(set(ids)) or "required" in ids:
            raise ValueError("Setup group IDs must be unique and cannot use 'required'")
        total = len(self.required.conditions) + sum(
            len(group.conditions) for group in self.groups
        )
        if total > 100:
            raise ValueError("An expression can contain at most 100 conditions")
        return self

    def to_expression(self) -> FilterExpression:
        return FilterExpression(
            required=self.required.to_domain(),
            group_join=MatchOperator(self.group_join),
            groups=tuple(group.to_domain() for group in self.groups),
            version=self.expression_version,
        )


__all__ = [
    "BOOLEAN_FIELDS",
    "CATEGORICAL_FIELDS",
    "RANGE_FIELDS",
    "TEXT_FIELDS",
    "FilterGroupRequest",
    "ScanQueryRequest",
]
