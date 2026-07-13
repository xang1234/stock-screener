"""Versioned HTTP shape for guided scan-result filter expressions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from app.domain.common.query import PageSpec, SortOrder, SortSpec
from app.domain.scanning.filter_capabilities import SORT_FIELDS
from app.domain.scanning.filter_expression import (
    FilterExpressionDecodePolicy,
    decode_filter_expression,
)
from app.domain.scanning.filter_expression_model import FilterExpression


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RangeConditionRequest(_ContractModel):
    kind: Literal["range"]
    field: str
    min: int | float | str | None = None
    max: int | float | str | None = None


class CategoricalConditionRequest(_ContractModel):
    kind: Literal["categorical"]
    field: str
    values: list[str] = Field(min_length=1, max_length=100)
    mode: Literal["include", "exclude"] = "include"


class BooleanConditionRequest(_ContractModel):
    kind: Literal["boolean"]
    field: str
    value: bool


class TextConditionRequest(_ContractModel):
    kind: Literal["text"]
    field: str
    pattern: str = Field(min_length=1, max_length=100)


class ListingDiscoveryConditionRequest(_ContractModel):
    kind: Literal["listing_discovery"]
    min_volume: int | float


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
    conditions: list[FilterConditionRequest] = Field(
        default_factory=list, max_length=20
    )
    enabled: bool = True


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

    _expression: FilterExpression = PrivateAttr()

    @model_validator(mode="after")
    def decode_expression(self):
        self._expression = decode_filter_expression(
            self.model_dump(
                include={
                    "expression_version",
                    "required",
                    "group_join",
                    "groups",
                }
            ),
            policy=FilterExpressionDecodePolicy.API,
        )
        return self

    def to_expression(self) -> FilterExpression:
        return self._expression


__all__ = ["FilterGroupRequest", "ScanQueryRequest"]
