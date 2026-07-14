"""Versioned HTTP shape for guided scan-result filter expressions."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from app.contracts.filter_expression import (
    ExpressionPayloadModel,
    FilterExpressionFieldPolicy,
    FilterExpressionPayload,
    FilterGroupPayload,
)
from app.domain.common.query import PageSpec, SortOrder, SortSpec
from app.domain.scanning.filter_capabilities import SORT_FIELDS
from app.domain.scanning.filter_expression_model import FilterExpression


class SortRequest(ExpressionPayloadModel):
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


class PageRequest(ExpressionPayloadModel):
    number: int = Field(default=1, ge=1)
    size: int = Field(default=50, ge=1, le=100)

    def to_domain(self) -> PageSpec:
        return PageSpec(page=self.number, per_page=self.size)


class QueryOptionsRequest(ExpressionPayloadModel):
    detail_level: Literal["table", "full"] = "table"
    include_sparklines: bool = True


class ScanQueryRequest(FilterExpressionPayload):
    sort: SortRequest = Field(default_factory=SortRequest)
    page: PageRequest | None = None
    options: QueryOptionsRequest = Field(default_factory=QueryOptionsRequest)
    passes_only: bool = False

    @model_validator(mode="after")
    def validate_domain_expression(self):
        self.to_domain_expression(FilterExpressionFieldPolicy.API)
        return self

    def to_expression(self) -> FilterExpression:
        return self.to_domain_expression(FilterExpressionFieldPolicy.API)


FilterGroupRequest = FilterGroupPayload


__all__ = ["FilterGroupRequest", "ScanQueryRequest"]
