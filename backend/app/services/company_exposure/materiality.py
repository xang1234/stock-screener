"""Disclosed and reproducibly calculated materiality (spec §6).

Materiality is typed evidence: basis (disclosed / calculated / qualitative /
unknown), metric, Decimal value, unit, currency, period, reporting scope,
denominator and operand citations. It is never ``exposure_strength`` or a
confidence score, and nothing here estimates a number the source did not
disclose.

V1 calculations: an explicit ratio of two compatible disclosed quantities,
and a sum of explicitly non-overlapping quantities. Any mismatch in period,
unit, currency, accounting basis or scope holds the derived value while the
original reported numbers are retained. A segment or subsidiary share is
reported as that scope's share, never relabelled as a theme share.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from app.domain.company_exposure.contracts import (
    UNKNOWN_MATERIALITY_WORDING,
    MaterialityBasis,
    QualitativeMateriality,
)

SHARE_METRICS = frozenset({"revenue_share", "profit_share", "capacity_share", "backlog_share", "asset_share"})
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class Operand:
    """One disclosed quantity with its provenance."""

    value: Decimal
    unit: str
    period: str
    scope: str
    label: str
    currency: str | None = None
    accounting_basis: str | None = None
    passage_id: str | None = None
    quote: str | None = None
    forecast: bool = False

    def compatibility_key(self) -> tuple:
        return (
            _norm(self.unit),
            (self.currency or "").upper(),
            _norm(self.period),
            _norm(self.accounting_basis or ""),
            self.forecast,
        )


@dataclass(frozen=True, slots=True)
class MaterialityMeasureResult:
    basis: MaterialityBasis
    metric: str | None = None
    value: Decimal | None = None
    value_high: Decimal | None = None
    unit: str | None = None
    currency: str | None = None
    period: str | None = None
    reporting_scope: str = "issuer_consolidated"
    scope_label: str | None = None
    denominator_definition: str | None = None
    qualitative_label: QualitativeMateriality | None = None
    formula: dict = field(default_factory=dict)
    operands: tuple[Operand, ...] = ()
    raw_reported: dict = field(default_factory=dict)
    hold_reasons: tuple[str, ...] = ()
    theme_specific: bool = False

    @property
    def held(self) -> bool:
        return bool(self.hold_reasons)

    @property
    def display(self) -> str:
        if self.basis == MaterialityBasis.UNKNOWN or self.held:
            return UNKNOWN_MATERIALITY_WORDING
        if self.value is None:
            return str(self.qualitative_label.value if self.qualitative_label else "")
        scope = f" of {self.scope_label}" if self.scope_label else ""
        return f"{self.metric} {self.value.normalize():f}{scope} ({self.period})"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


def parse_decimal(text: str) -> Decimal:
    """Parse a disclosed number exactly (commas removed, never via float)."""

    match = _NUMBER.search(str(text))
    if match is None:
        raise ValueError("no_number")
    try:
        value = Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        raise ValueError("invalid_number") from None
    if not value.is_finite():
        raise ValueError("invalid_number")
    return value


def quote_contains_value(quote: str | None, value: Decimal) -> bool:
    """A cited operand must appear in its quote (exact decimal equality)."""

    if not quote:
        return False
    for token in _NUMBER.findall(quote):
        try:
            if Decimal(token.replace(",", "")) == value:
                return True
        except InvalidOperation:
            continue
    return False


def unknown_materiality(reason: str | None = None) -> MaterialityMeasureResult:
    return MaterialityMeasureResult(
        basis=MaterialityBasis.UNKNOWN,
        raw_reported={"note": UNKNOWN_MATERIALITY_WORDING, "reason": reason},
    )


def validate_measure(
    *,
    metric: str,
    value: Decimal,
    unit: str,
    period: str,
    scope: str,
    scope_label: str | None,
    quote: str,
    passage_id: str | None,
    theme_terms: tuple[str, ...] = (),
    currency: str | None = None,
) -> MaterialityMeasureResult:
    """A directly disclosed figure (e.g. "Memory test was 20% of revenue")."""

    holds = []
    if not quote_contains_value(quote, value):
        holds.append("value_not_in_quote")
    if metric.endswith("_share") and not (Decimal(0) <= value <= Decimal(1)):
        holds.append("share_out_of_range")
    theme_specific = bool(scope_label) and any(
        term.casefold() in scope_label.casefold() for term in theme_terms
    )
    return MaterialityMeasureResult(
        basis=MaterialityBasis.DISCLOSED,
        metric=metric,
        value=value,
        unit=unit,
        currency=currency,
        period=period,
        reporting_scope=scope,
        scope_label=scope_label,
        operands=(
            Operand(value, unit, period, scope, scope_label or "", currency, None, passage_id, quote),
        ),
        raw_reported={"quote": quote, "value": format(value, "f"), "unit": unit},
        hold_reasons=tuple(holds),
        theme_specific=theme_specific and not holds,
    )


def compatible_ratio(
    *,
    numerator: Decimal,
    denominator: Decimal,
    numerator_scope: str,
    denominator_scope: str,
    period: str,
    unit: str,
    metric: str,
    operands_compatible: bool,
) -> MaterialityMeasureResult:
    """Exact ratio after provenance compatibility has been established."""

    holds = []
    if not operands_compatible:
        holds.append("operands_incompatible")
    if denominator <= 0:
        holds.append("nonpositive_denominator_review_required")
    value = None if holds else numerator / denominator
    if value is not None and metric in SHARE_METRICS and not (Decimal(0) <= value <= Decimal(1)):
        holds.append("share_out_of_range")
        value = None
    return MaterialityMeasureResult(
        basis=MaterialityBasis.CALCULATED,
        metric=metric,
        value=value,
        unit="ratio",
        period=period,
        reporting_scope="issuer_consolidated",
        scope_label=numerator_scope,
        denominator_definition=denominator_scope,
        formula={"kind": "ratio", "expression": "numerator / denominator"},
        raw_reported={
            "numerator": format(numerator, "f"),
            "denominator": format(denominator, "f"),
            "unit": unit,
        },
        hold_reasons=tuple(holds),
    )


def calculate_materiality(
    *,
    metric: str,
    numerator: Operand,
    denominator: Operand,
    theme_terms: tuple[str, ...] = (),
) -> MaterialityMeasureResult:
    """Validated ratio of two cited disclosed quantities (E05/E06/I03)."""

    holds = []
    for role, operand in (("numerator", numerator), ("denominator", denominator)):
        if not quote_contains_value(operand.quote, operand.value):
            holds.append(f"{role}_value_not_in_quote")
    if numerator.forecast or denominator.forecast:
        holds.append("forecast_operand")
    if _norm(numerator.period) != _norm(denominator.period):
        holds.append("period_mismatch")
    if _norm(numerator.unit) != _norm(denominator.unit):
        holds.append("unit_mismatch")
    if (numerator.currency or "").upper() != (denominator.currency or "").upper():
        holds.append("currency_mismatch_requires_approved_conversion")
    if _norm(numerator.accounting_basis or "") != _norm(denominator.accounting_basis or ""):
        holds.append("accounting_basis_mismatch")
    # A subsidiary/segment figure may only be divided by a denominator of the
    # same reporting entity; never by the consolidated parent (I03).
    if numerator.scope != denominator.scope and denominator.scope != "issuer_consolidated":
        holds.append("scope_mismatch")
    if numerator.scope == "segment_or_subsidiary" and denominator.scope == "issuer_consolidated" and (
        "subsidiary" in _norm(numerator.label)
    ):
        holds.append("subsidiary_share_of_parent_requires_consolidation_evidence")
    result = compatible_ratio(
        numerator=numerator.value,
        denominator=denominator.value,
        numerator_scope=numerator.label,
        denominator_scope=denominator.label,
        period=numerator.period,
        unit=numerator.unit,
        metric=metric,
        operands_compatible=not holds,
    )
    all_holds = tuple(dict.fromkeys([*holds, *result.hold_reasons]))
    theme_specific = any(term.casefold() in numerator.label.casefold() for term in theme_terms)
    return MaterialityMeasureResult(
        basis=MaterialityBasis.CALCULATED,
        metric=metric,
        value=None if all_holds else result.value,
        unit="ratio",
        currency=numerator.currency,
        period=numerator.period,
        reporting_scope=numerator.scope,
        scope_label=numerator.label,
        denominator_definition=denominator.label,
        formula={
            "kind": "ratio",
            "expression": "numerator / denominator",
            "numerator_passage": numerator.passage_id,
            "denominator_passage": denominator.passage_id,
        },
        operands=(numerator, denominator),
        raw_reported={
            "numerator": {"value": format(numerator.value, "f"), "unit": numerator.unit, "label": numerator.label},
            "denominator": {"value": format(denominator.value, "f"), "unit": denominator.unit, "label": denominator.label},
        },
        hold_reasons=all_holds,
        theme_specific=theme_specific and not all_holds,
    )


def sum_non_overlapping(operands: tuple[Operand, ...], *, overlapping: bool) -> Decimal | None:
    """Sum only when the source states the components do not overlap."""

    if overlapping or not operands:
        return None
    keys = {operand.compatibility_key() for operand in operands}
    if len(keys) != 1:
        return None
    return sum((operand.value for operand in operands), Decimal(0))


def qualitative_measure(label: str, quote: str) -> MaterialityMeasureResult:
    parsed = QualitativeMateriality(label)
    holds = () if quote.strip() else ("qualitative_label_without_primary_wording",)
    return MaterialityMeasureResult(
        basis=MaterialityBasis.QUALITATIVE,
        qualitative_label=parsed,
        raw_reported={"quote": quote},
        hold_reasons=holds,
    )
