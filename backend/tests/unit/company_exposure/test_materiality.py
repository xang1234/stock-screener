from __future__ import annotations

from decimal import Decimal

import pytest

from app.domain.company_exposure.contracts import MaterialityBasis
from app.services.company_exposure.materiality import (
    Operand,
    calculate_materiality,
    compatible_ratio,
    parse_decimal,
    sum_non_overlapping,
    unknown_materiality,
    validate_measure,
)

THEME = ("HBM", "memory test")


def operand(value, **kw):
    base = dict(
        value=parse_decimal(value), unit="USD million", period="FY2025",
        scope="issuer_consolidated", label="Memory test revenue",
        currency="USD", passage_id="p", quote=f"revenue of {value}",
    )
    base.update(kw)
    return Operand(**base)


def test_compatible_disclosure_ratio_is_exact():
    result = compatible_ratio(
        numerator=Decimal(200), denominator=Decimal(1000),
        numerator_scope="issuer_consolidated:ai_memory",
        denominator_scope="issuer_consolidated:total",
        period="FY2025", unit="USD_million", metric="revenue_share",
        operands_compatible=True,
    )
    assert result.value == Decimal("0.2")
    assert result.basis == MaterialityBasis.CALCULATED


@pytest.mark.case("E04")
@pytest.mark.exposure_layer("unit")
def test_e04_segment_not_theme_share():
    result = validate_measure(
        metric="revenue_share", value=Decimal("0.30"), unit="ratio",
        period="FY2025", scope="segment_or_subsidiary", scope_label="Server segment",
        quote="The server segment represented 0.30 of revenue.", passage_id="p",
        theme_terms=THEME,
    )
    assert result.value == Decimal("0.30")
    assert result.scope_label == "Server segment"
    assert result.theme_specific is False  # never relabelled as an AI Memory share


@pytest.mark.case("E05")
@pytest.mark.exposure_layer("unit")
def test_e05_compatible_disclosed_ratio():
    result = calculate_materiality(
        metric="revenue_share",
        numerator=operand("200", passage_id="p-num", quote="Memory test revenue was 200"),
        denominator=operand("1,000", label="Total revenue", passage_id="p-den", quote="Total revenue was 1,000"),
        theme_terms=THEME,
    )
    assert result.value == Decimal("0.2")
    assert result.hold_reasons == ()
    assert result.formula["numerator_passage"] == "p-num"
    assert result.formula["denominator_passage"] == "p-den"
    assert result.period == "FY2025" and result.theme_specific


@pytest.mark.case("E06")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    ("numerator", "denominator", "hold"),
    [
        (operand("200"), operand("1000", period="FY2024", label="Total"), "period_mismatch"),
        (operand("200"), operand("1000", currency="JPY", label="Total"), "currency_mismatch_requires_approved_conversion"),
        (operand("200"), operand("0", label="Operating profit", quote="profit of 0"), "nonpositive_denominator_review_required"),
        (operand("200"), operand("-50", label="Operating profit", quote="loss of -50"), "nonpositive_denominator_review_required"),
        (operand("200", forecast=True), operand("1000", label="Total"), "forecast_operand"),
        (operand("200", quote="no number here"), operand("1000", label="Total"), "numerator_value_not_in_quote"),
        (operand("1500"), operand("1000", label="Total"), "share_out_of_range"),
    ],
)
def test_e06_incompatible_inputs_held(numerator, denominator, hold):
    result = calculate_materiality(metric="revenue_share", numerator=numerator, denominator=denominator)
    assert result.value is None
    assert hold in result.hold_reasons
    assert result.raw_reported["numerator"]["value"]  # original numbers retained


@pytest.mark.case("E06")
@pytest.mark.exposure_layer("unit")
def test_e06_overlapping_segments_are_not_summed():
    parts = (operand("100"), operand("50"))
    assert sum_non_overlapping(parts, overlapping=True) is None
    assert sum_non_overlapping(parts, overlapping=False) == Decimal(150)


@pytest.mark.case("I03")
@pytest.mark.exposure_layer("unit")
def test_i03_subsidiary_not_parent_share():
    result = calculate_materiality(
        metric="revenue_share",
        numerator=operand("80", scope="segment_or_subsidiary", label="Subsidiary HBM revenue"),
        denominator=operand("1000", label="Group revenue"),
    )
    assert result.value is None
    assert "subsidiary_share_of_parent_requires_consolidation_evidence" in result.hold_reasons


@pytest.mark.case("E08")
@pytest.mark.exposure_layer("unit")
def test_unknown_materiality_wording():
    result = unknown_materiality()
    assert result.basis == MaterialityBasis.UNKNOWN
    assert result.display == "Not separately disclosed in reviewed evidence"


def test_decimals_are_parsed_exactly():
    assert parse_decimal("1,234.50") == Decimal("1234.50")
    with pytest.raises(ValueError):
        parse_decimal("n/a")
