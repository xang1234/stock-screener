"""Unit tests for market-aware fundamentals completeness and provenance."""
from __future__ import annotations

import math

import pytest

from app.services.fundamentals_completeness import (
    CORE_FIELDS,
    ENHANCED_FIELDS,
    STANDARD_FIELDS,
    compute_completeness_score,
    derive_field_provenance,
    expected_fields,
)
from app.services.provider_routing_policy import (
    MARKET_HK,
    MARKET_JP,
    MARKET_TW,
    MARKET_US,
    PROVIDER_FINVIZ,
    PROVIDER_YFINANCE,
)


def _full_us_payload() -> dict:
    """Return a dict with every expected US field populated with a non-null value."""
    return {f: 1.0 for f in CORE_FIELDS | STANDARD_FIELDS | ENHANCED_FIELDS}


def _yfinance_only_payload() -> dict:
    """Return a dict populated for yfinance-coverage fields only (no finviz)."""
    return {f: 1.0 for f in CORE_FIELDS | STANDARD_FIELDS}


class TestExpectedFields:
    def test_us_expects_all_three_tiers(self):
        fields = expected_fields(MARKET_US)
        assert CORE_FIELDS <= fields
        assert STANDARD_FIELDS <= fields
        assert ENHANCED_FIELDS <= fields

    @pytest.mark.parametrize("market", [MARKET_HK, MARKET_JP, MARKET_TW])
    def test_asia_excludes_enhanced(self, market):
        fields = expected_fields(market)
        assert CORE_FIELDS <= fields
        assert STANDARD_FIELDS <= fields
        # Enhanced tier (finviz-only) must not be expected for non-US.
        assert not (ENHANCED_FIELDS & fields)

    def test_none_market_defaults_to_us(self):
        # Legacy callers should get US semantics.
        assert expected_fields(None) == expected_fields(MARKET_US)


class TestCompletenessScore:
    def test_empty_payload_scores_zero(self):
        assert compute_completeness_score({}, MARKET_US) == 0
        assert compute_completeness_score(None, MARKET_US) == 0

    def test_full_us_payload_scores_100(self):
        assert compute_completeness_score(_full_us_payload(), MARKET_US) == 100

    @pytest.mark.parametrize("market", [MARKET_HK, MARKET_JP, MARKET_TW])
    def test_yfinance_only_payload_scores_100_for_asia(self, market):
        """Key T2 property: HK/JP/TW should achieve full score without finviz."""
        assert compute_completeness_score(_yfinance_only_payload(), market) == 100

    def test_yfinance_only_payload_scores_less_than_100_for_us(self):
        """Same payload on US should score below 100 since finviz fields are expected."""
        score = compute_completeness_score(_yfinance_only_payload(), MARKET_US)
        assert 0 < score < 100

    def test_missing_core_field_hits_score_harder_than_missing_standard(self):
        payload = _yfinance_only_payload()
        no_core = dict(payload)
        no_core.pop(next(iter(CORE_FIELDS)))
        no_standard = dict(payload)
        no_standard.pop(next(iter(STANDARD_FIELDS)))

        score_no_core = compute_completeness_score(no_core, MARKET_HK)
        score_no_standard = compute_completeness_score(no_standard, MARKET_HK)
        assert score_no_core < score_no_standard

    def test_none_values_are_missing(self):
        payload = {f: None for f in CORE_FIELDS | STANDARD_FIELDS}
        assert compute_completeness_score(payload, MARKET_HK) == 0

    def test_empty_string_is_missing(self):
        payload = _yfinance_only_payload()
        field = next(iter(CORE_FIELDS))
        payload[field] = ""
        full_score = compute_completeness_score(_yfinance_only_payload(), MARKET_HK)
        partial = compute_completeness_score(payload, MARKET_HK)
        assert partial < full_score

    def test_nan_is_missing(self):
        payload = _yfinance_only_payload()
        payload[next(iter(STANDARD_FIELDS))] = float("nan")
        partial = compute_completeness_score(payload, MARKET_HK)
        full = compute_completeness_score(_yfinance_only_payload(), MARKET_HK)
        assert partial < full

    def test_zero_is_present(self):
        """Legitimate zero values must count — e.g., dividend_yield can be 0."""
        payload = {f: 0 for f in CORE_FIELDS | STANDARD_FIELDS}
        assert compute_completeness_score(payload, MARKET_HK) == 100


class TestFieldSourceMapConsistency:
    """Every field in the source map must be expected by at least one market,
    otherwise the map entry is dead code and its field is silently ignored."""

    def test_every_source_map_field_is_expected_somewhere(self):
        # Expected fields across ALL markets is the US set (superset).
        from app.services.fundamentals_completeness import _FIELD_SOURCE
        us_expected = expected_fields(MARKET_US)
        orphans = set(_FIELD_SOURCE.keys()) - set(us_expected)
        assert not orphans, (
            f"Fields in _FIELD_SOURCE but not in any expected tier: {orphans}. "
            "Either add them to CORE/STANDARD/ENHANCED or drop the map entry."
        )


class TestProvenance:
    def test_provenance_tags_finviz_fields_on_us(self):
        payload = _full_us_payload()
        prov = derive_field_provenance(payload, MARKET_US)
        # At least one enhanced field should be tagged finviz.
        finviz_fields = [f for f, src in prov.items() if src == PROVIDER_FINVIZ]
        assert finviz_fields
        # All enhanced fields present in payload should be tagged finviz.
        for f in ENHANCED_FIELDS:
            assert prov[f] == PROVIDER_FINVIZ

    def test_provenance_excludes_finviz_fields_on_asia(self):
        """Enhanced fields must not appear in provenance for non-US markets,
        even if the caller accidentally included them in the payload."""
        payload = _full_us_payload()
        prov = derive_field_provenance(payload, MARKET_HK)
        for f in ENHANCED_FIELDS:
            assert f not in prov

    def test_provenance_tags_yfinance_fields(self):
        payload = _yfinance_only_payload()
        prov = derive_field_provenance(payload, MARKET_HK)
        # market_cap is a yfinance-tagged core field.
        assert prov.get("market_cap") == PROVIDER_YFINANCE

    def test_provenance_tags_technicals(self):
        payload = {"rsi_14": 55.0, "sma_50": 100.0}
        prov = derive_field_provenance(payload, MARKET_HK)
        assert prov["rsi_14"] == "technicals"
        assert prov["sma_50"] == "technicals"

    def test_provenance_skips_missing_values(self):
        payload = {"market_cap": None, "pe_ratio": 15.0, "sector": ""}
        prov = derive_field_provenance(payload, MARKET_US)
        assert "market_cap" not in prov
        assert "sector" not in prov
        assert "pe_ratio" in prov

    def test_empty_payload_yields_empty_provenance(self):
        assert derive_field_provenance({}, MARKET_US) == {}
        assert derive_field_provenance(None, MARKET_US) == {}

    def test_unknown_market_defaults_to_us_provenance(self):
        payload = _full_us_payload()
        prov_unknown = derive_field_provenance(payload, "XX")
        prov_us = derive_field_provenance(payload, MARKET_US)
        assert prov_unknown == prov_us
