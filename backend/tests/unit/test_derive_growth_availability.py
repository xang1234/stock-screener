"""Unit tests for ``derive_growth_availability``.

Covers the growth-cadence states that map onto ``field_availability``:
- BASIS_UNAVAILABLE + CADENCE_INSUFFICIENT → insufficient_history
- BASIS_UNAVAILABLE + non-CADENCE_INSUFFICIENT → market_policy_excludes_qoq
- BASIS_COMPARABLE_YOY → computed (comparable-period YoY fallback)
- BASIS_QOQ / None → no entries (normal case)
"""

from __future__ import annotations

from app.services.growth_cadence_service import (
    BASIS_COMPARABLE_YOY,
    BASIS_QOQ,
    BASIS_UNAVAILABLE,
    CADENCE_INSUFFICIENT,
    CADENCE_QUARTERLY,
    CADENCE_SEMIANNUAL,
    REASON_COMPARABLE_YOY_FALLBACK,
    REASON_INSUFFICIENT_HISTORY,
    REASON_MARKET_POLICY_EXCLUDES_QOQ,
    derive_growth_availability,
)


class TestDeriveGrowthAvailability:
    def test_unavailable_basis_with_insufficient_history_cadence(self):
        result = derive_growth_availability(BASIS_UNAVAILABLE, CADENCE_INSUFFICIENT)
        assert set(result.keys()) == {"eps_growth_qq", "sales_growth_qq"}
        for entry in result.values():
            assert entry["status"] == "unsupported"
            assert entry["reason_code"] == REASON_INSUFFICIENT_HISTORY
            assert entry["support_state"] == "unsupported"
            assert entry["cadence"] == CADENCE_INSUFFICIENT

    def test_unavailable_basis_with_quarterly_cadence_emits_market_policy_reason(self):
        # BASIS_UNAVAILABLE + non-INSUFFICIENT cadence = market doesn't support
        # QoQ and isn't in the comparable-period-primary set (TW for example).
        # The reason should distinguish this from "not enough history".
        result = derive_growth_availability(BASIS_UNAVAILABLE, CADENCE_QUARTERLY)
        assert set(result.keys()) == {"eps_growth_qq", "sales_growth_qq"}
        for entry in result.values():
            assert entry["reason_code"] == REASON_MARKET_POLICY_EXCLUDES_QOQ
            assert entry["status"] == "unsupported"

    def test_unavailable_basis_with_none_cadence_emits_market_policy_reason(self):
        # Cadence unknown/None and basis unavailable → market policy is the
        # most conservative default when we can't distinguish the cause.
        result = derive_growth_availability(BASIS_UNAVAILABLE, None)
        for entry in result.values():
            assert entry["reason_code"] == REASON_MARKET_POLICY_EXCLUDES_QOQ

    def test_comparable_yoy_basis_emits_computed_with_fallback_reason(self):
        # This is the HK/JP path — QoQ fields carry comparable-period YoY
        # values, and clients should see that as "computed" not "QoQ".
        result = derive_growth_availability(BASIS_COMPARABLE_YOY, CADENCE_SEMIANNUAL)
        assert set(result.keys()) == {"eps_growth_qq", "sales_growth_qq"}
        for entry in result.values():
            assert entry["status"] == "computed"
            assert entry["reason_code"] == REASON_COMPARABLE_YOY_FALLBACK
            assert entry["support_state"] == "computed"
            assert entry["cadence"] == CADENCE_SEMIANNUAL

    def test_qoq_basis_emits_empty(self):
        # Quarterly QoQ is the supported-normal path; no transparency entry
        # needed, the cell just shows the raw value.
        assert derive_growth_availability(BASIS_QOQ, "quarterly") == {}

    def test_none_basis_emits_empty(self):
        assert derive_growth_availability(None, None) == {}

    def test_unknown_basis_string_emits_empty(self):
        # Safety: unexpected values should not synthesize bogus entries.
        assert derive_growth_availability("something_new", "quarterly") == {}
