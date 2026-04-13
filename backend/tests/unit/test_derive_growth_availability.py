"""Unit tests for ``derive_growth_availability``.

Covers the three growth-cadence states that map onto ``field_availability``:
unavailable (insufficient_history), computed (comparable-period YoY fallback),
and QoQ / None (no entries — normal case).
"""

from __future__ import annotations

from app.services.growth_cadence_service import (
    BASIS_COMPARABLE_YOY,
    BASIS_QOQ,
    BASIS_UNAVAILABLE,
    CADENCE_INSUFFICIENT,
    CADENCE_SEMIANNUAL,
    REASON_COMPARABLE_YOY_FALLBACK,
    REASON_INSUFFICIENT_HISTORY,
    derive_growth_availability,
)


class TestDeriveGrowthAvailability:
    def test_unavailable_basis_emits_insufficient_history(self):
        result = derive_growth_availability(BASIS_UNAVAILABLE, CADENCE_INSUFFICIENT)
        assert set(result.keys()) == {"eps_growth_qq", "sales_growth_qq"}
        for entry in result.values():
            assert entry["status"] == "unsupported"
            assert entry["reason_code"] == REASON_INSUFFICIENT_HISTORY
            assert entry["support_state"] == "unsupported"
            assert entry["cadence"] == CADENCE_INSUFFICIENT

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
