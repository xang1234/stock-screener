"""Tests for operational invalidation flags (SE-E7).

Covers:
- Each flag independently: threshold boundary (at, above, below), None permissiveness
- Custom parameter overrides
- is_hard values on returned InvalidationFlag objects
- Multiple flags accumulating
- earnings_soon with various date scenarios
- breaks_50d_support cushion semantics
"""

from __future__ import annotations

from datetime import date

import pytest

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.operational_flags import (
    OperationalFlagInputs,
    compute_operational_flags,
)
from app.analysis.patterns.report import InvalidationFlag


DEFAULT_PARAMS = SetupEngineParameters()


class TestTooExtended:
    """Flag: too_extended — distance_to_pivot_pct > threshold."""

    def test_above_threshold_triggers(self):
        inputs = OperationalFlagInputs(distance_to_pivot_pct=10.1)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert any(f.code == "too_extended" for f in flags)

    def test_at_threshold_does_not_trigger(self):
        """Strictly greater — exactly at threshold should NOT flag."""
        inputs = OperationalFlagInputs(distance_to_pivot_pct=10.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "too_extended" for f in flags)

    def test_below_threshold_does_not_trigger(self):
        inputs = OperationalFlagInputs(distance_to_pivot_pct=5.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "too_extended" for f in flags)

    def test_none_skipped(self):
        inputs = OperationalFlagInputs(distance_to_pivot_pct=None)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "too_extended" for f in flags)

    def test_is_soft(self):
        inputs = OperationalFlagInputs(distance_to_pivot_pct=15.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        flag = next(f for f in flags if f.code == "too_extended")
        assert flag.is_hard is False

    def test_custom_threshold(self):
        params = SetupEngineParameters(too_extended_pivot_distance_pct=5.0)
        inputs = OperationalFlagInputs(distance_to_pivot_pct=6.0)
        flags = compute_operational_flags(inputs, params)
        assert any(f.code == "too_extended" for f in flags)

    def test_negative_distance_does_not_trigger(self):
        """Negative distance means price is below pivot — not extended."""
        inputs = OperationalFlagInputs(distance_to_pivot_pct=-3.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "too_extended" for f in flags)


class TestBreaks50dSupport:
    """Flag: breaks_50d_support — price < ma_50 * (1 - cushion/100)."""

    def test_below_ma50_strict_triggers(self):
        """Default cushion=0: any price below 50d MA flags."""
        inputs = OperationalFlagInputs(current_price=99.0, ma_50=100.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert any(f.code == "breaks_50d_support" for f in flags)

    def test_at_ma50_strict_does_not_trigger(self):
        inputs = OperationalFlagInputs(current_price=100.0, ma_50=100.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "breaks_50d_support" for f in flags)

    def test_above_ma50_does_not_trigger(self):
        inputs = OperationalFlagInputs(current_price=105.0, ma_50=100.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "breaks_50d_support" for f in flags)

    def test_cushion_2pct_allows_slight_break(self):
        """cushion=2 means threshold = 100 * (1 - 0.02) = 98. Price 99 > 98 → no flag."""
        params = SetupEngineParameters(breaks_50d_support_cushion_pct=2.0)
        inputs = OperationalFlagInputs(current_price=99.0, ma_50=100.0)
        flags = compute_operational_flags(inputs, params)
        assert not any(f.code == "breaks_50d_support" for f in flags)

    def test_cushion_2pct_deeper_break_triggers(self):
        """cushion=2 means threshold = 100 * 0.98 = 98. Price 97 < 98 → flag."""
        params = SetupEngineParameters(breaks_50d_support_cushion_pct=2.0)
        inputs = OperationalFlagInputs(current_price=97.0, ma_50=100.0)
        flags = compute_operational_flags(inputs, params)
        assert any(f.code == "breaks_50d_support" for f in flags)

    def test_none_price_skipped(self):
        inputs = OperationalFlagInputs(current_price=None, ma_50=100.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "breaks_50d_support" for f in flags)

    def test_none_ma50_skipped(self):
        inputs = OperationalFlagInputs(current_price=99.0, ma_50=None)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "breaks_50d_support" for f in flags)

    def test_is_hard(self):
        inputs = OperationalFlagInputs(current_price=95.0, ma_50=100.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        flag = next(f for f in flags if f.code == "breaks_50d_support")
        assert flag.is_hard is True


class TestLowLiquidity:
    """Flag: low_liquidity — adtv_usd < threshold."""

    def test_below_threshold_triggers(self):
        inputs = OperationalFlagInputs(adtv_usd=500_000.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert any(f.code == "low_liquidity" for f in flags)

    def test_at_threshold_does_not_trigger(self):
        """Strictly less — exactly at threshold should NOT flag."""
        inputs = OperationalFlagInputs(adtv_usd=1_000_000.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "low_liquidity" for f in flags)

    def test_above_threshold_does_not_trigger(self):
        inputs = OperationalFlagInputs(adtv_usd=5_000_000.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "low_liquidity" for f in flags)

    def test_none_skipped(self):
        inputs = OperationalFlagInputs(adtv_usd=None)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "low_liquidity" for f in flags)

    def test_is_soft(self):
        inputs = OperationalFlagInputs(adtv_usd=100_000.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        flag = next(f for f in flags if f.code == "low_liquidity")
        assert flag.is_hard is False

    def test_custom_threshold(self):
        params = SetupEngineParameters(low_liquidity_adtv_min_usd=10_000_000.0)
        inputs = OperationalFlagInputs(adtv_usd=5_000_000.0)
        flags = compute_operational_flags(inputs, params)
        assert any(f.code == "low_liquidity" for f in flags)


class TestEarningsSoon:
    """Flag: earnings_soon — within window_days of next earnings."""

    def test_within_window_triggers(self):
        inputs = OperationalFlagInputs(
            next_earnings_date=date(2026, 3, 10),
            reference_date=date(2026, 2, 22),
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert any(f.code == "earnings_soon" for f in flags)

    def test_same_day_triggers(self):
        inputs = OperationalFlagInputs(
            next_earnings_date=date(2026, 3, 1),
            reference_date=date(2026, 3, 1),
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert any(f.code == "earnings_soon" for f in flags)

    def test_outside_window_does_not_trigger(self):
        inputs = OperationalFlagInputs(
            next_earnings_date=date(2026, 6, 1),
            reference_date=date(2026, 2, 22),
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "earnings_soon" for f in flags)

    def test_past_earnings_does_not_trigger(self):
        """Earnings already passed (negative days) should not flag."""
        inputs = OperationalFlagInputs(
            next_earnings_date=date(2026, 2, 15),
            reference_date=date(2026, 2, 22),
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "earnings_soon" for f in flags)

    def test_none_earnings_date_skipped(self):
        inputs = OperationalFlagInputs(
            next_earnings_date=None,
            reference_date=date(2026, 2, 22),
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "earnings_soon" for f in flags)

    def test_none_reference_date_skipped(self):
        inputs = OperationalFlagInputs(
            next_earnings_date=date(2026, 3, 1),
            reference_date=None,
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert not any(f.code == "earnings_soon" for f in flags)

    def test_is_soft(self):
        inputs = OperationalFlagInputs(
            next_earnings_date=date(2026, 3, 5),
            reference_date=date(2026, 2, 22),
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        flag = next(f for f in flags if f.code == "earnings_soon")
        assert flag.is_hard is False

    def test_custom_window(self):
        """With 7-day window, 10 days out should not trigger."""
        params = SetupEngineParameters(earnings_soon_window_days=7.0)
        inputs = OperationalFlagInputs(
            next_earnings_date=date(2026, 3, 4),
            reference_date=date(2026, 2, 22),
        )
        flags = compute_operational_flags(inputs, params)
        assert not any(f.code == "earnings_soon" for f in flags)

    def test_at_boundary_triggers(self):
        """Exactly 21 days away (at boundary) should trigger."""
        inputs = OperationalFlagInputs(
            next_earnings_date=date(2026, 3, 15),
            reference_date=date(2026, 2, 22),
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert any(f.code == "earnings_soon" for f in flags)


class TestMultipleFlags:
    """Multiple flags can accumulate from a single input."""

    def test_two_flags_accumulate(self):
        inputs = OperationalFlagInputs(
            distance_to_pivot_pct=15.0,
            adtv_usd=500_000.0,
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        codes = {f.code for f in flags}
        assert "too_extended" in codes
        assert "low_liquidity" in codes
        assert len(flags) == 2

    def test_all_four_flags(self):
        inputs = OperationalFlagInputs(
            distance_to_pivot_pct=15.0,
            current_price=95.0,
            ma_50=100.0,
            adtv_usd=500_000.0,
            next_earnings_date=date(2026, 3, 5),
            reference_date=date(2026, 2, 22),
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        codes = {f.code for f in flags}
        assert codes == {"too_extended", "breaks_50d_support", "low_liquidity", "earnings_soon"}

    def test_no_flags_on_healthy_stock(self):
        inputs = OperationalFlagInputs(
            distance_to_pivot_pct=3.0,
            current_price=110.0,
            ma_50=100.0,
            adtv_usd=50_000_000.0,
        )
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert flags == []

    def test_empty_inputs_no_flags(self):
        inputs = OperationalFlagInputs()
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        assert flags == []


class TestToPayload:
    """Verify InvalidationFlag.to_payload() integration."""

    def test_flag_serializes_to_code_string(self):
        inputs = OperationalFlagInputs(distance_to_pivot_pct=15.0)
        flags = compute_operational_flags(inputs, DEFAULT_PARAMS)
        payloads = [f.to_payload() for f in flags]
        assert "too_extended" in payloads
