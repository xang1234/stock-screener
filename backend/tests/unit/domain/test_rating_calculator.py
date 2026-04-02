"""Tests for calculate_overall_rating.

Verifies:
- Empty screener outputs edge case
- Threshold-based rating mapping (exact boundaries and just-below)
- Pass-rate force-PASS (zero screeners pass)
- Pass-rate one-level downgrade (fewer than half pass)
- No downgrade when ≥ half pass (strict < comparison)
- Parity with ScanOrchestrator.scan_stock_multi end-to-end
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.domain.scanning.models import (
    CompositeMethod,
    RatingCategory,
    ScreenerOutputDomain,
)
from app.domain.scanning.scoring import (
    calculate_composite_score,
    calculate_overall_rating,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_output(
    score: float, passes: bool = True, name: str = "screener"
) -> ScreenerOutputDomain:
    return ScreenerOutputDomain(
        screener_name=name,
        score=score,
        passes=passes,
        rating="Strong Buy" if passes else "Pass",
        breakdown={},
        details={},
    )


def _make_outputs(
    scores_and_passes: list[tuple[float, bool]],
) -> dict[str, ScreenerOutputDomain]:
    """Build a screener_outputs dict from (score, passes) pairs."""
    return {
        f"screener_{i}": _make_output(score, passes, name=f"screener_{i}")
        for i, (score, passes) in enumerate(scores_and_passes)
    }


# ── Empty Outputs ────────────────────────────────────────────────────


class TestEmptyOutputs:
    """Empty screener_outputs dict triggers pass_count=0 → PASS."""

    def test_empty_outputs_returns_pass(self):
        result = calculate_overall_rating(0.0, {})
        assert result is RatingCategory.PASS

    def test_empty_outputs_with_high_score_still_pass(self):
        """Even a high composite score returns PASS when no screeners ran."""
        result = calculate_overall_rating(95.0, {})
        assert result is RatingCategory.PASS


# ── Threshold Mapping ────────────────────────────────────────────────


class TestThresholdMapping:
    """Verify score→rating mapping at exact boundaries and just below.

    All outputs pass to avoid pass-rate adjustments interfering.
    """

    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (100.0, RatingCategory.STRONG_BUY),
            (80.0, RatingCategory.STRONG_BUY),   # exact boundary
            (79.99, RatingCategory.BUY),          # just below
            (70.0, RatingCategory.BUY),           # exact boundary
            (69.99, RatingCategory.WATCH),         # just below
            (60.0, RatingCategory.WATCH),          # exact boundary
            (59.99, RatingCategory.PASS),          # just below
            (0.0, RatingCategory.PASS),
        ],
        ids=[
            "100→STRONG_BUY",
            "80→STRONG_BUY",
            "79.99→BUY",
            "70→BUY",
            "69.99→WATCH",
            "60→WATCH",
            "59.99→PASS",
            "0→PASS",
        ],
    )
    def test_threshold_boundary(self, score: float, expected: RatingCategory):
        # Single passing screener — avoids pass-rate adjustments
        outputs = {"alpha": _make_output(score, passes=True)}
        assert calculate_overall_rating(score, outputs) is expected


# ── Pass Rate: Force PASS ────────────────────────────────────────────


class TestPassRateForcePass:
    """When zero screeners pass, result is PASS regardless of score."""

    def test_zero_passes_high_score(self):
        """Score=90 but 0/2 pass → PASS."""
        outputs = _make_outputs([(90.0, False), (90.0, False)])
        assert calculate_overall_rating(90.0, outputs) is RatingCategory.PASS

    def test_single_screener_does_not_pass(self):
        """Single screener that doesn't pass → PASS."""
        outputs = {"alpha": _make_output(85.0, passes=False)}
        assert calculate_overall_rating(85.0, outputs) is RatingCategory.PASS


# ── Pass Rate: Downgrade ─────────────────────────────────────────────


class TestPassRateDowngrade:
    """When fewer than half of screeners pass, downgrade one level."""

    def test_strong_buy_downgraded_to_buy(self):
        """Score=85 (STRONG_BUY), 1/3 pass → downgrade to BUY."""
        outputs = _make_outputs([(85.0, True), (85.0, False), (85.0, False)])
        result = calculate_overall_rating(85.0, outputs)
        assert result is RatingCategory.BUY

    def test_buy_downgraded_to_watch(self):
        """Score=75 (BUY), 1/3 pass → downgrade to WATCH."""
        outputs = _make_outputs([(75.0, True), (75.0, False), (75.0, False)])
        result = calculate_overall_rating(75.0, outputs)
        assert result is RatingCategory.WATCH

    def test_watch_stays_watch(self):
        """Score=65 (WATCH), 1/3 pass → stays WATCH (no further downgrade)."""
        outputs = _make_outputs([(65.0, True), (65.0, False), (65.0, False)])
        result = calculate_overall_rating(65.0, outputs)
        assert result is RatingCategory.WATCH

    def test_pass_stays_pass(self):
        """Score=50 (PASS), 1/3 pass → stays PASS."""
        outputs = _make_outputs([(50.0, True), (50.0, False), (50.0, False)])
        result = calculate_overall_rating(50.0, outputs)
        assert result is RatingCategory.PASS


# ── Pass Rate: No Downgrade ──────────────────────────────────────────


class TestPassRateNoDowngrade:
    """Exactly half or more passing does NOT trigger downgrade (strict <)."""

    def test_exactly_half_no_downgrade_two(self):
        """1/2 pass → pass_count (1) is NOT < total/2 (1.0) → no downgrade."""
        outputs = _make_outputs([(85.0, True), (85.0, False)])
        result = calculate_overall_rating(85.0, outputs)
        assert result is RatingCategory.STRONG_BUY

    def test_exactly_half_no_downgrade_four(self):
        """2/4 pass → pass_count (2) is NOT < total/2 (2.0) → no downgrade."""
        outputs = _make_outputs([
            (80.0, True), (80.0, True), (80.0, False), (80.0, False)
        ])
        result = calculate_overall_rating(80.0, outputs)
        assert result is RatingCategory.STRONG_BUY

    def test_more_than_half_no_downgrade(self):
        """3/4 pass → well above threshold → no downgrade."""
        outputs = _make_outputs([
            (75.0, True), (75.0, True), (75.0, True), (75.0, False)
        ])
        result = calculate_overall_rating(75.0, outputs)
        assert result is RatingCategory.BUY

    def test_all_pass_no_downgrade(self):
        """All screeners pass → base rating preserved."""
        outputs = _make_outputs([(85.0, True), (85.0, True)])
        result = calculate_overall_rating(85.0, outputs)
        assert result is RatingCategory.STRONG_BUY


# ── Parity with ScanOrchestrator ─────────────────────────────────────


class TestParityWithOrchestrator:
    """Domain functions produce identical results to ScanOrchestrator.scan_stock_multi.

    These tests duplicate minimal orchestrator helpers to run the same
    inputs through both paths and assert the outputs match.
    """

    # -- Duplicated helpers (not in a shared module) --

    @staticmethod
    def _make_stock_data(symbol: str = "TEST", n_days: int = 200):
        from app.scanners.base_screener import StockData

        dates = pd.date_range(end="2026-01-15", periods=n_days, freq="B")
        df = pd.DataFrame(
            {
                "Open": 100.0,
                "High": 105.0,
                "Low": 95.0,
                "Close": 100.0,
                "Volume": 1_000_000,
            },
            index=dates,
        )
        return StockData(symbol=symbol, price_data=df, benchmark_data=df.copy())

    @staticmethod
    def _make_fake_screener_class(name: str, score: float, passes: bool):
        from app.scanners.base_screener import (
            BaseStockScreener,
            DataRequirements,
            ScreenerResult,
        )

        class FakeScreener(BaseStockScreener):
            @property
            def screener_name(self) -> str:
                return name

            def get_data_requirements(self, criteria=None) -> DataRequirements:
                return DataRequirements()

            def scan_stock(self, symbol, data, criteria=None) -> ScreenerResult:
                rating = "Strong Buy" if passes else "Pass"
                return ScreenerResult(
                    score=score,
                    passes=passes,
                    rating=rating,
                    breakdown={"test": score},
                    details={"test_detail": True},
                    screener_name=name,
                )

            def calculate_rating(self, score, details) -> str:
                return "Strong Buy" if score >= 80 else "Pass"

        FakeScreener.__name__ = f"FakeScreener_{name}"
        FakeScreener.__qualname__ = f"FakeScreener_{name}"
        return FakeScreener

    def _run_both(
        self,
        screener_configs: list[tuple[str, float, bool]],
        composite_method_str: str,
    ) -> tuple[float, str, float, RatingCategory]:
        """Run inputs through orchestrator and domain functions, return both.

        Returns:
            (orch_score, orch_rating, domain_score, domain_rating)
        """
        from app.domain.scanning.ports import StockDataProvider
        from app.scanners.scan_orchestrator import ScanOrchestrator
        from app.scanners.screener_registry import ScreenerRegistry

        stock_data = self._make_stock_data("TEST")

        class LocalFakeProvider(StockDataProvider):
            def prepare_data(self, symbol, requirements):
                return stock_data

            def prepare_data_bulk(
                self,
                symbols,
                requirements,
                *,
                allow_partial=True,
                batch_only_prices=False,
                batch_only_fundamentals=False,
            ):
                return {s: stock_data for s in symbols}

        provider = LocalFakeProvider()
        registry = ScreenerRegistry()
        for name, score, passes in screener_configs:
            registry.register(self._make_fake_screener_class(name, score, passes))

        orch = ScanOrchestrator(data_provider=provider, registry=registry)
        names = [cfg[0] for cfg in screener_configs]
        orch_result = orch.scan_stock_multi(
            "TEST", names, composite_method=composite_method_str
        )

        # Domain path
        method = CompositeMethod(composite_method_str)
        outputs = {
            name: _make_output(score, passes, name=name)
            for name, score, passes in screener_configs
        }
        domain_score = calculate_composite_score(outputs, method)
        domain_rating = calculate_overall_rating(domain_score, outputs)

        return (
            orch_result["composite_score"],
            orch_result["rating"],
            domain_score,
            domain_rating,
        )

    def test_parity_single_screener(self):
        orch_score, orch_rating, dom_score, dom_rating = self._run_both(
            [("alpha", 75.0, True)], "weighted_average"
        )
        assert orch_score == dom_score
        assert orch_rating == dom_rating.value

    def test_parity_multi_weighted_average(self):
        orch_score, orch_rating, dom_score, dom_rating = self._run_both(
            [("alpha", 80.0, True), ("beta", 60.0, True)], "weighted_average"
        )
        assert orch_score == dom_score
        assert orch_rating == dom_rating.value

    def test_parity_maximum(self):
        orch_score, orch_rating, dom_score, dom_rating = self._run_both(
            [("alpha", 90.0, True), ("beta", 60.0, True)], "maximum"
        )
        assert orch_score == dom_score
        assert orch_rating == dom_rating.value

    def test_parity_minimum(self):
        orch_score, orch_rating, dom_score, dom_rating = self._run_both(
            [("alpha", 90.0, True), ("beta", 60.0, True)], "minimum"
        )
        assert orch_score == dom_score
        assert orch_rating == dom_rating.value

    def test_parity_mixed_pass_fail(self):
        """1/3 pass with high scores → downgrade happens in both paths."""
        orch_score, orch_rating, dom_score, dom_rating = self._run_both(
            [("alpha", 95.0, True), ("beta", 70.0, False), ("gamma", 70.0, False)],
            "weighted_average",
        )
        assert orch_score == pytest.approx(dom_score, abs=0.01)
        assert orch_rating == dom_rating.value

    def test_parity_all_fail(self):
        """All screeners fail → PASS in both paths."""
        orch_score, orch_rating, dom_score, dom_rating = self._run_both(
            [("alpha", 90.0, False), ("beta", 85.0, False)], "weighted_average"
        )
        assert orch_score == pytest.approx(dom_score, abs=0.01)
        assert orch_rating == dom_rating.value
