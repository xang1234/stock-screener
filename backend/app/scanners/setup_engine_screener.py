"""SetupEngineScanner — runtime entrypoint for Setup Engine analysis pipeline.

Thin orchestration class that bridges StockData → analysis pipeline → ScreenerResult,
delegating all pattern detection, aggregation, calibration, and readiness computation
to existing modules in ``app.analysis.patterns``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import pandas as pd

from app.analysis.patterns.aggregator import (
    AggregatedPatternOutput,
    SetupEngineAggregator,
)
from app.analysis.patterns.config import (
    DEFAULT_SETUP_ENGINE_PARAMETERS,
    build_setup_engine_parameters,
)
from app.analysis.patterns.detectors import PatternDetectorInput
from app.analysis.patterns.policy import (
    SetupEngineDataPolicyResult,
    evaluate_setup_engine_data_policy,
)
from app.analysis.patterns.readiness import compute_breakout_readiness_features
from app.analysis.patterns.technicals import resample_ohlcv
from app.scanners.base_screener import (
    BaseStockScreener,
    DataRequirements,
    ScreenerResult,
    StockData,
)
from app.scanners.screener_registry import register_screener
from app.scanners.setup_engine_scanner import build_setup_engine_payload

logger = logging.getLogger(__name__)


@register_screener
class SetupEngineScanner(BaseStockScreener):
    """Run the full Setup Engine pipeline as a selectable screener."""

    @property
    def screener_name(self) -> str:
        return "setup_engine"

    def __init__(self) -> None:
        self._aggregator = SetupEngineAggregator()

    def get_data_requirements(self, criteria: Optional[Dict] = None) -> DataRequirements:
        return DataRequirements(
            price_period="2y",
            needs_fundamentals=False,
            needs_quarterly_growth=False,
            needs_benchmark=True,
            needs_earnings_history=False,
        )

    def scan_stock(
        self,
        symbol: str,
        data: StockData,
        criteria: Optional[Dict] = None,
    ) -> ScreenerResult:
        try:
            return self._scan_stock_inner(symbol, data, criteria)
        except Exception as exc:
            logger.exception("SetupEngineScanner error for %s: %s", symbol, exc)
            return self._error_result(symbol, str(exc))

    # ------------------------------------------------------------------
    # Rating
    # ------------------------------------------------------------------

    def calculate_rating(self, score: float, details: Dict[str, Any]) -> str:
        se = details.get("setup_engine") or {}
        setup_ready = se.get("setup_ready", False)
        if score >= 80 and setup_ready:
            return "Strong Buy"
        if score >= 65 and setup_ready:
            return "Buy"
        if score >= 50:
            return "Watch"
        return "Pass"

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _scan_stock_inner(
        self,
        symbol: str,
        data: StockData,
        criteria: Optional[Dict],
    ) -> ScreenerResult:
        t_total = time.perf_counter()

        # ── Phase A: prep (data policy, parameter build, detector input) ──
        t_prep = time.perf_counter()

        # 1. Quick guard on absolute minimum data
        if not data.has_sufficient_data(min_days=100):
            return self._insufficient_data_result(
                symbol, f"Only {len(data.price_data)} daily bars (need ≥100)"
            )

        price_data = data.price_data

        # 2. Resample daily → weekly
        weekly_data = resample_ohlcv(
            price_data, exclude_incomplete_last_period=True
        )

        # 3. Count bars
        daily_bars = len(price_data)
        weekly_bars = len(weekly_data)
        benchmark_bars = len(data.benchmark_data) if not data.benchmark_data.empty else 0

        # 4. Resolve benchmark
        spy_close: pd.Series | None = None
        if not data.benchmark_data.empty and "Close" in data.benchmark_data.columns:
            spy_close = data.benchmark_data["Close"]

        # 5. Count current-week sessions
        current_week_sessions = _count_current_week_sessions(price_data)

        # 6. Evaluate data policy
        policy_result = evaluate_setup_engine_data_policy(
            daily_bars=daily_bars,
            weekly_bars=weekly_bars,
            benchmark_bars=benchmark_bars,
            current_week_sessions=current_week_sessions,
        )

        # 7. If policy says insufficient, skip aggregator but still build payload
        if policy_result["status"] == "insufficient":
            return self._build_insufficient_data_with_payload(
                symbol,
                "; ".join(policy_result["failed_reasons"]),
                policy_result,
            )

        # 8. Build parameters (with optional overrides from criteria)
        criteria = criteria or {}
        overrides = criteria.get("setup_engine_parameters")
        parameters = (
            build_setup_engine_parameters(overrides)
            if overrides
            else DEFAULT_SETUP_ENGINE_PARAMETERS
        )

        # 9. Construct detector input (both daily AND weekly OHLCV)
        detector_input = PatternDetectorInput(
            symbol=symbol,
            timeframe="daily",
            daily_bars=daily_bars,
            weekly_bars=weekly_bars,
            features={
                "daily_ohlcv": price_data,
                "weekly_ohlcv": weekly_data,
            },
        )

        prep_ms = (time.perf_counter() - t_prep) * 1000.0

        # ── Phase B: detectors (aggregation pipeline) ──
        t_detectors = time.perf_counter()

        # 10. Run aggregation pipeline
        agg_output: AggregatedPatternOutput = self._aggregator.aggregate(
            detector_input,
            parameters=parameters,
            policy_result=policy_result,
        )

        detectors_ms = (time.perf_counter() - t_detectors) * 1000.0

        # Per-detector timing at DEBUG level (avoids 3500+ lines per full scan)
        for trace in agg_output.detector_traces:
            logger.debug(
                "SE detector %s: outcome=%s candidates=%d elapsed=%.1fms | %s",
                trace.detector_name, trace.outcome, trace.candidate_count,
                trace.elapsed_ms, symbol,
            )

        # ── Phase C: readiness + payload assembly ──
        t_readiness = time.perf_counter()

        # 11. Extract pivot price for readiness computation
        pivot_price = agg_output.pivot_price

        # 12. Compute readiness features
        readiness_features = compute_breakout_readiness_features(
            price_data,
            pivot_price=pivot_price,
            benchmark_close=spy_close,
        )

        # 13. Build canonical payload
        payload = build_setup_engine_payload(
            pattern_primary=agg_output.pattern_primary,
            pattern_confidence=agg_output.pattern_confidence,
            pivot_price=agg_output.pivot_price,
            pivot_type=agg_output.pivot_type,
            pivot_date=agg_output.pivot_date,
            candidates=list(agg_output.candidates),
            passed_checks=list(agg_output.passed_checks),
            failed_checks=list(agg_output.failed_checks),
            key_levels=agg_output.key_levels,
            invalidation_flags=list(agg_output.invalidation_flags),
            readiness_features=readiness_features,
            parameters=parameters,
            data_policy_result=policy_result,
        )

        readiness_ms = (time.perf_counter() - t_readiness) * 1000.0

        # 14. Package into ScreenerResult
        score = payload["setup_score"] if payload["setup_score"] is not None else 0.0
        details: Dict[str, Any] = {"setup_engine": payload}

        total_ms = (time.perf_counter() - t_total) * 1000.0

        # Scanner-level timing breakdown at INFO (1 line per symbol)
        error_count = sum(1 for t in agg_output.detector_traces if t.outcome == "error")
        no_data_count = sum(1 for t in agg_output.detector_traces if t.outcome == "no_data")
        logger.info(
            "SE timing %s: prep=%.1fms detectors=%.1fms readiness=%.1fms total=%.1fms "
            "score=%s errors=%d no_data=%d",
            symbol, prep_ms, detectors_ms, readiness_ms, total_ms,
            payload.get("setup_score"), error_count, no_data_count,
        )

        return ScreenerResult(
            score=score,
            passes=payload["setup_ready"],
            rating=self.calculate_rating(score, details),
            breakdown=self._extract_breakdown(payload),
            details=details,
            screener_name="setup_engine",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_breakdown(payload: Dict[str, Any]) -> Dict[str, float]:
        """Extract key sub-scores for the breakdown summary."""
        breakdown: Dict[str, float] = {}
        for key in ("quality_score", "readiness_score", "setup_score"):
            val = payload.get(key)
            if val is not None:
                breakdown[key] = float(val)
        return breakdown

    @staticmethod
    def _insufficient_data_result(symbol: str, reason: str) -> ScreenerResult:
        return ScreenerResult(
            score=0.0,
            passes=False,
            rating="Insufficient Data",
            breakdown={},
            details={"reason": reason},
            screener_name="setup_engine",
        )

    @staticmethod
    def _error_result(symbol: str, error: str) -> ScreenerResult:
        return ScreenerResult(
            score=0.0,
            passes=False,
            rating="Error",
            breakdown={},
            details={"error": error},
            screener_name="setup_engine",
        )

    @staticmethod
    def _build_insufficient_data_with_payload(
        symbol: str,
        reason: str,
        policy_result: SetupEngineDataPolicyResult,
    ) -> ScreenerResult:
        """Build a proper payload documenting why data was insufficient."""
        payload = build_setup_engine_payload(
            data_policy_result=policy_result,
        )
        return ScreenerResult(
            score=0.0,
            passes=False,
            rating="Insufficient Data",
            breakdown={},
            details={"setup_engine": payload, "reason": reason},
            screener_name="setup_engine",
        )


def _count_current_week_sessions(price_data: pd.DataFrame) -> int:
    """Count trading sessions in the current (most recent) week."""
    if price_data.empty:
        return 0
    last_date = price_data.index[-1]
    week_start = (last_date - pd.Timedelta(days=last_date.weekday())).normalize()
    return int((price_data.index >= week_start).sum())
