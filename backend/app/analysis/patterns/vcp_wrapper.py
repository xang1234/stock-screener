"""VCP detector entrypoint wrapper.

Expected input orientation:
- Chronological features (oldest -> newest).
- No look-ahead assumptions.
"""

from __future__ import annotations

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.models import PatternCandidateModel
from app.analysis.patterns.legacy_vcp_detection import VCPDetector
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv


class VCPWrapperDetector(PatternDetector):
    """Compile-safe entrypoint for VCP integration."""

    name = "vcp"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="daily",
            min_bars=120,
            feature_key="daily_ohlcv",
            fallback_bar_count=detector_input.daily_bars,
        )
        if not normalized.prerequisites_ok:
            return PatternDetectorResult.insufficient_data(
                self.name, normalized=normalized
            )

        if normalized.frame is None:
            return PatternDetectorResult.insufficient_data(
                self.name,
                failed_checks=("missing_daily_ohlcv_for_vcp_wrapper",),
                warnings=normalized.warnings,
            )

        frame = normalized.frame
        prices = frame["Close"].iloc[::-1].reset_index(drop=True)
        volumes = frame["Volume"].iloc[::-1].reset_index(drop=True)

        detector = VCPDetector()
        legacy = detector.detect_vcp(prices, volumes)

        if not bool(legacy.get("vcp_detected", False)):
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=_legacy_failed_checks(legacy),
                warnings=normalized.warnings,
            )

        pivot_info = legacy.get("pivot_info", {}) if isinstance(
            legacy.get("pivot_info"), dict
        ) else {}
        pivot_price = _as_float(pivot_info.get("pivot"))
        current_price = _as_float(legacy.get("current_price"))
        if pivot_price is not None and current_price is not None and current_price != 0.0:
            distance_to_pivot_pct = (
                (pivot_price - current_price) / current_price
            ) * 100.0
        else:
            distance_to_pivot_pct = _as_float(pivot_info.get("distance_pct"))

        vcp_score = _bounded_score(_as_float(legacy.get("vcp_score")))
        readiness_score = _bounded_score(
            (
                vcp_score * 0.7
                + (25.0 if bool(pivot_info.get("ready_for_breakout")) else 0.0)
                - max(0.0, _as_float(distance_to_pivot_pct) or 0.0) * 2.0
            )
        )
        confidence = min(0.95, max(0.05, vcp_score / 100.0))

        candidate = PatternCandidateModel(
            pattern=self.name,
            timeframe="daily",
            source_detector=self.name,
            pivot_price=pivot_price,
            pivot_type="vcp_pivot",
            pivot_date=None,
            distance_to_pivot_pct=distance_to_pivot_pct,
            quality_score=vcp_score,
            readiness_score=readiness_score,
            confidence=confidence,
            metrics={
                "num_bases": int(legacy.get("num_bases", 0)),
                "vcp_score": round(vcp_score, 4),
                "contraction_ratio": _as_float(legacy.get("contraction_ratio")),
                "depth_score": _as_float(legacy.get("depth_score")),
                "volume_score": _as_float(legacy.get("volume_score")),
                "tightness_score": _as_float(legacy.get("tightness_score")),
                "atr_score": _as_float(legacy.get("atr_score")),
                "atr_contraction_ratio": _as_float(
                    legacy.get("atr_contraction_ratio")
                ),
                "distance_from_high_pct": _as_float(
                    legacy.get("distance_from_high_pct")
                ),
                "distance_to_pivot_pct_legacy": _as_float(
                    pivot_info.get("distance_pct")
                ),
                "ready_for_breakout": bool(
                    pivot_info.get("ready_for_breakout", False)
                ),
            },
            checks={
                "vcp_detected_by_legacy": True,
                "contracting_depth": bool(
                    legacy.get("contracting_depth", False)
                ),
                "contracting_volume": bool(
                    legacy.get("contracting_volume", False)
                ),
                "tight_near_highs": bool(
                    legacy.get("tight_near_highs", False)
                ),
                "pivot_available": pivot_price is not None,
            },
            notes=("legacy_vcp_wrapper_no_logic_fork",),
        )
        return PatternDetectorResult.detected(
            self.name,
            candidate,
            passed_checks=("legacy_vcp_detection_passed",),
            warnings=normalized.warnings,
        )


def _legacy_failed_checks(legacy: dict[str, object]) -> tuple[str, ...]:
    failed: list[str] = []
    if not bool(legacy.get("contracting_depth", False)):
        failed.append("vcp_contracting_depth_failed")
    if not bool(legacy.get("tight_near_highs", False)):
        failed.append("vcp_tight_near_highs_failed")
    if _as_float(legacy.get("vcp_score")) is not None and (
        (_as_float(legacy.get("vcp_score")) or 0.0) < 65.0
    ):
        failed.append("vcp_score_below_legacy_threshold")
    if int(legacy.get("num_bases", 0) or 0) < 3:
        failed.append("vcp_insufficient_bases")
    if not failed:
        failed.append("vcp_not_detected")
    return tuple(failed)


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bounded_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return min(100.0, max(0.0, value))
