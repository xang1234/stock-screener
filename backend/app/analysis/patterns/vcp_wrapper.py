"""VCP detector entrypoint wrapper.

Expected input orientation:
- Chronological features (oldest -> newest).
- No look-ahead assumptions.

TODO(SE-C1): Wrap existing criteria/vcp_detection implementation without logic fork.
"""

from __future__ import annotations

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
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
            return PatternDetectorResult(
                detector_name=self.name,
                candidate=None,
                failed_checks=("insufficient_data", *normalized.failed_checks),
                warnings=("vcp_wrapper_insufficient_data", *normalized.warnings),
            )

        return PatternDetectorResult(
            detector_name=self.name,
            candidate=None,
            failed_checks=("detector_not_implemented",),
            warnings=("vcp_wrapper_stub", *normalized.warnings),
        )
