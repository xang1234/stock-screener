"""First-pullback / trend-resumption detector entrypoint.

Expected input orientation:
- Chronological bars with MA features.
- Distinct touch counting must avoid clustered double-counts.

TODO(SE-C6a): Implement MA-touch/test counting and orderliness metrics.
TODO(SE-C6b): Implement trigger and pivot-choice logic.
"""

from __future__ import annotations

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv


class FirstPullbackDetector(PatternDetector):
    """Compile-safe entrypoint for first-pullback detection."""

    name = "first_pullback"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="daily",
            min_bars=60,
            feature_key="daily_ohlcv",
            fallback_bar_count=detector_input.daily_bars,
        )
        if not normalized.prerequisites_ok:
            return PatternDetectorResult(
                detector_name=self.name,
                candidate=None,
                failed_checks=("insufficient_data", *normalized.failed_checks),
                warnings=("first_pullback_insufficient_data", *normalized.warnings),
            )

        return PatternDetectorResult(
            detector_name=self.name,
            candidate=None,
            failed_checks=("detector_not_implemented",),
            warnings=("first_pullback_stub", *normalized.warnings),
        )
