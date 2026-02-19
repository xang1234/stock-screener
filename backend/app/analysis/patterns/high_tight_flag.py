"""High-Tight-Flag detector entrypoint.

Expected input orientation:
- Chronological daily bars (oldest -> newest).
- Pole and flag phases use strictly historical windows.

TODO(SE-C3a): Implement pole candidate identification over configurable windows.
TODO(SE-C3b): Implement flag validation and pivot extraction.
"""

from __future__ import annotations

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv


class HighTightFlagDetector(PatternDetector):
    """Compile-safe entrypoint for HTF detection."""

    name = "high_tight_flag"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="daily",
            min_bars=180,
            feature_key="daily_ohlcv",
            fallback_bar_count=detector_input.daily_bars,
        )
        if not normalized.prerequisites_ok:
            return PatternDetectorResult(
                detector_name=self.name,
                candidate=None,
                failed_checks=("insufficient_data", *normalized.failed_checks),
                warnings=("high_tight_flag_insufficient_data", *normalized.warnings),
            )

        return PatternDetectorResult(
            detector_name=self.name,
            candidate=None,
            failed_checks=("detector_not_implemented",),
            warnings=("high_tight_flag_stub", *normalized.warnings),
        )
