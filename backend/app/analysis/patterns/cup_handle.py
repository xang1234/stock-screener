"""Cup-with-handle detector entrypoint.

Expected input orientation:
- Weekly swing features in chronological order.
- Candidate enumeration must stay deterministic.

TODO(SE-C4a): Implement cup structure parsing.
TODO(SE-C4b): Implement handle detection and upper-half constraints.
"""

from __future__ import annotations

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv


class CupHandleDetector(PatternDetector):
    """Compile-safe entrypoint for cup-with-handle detection."""

    name = "cup_with_handle"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="weekly",
            min_bars=20,
            feature_key="weekly_ohlcv",
            fallback_bar_count=detector_input.weekly_bars,
        )
        if not normalized.prerequisites_ok:
            return PatternDetectorResult(
                detector_name=self.name,
                candidate=None,
                failed_checks=("insufficient_data", *normalized.failed_checks),
                warnings=("cup_handle_insufficient_data", *normalized.warnings),
            )

        return PatternDetectorResult(
            detector_name=self.name,
            candidate=None,
            failed_checks=("detector_not_implemented",),
            warnings=("cup_handle_stub", *normalized.warnings),
        )
