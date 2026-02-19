"""NR7/Inside-Day trigger detector entrypoint.

Expected input orientation:
- Daily bars in chronological order.
- Trigger bars are evaluated on completed bars only.

TODO(SE-C5): Implement NR7, inside-day, and combined trigger subtype logic.
"""

from __future__ import annotations

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv


class NR7InsideDayDetector(PatternDetector):
    """Compile-safe entrypoint for trigger-family detection."""

    name = "nr7_inside_day"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="daily",
            min_bars=10,
            feature_key="daily_ohlcv",
            fallback_bar_count=detector_input.daily_bars,
        )
        if not normalized.prerequisites_ok:
            return PatternDetectorResult(
                detector_name=self.name,
                candidate=None,
                failed_checks=("insufficient_data", *normalized.failed_checks),
                warnings=("nr7_inside_day_insufficient_data", *normalized.warnings),
            )

        return PatternDetectorResult(
            detector_name=self.name,
            candidate=None,
            failed_checks=("detector_not_implemented",),
            warnings=("nr7_inside_day_stub", *normalized.warnings),
        )
