"""Cup-with-handle detector stub.

This module exists to establish import boundaries; detection logic lands in SE-B6.
"""

from __future__ import annotations

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)


class CupWithHandleDetector(PatternDetector):
    """Placeholder detector implementation."""

    name = "cup_with_handle"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del detector_input, parameters
        return PatternDetectorResult(
            detector_name=self.name,
            candidate=None,
            failed_checks=("detector_not_implemented",),
            warnings=("cup_with_handle_detector_stub",),
        )
