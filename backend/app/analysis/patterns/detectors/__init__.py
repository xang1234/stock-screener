"""Pattern detector registry for Setup Engine analysis layer."""

from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.detectors.cup_with_handle import CupWithHandleDetector
from app.analysis.patterns.detectors.double_bottom import DoubleBottomDetector
from app.analysis.patterns.detectors.vcp import VCPDetector


def default_pattern_detectors() -> tuple[PatternDetector, ...]:
    """Return the default v1 detector set in stable execution order."""
    return (
        CupWithHandleDetector(),
        VCPDetector(),
        DoubleBottomDetector(),
    )


__all__ = [
    "PatternDetector",
    "PatternDetectorInput",
    "PatternDetectorResult",
    "CupWithHandleDetector",
    "DoubleBottomDetector",
    "VCPDetector",
    "default_pattern_detectors",
]
