"""Stable public APIs for Setup Engine pattern analysis."""

from .aggregator import AggregatedPatternOutput, SetupEngineAggregator
from .cup_handle import CupHandleDetector
from .detectors import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
    default_pattern_detectors,
)
from .first_pullback import FirstPullbackDetector
from .high_tight_flag import HighTightFlagDetector
from .models import (
    PatternCandidate,
    PatternCandidateModel,
    coerce_pattern_candidate,
    validate_pattern_candidate,
)
from .normalization import (
    REQUIRED_OHLCV_COLUMNS,
    NormalizedOHLCV,
    normalize_detector_input_ohlcv,
    normalize_ohlcv_frame,
)
from .nr7_inside_day import NR7InsideDayDetector
from .report import (
    ExplainPayload,
    InvalidationFlag,
    KeyLevels,
    SetupEngineReport,
    assert_valid_setup_engine_report_payload,
    canonical_setup_engine_report_examples,
    validate_setup_engine_report_payload,
)
from .technicals import (
    average_true_range,
    bollinger_band_width_percent,
    bollinger_bands,
    detect_swings,
    resample_ohlcv,
    rolling_linear_regression,
    rolling_percentile_rank,
    rolling_slope,
    true_range,
    true_range_from_ohlc,
    true_range_percent,
)
from .three_weeks_tight import ThreeWeeksTightDetector
from .vcp_wrapper import VCPWrapperDetector

__all__ = [
    "AggregatedPatternOutput",
    "SetupEngineAggregator",
    "PatternCandidate",
    "PatternCandidateModel",
    "coerce_pattern_candidate",
    "validate_pattern_candidate",
    "REQUIRED_OHLCV_COLUMNS",
    "NormalizedOHLCV",
    "normalize_ohlcv_frame",
    "normalize_detector_input_ohlcv",
    "PatternDetector",
    "PatternDetectorInput",
    "PatternDetectorResult",
    "default_pattern_detectors",
    "VCPWrapperDetector",
    "ThreeWeeksTightDetector",
    "HighTightFlagDetector",
    "CupHandleDetector",
    "NR7InsideDayDetector",
    "FirstPullbackDetector",
    "SetupEngineReport",
    "ExplainPayload",
    "KeyLevels",
    "InvalidationFlag",
    "validate_setup_engine_report_payload",
    "assert_valid_setup_engine_report_payload",
    "canonical_setup_engine_report_examples",
    "resample_ohlcv",
    "true_range",
    "true_range_from_ohlc",
    "true_range_percent",
    "average_true_range",
    "bollinger_bands",
    "bollinger_band_width_percent",
    "rolling_linear_regression",
    "rolling_slope",
    "rolling_percentile_rank",
    "detect_swings",
]
