"""Stable public APIs for Setup Engine pattern analysis."""

from .aggregator import (
    AggregatedPatternOutput,
    DetectorExecutionTrace,
    SetupEngineAggregator,
)
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
from .readiness import (
    BreakoutReadinessFeatures,
    BreakoutReadinessTraceInputs,
    compute_breakout_readiness_features,
    compute_breakout_readiness_features_with_trace,
    readiness_features_to_payload_fields,
)
from .trace import build_null_score_trace, build_score_trace
from .technicals import (
    average_true_range,
    bollinger_band_width_percent,
    bollinger_bands,
    detect_swings,
    has_incomplete_last_period,
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
    "DetectorExecutionTrace",
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
    "BreakoutReadinessFeatures",
    "BreakoutReadinessTraceInputs",
    "compute_breakout_readiness_features",
    "compute_breakout_readiness_features_with_trace",
    "readiness_features_to_payload_fields",
    "build_score_trace",
    "build_null_score_trace",
    "resample_ohlcv",
    "has_incomplete_last_period",
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
