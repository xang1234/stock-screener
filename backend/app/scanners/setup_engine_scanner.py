"""Setup Engine payload assembly helpers.

This module defines the canonical assembly path for ``details['setup_engine']``.
The values here are intentionally decoupled from concrete detector logic so
future detectors can feed a stable persistence/query contract.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Mapping, Sequence, cast

from app.analysis.patterns.config import (
    DEFAULT_SETUP_ENGINE_PARAMETERS,
    SETUP_SCORE_WEIGHTS,
    SetupEngineParameters,
    assert_valid_setup_engine_parameters,
)
from app.analysis.patterns.models import (
    SETUP_ENGINE_ALLOWED_TIMEFRAMES,
    SETUP_ENGINE_DEFAULT_SCHEMA_VERSION,
    PatternCandidate,
    PatternCandidateModel,
    coerce_pattern_candidate,
    SetupEnginePayload,
    assert_valid_setup_engine_payload,
    is_snake_case,
    normalize_iso_date,
)
from app.analysis.patterns.report import (
    KeyLevels,
    SetupEngineReport,
    assert_valid_setup_engine_report_payload,
)
from app.analysis.patterns.policy import (
    SetupEngineDataPolicyResult,
    policy_failed_checks,
    policy_invalidation_flags,
)
from app.analysis.patterns.readiness import (
    BreakoutReadinessFeatures,
    BreakoutReadinessTraceInputs,
    readiness_features_to_payload_fields,
)
from app.analysis.patterns.explain_builder import (
    ExplainBuilderInput,
    build_explain_payload,
)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric fields")
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    raise ValueError(f"Expected numeric value, got {type(value)!r}")


def _normalize_text_list(values: Sequence[Any] | None) -> list[str]:
    if values is None:
        return []
    return [str(v) for v in values if v is not None and str(v) != ""]


def _normalize_invalidation_flags(values: Sequence[Any] | None) -> list[Any]:
    if values is None:
        return []
    normalized: list[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        normalized.append(value)
    return normalized


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _normalize_key_levels(values: Mapping[str, Any] | None) -> dict[str, float | None]:
    if values is None:
        return {}

    normalized: dict[str, float | None] = {}
    for key, value in values.items():
        if not is_snake_case(key):
            raise ValueError(f"key_levels key must be snake_case: {key}")
        normalized[key] = _to_float(value)
    return normalized


def _normalize_candidates(
    candidates: Sequence[Mapping[str, Any] | PatternCandidateModel] | None,
    *,
    default_timeframe: str,
) -> list[PatternCandidate]:
    if candidates is None:
        return []

    normalized: list[PatternCandidate] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            for key in candidate.keys():
                if not is_snake_case(str(key)):
                    raise ValueError(f"Candidate key must be snake_case: {key}")

        normalized.append(
            coerce_pattern_candidate(
                candidate,
                default_timeframe=default_timeframe,
            )
        )

    return normalized


def _coerce_readiness_features(
    value: BreakoutReadinessFeatures | Mapping[str, Any] | None,
) -> BreakoutReadinessFeatures | None:
    if value is None:
        return None
    if isinstance(value, BreakoutReadinessFeatures):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("readiness_features must be a mapping or BreakoutReadinessFeatures")

    def _num_or_none(key: str) -> float | None:
        return _to_float(value.get(key))

    return BreakoutReadinessFeatures(
        distance_to_pivot_pct=_num_or_none("distance_to_pivot_pct"),
        atr14_pct=_num_or_none("atr14_pct"),
        atr14_pct_trend=_num_or_none("atr14_pct_trend"),
        bb_width_pct=_num_or_none("bb_width_pct"),
        bb_width_pctile_252=_num_or_none("bb_width_pctile_252"),
        volume_vs_50d=_num_or_none("volume_vs_50d"),
        rs=_num_or_none("rs"),
        rs_line_new_high=bool(value.get("rs_line_new_high", False)),
        rs_vs_spy_65d=_num_or_none("rs_vs_spy_65d"),
        rs_vs_spy_trend_20d=_num_or_none("rs_vs_spy_trend_20d"),
        bb_squeeze=bool(value.get("bb_squeeze", False)),
        quiet_days_10d=_to_int(value.get("quiet_days_10d")),
        up_down_volume_ratio_10d=_num_or_none("up_down_volume_ratio_10d"),
    )


def _primary_candidate_mapping(
    candidates: Sequence[Mapping[str, Any] | Any],
    pattern_primary: str,
) -> Mapping[str, Any] | None:
    for candidate in candidates:
        mapping: Mapping[str, Any] = (
            candidate if isinstance(candidate, Mapping) else {}
        )
        if mapping.get("pattern") == pattern_primary:
            return mapping
    return None


def _extract_primary_candidate_scores(
    candidates: Sequence[Mapping[str, Any] | Any],
    pattern_primary: str,
) -> tuple[float | None, float | None, float | None, bool]:
    """Extract calibrated scores from the candidate matching pattern_primary.

    Returns (quality_score, readiness_score, confidence, stage_ok).
    Falls back to (None, None, None, True) when no match is found.
    """
    mapping = _primary_candidate_mapping(candidates, pattern_primary)
    if mapping is None:
        return (None, None, None, True)

    quality = mapping.get("quality_score")
    readiness = mapping.get("readiness_score")
    confidence_pct = mapping.get("confidence_pct")
    confidence = mapping.get("confidence")
    checks = mapping.get("checks") or {}
    stage_ok = checks.get("stage_ok", True)

    normalized_confidence_pct = (
        float(confidence_pct)
        if confidence_pct is not None and not isinstance(confidence_pct, bool)
        else None
    )
    if normalized_confidence_pct is None and confidence is not None and not isinstance(confidence, bool):
        normalized_confidence_pct = float(confidence) * 100.0

    return (
        float(quality) if quality is not None and not isinstance(quality, bool) else None,
        float(readiness) if readiness is not None and not isinstance(readiness, bool) else None,
        normalized_confidence_pct,
        bool(stage_ok),
    )


def _coalesce(*values: float | int | None) -> float | None:
    for value in values:
        if value is None:
            continue
        return float(value)
    return None


def _extract_base_structure_metrics(
    candidate: Mapping[str, Any] | None,
) -> tuple[float | None, float | None, int | None, int | None]:
    if candidate is None:
        return (None, None, None, None)

    metrics_raw = candidate.get("metrics")
    metrics: Mapping[str, Any] = metrics_raw if isinstance(metrics_raw, Mapping) else {}

    start_date = normalize_iso_date(cast(str | date | datetime | None, candidate.get("start_date")))
    end_date = normalize_iso_date(cast(str | date | datetime | None, candidate.get("end_date")))
    if start_date is None:
        start_date = normalize_iso_date(cast(str | date | datetime | None, metrics.get("run_start_date")))
    if end_date is None:
        end_date = normalize_iso_date(cast(str | date | datetime | None, metrics.get("run_end_date")))
    if start_date is None:
        start_date = normalize_iso_date(cast(str | date | datetime | None, metrics.get("handle_start_date")))
    if end_date is None:
        end_date = normalize_iso_date(cast(str | date | datetime | None, metrics.get("handle_end_date")))

    base_length_weeks = _coalesce(
        metrics.get("base_length_weeks"),
        metrics.get("cup_duration_weeks"),
        metrics.get("weeks_tight"),
    )
    if base_length_weeks is None:
        flag_duration_bars = _to_float(metrics.get("flag_duration_bars"))
        if flag_duration_bars is not None:
            base_length_weeks = flag_duration_bars / 5.0
    if base_length_weeks is None:
        pullback_span_bars = _to_float(metrics.get("pullback_span_bars"))
        if pullback_span_bars is not None:
            base_length_weeks = pullback_span_bars / 5.0
    if base_length_weeks is None and start_date and end_date:
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
        span_days = (end_dt - start_dt).days
        if span_days >= 0:
            base_length_weeks = span_days / 7.0

    base_depth_pct = _coalesce(
        metrics.get("base_depth_pct"),
        metrics.get("cup_depth_pct"),
        metrics.get("pullback_depth_pct_from_high"),
        metrics.get("flag_depth_pct"),
        metrics.get("tight_range_pct"),
        metrics.get("distance_from_high_pct"),
    )

    support_tests_count = _to_int(
        metrics.get("support_tests_count")
        if metrics.get("support_tests_count") is not None
        else metrics.get("tests_count")
    )
    tight_closes_count = _to_int(
        metrics.get("tight_closes_count")
        if metrics.get("tight_closes_count") is not None
        else metrics.get("weeks_tight")
    )

    return (base_length_weeks, base_depth_pct, support_tests_count, tight_closes_count)


def build_setup_engine_payload(
    *,
    setup_score: int | float | None = None,
    quality_score: int | float | None = None,
    readiness_score: int | float | None = None,
    setup_ready: bool | None = None,
    pattern_primary: str | None = None,
    pattern_confidence: int | float | None = None,
    pivot_price: int | float | None = None,
    pivot_type: str | None = None,
    pivot_date: str | date | datetime | None = None,
    distance_to_pivot_pct: int | float | None = None,
    in_early_zone: bool | None = None,
    extended_from_pivot: bool | None = None,
    base_length_weeks: int | float | None = None,
    base_depth_pct: int | float | None = None,
    support_tests_count: int | float | None = None,
    tight_closes_count: int | float | None = None,
    atr14_pct: int | float | None = None,
    atr14_pct_trend: int | float | None = None,
    bb_width_pct: int | float | None = None,
    bb_width_pctile_252: int | float | None = None,
    bb_squeeze: bool | None = None,
    volume_vs_50d: int | float | None = None,
    up_down_volume_ratio_10d: int | float | None = None,
    quiet_days_10d: int | float | None = None,
    rs: int | float | None = None,
    rs_line_new_high: bool | None = None,
    rs_vs_spy_65d: int | float | None = None,
    rs_vs_spy_trend_20d: int | float | None = None,
    readiness_features: BreakoutReadinessFeatures | Mapping[str, Any] | None = None,
    candidates: Sequence[Mapping[str, Any] | PatternCandidateModel] | None = None,
    passed_checks: Sequence[Any] | None = None,
    failed_checks: Sequence[Any] | None = None,
    key_levels: Mapping[str, Any] | None = None,
    invalidation_flags: Sequence[Any] | None = None,
    timeframe: str = "daily",
    schema_version: str = SETUP_ENGINE_DEFAULT_SCHEMA_VERSION,
    stage: int | None = None,
    ma_alignment_score: int | float | None = None,
    rs_rating: int | float | None = None,
    readiness_threshold_pct: float | None = None,
    parameters: SetupEngineParameters = DEFAULT_SETUP_ENGINE_PARAMETERS,
    data_policy_result: SetupEngineDataPolicyResult | None = None,
    include_score_trace: bool = False,
    readiness_trace_inputs: BreakoutReadinessTraceInputs | None = None,
) -> SetupEnginePayload:
    """Build the canonical ``setup_engine`` payload.

    Bool semantics:
    - ``setup_ready`` is derived when not explicitly set.
    - Derived readiness requires ``readiness_score >= readiness_threshold_pct``
      and zero failed checks.

    Date semantics:
    - ``pivot_date`` and candidate ``pivot_date`` use ``YYYY-MM-DD``.

    Nullability semantics:
    - Numeric fields are null when detectors cannot produce a value.
    - Lists and explain object are always present.
    """

    if timeframe not in SETUP_ENGINE_ALLOWED_TIMEFRAMES:
        raise ValueError(
            f"timeframe must be one of {sorted(SETUP_ENGINE_ALLOWED_TIMEFRAMES)}"
        )

    assert_valid_setup_engine_parameters(parameters)
    threshold_pct = (
        float(readiness_threshold_pct)
        if readiness_threshold_pct is not None
        else float(parameters.readiness_score_ready_min_pct)
    )

    normalized_passed = _normalize_text_list(passed_checks)
    normalized_failed = _normalize_text_list(failed_checks)
    normalized_flags = _normalize_invalidation_flags(invalidation_flags)

    normalized_setup_score = _to_float(setup_score)
    normalized_quality_score = _to_float(quality_score)
    normalized_readiness_score = _to_float(readiness_score)
    normalized_pattern_confidence = _to_float(pattern_confidence)

    # Primary candidate extraction: fill quality/readiness/confidence fallbacks.
    primary_candidate: Mapping[str, Any] | None = None
    extracted_confidence: float | None = None
    if candidates and pattern_primary:
        primary_candidate = _primary_candidate_mapping(candidates, pattern_primary)
        ext_q, ext_r, ext_c, _ = _extract_primary_candidate_scores(
            candidates,
            pattern_primary,
        )
        if normalized_quality_score is None:
            normalized_quality_score = ext_q
        if normalized_readiness_score is None:
            normalized_readiness_score = ext_r
        extracted_confidence = ext_c

    if normalized_pattern_confidence is None and extracted_confidence is not None:
        normalized_pattern_confidence = extracted_confidence

    normalized_readiness = _coerce_readiness_features(readiness_features)
    if normalized_readiness is not None:
        readiness_values = readiness_features_to_payload_fields(normalized_readiness)
        distance_to_pivot_pct = (
            distance_to_pivot_pct
            if distance_to_pivot_pct is not None
            else readiness_values["distance_to_pivot_pct"]
        )
        atr14_pct = atr14_pct if atr14_pct is not None else readiness_values["atr14_pct"]
        atr14_pct_trend = (
            atr14_pct_trend
            if atr14_pct_trend is not None
            else readiness_values["atr14_pct_trend"]
        )
        bb_width_pct = (
            bb_width_pct
            if bb_width_pct is not None
            else readiness_values["bb_width_pct"]
        )
        bb_width_pctile_252 = (
            bb_width_pctile_252
            if bb_width_pctile_252 is not None
            else readiness_values["bb_width_pctile_252"]
        )
        bb_squeeze = (
            bb_squeeze
            if bb_squeeze is not None
            else cast(bool | None, readiness_values["bb_squeeze"])
        )
        volume_vs_50d = (
            volume_vs_50d
            if volume_vs_50d is not None
            else readiness_values["volume_vs_50d"]
        )
        up_down_volume_ratio_10d = (
            up_down_volume_ratio_10d
            if up_down_volume_ratio_10d is not None
            else cast(float | int | None, readiness_values["up_down_volume_ratio_10d"])
        )
        quiet_days_10d = (
            quiet_days_10d
            if quiet_days_10d is not None
            else cast(float | int | None, readiness_values["quiet_days_10d"])
        )
        rs = rs if rs is not None else readiness_values["rs"]
        rs_line_new_high = (
            rs_line_new_high
            if rs_line_new_high is not None
            else bool(readiness_values["rs_line_new_high"])
        )
        rs_vs_spy_65d = (
            rs_vs_spy_65d
            if rs_vs_spy_65d is not None
            else readiness_values["rs_vs_spy_65d"]
        )
        rs_vs_spy_trend_20d = (
            rs_vs_spy_trend_20d
            if rs_vs_spy_trend_20d is not None
            else readiness_values["rs_vs_spy_trend_20d"]
        )

    if data_policy_result is not None:
        normalized_failed.extend(policy_failed_checks(data_policy_result))
        normalized_flags.extend(policy_invalidation_flags(data_policy_result))
        if data_policy_result["status"] == "insufficient":
            # Deterministic degradation path for insufficient inputs.
            normalized_setup_score = None
            normalized_quality_score = None
            normalized_readiness_score = None
            pattern_primary = None
            normalized_pattern_confidence = None
            pivot_price = None
            pivot_type = None
            pivot_date = None
            distance_to_pivot_pct = None
            in_early_zone = None
            extended_from_pivot = None
            base_length_weeks = None
            base_depth_pct = None
            support_tests_count = None
            tight_closes_count = None
            atr14_pct = None
            atr14_pct_trend = None
            bb_width_pct = None
            bb_width_pctile_252 = None
            bb_squeeze = None
            volume_vs_50d = None
            up_down_volume_ratio_10d = None
            quiet_days_10d = None
            rs = None
            candidates = []
            rs_line_new_high = False
            rs_vs_spy_65d = None
            rs_vs_spy_trend_20d = None
            stage = None
            ma_alignment_score = None
            rs_rating = None

    # ── Setup score synthesis ─────────────────────────
    if normalized_setup_score is None and normalized_quality_score is not None and normalized_readiness_score is not None:
        wq, wr = SETUP_SCORE_WEIGHTS
        normalized_setup_score = round(
            min(100.0, max(0.0, wq * normalized_quality_score + wr * normalized_readiness_score)),
            6,
        )

    if (
        base_length_weeks is None
        or base_depth_pct is None
        or support_tests_count is None
        or tight_closes_count is None
    ):
        (
            extracted_base_length_weeks,
            extracted_base_depth_pct,
            extracted_support_tests_count,
            extracted_tight_closes_count,
        ) = _extract_base_structure_metrics(primary_candidate)
        if base_length_weeks is None:
            base_length_weeks = extracted_base_length_weeks
        if base_depth_pct is None:
            base_depth_pct = extracted_base_depth_pct
        if support_tests_count is None:
            support_tests_count = extracted_support_tests_count
        if tight_closes_count is None:
            tight_closes_count = extracted_tight_closes_count

    # ── Readiness gate evaluation via explain builder ──
    norm_distance = _to_float(distance_to_pivot_pct)
    norm_atr14 = _to_float(atr14_pct)
    norm_volume = _to_float(volume_vs_50d)
    norm_rs_vs_spy = _to_float(rs_vs_spy_65d)
    norm_base_length_weeks = _to_float(base_length_weeks)
    norm_base_depth_pct = _to_float(base_depth_pct)
    norm_up_down_volume_ratio_10d = _to_float(up_down_volume_ratio_10d)
    norm_quiet_days_10d = _to_int(quiet_days_10d)
    norm_support_tests_count = _to_int(support_tests_count)
    norm_tight_closes_count = _to_int(tight_closes_count)

    if in_early_zone is None:
        in_early_zone = (
            (
                parameters.early_zone_distance_to_pivot_pct_min
                <= norm_distance
                <= parameters.early_zone_distance_to_pivot_pct_max
            )
            if norm_distance is not None
            else None
        )
    if extended_from_pivot is None:
        extended_from_pivot = (
            norm_distance > parameters.too_extended_pivot_distance_pct
            if norm_distance is not None
            else None
        )

    explain_input = ExplainBuilderInput(
        pre_existing_passed_checks=tuple(normalized_passed),
        pre_existing_failed_checks=tuple(normalized_failed),
        key_levels=KeyLevels(levels=_normalize_key_levels(key_levels)),
        pre_existing_invalidation_flags=tuple(normalized_flags),
        setup_score=normalized_setup_score,
        quality_score=normalized_quality_score,
        readiness_score=normalized_readiness_score,
        distance_to_pivot_pct=norm_distance,
        atr14_pct=norm_atr14,
        volume_vs_50d=norm_volume,
        rs_vs_spy_65d=norm_rs_vs_spy,
        rs_line_new_high=bool(rs_line_new_high) if rs_line_new_high is not None else False,
        stage=stage,
        ma_alignment_score=_to_float(ma_alignment_score),
        rs_rating=_to_float(rs_rating),
        parameters=parameters,
        readiness_threshold_pct=threshold_pct,
        include_score_trace=include_score_trace,
        readiness_features=normalized_readiness,
        readiness_trace_inputs=readiness_trace_inputs,
    )
    explain_result = build_explain_payload(explain_input)

    payload: SetupEnginePayload = {
        "schema_version": schema_version,
        "timeframe": cast(Any, timeframe),
        "setup_score": normalized_setup_score,
        "quality_score": normalized_quality_score,
        "readiness_score": normalized_readiness_score,
        "setup_ready": bool(setup_ready) if setup_ready is not None else explain_result.derived_ready,
        "pattern_primary": pattern_primary,
        "pattern_confidence": normalized_pattern_confidence,
        "pivot_price": _to_float(pivot_price),
        "pivot_type": pivot_type,
        "pivot_date": normalize_iso_date(pivot_date),
        "distance_to_pivot_pct": norm_distance,
        "in_early_zone": in_early_zone,
        "extended_from_pivot": extended_from_pivot,
        "base_length_weeks": norm_base_length_weeks,
        "base_depth_pct": norm_base_depth_pct,
        "support_tests_count": norm_support_tests_count,
        "tight_closes_count": norm_tight_closes_count,
        "atr14_pct": norm_atr14,
        "atr14_pct_trend": _to_float(atr14_pct_trend),
        "bb_width_pct": _to_float(bb_width_pct),
        "bb_width_pctile_252": _to_float(bb_width_pctile_252),
        "bb_squeeze": bb_squeeze,
        "volume_vs_50d": norm_volume,
        "up_down_volume_ratio_10d": norm_up_down_volume_ratio_10d,
        "quiet_days_10d": norm_quiet_days_10d,
        "rs": _to_float(rs),
        "rs_line_new_high": bool(rs_line_new_high),
        "rs_vs_spy_65d": norm_rs_vs_spy,
        "rs_vs_spy_trend_20d": _to_float(rs_vs_spy_trend_20d),
        "stage": stage,
        "ma_alignment_score": _to_float(ma_alignment_score),
        "rs_rating": _to_float(rs_rating),
        "candidates": _normalize_candidates(
            candidates,
            default_timeframe=timeframe,
        ),
        "explain": explain_result.explain.to_payload(),
    }

    assert_valid_setup_engine_payload(payload)
    assert_valid_setup_engine_report_payload(payload)
    return payload


def attach_setup_engine(
    details: Mapping[str, Any] | None,
    setup_engine: SetupEnginePayload,
) -> dict[str, Any]:
    """Return a details payload with ``setup_engine`` attached at top level."""
    merged = dict(details or {})
    merged["setup_engine"] = setup_engine
    return merged


def build_setup_engine_payload_from_report(
    report: SetupEngineReport,
) -> SetupEnginePayload:
    """Serialize a typed report container into canonical setup_engine payload."""
    payload = report.to_payload()
    assert_valid_setup_engine_report_payload(payload)
    return payload
