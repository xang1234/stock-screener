"""Setup Engine payload assembly helpers.

This module defines the canonical assembly path for ``details['setup_engine']``.
The values here are intentionally decoupled from concrete detector logic so
future detectors can feed a stable persistence/query contract.
"""

from __future__ import annotations

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
    SetupEngineExplain,
    SetupEnginePayload,
    assert_valid_setup_engine_payload,
    is_snake_case,
    normalize_iso_date,
)
from app.analysis.patterns.report import (
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
from app.analysis.patterns.trace import build_score_trace


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric fields")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected numeric value, got {type(value)!r}")


def _normalize_text_list(values: Sequence[Any] | None) -> list[str]:
    if values is None:
        return []
    return [str(v) for v in values if v is not None and str(v) != ""]


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
    )


def _extract_primary_candidate_scores(
    candidates: Sequence[Mapping[str, Any] | Any],
    pattern_primary: str,
) -> tuple[float | None, float | None, float | None, bool]:
    """Extract calibrated scores from the candidate matching pattern_primary.

    Returns (quality_score, readiness_score, confidence, stage_ok).
    Falls back to (None, None, None, True) when no match is found.
    """
    for candidate in candidates:
        mapping: Mapping[str, Any] = (
            candidate if isinstance(candidate, Mapping) else {}
        )
        if mapping.get("pattern") == pattern_primary:
            quality = mapping.get("quality_score")
            readiness = mapping.get("readiness_score")
            confidence = mapping.get("confidence")
            checks = mapping.get("checks") or {}
            stage_ok = checks.get("stage_ok", True)
            return (
                float(quality) if quality is not None and not isinstance(quality, bool) else None,
                float(readiness) if readiness is not None and not isinstance(readiness, bool) else None,
                float(confidence) if confidence is not None and not isinstance(confidence, bool) else None,
                bool(stage_ok),
            )
    return (None, None, None, True)


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
    atr14_pct: int | float | None = None,
    atr14_pct_trend: int | float | None = None,
    bb_width_pct: int | float | None = None,
    bb_width_pctile_252: int | float | None = None,
    volume_vs_50d: int | float | None = None,
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
    normalized_flags = _normalize_text_list(invalidation_flags)

    normalized_setup_score = _to_float(setup_score)
    normalized_quality_score = _to_float(quality_score)
    normalized_readiness_score = _to_float(readiness_score)

    # Primary candidate extraction: fill quality/readiness/confidence fallbacks.
    extracted_stage_ok = True
    extracted_confidence: float | None = None
    if (normalized_quality_score is None or normalized_readiness_score is None) and candidates and pattern_primary:
        ext_q, ext_r, ext_c, extracted_stage_ok = _extract_primary_candidate_scores(
            candidates, pattern_primary,
        )
        if normalized_quality_score is None:
            normalized_quality_score = ext_q
        if normalized_readiness_score is None:
            normalized_readiness_score = ext_r
        extracted_confidence = ext_c
    elif candidates and pattern_primary:
        _, _, _, extracted_stage_ok = _extract_primary_candidate_scores(
            candidates, pattern_primary,
        )

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
        volume_vs_50d = (
            volume_vs_50d
            if volume_vs_50d is not None
            else readiness_values["volume_vs_50d"]
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
            pattern_confidence = None
            pivot_price = None
            pivot_type = None
            pivot_date = None
            distance_to_pivot_pct = None
            atr14_pct = None
            atr14_pct_trend = None
            bb_width_pct = None
            bb_width_pctile_252 = None
            volume_vs_50d = None
            rs = None
            candidates = []
            rs_line_new_high = False
            rs_vs_spy_65d = None
            rs_vs_spy_trend_20d = None

    # ── Setup score synthesis ─────────────────────────
    if normalized_setup_score is None and normalized_quality_score is not None and normalized_readiness_score is not None:
        wq, wr = SETUP_SCORE_WEIGHTS
        normalized_setup_score = round(
            min(100.0, max(0.0, wq * normalized_quality_score + wr * normalized_readiness_score)),
            6,
        )

    # ── Readiness gate evaluation ──────────────────────
    # Normalize context-relevant floats before gate checks.
    norm_distance = _to_float(distance_to_pivot_pct)
    norm_atr14 = _to_float(atr14_pct)
    norm_volume = _to_float(volume_vs_50d)
    norm_rs_vs_spy = _to_float(rs_vs_spy_65d)

    if normalized_setup_score is None:
        # Insufficient data: skip all context gates, force not ready.
        derived_ready = False
    else:
        # Gate 1: setup_score threshold
        if normalized_setup_score >= parameters.setup_score_min_pct:
            normalized_passed.append("setup_score_ok")
        else:
            normalized_failed.append("setup_score_below_threshold")

        # Gate 2: quality floor
        if normalized_quality_score is not None and normalized_quality_score >= parameters.quality_score_min_pct:
            normalized_passed.append("quality_floor_ok")
        else:
            normalized_failed.append("quality_below_threshold")

        # Gate 3: readiness floor
        if normalized_readiness_score is not None and normalized_readiness_score >= threshold_pct:
            normalized_passed.append("readiness_floor_ok")
        else:
            normalized_failed.append("readiness_below_threshold")

        # Gate 4: early zone
        if norm_distance is not None:
            if (
                parameters.early_zone_distance_to_pivot_pct_min
                <= norm_distance
                <= parameters.early_zone_distance_to_pivot_pct_max
            ):
                normalized_passed.append("in_early_zone")
            else:
                normalized_failed.append("outside_early_zone")
        else:
            normalized_failed.append("outside_early_zone")

        # Gate 5: ATR14 cap (permissive when None)
        if norm_atr14 is not None:
            if norm_atr14 <= parameters.atr14_pct_max_for_ready:
                normalized_passed.append("atr14_within_limit")
            else:
                normalized_failed.append("atr14_pct_exceeds_limit")
        else:
            normalized_passed.append("atr14_within_limit")

        # Gate 6: Volume floor (permissive when None)
        if norm_volume is not None:
            if norm_volume >= parameters.volume_vs_50d_min_for_ready:
                normalized_passed.append("volume_sufficient")
            else:
                normalized_failed.append("volume_below_minimum")
        else:
            normalized_passed.append("volume_sufficient")

        # Gate 7: RS leadership (permissive when both None)
        rs_line_high = bool(rs_line_new_high) if rs_line_new_high is not None else False
        if norm_rs_vs_spy is not None or rs_line_high:
            rs_ok = (norm_rs_vs_spy is not None and norm_rs_vs_spy > 0) or rs_line_high
            if rs_ok:
                normalized_passed.append("rs_leadership_ok")
            else:
                normalized_failed.append("rs_leadership_insufficient")
        else:
            # Both unavailable → permissive
            normalized_passed.append("rs_leadership_ok")

        # Gate 8: Stage OK (from primary candidate checks)
        if extracted_stage_ok:
            normalized_passed.append("stage_ok")
        else:
            normalized_failed.append("stage_not_ok")

        # derived_ready = all gates passed (no new failures from gates)
        derived_ready = len(normalized_failed) == 0

    explain: SetupEngineExplain = {
        "passed_checks": normalized_passed,
        "failed_checks": normalized_failed,
        "key_levels": _normalize_key_levels(key_levels),
        "invalidation_flags": normalized_flags,
    }

    # ── Optional score trace ──────────────────────
    if (
        include_score_trace
        and readiness_trace_inputs is not None
        and normalized_readiness is not None
    ):
        score_trace = build_score_trace(
            normalized_readiness,
            readiness_trace_inputs,
            quality_score=normalized_quality_score,
            readiness_score=normalized_readiness_score,
            setup_score=normalized_setup_score,
        )
        explain["score_trace"] = score_trace  # type: ignore[typeddict-unknown-key]

    payload: SetupEnginePayload = {
        "schema_version": schema_version,
        "timeframe": cast(Any, timeframe),
        "setup_score": normalized_setup_score,
        "quality_score": normalized_quality_score,
        "readiness_score": normalized_readiness_score,
        "setup_ready": bool(setup_ready) if setup_ready is not None else derived_ready,
        "pattern_primary": pattern_primary,
        "pattern_confidence": _to_float(pattern_confidence),
        "pivot_price": _to_float(pivot_price),
        "pivot_type": pivot_type,
        "pivot_date": normalize_iso_date(pivot_date),
        "distance_to_pivot_pct": norm_distance,
        "atr14_pct": norm_atr14,
        "atr14_pct_trend": _to_float(atr14_pct_trend),
        "bb_width_pct": _to_float(bb_width_pct),
        "bb_width_pctile_252": _to_float(bb_width_pctile_252),
        "volume_vs_50d": norm_volume,
        "rs": _to_float(rs),
        "rs_line_new_high": bool(rs_line_new_high),
        "rs_vs_spy_65d": norm_rs_vs_spy,
        "rs_vs_spy_trend_20d": _to_float(rs_vs_spy_trend_20d),
        "candidates": _normalize_candidates(
            candidates,
            default_timeframe=timeframe,
        ),
        "explain": explain,
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
