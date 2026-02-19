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
    SetupEngineParameters,
    assert_valid_setup_engine_parameters,
)
from app.analysis.patterns.models import (
    SETUP_ENGINE_ALLOWED_TIMEFRAMES,
    SETUP_ENGINE_DEFAULT_SCHEMA_VERSION,
    PatternCandidate,
    SetupEngineExplain,
    SetupEnginePayload,
    assert_valid_setup_engine_payload,
    is_snake_case,
    normalize_iso_date,
)
from app.analysis.patterns.policy import (
    SetupEngineDataPolicyResult,
    policy_failed_checks,
    policy_invalidation_flags,
)


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
    candidates: Sequence[Mapping[str, Any]] | None,
    *,
    default_timeframe: str,
) -> list[PatternCandidate]:
    if candidates is None:
        return []

    normalized: list[PatternCandidate] = []
    for candidate in candidates:
        for key in candidate.keys():
            if not is_snake_case(str(key)):
                raise ValueError(f"Candidate key must be snake_case: {key}")

        timeframe = cast(str | None, candidate.get("timeframe")) or default_timeframe
        if timeframe not in SETUP_ENGINE_ALLOWED_TIMEFRAMES:
            raise ValueError(
                f"Candidate timeframe must be one of {sorted(SETUP_ENGINE_ALLOWED_TIMEFRAMES)}"
            )

        normalized.append(
            PatternCandidate(
                pattern=cast(str, candidate.get("pattern") or "unknown"),
                confidence_pct=_to_float(candidate.get("confidence_pct")),
                pivot_price=_to_float(candidate.get("pivot_price")),
                pivot_type=cast(str | None, candidate.get("pivot_type")),
                pivot_date=normalize_iso_date(
                    cast(str | date | datetime | None, candidate.get("pivot_date"))
                ),
                distance_to_pivot_pct=_to_float(candidate.get("distance_to_pivot_pct")),
                setup_score=_to_float(candidate.get("setup_score")),
                quality_score=_to_float(candidate.get("quality_score")),
                readiness_score=_to_float(candidate.get("readiness_score")),
                timeframe=cast(Any, timeframe),
            )
        )

    return normalized


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
    bb_width_pctile_252: int | float | None = None,
    volume_vs_50d: int | float | None = None,
    rs_line_new_high: bool = False,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    passed_checks: Sequence[Any] | None = None,
    failed_checks: Sequence[Any] | None = None,
    key_levels: Mapping[str, Any] | None = None,
    invalidation_flags: Sequence[Any] | None = None,
    timeframe: str = "daily",
    schema_version: str = SETUP_ENGINE_DEFAULT_SCHEMA_VERSION,
    readiness_threshold_pct: float | None = None,
    parameters: SetupEngineParameters = DEFAULT_SETUP_ENGINE_PARAMETERS,
    data_policy_result: SetupEngineDataPolicyResult | None = None,
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
            bb_width_pctile_252 = None
            volume_vs_50d = None
            candidates = []
            rs_line_new_high = False

    derived_ready = (
        (normalized_readiness_score is not None)
        and (normalized_readiness_score >= threshold_pct)
        and (len(normalized_failed) == 0)
    )

    explain: SetupEngineExplain = {
        "passed_checks": normalized_passed,
        "failed_checks": normalized_failed,
        "key_levels": _normalize_key_levels(key_levels),
        "invalidation_flags": normalized_flags,
    }

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
        "distance_to_pivot_pct": _to_float(distance_to_pivot_pct),
        "atr14_pct": _to_float(atr14_pct),
        "bb_width_pctile_252": _to_float(bb_width_pctile_252),
        "volume_vs_50d": _to_float(volume_vs_50d),
        "rs_line_new_high": bool(rs_line_new_high),
        "candidates": _normalize_candidates(
            candidates,
            default_timeframe=timeframe,
        ),
        "explain": explain,
    }

    assert_valid_setup_engine_payload(payload)
    return payload


def attach_setup_engine(
    details: Mapping[str, Any] | None,
    setup_engine: SetupEnginePayload,
) -> dict[str, Any]:
    """Return a details payload with ``setup_engine`` attached at top level."""
    merged = dict(details or {})
    merged["setup_engine"] = setup_engine
    return merged
