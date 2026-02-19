"""Typed setup_engine report schemas and serialization guards.

Producer boundary: scanner payload assembly.
Persistence boundary: repository writes.
Query boundary: JSON payload consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Mapping, Sequence

from app.analysis.patterns.models import (
    SETUP_ENGINE_DEFAULT_SCHEMA_VERSION,
    SETUP_ENGINE_FIELD_SPECS,
    SETUP_ENGINE_NUMERIC_UNITS,
    PatternCandidate,
    PatternCandidateModel,
    SetupEngineExplain,
    SetupEnginePayload,
    coerce_pattern_candidate,
    is_snake_case,
    normalize_iso_date,
    validate_setup_engine_payload,
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_pct(name: str, value: float | None) -> None:
    if value is None:
        return
    if value < 0.0 or value > 100.0:
        raise ValueError(f"{name} must be in [0, 100]")


@dataclass(frozen=True)
class InvalidationFlag:
    """Typed invalidation flag emitted by explain payload."""

    code: str
    detail: str | None = None
    is_hard: bool = True

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("InvalidationFlag.code is required")
        if not is_snake_case(self.code):
            raise ValueError("InvalidationFlag.code must be snake_case")

    def to_payload(self) -> str:
        if self.detail:
            return f"{self.code}:{self.detail}"
        return self.code


@dataclass(frozen=True)
class KeyLevels:
    """Typed key-levels container used by explain payload."""

    levels: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.levels.items():
            if not is_snake_case(str(key)):
                raise ValueError(f"key_levels key must be snake_case: {key}")
            if value is not None and not _is_number(value):
                raise ValueError(f"key_levels[{key}] must be numeric or null")

    def to_payload(self) -> dict[str, float | None]:
        return {str(key): (float(value) if value is not None else None) for key, value in self.levels.items()}


@dataclass(frozen=True)
class ExplainPayload:
    """Typed explain payload schema."""

    passed_checks: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    key_levels: KeyLevels = field(default_factory=KeyLevels)
    invalidation_flags: tuple[InvalidationFlag | str, ...] = ()

    def __post_init__(self) -> None:
        for check in (*self.passed_checks, *self.failed_checks):
            if check and not is_snake_case(check):
                raise ValueError(f"check key must be snake_case: {check}")

    def to_payload(self) -> SetupEngineExplain:
        flags: list[str] = []
        for flag in self.invalidation_flags:
            if isinstance(flag, InvalidationFlag):
                flags.append(flag.to_payload())
            else:
                if not flag:
                    continue
                flags.append(str(flag))

        return SetupEngineExplain(
            passed_checks=[c for c in self.passed_checks if c],
            failed_checks=[c for c in self.failed_checks if c],
            key_levels=self.key_levels.to_payload(),
            invalidation_flags=flags,
        )


@dataclass(frozen=True)
class SetupEngineReport:
    """Typed container for final setup_engine report serialization."""

    timeframe: Literal["daily", "weekly"]
    setup_ready: bool

    setup_score: float | None = None
    quality_score: float | None = None
    readiness_score: float | None = None

    pattern_primary: str | None = None
    pattern_confidence_pct: float | None = None
    pivot_price: float | None = None
    pivot_type: str | None = None
    pivot_date: str | date | datetime | None = None

    distance_to_pivot_pct: float | None = None
    atr14_pct: float | None = None
    bb_width_pctile_252: float | None = None
    volume_vs_50d: float | None = None
    rs_line_new_high: bool = False

    candidates: tuple[PatternCandidateModel | Mapping[str, Any], ...] = ()
    explain: ExplainPayload = field(default_factory=ExplainPayload)
    schema_version: str = SETUP_ENGINE_DEFAULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_pct("setup_score", self.setup_score)
        _validate_pct("quality_score", self.quality_score)
        _validate_pct("readiness_score", self.readiness_score)
        _validate_pct("pattern_confidence_pct", self.pattern_confidence_pct)

    def to_payload(self) -> SetupEnginePayload:
        normalized_candidates: list[PatternCandidate] = [
            coerce_pattern_candidate(c, default_timeframe=self.timeframe)
            for c in self.candidates
        ]

        payload: SetupEnginePayload = {
            "schema_version": self.schema_version,
            "timeframe": self.timeframe,
            "setup_score": _as_float(self.setup_score),
            "quality_score": _as_float(self.quality_score),
            "readiness_score": _as_float(self.readiness_score),
            "setup_ready": bool(self.setup_ready),
            "pattern_primary": self.pattern_primary,
            "pattern_confidence": _as_float(self.pattern_confidence_pct),
            "pivot_price": _as_float(self.pivot_price),
            "pivot_type": self.pivot_type,
            "pivot_date": normalize_iso_date(self.pivot_date),
            "distance_to_pivot_pct": _as_float(self.distance_to_pivot_pct),
            "atr14_pct": _as_float(self.atr14_pct),
            "bb_width_pctile_252": _as_float(self.bb_width_pctile_252),
            "volume_vs_50d": _as_float(self.volume_vs_50d),
            "rs_line_new_high": bool(self.rs_line_new_high),
            "candidates": normalized_candidates,
            "explain": self.explain.to_payload(),
        }

        assert_valid_setup_engine_report_payload(payload)
        return payload


def _as_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _validate_json_primitives(value: Any, path: str = "$", errors: list[str] | None = None) -> list[str]:
    if errors is None:
        errors = []

    if value is None or isinstance(value, (str, int, float, bool)):
        return errors

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(f"{path}: JSON object keys must be strings")
                continue
            _validate_json_primitives(item, f"{path}.{key}", errors)
        return errors

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, item in enumerate(value):
            _validate_json_primitives(item, f"{path}[{idx}]", errors)
        return errors

    errors.append(f"{path}: non-JSON-serializable type {type(value)!r}")
    return errors


def validate_setup_engine_report_payload(payload: Mapping[str, Any]) -> list[str]:
    """Validate payload strictness for producer/persistence/query boundaries."""
    errors = list(validate_setup_engine_payload(payload))
    errors.extend(_validate_json_primitives(payload))

    # Numeric fields with units must be numeric/null at boundary.
    for spec in SETUP_ENGINE_FIELD_SPECS:
        if spec.unit is None:
            continue
        if spec.unit not in SETUP_ENGINE_NUMERIC_UNITS:
            errors.append(f"Unsupported unit in schema spec: {spec.name}={spec.unit}")
            continue

        if "." in spec.name:
            continue  # nested structures handled in base validator

        value = payload.get(spec.name)
        if value is not None and not _is_number(value):
            errors.append(f"{spec.name} must be numeric or null")

    # Candidate confidence ratio/pct consistency checks.
    candidates = payload.get("candidates")
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes, bytearray)):
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                continue
            ratio = candidate.get("confidence")
            pct = candidate.get("confidence_pct")
            if ratio is not None:
                if not _is_number(ratio) or float(ratio) < 0.0 or float(ratio) > 1.0:
                    errors.append(f"candidates[{idx}].confidence must be in [0,1]")
            if pct is not None:
                if not _is_number(pct) or float(pct) < 0.0 or float(pct) > 100.0:
                    errors.append(f"candidates[{idx}].confidence_pct must be in [0,100]")
            if ratio is not None and pct is not None and _is_number(ratio) and _is_number(pct):
                if abs(float(ratio) * 100.0 - float(pct)) > 1e-6:
                    errors.append(f"candidates[{idx}] confidence and confidence_pct are inconsistent")

    return errors


def assert_valid_setup_engine_report_payload(payload: Mapping[str, Any]) -> None:
    """Raise ValueError if report payload fails strict boundary validation."""
    errors = validate_setup_engine_report_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))


def canonical_setup_engine_report_examples() -> tuple[SetupEnginePayload, ...]:
    """Return canonical valid examples used for regression tests."""
    example = SetupEngineReport(
        timeframe="daily",
        setup_ready=False,
        setup_score=74.0,
        quality_score=68.0,
        readiness_score=59.0,
        pattern_primary="three_weeks_tight",
        pattern_confidence_pct=63.0,
        pivot_price=101.25,
        pivot_type="breakout",
        pivot_date="2026-02-13",
        distance_to_pivot_pct=-1.3,
        atr14_pct=3.1,
        bb_width_pctile_252=28.0,
        volume_vs_50d=1.04,
        rs_line_new_high=False,
        candidates=(
            PatternCandidateModel(
                pattern="three_weeks_tight",
                timeframe="daily",
                confidence=0.63,
                setup_score=74.0,
                quality_score=68.0,
                readiness_score=59.0,
                metrics={"weeks_tight": 3, "tight_band_pct": 1.4},
                checks={"tight_band_ok": True},
                notes=("strict_mode",),
            ),
        ),
        explain=ExplainPayload(
            passed_checks=("tight_band_ok",),
            failed_checks=("breakout_volume_unconfirmed",),
            key_levels=KeyLevels(levels={"pivot_price": 101.25}),
            invalidation_flags=(
                InvalidationFlag("breakout_volume_unconfirmed", is_hard=False),
            ),
        ),
    ).to_payload()

    return (example,)
