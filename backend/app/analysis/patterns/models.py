"""Canonical Setup Engine schema contract and validation helpers.

This module is the single source of truth for the ``setup_engine`` payload:
field names, types, units, nullability, and naming policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import re
from typing import Any, Literal, Mapping, Sequence, TypedDict, cast


SETUP_ENGINE_DEFAULT_SCHEMA_VERSION = "v1"
SETUP_ENGINE_ALLOWED_TIMEFRAMES = frozenset({"daily", "weekly"})
SETUP_ENGINE_NUMERIC_UNITS = frozenset({"pct", "ratio", "days", "weeks", "price", "usd"})

PATTERN_SCORE_MIN = 0.0
PATTERN_SCORE_MAX = 100.0
PATTERN_CONFIDENCE_MIN = 0.0
PATTERN_CONFIDENCE_MAX = 1.0

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


JsonScalar = str | int | float | bool | None


class FieldTrace(TypedDict):
    """One field's formula, raw inputs, computed output, and unit."""

    formula: str
    inputs: dict[str, JsonScalar]
    output: JsonScalar
    unit: str


ScoreTrace = dict[str, FieldTrace]


class PatternCandidate(TypedDict, total=False):
    """Canonical candidate shape emitted by all detectors.

    Conventions:
    - ``setup_score``/``quality_score``/``readiness_score`` are 0..100.
    - ``confidence`` is 0..1; ``confidence_pct`` is the derived 0..100 alias.
    """

    pattern: str
    timeframe: Literal["daily", "weekly"]
    source_detector: str | None

    pivot_price: float | None
    pivot_type: str | None
    pivot_date: str | None
    start_date: str | None
    end_date: str | None

    distance_to_pivot_pct: float | None
    setup_score: float | None
    quality_score: float | None
    readiness_score: float | None

    confidence: float | None
    confidence_pct: float | None

    metrics: dict[str, JsonScalar]
    checks: dict[str, bool]
    notes: list[str]


class InvalidationFlagPayload(TypedDict):
    """Structured invalidation flag for explain payload."""

    code: str
    message: str
    severity: Literal["low", "medium", "high"]


class SetupEngineExplain(TypedDict):
    """Human-readable checks and key levels used by setup_engine."""

    passed_checks: list[str]
    failed_checks: list[str]
    key_levels: dict[str, float | None]
    invalidation_flags: list[InvalidationFlagPayload]


class SetupEnginePayload(TypedDict):
    """Top-level payload stored under ``details.setup_engine``."""

    schema_version: str
    timeframe: Literal["daily", "weekly"]

    setup_score: float | None
    quality_score: float | None
    readiness_score: float | None
    setup_ready: bool

    pattern_primary: str | None
    pattern_confidence: float | None
    pivot_price: float | None
    pivot_type: str | None
    pivot_date: str | None

    distance_to_pivot_pct: float | None
    in_early_zone: bool | None
    extended_from_pivot: bool | None
    base_length_weeks: float | None
    base_depth_pct: float | None
    support_tests_count: int | None
    tight_closes_count: int | None
    atr14_pct: float | None
    atr14_pct_trend: float | None
    bb_width_pct: float | None
    bb_width_pctile_252: float | None
    bb_squeeze: bool | None
    volume_vs_50d: float | None
    up_down_volume_ratio_10d: float | None
    quiet_days_10d: int | None
    rs: float | None
    rs_line_new_high: bool
    rs_vs_spy_65d: float | None
    rs_vs_spy_trend_20d: float | None

    stage: int | None
    ma_alignment_score: float | None
    rs_rating: float | None

    candidates: list[PatternCandidate]
    explain: SetupEngineExplain


@dataclass(frozen=True)
class PatternCandidateModel:
    """Typed model for detector outputs before payload serialization."""

    pattern: str
    timeframe: Literal["daily", "weekly"]
    source_detector: str | None = None

    pivot_price: float | None = None
    pivot_type: str | None = None
    pivot_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    distance_to_pivot_pct: float | None = None
    setup_score: float | None = None
    quality_score: float | None = None
    readiness_score: float | None = None

    confidence: float | None = None

    metrics: dict[str, JsonScalar] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("pattern is required")
        if self.timeframe not in SETUP_ENGINE_ALLOWED_TIMEFRAMES:
            raise ValueError("timeframe must be daily or weekly")

        for key in self.metrics.keys():
            if not is_snake_case(key):
                raise ValueError(f"metrics key must be snake_case: {key}")
        for key, value in self.checks.items():
            if not is_snake_case(key):
                raise ValueError(f"checks key must be snake_case: {key}")
            if not isinstance(value, bool):
                raise ValueError(f"checks[{key}] must be bool")

        _validate_score("setup_score", self.setup_score)
        _validate_score("quality_score", self.quality_score)
        _validate_score("readiness_score", self.readiness_score)
        _validate_confidence(self.confidence)

        if self.pivot_date is not None:
            normalize_iso_date(self.pivot_date)
        if self.start_date is not None:
            normalize_iso_date(self.start_date)
        if self.end_date is not None:
            normalize_iso_date(self.end_date)

    @property
    def confidence_pct(self) -> float | None:
        if self.confidence is None:
            return None
        return self.confidence * 100.0

    def to_payload(self) -> PatternCandidate:
        """Serialize to the canonical payload candidate shape."""
        return PatternCandidate(
            pattern=self.pattern,
            timeframe=self.timeframe,
            source_detector=self.source_detector,
            pivot_price=self.pivot_price,
            pivot_type=self.pivot_type,
            pivot_date=self.pivot_date,
            start_date=self.start_date,
            end_date=self.end_date,
            distance_to_pivot_pct=self.distance_to_pivot_pct,
            setup_score=self.setup_score,
            quality_score=self.quality_score,
            readiness_score=self.readiness_score,
            confidence=self.confidence,
            confidence_pct=self.confidence_pct,
            metrics=dict(self.metrics),
            checks=dict(self.checks),
            notes=list(self.notes),
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        default_timeframe: str = "daily",
    ) -> PatternCandidateModel:
        """Create a validated model from a mapping with legacy aliases."""
        timeframe = cast(str | None, raw.get("timeframe")) or default_timeframe

        confidence_raw = raw.get("confidence")
        if confidence_raw is None and raw.get("confidence_pct") is not None:
            confidence_raw = float(raw["confidence_pct"]) / 100.0

        metrics_raw = raw.get("metrics")
        metrics_mapping: Mapping[str, Any] = (
            metrics_raw if isinstance(metrics_raw, Mapping) else {}
        )
        start_date_raw = (
            raw.get("start_date")
            or metrics_mapping.get("run_start_date")
            or metrics_mapping.get("handle_start_date")
            or metrics_mapping.get("flag_start_date")
            or metrics_mapping.get("pole_start_date")
            or metrics_mapping.get("left_lip_date")
            or metrics_mapping.get("pullback_high_date")
        )
        end_date_raw = (
            raw.get("end_date")
            or metrics_mapping.get("run_end_date")
            or metrics_mapping.get("handle_end_date")
            or metrics_mapping.get("flag_end_date")
            or metrics_mapping.get("pole_end_date")
            or metrics_mapping.get("right_lip_date")
            or metrics_mapping.get("resumption_high_date")
        )

        return cls(
            pattern=cast(str, raw.get("pattern") or "unknown"),
            timeframe=cast(Any, timeframe),
            source_detector=cast(str | None, raw.get("source_detector")),
            pivot_price=_as_float(raw.get("pivot_price")),
            pivot_type=cast(str | None, raw.get("pivot_type")),
            pivot_date=normalize_iso_date(
                cast(str | date | datetime | None, raw.get("pivot_date"))
            ),
            start_date=normalize_iso_date(
                cast(str | date | datetime | None, start_date_raw)
            ),
            end_date=normalize_iso_date(
                cast(str | date | datetime | None, end_date_raw)
            ),
            distance_to_pivot_pct=_as_float(raw.get("distance_to_pivot_pct")),
            setup_score=_as_float(raw.get("setup_score")),
            quality_score=_as_float(raw.get("quality_score")),
            readiness_score=_as_float(raw.get("readiness_score")),
            confidence=_as_float(confidence_raw),
            metrics=_normalize_metrics(raw.get("metrics")),
            checks=_normalize_checks(raw.get("checks")),
            notes=tuple(_normalize_notes(raw.get("notes"))),
        )


@dataclass(frozen=True)
class SetupEngineFieldSpec:
    """Schema reference row for contract docs and review."""

    name: str
    type_name: str
    nullable: bool
    unit: str | None
    source_module: str
    description: str


SETUP_ENGINE_FIELD_SPECS: tuple[SetupEngineFieldSpec, ...] = (
    SetupEngineFieldSpec(
        name="schema_version",
        type_name="str",
        nullable=False,
        unit=None,
        source_module="backend/app/analysis/patterns/models.py",
        description="Schema version for compatibility gates.",
    ),
    SetupEngineFieldSpec(
        name="timeframe",
        type_name="Literal['daily','weekly']",
        nullable=False,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Data horizon used for pattern classification.",
    ),
    SetupEngineFieldSpec(
        name="setup_score",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Composite setup quality score (0.60*quality + 0.40*readiness). Per-candidate setup_score uses a different blend including confidence.",
    ),
    SetupEngineFieldSpec(
        name="quality_score",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Pattern quality score on 0..100 scale.",
    ),
    SetupEngineFieldSpec(
        name="readiness_score",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Breakout readiness score on 0..100 scale.",
    ),
    SetupEngineFieldSpec(
        name="setup_ready",
        type_name="bool",
        nullable=False,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="True when readiness threshold is met and no failed checks remain.",
    ),
    SetupEngineFieldSpec(
        name="pattern_primary",
        type_name="str",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Primary detected pattern label.",
    ),
    SetupEngineFieldSpec(
        name="pattern_confidence",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Confidence in pattern_primary on 0..100 scale.",
    ),
    SetupEngineFieldSpec(
        name="pivot_price",
        type_name="float",
        nullable=True,
        unit="price",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Canonical pivot price for the primary pattern.",
    ),
    SetupEngineFieldSpec(
        name="pivot_type",
        type_name="str",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Pivot family (breakout, pullback, reclaim, etc).",
    ),
    SetupEngineFieldSpec(
        name="pivot_date",
        type_name="str(YYYY-MM-DD)",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Date associated with pivot_price.",
    ),
    SetupEngineFieldSpec(
        name="distance_to_pivot_pct",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Percent distance from current price to pivot.",
    ),
    SetupEngineFieldSpec(
        name="in_early_zone",
        type_name="bool",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="True when distance_to_pivot_pct is within the configured early zone window.",
    ),
    SetupEngineFieldSpec(
        name="extended_from_pivot",
        type_name="bool",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="True when distance_to_pivot_pct is beyond extended-entry threshold.",
    ),
    SetupEngineFieldSpec(
        name="base_length_weeks",
        type_name="float",
        nullable=True,
        unit="weeks",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Estimated base duration in weeks for the selected primary pattern.",
    ),
    SetupEngineFieldSpec(
        name="base_depth_pct",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Estimated base depth percent for the selected primary pattern.",
    ),
    SetupEngineFieldSpec(
        name="support_tests_count",
        type_name="int",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Estimated count of support tests in the detected base.",
    ),
    SetupEngineFieldSpec(
        name="tight_closes_count",
        type_name="int",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Estimated count of tight closes within the detected base.",
    ),
    SetupEngineFieldSpec(
        name="atr14_pct",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="ATR(14) as a percentage of current price.",
    ),
    SetupEngineFieldSpec(
        name="atr14_pct_trend",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="20-bar slope of ATR14 percent.",
    ),
    SetupEngineFieldSpec(
        name="bb_width_pct",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Current Bollinger width percent ((upper-lower)/middle*100).",
    ),
    SetupEngineFieldSpec(
        name="bb_width_pctile_252",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="252-session Bollinger width percentile on 0..100.",
    ),
    SetupEngineFieldSpec(
        name="bb_squeeze",
        type_name="bool",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="True when Bollinger width percentile is in configured squeeze zone.",
    ),
    SetupEngineFieldSpec(
        name="volume_vs_50d",
        type_name="float",
        nullable=True,
        unit="ratio",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Volume / 50-day average volume ratio.",
    ),
    SetupEngineFieldSpec(
        name="up_down_volume_ratio_10d",
        type_name="float",
        nullable=True,
        unit="ratio",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="10-session up-volume to down-volume ratio.",
    ),
    SetupEngineFieldSpec(
        name="quiet_days_10d",
        type_name="float",
        nullable=True,
        unit="days",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Count of quiet consolidation sessions in the last 10 bars.",
    ),
    SetupEngineFieldSpec(
        name="rs",
        type_name="float",
        nullable=True,
        unit="ratio",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Relative-strength line ratio (close / benchmark close).",
    ),
    SetupEngineFieldSpec(
        name="rs_line_new_high",
        type_name="bool",
        nullable=False,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="True when RS line has made a new high for timeframe.",
    ),
    SetupEngineFieldSpec(
        name="rs_vs_spy_65d",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="65-session percent change in RS line.",
    ),
    SetupEngineFieldSpec(
        name="rs_vs_spy_trend_20d",
        type_name="float",
        nullable=True,
        unit="ratio",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="20-session slope of RS line.",
    ),
    SetupEngineFieldSpec(
        name="stage",
        type_name="int",
        nullable=True,
        unit=None,
        source_module="backend/app/scanners/setup_engine_screener.py",
        description="Weinstein stage 1-4 via quick_stage_check() MA structure analysis.",
    ),
    SetupEngineFieldSpec(
        name="ma_alignment_score",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_screener.py",
        description="Minervini MA alignment score (50>150>200) on 0..100 scale.",
    ),
    SetupEngineFieldSpec(
        name="rs_rating",
        type_name="float",
        nullable=True,
        unit="pct",
        source_module="backend/app/scanners/setup_engine_screener.py",
        description="Multi-period weighted RS rating on 0..100 linear scale (50 = outperforming SPY).",
    ),
    SetupEngineFieldSpec(
        name="candidates",
        type_name="list[PatternCandidate]",
        nullable=False,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Alternative pattern candidates for explainability.",
    ),
    SetupEngineFieldSpec(
        name="explain",
        type_name="SetupEngineExplain",
        nullable=False,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Structured pass/fail checks and key levels.",
    ),
    SetupEngineFieldSpec(
        name="explain.passed_checks",
        type_name="list[str]",
        nullable=False,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Checks passed by the setup.",
    ),
    SetupEngineFieldSpec(
        name="explain.failed_checks",
        type_name="list[str]",
        nullable=False,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Checks failed by the setup.",
    ),
    SetupEngineFieldSpec(
        name="explain.key_levels",
        type_name="dict[str, float|None]",
        nullable=False,
        unit="price",
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Named price levels (pivot, support, invalidation, etc).",
    ),
    SetupEngineFieldSpec(
        name="explain.invalidation_flags",
        type_name="list[InvalidationFlagPayload]",
        nullable=False,
        unit=None,
        source_module="backend/app/scanners/setup_engine_scanner.py",
        description="Reasons this setup should not be acted on.",
    ),
    SetupEngineFieldSpec(
        name="explain.score_trace",
        type_name="ScoreTrace",
        nullable=True,
        unit=None,
        source_module="backend/app/analysis/patterns/trace.py",
        description="Optional per-field calculation trace for auditability.",
    ),
)

SETUP_ENGINE_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "timeframe",
    "setup_score",
    "quality_score",
    "readiness_score",
    "setup_ready",
    "pattern_primary",
    "pattern_confidence",
    "pivot_price",
    "pivot_type",
    "pivot_date",
    "distance_to_pivot_pct",
    "in_early_zone",
    "extended_from_pivot",
    "base_length_weeks",
    "base_depth_pct",
    "support_tests_count",
    "tight_closes_count",
    "atr14_pct",
    "atr14_pct_trend",
    "bb_width_pct",
    "bb_width_pctile_252",
    "bb_squeeze",
    "volume_vs_50d",
    "up_down_volume_ratio_10d",
    "quiet_days_10d",
    "rs",
    "rs_line_new_high",
    "rs_vs_spy_65d",
    "rs_vs_spy_trend_20d",
    "stage",
    "ma_alignment_score",
    "rs_rating",
    "candidates",
    "explain",
)


def is_snake_case(name: str) -> bool:
    """Return True when *name* follows snake_case."""
    return bool(_SNAKE_CASE_RE.fullmatch(name))


def normalize_iso_date(value: str | date | datetime | None) -> str | None:
    """Normalize date-ish input to ``YYYY-MM-DD`` or return None."""
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    if not isinstance(value, str):
        raise ValueError(f"Date value must be str/date/datetime, got {type(value)!r}")

    if not _ISO_DATE_RE.fullmatch(value):
        raise ValueError("Date value must use YYYY-MM-DD format")

    # Verifies calendar validity (e.g. 2026-02-30 is invalid).
    date.fromisoformat(value)
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric fields")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected numeric value, got {type(value)!r}")


def _validate_score(name: str, value: float | None) -> None:
    if value is None:
        return
    if value < PATTERN_SCORE_MIN or value > PATTERN_SCORE_MAX:
        raise ValueError(
            f"{name} must be in [{PATTERN_SCORE_MIN}, {PATTERN_SCORE_MAX}]"
        )


def _validate_confidence(value: float | None) -> None:
    if value is None:
        return
    if value < PATTERN_CONFIDENCE_MIN or value > PATTERN_CONFIDENCE_MAX:
        raise ValueError(
            f"confidence must be in [{PATTERN_CONFIDENCE_MIN}, {PATTERN_CONFIDENCE_MAX}]"
        )


def _normalize_metrics(raw: Any) -> dict[str, JsonScalar]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("metrics must be an object")

    metrics: dict[str, JsonScalar] = {}
    for key, value in raw.items():
        key_s = str(key)
        if not is_snake_case(key_s):
            raise ValueError(f"metrics key must be snake_case: {key_s}")
        if not (
            value is None
            or isinstance(value, (str, int, float, bool))
        ):
            raise ValueError(f"metrics[{key_s}] must be JSON scalar")
        metrics[key_s] = cast(JsonScalar, value)
    return metrics


def _normalize_checks(raw: Any) -> dict[str, bool]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("checks must be an object")

    checks: dict[str, bool] = {}
    for key, value in raw.items():
        key_s = str(key)
        if not is_snake_case(key_s):
            raise ValueError(f"checks key must be snake_case: {key_s}")
        if not isinstance(value, bool):
            raise ValueError(f"checks[{key_s}] must be bool")
        checks[key_s] = value
    return checks


def _normalize_notes(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("notes must be a list")
    return [str(item) for item in raw if item is not None and str(item) != ""]


def validate_pattern_candidate(candidate: Mapping[str, Any]) -> list[str]:
    """Return validation errors for one candidate mapping."""
    errors: list[str] = []

    try:
        PatternCandidateModel.from_mapping(candidate)
    except ValueError as exc:
        errors.append(str(exc))

    return errors


def coerce_pattern_candidate(
    candidate: Mapping[str, Any] | PatternCandidateModel,
    *,
    default_timeframe: str = "daily",
) -> PatternCandidate:
    """Normalize candidate to canonical payload shape."""
    if isinstance(candidate, PatternCandidateModel):
        model = candidate
    else:
        model = PatternCandidateModel.from_mapping(
            candidate,
            default_timeframe=default_timeframe,
        )

    return model.to_payload()


def validate_setup_engine_payload(payload: Mapping[str, Any]) -> list[str]:
    """Validate contract compliance and return human-readable errors."""
    errors: list[str] = []

    for key in SETUP_ENGINE_REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"Missing required top-level key: {key}")

    for key in payload.keys():
        if not is_snake_case(key):
            errors.append(f"Top-level key is not snake_case: {key}")

    timeframe = payload.get("timeframe")
    if timeframe not in SETUP_ENGINE_ALLOWED_TIMEFRAMES:
        errors.append("timeframe must be one of: daily, weekly")

    pivot_date = payload.get("pivot_date")
    try:
        normalize_iso_date(cast(str | date | datetime | None, pivot_date))
    except ValueError as exc:
        errors.append(f"pivot_date invalid: {exc}")

    explain = payload.get("explain")
    if not isinstance(explain, Mapping):
        errors.append("explain must be an object")
    else:
        for required in ("passed_checks", "failed_checks", "key_levels", "invalidation_flags"):
            if required not in explain:
                errors.append(f"explain missing key: {required}")

        key_levels = explain.get("key_levels")
        if isinstance(key_levels, Mapping):
            for level_name, level_value in key_levels.items():
                if not is_snake_case(str(level_name)):
                    errors.append(
                        f"explain.key_levels key must be snake_case: {level_name}"
                    )
                if level_value is not None and not _is_number(level_value):
                    errors.append(
                        f"explain.key_levels[{level_name}] must be numeric or null"
                    )
        else:
            errors.append("explain.key_levels must be an object")

        invalidation_flags = explain.get("invalidation_flags")
        if not isinstance(invalidation_flags, list):
            errors.append("explain.invalidation_flags must be a list")
        else:
            valid_severities = {"low", "medium", "high"}
            for idx, flag in enumerate(invalidation_flags):
                if not isinstance(flag, Mapping):
                    errors.append(
                        f"explain.invalidation_flags[{idx}] must be an object"
                    )
                    continue
                code = flag.get("code")
                message = flag.get("message")
                severity = flag.get("severity")
                if not isinstance(code, str) or not is_snake_case(code):
                    errors.append(
                        f"explain.invalidation_flags[{idx}].code must be snake_case string"
                    )
                if not isinstance(message, str) or not message:
                    errors.append(
                        f"explain.invalidation_flags[{idx}].message must be non-empty string"
                    )
                if severity not in valid_severities:
                    errors.append(
                        f"explain.invalidation_flags[{idx}].severity must be one of: low, medium, high"
                    )

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        errors.append("candidates must be a list")
    else:
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                errors.append(f"candidates[{index}] must be an object")
                continue

            for key in candidate.keys():
                if not is_snake_case(str(key)):
                    errors.append(
                        f"candidates[{index}] key is not snake_case: {key}"
                    )

            candidate_errors = validate_pattern_candidate(candidate)
            errors.extend(
                [f"candidates[{index}] {msg}" for msg in candidate_errors]
            )

    return errors


def assert_valid_setup_engine_payload(payload: Mapping[str, Any]) -> None:
    """Raise ValueError if payload does not satisfy the contract."""
    errors = validate_setup_engine_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
