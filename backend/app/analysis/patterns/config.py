"""Setup Engine parameter governance.

This module externalizes threshold defaults, bounds, and rationale notes so
runtime behavior can be tuned without hidden code changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from app.analysis.patterns.models import SETUP_ENGINE_NUMERIC_UNITS


@dataclass(frozen=True)
class SetupEngineParameterSpec:
    """Metadata for one configurable threshold."""

    name: str
    default_value: float
    min_value: float
    max_value: float
    unit: str
    profile: str
    rationale: str


@dataclass(frozen=True)
class SetupEngineParameters:
    """Calibratable Setup Engine thresholds and guardrails."""

    # Three-weeks-tight contraction bands (% drawdown from pivot).
    three_weeks_tight_max_contraction_pct_strict: float = 1.0
    three_weeks_tight_max_contraction_pct_relaxed: float = 1.5

    # Early breakout zone window around pivot (% distance).
    early_zone_distance_to_pivot_pct_min: float = -2.0
    early_zone_distance_to_pivot_pct_max: float = 3.0

    # Volatility squeeze guardrails (Bollinger-width percentile).
    squeeze_bb_width_pctile_max_strict: float = 20.0
    squeeze_bb_width_pctile_max_relaxed: float = 35.0

    # Readiness and quality gates.
    readiness_score_ready_min_pct: float = 70.0
    quality_score_min_pct: float = 60.0
    pattern_confidence_min_pct: float = 55.0

    # Supplemental readiness controls.
    atr14_pct_max_for_ready: float = 8.0
    volume_vs_50d_min_for_ready: float = 1.0


SETUP_ENGINE_PARAMETER_SPECS: tuple[SetupEngineParameterSpec, ...] = (
    SetupEngineParameterSpec(
        name="three_weeks_tight_max_contraction_pct_strict",
        default_value=1.0,
        min_value=0.2,
        max_value=5.0,
        unit="pct",
        profile="strict",
        rationale="Strict 3WT mode should only allow very tight contractions.",
    ),
    SetupEngineParameterSpec(
        name="three_weeks_tight_max_contraction_pct_relaxed",
        default_value=1.5,
        min_value=0.5,
        max_value=8.0,
        unit="pct",
        profile="relaxed",
        rationale="Relaxed 3WT mode supports noisy leaders while preserving shape quality.",
    ),
    SetupEngineParameterSpec(
        name="early_zone_distance_to_pivot_pct_min",
        default_value=-2.0,
        min_value=-15.0,
        max_value=5.0,
        unit="pct",
        profile="baseline",
        rationale="Allows pre-breakout setups slightly below pivot without deep failure states.",
    ),
    SetupEngineParameterSpec(
        name="early_zone_distance_to_pivot_pct_max",
        default_value=3.0,
        min_value=-5.0,
        max_value=20.0,
        unit="pct",
        profile="baseline",
        rationale="Prevents late-chase entries from being labeled as early-zone ready.",
    ),
    SetupEngineParameterSpec(
        name="squeeze_bb_width_pctile_max_strict",
        default_value=20.0,
        min_value=1.0,
        max_value=50.0,
        unit="pct",
        profile="strict",
        rationale="Strict squeeze requires historically compressed volatility.",
    ),
    SetupEngineParameterSpec(
        name="squeeze_bb_width_pctile_max_relaxed",
        default_value=35.0,
        min_value=5.0,
        max_value=80.0,
        unit="pct",
        profile="relaxed",
        rationale="Relaxed squeeze tolerates broader consolidations in liquid names.",
    ),
    SetupEngineParameterSpec(
        name="readiness_score_ready_min_pct",
        default_value=70.0,
        min_value=40.0,
        max_value=95.0,
        unit="pct",
        profile="baseline",
        rationale="70% balances signal quality and setup throughput for v1 rollout.",
    ),
    SetupEngineParameterSpec(
        name="quality_score_min_pct",
        default_value=60.0,
        min_value=30.0,
        max_value=90.0,
        unit="pct",
        profile="baseline",
        rationale="Avoids classifying low-quality bases as tradeable setups.",
    ),
    SetupEngineParameterSpec(
        name="pattern_confidence_min_pct",
        default_value=55.0,
        min_value=20.0,
        max_value=95.0,
        unit="pct",
        profile="baseline",
        rationale="Ensures primary pattern selection is not dominated by weak detector noise.",
    ),
    SetupEngineParameterSpec(
        name="atr14_pct_max_for_ready",
        default_value=8.0,
        min_value=1.0,
        max_value=30.0,
        unit="pct",
        profile="baseline",
        rationale="Caps volatility to avoid breakout-ready flags in disorderly conditions.",
    ),
    SetupEngineParameterSpec(
        name="volume_vs_50d_min_for_ready",
        default_value=1.0,
        min_value=0.2,
        max_value=5.0,
        unit="ratio",
        profile="baseline",
        rationale="Requires at least baseline liquidity participation at decision time.",
    ),
)

DEFAULT_SETUP_ENGINE_PARAMETERS = SetupEngineParameters()


def validate_setup_engine_parameters(params: SetupEngineParameters) -> list[str]:
    """Return validation errors for parameter bounds and contradictions."""
    errors: list[str] = []

    values = asdict(params)
    spec_by_name = {spec.name: spec for spec in SETUP_ENGINE_PARAMETER_SPECS}

    for spec in SETUP_ENGINE_PARAMETER_SPECS:
        if spec.unit not in SETUP_ENGINE_NUMERIC_UNITS:
            errors.append(
                f"{spec.name} uses unsupported unit '{spec.unit}'"
            )

        value = values[spec.name]
        if value < spec.min_value or value > spec.max_value:
            errors.append(
                f"{spec.name}={value} outside [{spec.min_value}, {spec.max_value}]"
            )

    # Guardrails for contradictory runtime configs.
    if (
        params.three_weeks_tight_max_contraction_pct_strict
        > params.three_weeks_tight_max_contraction_pct_relaxed
    ):
        errors.append(
            "three_weeks_tight_max_contraction_pct_strict must be <= relaxed"
        )

    if params.early_zone_distance_to_pivot_pct_min > params.early_zone_distance_to_pivot_pct_max:
        errors.append(
            "early_zone_distance_to_pivot_pct_min must be <= early_zone_distance_to_pivot_pct_max"
        )

    if params.squeeze_bb_width_pctile_max_strict > params.squeeze_bb_width_pctile_max_relaxed:
        errors.append(
            "squeeze_bb_width_pctile_max_strict must be <= relaxed"
        )

    if params.quality_score_min_pct > params.readiness_score_ready_min_pct:
        errors.append(
            "quality_score_min_pct must be <= readiness_score_ready_min_pct"
        )

    # Defensive check for accidental schema drift.
    unknown_keys = sorted(set(values.keys()) - set(spec_by_name.keys()))
    if unknown_keys:
        errors.append(f"No spec metadata for parameters: {', '.join(unknown_keys)}")

    return errors


def assert_valid_setup_engine_parameters(params: SetupEngineParameters) -> None:
    """Raise ValueError if parameter values are out-of-policy."""
    errors = validate_setup_engine_parameters(params)
    if errors:
        raise ValueError("; ".join(errors))


def build_setup_engine_parameters(
    overrides: Mapping[str, int | float] | None = None,
) -> SetupEngineParameters:
    """Build validated parameters from defaults plus numeric overrides."""
    values = asdict(DEFAULT_SETUP_ENGINE_PARAMETERS)

    if overrides:
        for key, value in overrides.items():
            if key not in values:
                raise KeyError(f"Unknown setup_engine parameter: {key}")
            if isinstance(value, bool):
                raise ValueError(f"Parameter '{key}' expects a numeric value")
            values[key] = float(value)

    params = SetupEngineParameters(**values)
    assert_valid_setup_engine_parameters(params)
    return params
