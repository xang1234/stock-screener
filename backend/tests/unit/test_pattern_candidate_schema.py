"""Tests for shared PatternCandidate schema and normalization."""

import pytest

from app.analysis.patterns.models import (
    PatternCandidateModel,
    coerce_pattern_candidate,
    validate_pattern_candidate,
)


def test_candidate_model_accepts_confidence_ratio_and_derives_pct():
    candidate = PatternCandidateModel(
        pattern="vcp",
        timeframe="daily",
        confidence=0.72,
        setup_score=80.0,
        quality_score=75.0,
        readiness_score=70.0,
        metrics={"tight_band_pct": 2.4},
        checks={"volume_dry_up": True},
        notes=("healthy_contraction",),
    )

    payload = candidate.to_payload()
    assert payload["confidence"] == pytest.approx(0.72)
    assert payload["confidence_pct"] == pytest.approx(72.0)
    assert payload["metrics"]["tight_band_pct"] == pytest.approx(2.4)
    assert payload["checks"]["volume_dry_up"] is True


def test_candidate_from_mapping_supports_confidence_pct_alias():
    payload = coerce_pattern_candidate(
        {
            "pattern": "three_weeks_tight",
            "confidence_pct": 65.0,
            "timeframe": "weekly",
        },
        default_timeframe="daily",
    )

    assert payload["timeframe"] == "weekly"
    assert payload["confidence"] == pytest.approx(0.65)
    assert payload["confidence_pct"] == pytest.approx(65.0)


def test_candidate_validation_rejects_out_of_range_confidence():
    errors = validate_pattern_candidate(
        {
            "pattern": "vcp",
            "confidence": 1.2,
            "timeframe": "daily",
        }
    )
    assert any("confidence" in err for err in errors)


def test_candidate_validation_rejects_non_snake_case_metric_key():
    errors = validate_pattern_candidate(
        {
            "pattern": "vcp",
            "timeframe": "daily",
            "metrics": {"tightBand": 1.0},
        }
    )
    assert any("metrics key" in err for err in errors)


def test_candidate_validation_rejects_score_outside_0_100():
    errors = validate_pattern_candidate(
        {
            "pattern": "vcp",
            "timeframe": "daily",
            "setup_score": 120,
        }
    )
    assert any("setup_score" in err for err in errors)


def test_candidate_dates_can_be_inferred_from_metrics_aliases():
    payload = coerce_pattern_candidate(
        {
            "pattern": "three_weeks_tight",
            "timeframe": "weekly",
            "metrics": {
                "run_start_date": "2026-01-01",
                "run_end_date": "2026-01-29",
            },
        }
    )

    assert payload["start_date"] == "2026-01-01"
    assert payload["end_date"] == "2026-01-29"
