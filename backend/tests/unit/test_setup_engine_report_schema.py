"""Tests for SetupEngine report typed schemas and serialization guards."""

import pytest

from app.analysis.patterns.models import PatternCandidateModel
from app.analysis.patterns.report import (
    ExplainPayload,
    InvalidationFlag,
    KeyLevels,
    SetupEngineReport,
    assert_valid_setup_engine_report_payload,
    canonical_setup_engine_report_examples,
    validate_setup_engine_report_payload,
)


def test_setup_engine_report_to_payload_validates_successfully():
    report = SetupEngineReport(
        timeframe="daily",
        setup_ready=False,
        setup_score=72.0,
        quality_score=66.0,
        readiness_score=58.0,
        pattern_primary="vcp",
        pattern_confidence_pct=61.0,
        pivot_price=101.2,
        pivot_type="breakout",
        pivot_date="2026-02-13",
        candidates=(
            PatternCandidateModel(
                pattern="vcp",
                timeframe="daily",
                confidence=0.61,
                setup_score=72.0,
                quality_score=66.0,
                readiness_score=58.0,
                metrics={"contractions_count": 3},
                checks={"volume_dry_up": True},
            ),
        ),
        explain=ExplainPayload(
            passed_checks=("volume_dry_up",),
            failed_checks=("breakout_volume_unconfirmed",),
            key_levels=KeyLevels(levels={"pivot_price": 101.2}),
            invalidation_flags=(InvalidationFlag("breakout_volume_unconfirmed"),),
        ),
    )

    payload = report.to_payload()
    assert payload["pattern_confidence"] == pytest.approx(61.0)
    assert payload["candidates"][0]["confidence"] == pytest.approx(0.61)
    assert_valid_setup_engine_report_payload(payload)


def test_invalidation_flag_requires_snake_case_code():
    with pytest.raises(ValueError, match="snake_case"):
        InvalidationFlag("BadCode")


def test_report_validator_rejects_non_json_types():
    payload = canonical_setup_engine_report_examples()[0]
    payload["candidates"][0]["metrics"]["bad_value"] = {"nested"}

    errors = validate_setup_engine_report_payload(payload)
    assert any("non-JSON-serializable" in err for err in errors)


def test_report_validator_checks_confidence_consistency():
    payload = canonical_setup_engine_report_examples()[0]
    payload["candidates"][0]["confidence"] = 0.6
    payload["candidates"][0]["confidence_pct"] = 70.0

    errors = validate_setup_engine_report_payload(payload)
    assert any("inconsistent" in err for err in errors)


def test_canonical_examples_are_valid():
    for payload in canonical_setup_engine_report_examples():
        assert validate_setup_engine_report_payload(payload) == []
