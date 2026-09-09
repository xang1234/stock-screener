from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.domain.social_signals.records import (
    ComponentScore,
    SocialPostRecord,
    SocialReadRequest,
    SocialSnapshotRecord,
    SocialSourceOutcome,
    SocialSourceBatch,
    SourceTestOutcome,
    validate_utc_timestamp,
)


NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)


def _post(**changes):
    values = {
        "provider": "xui",
        "provider_post_id": "101",
        "source_id": "1522014550211457024",
        "text": "$NVDA base",
        "url": "https://x.com/a/status/101",
        "author_handle": "a",
        "created_at": NOW - timedelta(days=1),
        "observed_at": NOW,
        "likes": 0,
        "reposts": 0,
        "replies": 0,
        "quotes": None,
        "bookmarks": None,
        "views": None,
    }
    values.update(changes)
    return SocialPostRecord(**values)


def test_missing_metric_is_not_observed_zero():
    row = _post()

    assert row.likes == 0
    assert row.views is None


def test_future_post_is_rejected():
    with pytest.raises(ValueError, match="future_timestamp"):
        SocialPostRecord.from_untrusted(
            {"tweet_id": "101", "created_at": "2099-01-01T00:00:00Z"},
            provider="xui",
            source_id="1522014550211457024",
            observed_at=NOW,
        )


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"provider": "mystery"}, "unsupported_provider"),
        ({"provider_post_id": " "}, "blank_provider_post_id"),
        ({"source_id": ""}, "blank_source_id"),
        ({"text": "\t"}, "blank_text"),
        ({"url": ""}, "blank_url"),
        ({"created_at": datetime(2026, 9, 5)}, "naive_timestamp"),
        ({"likes": -1}, "negative_metric"),
    ],
)
def test_post_rejects_invalid_boundary_values(changes, code):
    with pytest.raises(ValueError, match=code):
        _post(**changes)


def test_post_allows_five_minute_clock_skew_but_not_more():
    assert _post(created_at=NOW + timedelta(minutes=5)).created_at > NOW

    with pytest.raises(ValueError, match="future_timestamp"):
        _post(created_at=NOW + timedelta(minutes=5, seconds=1))


def test_aware_non_utc_timestamp_is_rejected():
    non_utc = NOW.astimezone(timezone(timedelta(hours=8)))

    with pytest.raises(ValueError, match="non_utc_timestamp"):
        _post(created_at=non_utc)


def test_ingress_timestamp_validator_uses_explicit_trusted_reference():
    assert validate_utc_timestamp(
        NOW + timedelta(minutes=5), "tested_at", reference_at=NOW
    ) == NOW + timedelta(minutes=5)

    with pytest.raises(ValueError, match="future_timestamp:tested_at"):
        validate_utc_timestamp(
            NOW + timedelta(minutes=5, seconds=1),
            "tested_at",
            reference_at=NOW,
        )


def test_untrusted_post_rejects_unknown_payload_fields():
    with pytest.raises(ValueError, match="unexpected_payload_fields"):
        SocialPostRecord.from_untrusted(
            {
                "tweet_id": "101",
                "created_at": "2026-09-05T00:00:00Z",
                "secret_cookie": "must-not-cross-boundary",
            },
            provider="xui",
            source_id="1522014550211457024",
            observed_at=NOW,
        )


def test_records_are_immutable():
    row = _post()

    with pytest.raises(FrozenInstanceError):
        row.text = "changed"


@pytest.mark.parametrize("sample_count", [-1, 6])
def test_source_test_sample_count_is_bounded(sample_count):
    with pytest.raises(ValueError, match="invalid_sample_count"):
        SourceTestOutcome(
            provider="official",
            status="passed",
            sample_count=sample_count,
            tested_at=NOW,
            reason_code=None,
        )


def test_source_test_reason_code_cannot_contain_blank_provider_text():
    with pytest.raises(ValueError, match="blank_reason_code"):
        SourceTestOutcome(
            provider="official",
            status="failed",
            sample_count=0,
            tested_at=NOW,
            reason_code="  ",
        )


def test_history_limit_does_not_turn_success_into_failure():
    outcome = SocialSourceOutcome(
        read_status="success",
        processing_status="complete",
        history_status="limited",
        coverage_reason_codes=("post_cap_reached",),
        known_gap_intervals=(),
        observed_oldest_at=NOW - timedelta(days=10),
        observed_newest_at=NOW - timedelta(hours=1),
        received_count=1000,
        committed_progress="cursor-10",
        error_code=None,
    )

    assert outcome.read_status == "success"
    assert outcome.history_status == "limited"
    assert outcome.error_code is None


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_oldest_at": NOW - timedelta(days=1), "observed_newest_at": None},
        {"observed_oldest_at": NOW, "observed_newest_at": NOW - timedelta(days=1)},
        {"read_status": "success", "error_code": "provider_error"},
        {"read_status": "failed", "error_code": None},
        {"read_status": "failed", "processing_status": "complete", "error_code": "provider_error"},
        {"read_status": "failed", "committed_progress": "cursor-10", "error_code": "provider_error"},
        {"read_status": "failed", "history_status": "observed_window", "error_code": "provider_error"},
    ],
)
def test_source_outcome_rejects_contradictory_coverage(changes):
    values = {
        "read_status": "success",
        "processing_status": "complete",
        "history_status": "limited",
        "coverage_reason_codes": ("post_cap_reached",),
        "known_gap_intervals": (),
        "observed_oldest_at": NOW - timedelta(days=10),
        "observed_newest_at": NOW - timedelta(hours=1),
        "received_count": 1000,
        "committed_progress": "cursor-10",
        "error_code": None,
    }
    values.update(changes)

    with pytest.raises(ValueError, match="inconsistent_source_outcome"):
        SocialSourceOutcome(**values)


def test_failed_partial_read_may_retain_paired_bounds_and_count_for_audit():
    outcome = SocialSourceOutcome(
        read_status="failed",
        processing_status="failed",
        history_status="limited",
        coverage_reason_codes=("partial_read",),
        known_gap_intervals=((NOW - timedelta(days=14), NOW - timedelta(days=3)),),
        observed_oldest_at=NOW - timedelta(days=2),
        observed_newest_at=NOW - timedelta(days=1),
        received_count=3,
        committed_progress=None,
        error_code="provider_error",
    )

    assert outcome.received_count == 3
    assert outcome.observed_oldest_at is not None


def test_nested_mutable_collections_are_rejected():
    outcome = SocialSourceOutcome(
        read_status="success",
        processing_status="complete",
        history_status="limited",
        coverage_reason_codes=("post_cap_reached",),
        known_gap_intervals=(),
        observed_oldest_at=NOW - timedelta(days=10),
        observed_newest_at=NOW - timedelta(hours=1),
        received_count=1,
        committed_progress="cursor-10",
        error_code=None,
    )

    request = SocialReadRequest(
        request_id="request-1",
        source_id="source-1",
        list_id="1522014550211457024",
        intent="incremental",
        observed_at=NOW,
        limit=100,
        target_published_after=NOW - timedelta(days=14),
    )

    with pytest.raises(TypeError, match="immutable_tuple:posts"):
        SocialSourceBatch(request=request, posts=[_post()], outcome=outcome)


@pytest.mark.parametrize(
    ("social", "confirmation", "queue"),
    [
        (None, Decimal("50"), Decimal("20")),
        (Decimal("50"), None, Decimal("30")),
        (None, None, Decimal("0")),
        (Decimal("50"), Decimal("50"), None),
    ],
)
def test_snapshot_rejects_inconsistent_nullable_scores(social, confirmation, queue):
    with pytest.raises(ValueError, match="inconsistent_snapshot_scores"):
        SocialSnapshotRecord(
            run_id="run-1",
            candidate_id="candidate-1",
            symbol="NVDA",
            market="US",
            candidate_state="watch",
            social_score=social,
            confirmation_score=confirmation,
            queue_score=queue,
            pinned_inputs=(),
            coverage=(),
            latest_mention=NOW,
            canonical_symbol="NVDA",
            candidate_key="US:NVDA",
        )


def test_component_score_preserves_missing_value_and_weights():
    score = ComponentScore(
        value=None,
        available_weight=Decimal("0"),
        total_weight=Decimal("20"),
        reasons=("missing_history",),
    )

    assert score.value is None
    assert score.available_weight == Decimal("0")
