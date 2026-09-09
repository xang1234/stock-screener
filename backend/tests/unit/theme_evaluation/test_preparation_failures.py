import pytest
from app.services.theme_evaluation.preparation_failures import (
    PreparationFailure,
    classify_image_failure,
    retryable_failure,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("model_connection_failed", True),
        ("model_timeout", True),
        ("model_rate_limited", True),
        ("model_server_error", True),
        ("image_download_connection_failed", True),
        ("image_download_timeout", True),
        ("image_download_rate_limited", True),
        ("image_download_server_error", True),
        ("invalid_image", False),
        ("image_pixel_limit", False),
        ("model_auth_failed", False),
        ("model_response_incomplete", False),
        ("model_schema_invalid", False),
        ("unknown_provider_failure", False),
    ],
)
def test_retry_policy_is_a_closed_allowlist(code, expected):
    assert retryable_failure(code) is expected


def test_preparation_failure_exposes_only_safe_diagnostic_values():
    failure = PreparationFailure(
        "model_rate_limited",
        retryable=True,
        http_status=429,
        retry_after_seconds=2.5,
    )

    assert str(failure) == "model_rate_limited"
    assert failure.code == "model_rate_limited"
    assert failure.retryable is True
    assert failure.http_status == 429
    assert failure.retry_after_seconds == 2.5


@pytest.mark.parametrize("code", ["", "private body", "token=secret", "bad:detail"])
def test_preparation_failure_rejects_codes_that_could_carry_private_text(code):
    with pytest.raises(ValueError, match="invalid_preparation_failure_code"):
        PreparationFailure(code)


def test_legacy_generic_image_failure_is_not_given_an_invented_cause():
    assert (
        classify_image_failure("image_validation_or_processing_failed")
        == "legacy_image_failure_unknown"
    )
    assert classify_image_failure("model_timeout") == "model_timeout"


def test_caller_cannot_open_the_closed_retry_policy():
    with pytest.raises(ValueError, match="retry_policy_mismatch"):
        PreparationFailure("model_schema_invalid", retryable=True)
