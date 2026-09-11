"""Sanitized preparation failures and the closed retry policy."""

from __future__ import annotations

import math
import re

_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]*")
_RETRYABLE_CODES = frozenset(
    {
        "model_connection_failed",
        "model_timeout",
        "model_rate_limited",
        "model_server_error",
        "image_download_connection_failed",
        "image_download_timeout",
        "image_download_rate_limited",
        "image_download_server_error",
        "attachment_fetch_transient",
    }
)


def retryable_failure(code: str) -> bool:
    """Return whether a diagnosed failure may consume the single retry budget."""

    return code in _RETRYABLE_CODES


def classify_image_failure(code: str) -> str:
    """Keep old generic failures explicit without inventing a historical cause."""

    if code == "image_validation_or_processing_failed":
        return "legacy_image_failure_unknown"
    return code


class PreparationFailure(RuntimeError):
    """A preparation error safe to persist without a provider response or secret."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool | None = None,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        if not isinstance(code, str) or _SAFE_CODE.fullmatch(code) is None:
            raise ValueError("invalid_preparation_failure_code")
        if http_status is not None and (
            isinstance(http_status, bool)
            or not isinstance(http_status, int)
            or not 100 <= http_status <= 599
        ):
            raise ValueError("invalid_preparation_http_status")
        if retry_after_seconds is not None and (
            isinstance(retry_after_seconds, bool)
            or not isinstance(retry_after_seconds, (int, float))
            or not math.isfinite(retry_after_seconds)
            or retry_after_seconds < 0
        ):
            raise ValueError("invalid_retry_after")

        policy_retryable = retryable_failure(code)
        if retryable is not None and (
            not isinstance(retryable, bool) or retryable is not policy_retryable
        ):
            raise ValueError("retry_policy_mismatch")

        self.code = code
        self.retryable = policy_retryable
        self.http_status = http_status
        self.retry_after_seconds = (
            float(retry_after_seconds) if retry_after_seconds is not None else None
        )
        super().__init__(code)
