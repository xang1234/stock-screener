"""Shared bounded Kimi JSON transport for evidence image and text adapters."""

import json
import math
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from uuid import uuid4

import httpx

from app.config import settings

from .preparation_failures import PreparationFailure

_OPENCODE_GO_BASE = "https://opencode.ai/zen/go/v1"
_MAX_PROVIDER_RESPONSE_BYTES = 256 * 1024
_MAX_SESSION_ID_LENGTH = 128


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            return None
        delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    return delay if math.isfinite(delay) and delay >= 0 else None


def _session_id(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_SESSION_ID_LENGTH
        or any(not 33 <= ord(character) <= 126 for character in value)
    ):
        raise ValueError("opencode_go_session_id_invalid")
    return value


class OpenCodeGoKimi:
    provider = "opencode-go"
    model = "kimi-k2.6"

    def __init__(
        self,
        api_key: str,
        *,
        session_id: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("opencode_go_api_key_required")
        base = (getattr(settings, "opencode_go_api_base", None)
                or os.environ.get("OPENCODE_GO_API_BASE") or _OPENCODE_GO_BASE)
        self._endpoint = base.rstrip("/") + "/chat/completions"
        self._api_key = api_key.strip()
        self._transport = transport
        self._session_id = _session_id(session_id)

    def complete_json(
        self, messages: list[dict], *, max_tokens: int, read_timeout=20.0
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        timeout = httpx.Timeout(read_timeout, connect=5.0)

        try:
            with (
                httpx.Client(
                    transport=self._transport,
                    timeout=timeout,
                    trust_env=False,
                ) as client,
                client.stream(
                    "POST",
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "User-Agent": "stockscreen-evidence-preparation/1.0",
                        "x-opencode-session": self._session_id,
                    },
                    json=payload,
                ) as response,
            ):
                if not 200 <= response.status_code < 300:
                    status = response.status_code
                    retry_after = (
                        _retry_after_seconds(response.headers.get("retry-after"))
                        if status == 429
                        else None
                    )
                    code = (
                        "model_rate_limited"
                        if status == 429
                        else "model_auth_failed"
                        if status in {401, 403}
                        else "model_server_error"
                        if 500 <= status <= 599
                        else "model_http_error"
                    )
                    raise PreparationFailure(
                        code,
                        http_status=status,
                        retry_after_seconds=retry_after,
                    )
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        if int(content_length) > _MAX_PROVIDER_RESPONSE_BYTES:
                            raise PreparationFailure("model_response_too_large")
                    except ValueError:
                        raise PreparationFailure("model_response_invalid") from None
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > _MAX_PROVIDER_RESPONSE_BYTES:
                        raise PreparationFailure("model_response_too_large")
        except PreparationFailure:
            raise
        except httpx.TimeoutException:
            raise PreparationFailure("model_timeout") from None
        except httpx.HTTPError:
            raise PreparationFailure("model_connection_failed") from None

        try:
            envelope = json.loads(raw)
        except (TypeError, ValueError):
            raise PreparationFailure("model_json_invalid") from None

        try:
            choice = envelope["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise PreparationFailure("model_response_incomplete")
            content = choice["message"]["content"]
        except PreparationFailure:
            raise
        except (KeyError, IndexError, TypeError):
            raise PreparationFailure("model_response_invalid") from None

        try:
            result = json.loads(content)
        except (TypeError, ValueError):
            raise PreparationFailure("model_json_invalid") from None
        if not isinstance(result, dict):
            raise PreparationFailure("model_response_invalid")
        return result
