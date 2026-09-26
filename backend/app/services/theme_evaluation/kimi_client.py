"""Shared bounded Kimi JSON transport for evidence image and text adapters."""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from uuid import uuid4

import httpx

from app.config import settings

from .preparation_failures import PreparationFailure

_OPENCODE_GO_BASE = "https://opencode.ai/zen/go/v1"
_MAX_PROVIDER_RESPONSE_BYTES = 256 * 1024
_MAX_SESSION_ID_LENGTH = 128


def opencode_go_endpoint() -> str:
    """Use the same configured route for preparation and extraction fallback."""
    base = (getattr(settings, "opencode_go_api_base", None)
            or os.environ.get("OPENCODE_GO_API_BASE") or _OPENCODE_GO_BASE)
    return base.rstrip("/") + "/chat/completions"


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
        self._endpoint = opencode_go_endpoint()
        self._api_key = api_key.strip()
        self._transport = transport
        self._session_id = _session_id(session_id)

    def complete_json(
        self, messages: list[dict], *, max_tokens: int, read_timeout=20.0
    ) -> dict:
        return self.complete_json_response(
            messages, max_tokens=max_tokens, read_timeout=read_timeout
        ).data

    def complete_json_response(
        self, messages: list[dict], *, max_tokens: int, read_timeout=20.0
    ) -> "KimiJSONResponse":
        """One HTTP dispatch; failures carry an exact ``dispatch_phase``.

        Connect-phase failures (DNS, refused, TLS handshake, connect or pool
        timeout) are ``pre_dispatch``: no request bytes reached the provider.
        A received HTTP response is ``dispatched``. Anything that fails after
        the request started is ``uncertain``.
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        timeout = httpx.Timeout(read_timeout, connect=5.0)
        provider_request_id = None

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
                provider_request_id = _header_request_id(response.headers)
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
                        code, http_status=status, retry_after_seconds=retry_after
                    )
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared = int(content_length)
                    except ValueError:
                        raise PreparationFailure("model_response_invalid") from None
                    if declared > _MAX_PROVIDER_RESPONSE_BYTES:
                        raise PreparationFailure(
                            "model_response_too_large", dispatch_phase="uncertain"
                        )
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > _MAX_PROVIDER_RESPONSE_BYTES:
                        raise PreparationFailure(
                            "model_response_too_large", dispatch_phase="uncertain"
                        )
        except PreparationFailure as exc:
            # A response was received; only a size abort is less certain.
            raise exc.in_phase("dispatched")
        except httpx.TimeoutException as exc:
            raise PreparationFailure(
                "model_timeout", dispatch_phase=_transport_phase(exc)
            ) from None
        except httpx.HTTPError as exc:
            raise PreparationFailure(
                "model_connection_failed", dispatch_phase=_transport_phase(exc)
            ) from None

        try:
            return _parse_response(bytes(raw), self.model, provider_request_id)
        except PreparationFailure as exc:
            raise exc.in_phase("dispatched")


def _parse_response(
    raw: bytes, requested_model: str, header_request_id: str | None
) -> "KimiJSONResponse":
    """Validate the chat envelope and its JSON content."""

    try:
        envelope = json.loads(raw)
    except (TypeError, ValueError):
        raise PreparationFailure("model_json_invalid") from None
    try:
        choice = envelope["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise PreparationFailure("model_response_incomplete")
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, AttributeError):
        raise PreparationFailure("model_response_invalid") from None
    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        raise PreparationFailure("model_json_invalid") from None
    if not isinstance(result, dict):
        raise PreparationFailure("model_response_invalid")
    envelope_id = envelope.get("id")
    return KimiJSONResponse(
        data=result,
        provider_request_id=(
            envelope_id if isinstance(envelope_id, str) and envelope_id else None
        )
        or header_request_id,
        reported_usage=_reported_usage(envelope),
        response_hash=hashlib.sha256(raw).hexdigest(),
        finish_reason=str(choice.get("finish_reason")),
        model=str(envelope.get("model") or requested_model),
    )


@dataclass(frozen=True, slots=True)
class KimiJSONResponse:
    """Validated JSON content plus bounded provider metadata."""

    data: dict
    provider_request_id: str | None
    reported_usage: dict | None
    response_hash: str
    finish_reason: str
    model: str


# Exceptions raised before any request bytes can reach the provider.
_PRE_DISPATCH_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.UnsupportedProtocol,
)


def _transport_phase(exc: httpx.HTTPError) -> str:
    return "pre_dispatch" if isinstance(exc, _PRE_DISPATCH_ERRORS) else "uncertain"


def _header_request_id(headers) -> str | None:
    for name in ("x-request-id", "x-opencode-request-id", "request-id"):
        value = headers.get(name)
        if value and len(value) <= 200:
            return value
    return None


def _reported_usage(envelope) -> dict | None:
    usage = envelope.get("usage") if isinstance(envelope, dict) else None
    if not isinstance(usage, dict):
        return None
    reported = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            reported[key] = value
    return reported or None
