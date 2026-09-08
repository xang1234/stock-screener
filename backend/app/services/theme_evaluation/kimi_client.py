"""Shared bounded Kimi JSON transport for evidence image and text adapters."""

import json
from uuid import uuid4

import httpx

_OPENCODE_GO_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
_MAX_PROVIDER_RESPONSE_BYTES = 256 * 1024


class OpenCodeGoKimi:
    provider = "opencode-go"
    model = "kimi-k2.6"

    def __init__(self, api_key: str, *, transport: httpx.BaseTransport | None = None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("opencode_go_api_key_required")
        self._api_key = api_key.strip()
        self._transport = transport
        self._session_id = str(uuid4())

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
                    _OPENCODE_GO_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "User-Agent": "stockscreen-evidence-preparation/1.0",
                        "x-opencode-session": self._session_id,
                    },
                    json=payload,
                ) as response,
            ):
                if not 200 <= response.status_code < 300:
                    raise RuntimeError(f"opencode_go_http_error:{response.status_code}")
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        if int(content_length) > _MAX_PROVIDER_RESPONSE_BYTES:
                            raise RuntimeError("opencode_go_response_too_large")
                    except ValueError:
                        raise RuntimeError("opencode_go_response_invalid") from None
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > _MAX_PROVIDER_RESPONSE_BYTES:
                        raise RuntimeError("opencode_go_response_too_large")
        except httpx.HTTPError:
            raise RuntimeError("opencode_go_request_failed") from None

        try:
            envelope = json.loads(raw)
            choice = envelope["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise RuntimeError("opencode_go_response_incomplete")
            content = choice["message"]["content"]
            result = json.loads(content)
            if not isinstance(result, dict):
                raise TypeError
            return result
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("opencode_go_response_invalid") from None
