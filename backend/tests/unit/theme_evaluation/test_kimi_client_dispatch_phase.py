"""OpenCodeGoKimi records whether a failed request can have reached the provider."""

from __future__ import annotations

import json

import httpx
import pytest

from app.services.theme_evaluation.kimi_client import OpenCodeGoKimi
from app.services.theme_evaluation.preparation_failures import PreparationFailure


def _client(handler):
    return OpenCodeGoKimi(
        "test-key", session_id="session-1", transport=httpx.MockTransport(handler)
    )


def _raise(error):
    def handler(request):
        raise error

    return handler


def _ok(body, *, headers=None):
    def handler(request):
        return httpx.Response(200, json=body, headers=headers or {})

    return handler


GOOD = {
    "id": "resp-123",
    "model": "kimi-k2.6",
    "choices": [
        {"finish_reason": "stop", "message": {"content": json.dumps({"claims": []})}}
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
}


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx.ConnectError("refused"), "model_connection_failed"),
        (httpx.ConnectTimeout("connect"), "model_timeout"),
        (httpx.PoolTimeout("pool"), "model_timeout"),
    ],
)
def test_connect_phase_failures_are_pre_dispatch(error, code):
    with pytest.raises(PreparationFailure) as raised:
        _client(_raise(error)).complete_json([{"role": "user", "content": "x"}], max_tokens=10)
    assert (raised.value.code, raised.value.dispatch_phase) == (code, "pre_dispatch")


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx.ReadTimeout("read"), "model_timeout"),
        (httpx.WriteTimeout("write"), "model_timeout"),
        (httpx.ReadError("reset"), "model_connection_failed"),
        (httpx.WriteError("broken"), "model_connection_failed"),
        (httpx.RemoteProtocolError("eof"), "model_connection_failed"),
    ],
)
def test_post_send_failures_are_uncertain(error, code):
    with pytest.raises(PreparationFailure) as raised:
        _client(_raise(error)).complete_json([{"role": "user", "content": "x"}], max_tokens=10)
    assert (raised.value.code, raised.value.dispatch_phase) == (code, "uncertain")


@pytest.mark.parametrize(
    ("status", "code"),
    [(429, "model_rate_limited"), (401, "model_auth_failed"), (503, "model_server_error")],
)
def test_http_error_responses_are_dispatched_with_unchanged_codes(status, code):
    def handler(request):
        return httpx.Response(status, headers={"retry-after": "7"})

    with pytest.raises(PreparationFailure) as raised:
        _client(handler).complete_json([{"role": "user", "content": "x"}], max_tokens=10)
    assert (raised.value.code, raised.value.dispatch_phase) == (code, "dispatched")
    assert raised.value.retry_after_seconds == (7.0 if status == 429 else None)


def test_invalid_body_is_dispatched():
    def handler(request):
        return httpx.Response(200, content=b"not json")

    with pytest.raises(PreparationFailure) as raised:
        _client(handler).complete_json([{"role": "user", "content": "x"}], max_tokens=10)
    assert (raised.value.code, raised.value.dispatch_phase) == (
        "model_json_invalid",
        "dispatched",
    )


def test_response_metadata_is_returned_and_complete_json_is_unchanged():
    client = _client(_ok(GOOD))
    response = client.complete_json_response(
        [{"role": "user", "content": "x"}], max_tokens=10
    )
    assert response.data == {"claims": []}
    assert response.provider_request_id == "resp-123"
    assert response.reported_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
    assert len(response.response_hash) == 64
    assert client.complete_json([{"role": "user", "content": "x"}], max_tokens=10) == {
        "claims": []
    }


def test_missing_usage_is_unknown_not_zero():
    body = dict(GOOD)
    body.pop("usage")
    response = _client(_ok(body)).complete_json_response(
        [{"role": "user", "content": "x"}], max_tokens=10
    )
    assert response.reported_usage is None


def test_invalid_dispatch_phase_is_rejected():
    with pytest.raises(ValueError):
        PreparationFailure("model_timeout", dispatch_phase="maybe")
