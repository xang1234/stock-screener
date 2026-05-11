"""Tests for the Boerse Frankfurt request-signing helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.de_universe_source_signing import (
    build_signed_headers,
    client_date_header,
    client_trace_id,
    compute_x_security,
)


_FIXED_DATE = datetime(2026, 5, 10, 12, 34, 56, 789000, tzinfo=timezone.utc)
_FIXED_TRACE = "0123456789abcdef0123456789abcdef"
_FIXED_URL = "https://api.boerse-frankfurt.de/v1/search/equity_search?lang=en&offset=0&limit=100"


def test_client_date_header_uses_iso_with_zero_offset_no_colon():
    assert client_date_header(now=_FIXED_DATE) == "2026-05-10T12:34:56.789+0000"


def test_client_trace_id_returns_32_char_hex():
    trace = client_trace_id()
    assert len(trace) == 32
    int(trace, 16)  # raises if not hex


def test_compute_x_security_returns_64_char_hex():
    signature = compute_x_security(
        url=_FIXED_URL,
        client_date=client_date_header(now=_FIXED_DATE),
        trace_id=_FIXED_TRACE,
    )
    assert len(signature) == 64
    int(signature, 16)  # raises if not hex


def test_compute_x_security_is_deterministic():
    signature_a = compute_x_security(
        url=_FIXED_URL,
        client_date=client_date_header(now=_FIXED_DATE),
        trace_id=_FIXED_TRACE,
    )
    signature_b = compute_x_security(
        url=_FIXED_URL,
        client_date=client_date_header(now=_FIXED_DATE),
        trace_id=_FIXED_TRACE,
    )
    assert signature_a == signature_b


@pytest.mark.parametrize(
    "field,override",
    [
        ("url", "https://api.boerse-frankfurt.de/v1/search/equity_search?offset=100"),
        ("client_date", "2026-05-11T00:00:00.000+0000"),
        ("trace_id", "ffffffffffffffffffffffffffffffff"),
        ("salt", "rotated-salt-value"),
    ],
)
def test_compute_x_security_changes_when_any_input_changes(field, override):
    base_kwargs = dict(
        url=_FIXED_URL,
        client_date=client_date_header(now=_FIXED_DATE),
        trace_id=_FIXED_TRACE,
    )
    base_signature = compute_x_security(**base_kwargs)
    mutated_kwargs = dict(base_kwargs)
    mutated_kwargs[field] = override
    assert compute_x_security(**mutated_kwargs) != base_signature


def test_build_signed_headers_sets_required_keys():
    headers = build_signed_headers(_FIXED_URL, now=_FIXED_DATE, trace_id=_FIXED_TRACE)
    assert headers["Origin"] == "https://www.boerse-frankfurt.de"
    assert headers["Referer"] == "https://www.boerse-frankfurt.de/"
    assert headers["Accept"] == "application/json"
    assert headers["Client-Date"] == "2026-05-10T12:34:56.789+0000"
    assert headers["X-Client-TraceId"] == _FIXED_TRACE
    assert len(headers["X-Security"]) == 64


def test_build_signed_headers_signature_matches_compute_helper():
    headers = build_signed_headers(_FIXED_URL, now=_FIXED_DATE, trace_id=_FIXED_TRACE)
    expected = compute_x_security(
        url=_FIXED_URL,
        client_date=headers["Client-Date"],
        trace_id=headers["X-Client-TraceId"],
    )
    assert headers["X-Security"] == expected
