"""Tests for CPU-bounded parallel worker helpers."""

from __future__ import annotations

import pytest

from app.utils import parallelism


@pytest.mark.parametrize(
    ("cpu_count", "requested", "expected"),
    [
        (None, 8, 1),
        (1, 8, 1),
        (2, 8, 1),
        (4, 8, 2),
        (8, 8, 4),
        (16, 8, 8),
    ],
)
def test_bounded_symbol_workers_caps_requested_workers_by_half_cpu_count(
    monkeypatch,
    cpu_count,
    requested,
    expected,
):
    monkeypatch.setattr(parallelism.os, "cpu_count", lambda: cpu_count)

    assert parallelism.bounded_symbol_workers(requested) == expected


@pytest.mark.parametrize("requested", [0, -1])
def test_bounded_symbol_workers_rejects_non_positive_requests(requested):
    with pytest.raises(ValueError, match="requested must be >= 1"):
        parallelism.bounded_symbol_workers(requested)
