"""Regression checks for the static Alembic baseline revision."""

from __future__ import annotations

from pathlib import Path


def test_baseline_uses_quoted_manual_trigger_source_default():
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "20260408_0001_baseline.py"
    )
    source = baseline_path.read_text()

    assert "server_default='manual'" in source
    assert "server_default=sa.text('manual')" not in source
