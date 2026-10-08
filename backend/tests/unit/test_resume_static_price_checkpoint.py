"""The workflow's resume step imports a compatible checkpoint and never fails the job (#502)."""

from __future__ import annotations

from datetime import date

import pytest

import app.scripts.resume_static_price_checkpoint as resume_script


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.mark.parametrize("status", ["imported", "missing", "incompatible", "invalid"])
def test_resume_reports_the_status_for_the_exports_session(monkeypatch, capsys, status):
    calls = []
    monkeypatch.setattr(resume_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(resume_script, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(
        resume_script,
        "_resolve_latest_completed_trading_date",
        lambda market: date(2026, 10, 7),
    )
    monkeypatch.setattr(
        resume_script,
        "resume_price_checkpoint",
        lambda db, **kwargs: calls.append(kwargs) or {"status": status, "reason": "r"},
    )

    assert resume_script.main(["--market", "us"]) == 0

    assert calls == [{"market": "US", "as_of_date": date(2026, 10, 7)}]
    assert f"checkpoint status={status}" in capsys.readouterr().out


def test_an_unexpected_resume_error_is_reported_and_never_fails_the_job(monkeypatch, capsys):
    rollbacks = []

    class _Session(_FakeSession):
        def rollback(self):
            rollbacks.append(True)

    def boom(db, **kwargs):
        raise RuntimeError("database connection lost mid-import")

    monkeypatch.setattr(resume_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(resume_script, "SessionLocal", lambda: _Session())
    monkeypatch.setattr(
        resume_script, "_resolve_latest_completed_trading_date", lambda market: date(2026, 10, 7)
    )
    monkeypatch.setattr(resume_script, "resume_price_checkpoint", boom)

    assert resume_script.main(["--market", "US"]) == 0

    assert rollbacks == [True]
    assert "checkpoint status=invalid" in capsys.readouterr().out


def test_a_failing_rollback_does_not_hide_the_import_error(monkeypatch, capsys):
    class _Session(_FakeSession):
        def rollback(self):
            raise RuntimeError("connection already closed")

    def boom(db, **kwargs):
        raise RuntimeError("database connection lost mid-import")

    monkeypatch.setattr(resume_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(resume_script, "SessionLocal", lambda: _Session())
    monkeypatch.setattr(
        resume_script, "_resolve_latest_completed_trading_date", lambda market: date(2026, 10, 7)
    )
    monkeypatch.setattr(resume_script, "resume_price_checkpoint", boom)

    assert resume_script.main(["--market", "US"]) == 0

    out = capsys.readouterr().out
    assert "checkpoint status=invalid" in out and "mid-import" in out
