"""Operator CLI tests for balanced Market RS rollout."""

from __future__ import annotations

from argparse import Namespace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION


def _options(tmp_path: Path, *, activate: bool) -> Namespace:
    return Namespace(
        market="US",
        through_date=date(2026, 4, 10),
        start_date=None,
        static_staging_dir=tmp_path / "stage",
        activate=activate,
    )


def test_dry_run_prints_report_and_never_activates(monkeypatch, tmp_path):
    from app.scripts import backfill_market_rs as module

    report = SimpleNamespace(ok=True, failed_count=0, to_dict=lambda: {"ok": True})
    service = MagicMock()
    service.backfill.return_value = report
    monkeypatch.setattr(module, "get_market_rs_rollout_service", lambda: service)
    db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: db)

    result = module.execute_rollout(_options(tmp_path, activate=False))

    assert result == {"backfill": {"ok": True}, "activated": False}
    service.validate_activation.assert_not_called()
    service.activate.assert_not_called()
    db.close.assert_called_once_with()


def test_activate_requires_empty_non_serving_absolute_staging_directory(
    monkeypatch,
    tmp_path,
):
    from app.scripts import backfill_market_rs as module

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "existing.json").write_text("{}", encoding="utf-8")
    options = _options(tmp_path, activate=True)

    with pytest.raises(module.RolloutCommandFailed, match="must be empty"):
        module.execute_rollout(options)


def test_activate_stages_validates_then_atomically_switches(monkeypatch, tmp_path):
    from app.scripts import backfill_market_rs as module

    events: list[str] = []
    report = SimpleNamespace(
        ok=True,
        failed_count=0,
        to_dict=lambda: {"ok": True},
    )
    validation = SimpleNamespace(
        ok=True,
        errors=(),
        to_dict=lambda: {"ok": True},
    )
    service = MagicMock()
    service.backfill.side_effect = lambda *a, **k: events.append("backfill") or report
    service.validate_activation.side_effect = (
        lambda *a, **k: events.append("validate") or validation
    )
    service.activate.side_effect = lambda *a, **k: events.append("activate")
    monkeypatch.setattr(module, "get_market_rs_rollout_service", lambda: service)
    monkeypatch.setattr(
        module,
        "_build_balanced_feature_snapshot",
        lambda **kwargs: events.append("feature") or 99,
    )
    monkeypatch.setattr(
        module,
        "_export_static_v3",
        lambda **kwargs: events.append("static"),
    )
    monkeypatch.setattr(
        module,
        "_publish_live_groups",
        lambda market: events.append("publish_live"),
    )
    db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: db)

    result = module.execute_rollout(_options(tmp_path, activate=True))

    assert events == [
        "backfill",
        "feature",
        "static",
        "validate",
        "activate",
        "publish_live",
    ]
    assert result["activated"] is True
    assert result["formula_version"] == BALANCED_RS_FORMULA_VERSION


def test_activate_stops_before_feature_build_when_backfill_failed(monkeypatch, tmp_path):
    from app.scripts import backfill_market_rs as module

    report = SimpleNamespace(
        ok=False,
        failed_count=1,
        to_dict=lambda: {"ok": False, "failed_dates": ["2026-04-09"]},
    )
    service = MagicMock()
    service.backfill.return_value = report
    monkeypatch.setattr(module, "get_market_rs_rollout_service", lambda: service)
    monkeypatch.setattr(module, "SessionLocal", MagicMock())
    feature = MagicMock()
    monkeypatch.setattr(module, "_build_balanced_feature_snapshot", feature)

    with pytest.raises(module.RolloutCommandFailed, match="required backfill dates failed"):
        module.execute_rollout(_options(tmp_path, activate=True))

    feature.assert_not_called()
