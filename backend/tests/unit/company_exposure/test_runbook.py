from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models.company_exposure import IssuerSecurityLinkRevision
from scripts import company_exposure as cli
from tests.fixtures.company_exposure.research_harness import SHADOW, TICKERS, Harness

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNBOOK = REPO_ROOT / "docs/runbooks/company-exposure-map.md"


def test_runbook_covers_required_operations():
    text = RUNBOOK.read_text("utf-8")
    for required in (
        "mkdir -p ./data/exposure-evidence && sudo chown -R 1000:1000 ./data/exposure-evidence",
        "storage_not_writable",
        "review_required",
        "resolve-issuer",
        "official_registry_single_listing",
        "expired_uncertain",
        "tombstone",
        "EXPOSURE_RESEARCH_MODE=disabled",
        "shadow_preview",
        "not a live probe",
        "run_required_company_exposure_postgres.py",
    ):
        assert required in text, required


def _run(capsys, argv, **kwargs):
    code = cli.main(argv, **kwargs)
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


@pytest.fixture
def cli_kwargs(db_session, tmp_path):
    return {
        "session_factory": lambda: db_session,
        "config_loader": lambda: replace(SHADOW, document_store=str(tmp_path)),
    }


def test_status_reports_stages_without_secrets(capsys, cli_kwargs):
    code, payload = _run(capsys, ["status"], **cli_kwargs)
    assert code == 0
    stages = {s["stage"]: s for s in payload["activation"]}
    assert stages["shadow_verify_us"]["allowed"] is True
    assert stages["automatic_admission"]["reasons"] == ["not_installed"]
    assert payload["configuration"]["subscription_key_present"] is True
    assert "api_key" not in json.dumps(payload["configuration"]).lower().replace(
        "subscription_key_present", ""
    )


def test_review_required_job_is_resolved_and_resumed(
    capsys, cli_kwargs, db_session, tmp_path, clock, monkeypatch
):
    harness = Harness(db_session, tmp_path, clock)
    harness.serve_sec()
    harness.sec.serve_json(
        "https://www.sec.gov/files/company_tickers_exchange.json",
        {
            "fields": TICKERS["fields"],
            "data": [*TICKERS["data"], [7654321, "Other", "EXMP", "NYSE"]],
        },
    )
    ref = harness.request()
    assert harness.step().state == "review_required"

    def links():
        return db_session.execute(
            select(func.count()).select_from(IssuerSecurityLinkRevision)
        ).scalar()

    before = links()
    args = [
        "resolve-issuer",
        "--security-id",
        str(harness.security.id),
        "--cik",
        "1234567",
    ]
    code, dry = _run(capsys, args, **cli_kwargs)
    assert (code, dry["state"]) == (0, "dry_run")
    assert links() == before

    from app.config import settings

    monkeypatch.setattr(settings, "admin_principal_id", "")
    code, blocked = _run(capsys, [*args, "--apply"], **cli_kwargs)
    assert (code, blocked["reason"]) == (2, "admin_principal_unbound")

    monkeypatch.setattr(settings, "admin_principal_id", "ops:alice")
    code, applied = _run(capsys, [*args, "--apply"], **cli_kwargs)
    assert (code, applied["state"]) == (0, "accepted")

    code, resumed = _run(capsys, ["resume", str(ref.id), "--apply"], **cli_kwargs)
    assert resumed == {"state": "queued", "previous_state": "review_required"}
    step = harness.step()
    assert (step.stage, step.detail["source"]) == ("resolve_issuer", "accepted_link")


def test_process_reports_disabled_research_as_blocked(capsys, cli_kwargs):
    code, payload = _run(
        capsys,
        ["process"],
        **cli_kwargs,
        worker=lambda max_steps: {"status": "skipped", "reason": "research_disabled"},
    )
    assert (code, payload["reason"]) == (2, "research_disabled")


def test_bounded_arguments(capsys, cli_kwargs):
    assert cli.main(["process", "--max-steps", "99"], **cli_kwargs) == 64
    assert (
        cli.main(
            ["resolve-issuer", "--security-id", "1", "--cik", "12ab"], **cli_kwargs
        )
        == 64
    )


def test_refresh_holds_is_dry_run_by_default(capsys, cli_kwargs):
    code, payload = _run(capsys, ["refresh-holds"], **cli_kwargs)
    assert (code, payload["applied"], payload["new_holds"]) == (0, False, 0)
