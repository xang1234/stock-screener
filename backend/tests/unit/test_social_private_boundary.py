"""Trusted-CI and public/private dependency boundary contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
PRIVATE_WORKFLOW = ROOT / ".github" / "workflows" / "private-social-worker.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
VERIFIER = ROOT / "scripts" / "verify_social_private_boundary.py"


def _workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _load_verifier():
    spec = importlib.util.spec_from_file_location("social_private_boundary", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_private_workflow_has_only_trusted_manual_main_and_release_triggers():
    workflow = _workflow(PRIVATE_WORKFLOW)
    triggers = workflow["on"]

    assert "pull_request" not in triggers
    assert set(triggers) == {"workflow_dispatch", "push"}
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["push"]["tags"] == ["v*"]
    assert workflow["permissions"] == {"contents": "read"}


def test_private_workflow_delays_key_loading_and_uses_ephemeral_buildkit_inputs():
    text = PRIVATE_WORKFLOW.read_text(encoding="utf-8")
    workflow = _workflow(PRIVATE_WORKFLOW)
    jobs = workflow["jobs"]
    smoke_steps = jobs["smoke-private-image"]["steps"]
    names = [step.get("name", "") for step in smoke_steps]
    trusted_index = names.index("Validate trusted event and pinned commit")
    ssh_index = next(
        index
        for index, step in enumerate(smoke_steps)
        if step.get("uses") == "webfactory/ssh-agent@v0.9.0"
    )

    assert trusted_index < ssh_index
    assert "secrets.XUI_READER_DEPLOY_KEY" in text
    assert "secrets.GITHUB_KNOWN_HOSTS" in text
    assert "vars.XUI_READER_REF" in text
    assert "ssh: default" in text
    assert "github_known_hosts" in text
    assert "XUI_READER_DEPLOY_KEY" not in "\n".join(
        str(step.get("with", {}).get("build-args", "")) for step in smoke_steps
    )
    assert "docker/setup-qemu-action@v3" in text
    assert "docker/setup-buildx-action@v3" in text
    assert "docker/build-push-action@v6" in text
    assert "upload-artifact" not in text


def test_private_publish_is_multi_arch_private_and_has_no_latest_tag():
    text = PRIVATE_WORKFLOW.read_text(encoding="utf-8")
    workflow = _workflow(PRIVATE_WORKFLOW)
    publish = workflow["jobs"]["publish-private-image"]
    publish_names = [step.get("name", "") for step in publish["steps"]]

    assert publish["permissions"] == {
        "contents": "read",
        "packages": "write",
        "attestations": "write",
        "id-token": "write",
    }
    assert "docker/login-action@v3" in text
    assert "secrets.GITHUB_TOKEN" in text
    assert "ghcr.io/xang1234/stock-screener-social-xui:sha-${{ github.sha }}" in text
    assert "platforms: linux/amd64,linux/arm64" in text
    assert "provenance: mode=max" in text
    assert "sbom: true" in text
    assert "visibility" in text.lower() and "private" in text.lower()
    assert publish_names.index("Require an existing private GHCR package") < (
        publish_names.index("Build and push private multi-architecture image")
    )
    assert publish_names.index("Reconfirm private package visibility") > (
        publish_names.index("Build and push private multi-architecture image")
    )
    assert ":latest" not in text


def test_public_ci_runs_boundary_guard_before_backend_suites():
    workflow = _workflow(CI_WORKFLOW)
    jobs = workflow["jobs"]

    assert jobs["private-boundary"]["steps"][-1]["run"] == (
        "python3 scripts/verify_social_private_boundary.py"
    )
    assert "private-boundary" in jobs["backend"]["needs"]
    assert "private-boundary" in jobs["backend-unit"]["needs"]


def test_boundary_verifier_accepts_repo_and_rejects_public_dependency_leaks(tmp_path):
    verifier = _load_verifier()
    assert verifier.verify(ROOT) == []

    (tmp_path / "backend" / "app").mkdir(parents=True)
    (tmp_path / "frontend" / "src").mkdir(parents=True)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "backend" / "requirements-runtime.txt").write_text(
        "xui-reader==1.2.3\n", encoding="utf-8"
    )
    (tmp_path / "backend" / "Dockerfile").write_text(
        "FROM python:3.11 AS runtime-base\nRUN pip install xui-reader\n"
        "FROM builder AS social-xui-builder\n",
        encoding="utf-8",
    )
    (tmp_path / "backend" / "app" / "leak.py").write_text(
        "import xui_reader\n", encoding="utf-8"
    )
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "steps:\n  - run: pip install git+ssh://git@github.com/xang1234/xui.git\n",
        encoding="utf-8",
    )

    violations = verifier.verify(tmp_path)
    assert {violation.category for violation in violations} == {
        "application-import",
        "default-docker-stage",
        "public-dependency",
        "public-workflow",
    }
