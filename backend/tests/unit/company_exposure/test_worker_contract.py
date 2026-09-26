"""Research work runs only on the dedicated, opt-in exposure_research worker."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.celery_app import celery_app
from app.tasks import company_exposure_tasks

REPO_ROOT = Path(__file__).resolve().parents[4]
PREFIX = "app.tasks.company_exposure_tasks."


def _exposure_tasks():
    return sorted(name for name in celery_app.tasks if name.startswith(PREFIX))


def test_every_exposure_task_routes_to_the_dedicated_queue():
    assert "app.tasks.company_exposure_tasks" in celery_app.conf.include
    names = _exposure_tasks()
    assert names
    for name in names:
        assert celery_app.conf.task_routes[name] == {"queue": "exposure_research"}


def test_existing_workers_never_consume_the_research_queue():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    for name, service in compose["services"].items():
        command = service.get("command")
        text = command if isinstance(command, str) else " ".join(command or [])
        assert "exposure_research" not in text, name


def test_overlay_worker_has_narrow_environment_and_dedicated_queue():
    overlay = yaml.safe_load((REPO_ROOT / "docker-compose.exposure.yml").read_text())
    worker = overlay["services"]["celery-exposure-research"]
    assert worker["profiles"] == ["exposure-research"]
    assert "env_file" not in worker
    command = worker["command"]
    assert command[command.index("-Q") + 1] == "exposure_research"
    assert "--concurrency=1" in command and "--prefetch-multiplier=1" in command
    environment = worker["environment"]
    assert environment["EXPOSURE_RESEARCH_MODE"] == "${EXPOSURE_RESEARCH_MODE:-disabled}"
    for unrelated in ("ADMIN_API_KEY", "GROQ_API_KEY", "MINIMAX_API_KEY", "ZAI_API_KEY",
                      "TWITTER_BEARER_TOKEN", "SERVER_AUTH_PASSWORD"):
        assert unrelated not in environment
    assert worker["deploy"]["resources"]["limits"] == {"cpus": "1", "memory": "2G"}
    backend_mounts = overlay["services"]["backend"]["volumes"]
    assert backend_mounts == ["./data/exposure-evidence:/app/data/exposure-evidence:ro"]


def test_start_script_launches_worker_only_when_enabled():
    script = (REPO_ROOT / "backend/start_celery.sh").read_text()
    guard = 'if [[ "${EXPOSURE_WORKER_ENABLED:-false}" == "true" ]]; then'
    assert guard in script
    block = script[script.index(guard) :]
    assert "-Q exposure_research" in block.split("fi", 1)[0]


def test_task_claims_no_work_until_the_verification_stage_is_installed():
    outcome = company_exposure_tasks.process_exposure_work.run()
    assert outcome["status"] in {"skipped", "completed"}
