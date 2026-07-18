from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_backend_unit_shards_exclude_opt_in_markers_during_collection():
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    )
    steps = workflow["jobs"]["backend-unit"]["steps"]
    names = [step.get("name") for step in steps]

    assert "Safe test collection" not in names
    shard_script = next(
        step["run"]
        for step in steps
        if str(step.get("name", "")).startswith(
            "Comprehensive backend unit suite"
        )
    )
    assert (
        'python -m pytest tests/unit --collect-only -qq '
        '-m "not live_service and not load"'
    ) in shard_script
