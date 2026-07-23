from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_backend_unit_shards_use_cache_free_equal_weight_split_and_exclude_opt_in_markers():
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    )
    steps = workflow["jobs"]["backend-unit"]["steps"]
    names = [step.get("name") for step in steps]

    assert "Safe test collection" not in names
    assert "Restore backend unit duration cache" not in names
    assert "Save backend unit duration cache" not in names
    shard_script = next(
        step["run"]
        for step in steps
        if str(step.get("name", "")).startswith(
            "Comprehensive backend unit suite"
        )
    )
    assert 'make gate-unit-files > "${gate_unit_file}"' in shard_script
    assert 'ignore_args+=(--ignore "${path}")' in shard_script
    assert "python -m pytest tests/unit \\" in shard_script
    assert '-m "not live_service and not load" \\' in shard_script
    assert (
        "Intentionally cache-free: without duration data, pytest-split gives "
        "each test equal weight and all shards compute the same partition."
    ) in shard_script
    assert "--splits 4 \\" in shard_script
    assert '--group "${{ matrix.shard }}" \\' in shard_script
    assert "--splitting-algorithm least_duration \\" in shard_script
    assert "--collect-only" not in shard_script
    assert "index % 4" not in shard_script
