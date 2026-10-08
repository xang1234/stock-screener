from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml
from app.scripts import download_static_market_fallbacks as fallback_script
from app.scripts.download_static_market_fallbacks import (
    collect_current_markets,
    downloaded_market_is_compatible,
)

ROOT = Path(__file__).resolve().parents[3]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "static-site-publish.yml"


_ARTIFACT_LOOKUP_PRELUDE = """\
import json as _json
import sys as _sys

_args = _sys.argv[1:]
if _args[:3] == ["api", "--paginate", "--slurp"] and "actions/artifacts?name=" in _args[3]:
    _name = _args[3].split("name=", 1)[1].split("&", 1)[0]
    print(_json.dumps([{"artifacts": [
        {
            "name": _name,
            "expired": False,
            "created_at": created,
            "workflow_run": {
                "id": run_id,
                "head_branch": "main",
                "repository_id": 1,
                "head_repository_id": 1,
            },
        }
        for run_id, created, names in _ARTIFACT_RUNS
        if _name in names
    ]}]))
    _sys.exit(0)
"""


def _write_fake_gh(
    fake_gh: Path,
    payload: str,
    *,
    artifact_runs: list[tuple[int, str | None, list[str]]] | None = None,
) -> None:
    """Install a fake ``gh``; ``artifact_runs`` serves the by-name artifact API.

    Each ``(run_id, created_at, artifact_names)`` row is one workflow run of
    this repository's main branch that uploaded those artifacts.
    """
    prelude = (
        f"_ARTIFACT_RUNS = {artifact_runs!r}\n{_ARTIFACT_LOOKUP_PRELUDE}\n"
        if artifact_runs is not None
        else ""
    )
    payload_path = fake_gh.with_suffix(".py")
    payload_path.write_text(prelude + textwrap.dedent(payload), encoding="utf-8")
    fake_gh.write_text(
        "#!/bin/sh\n"
        f'exec {shlex.quote(sys.executable)} {shlex.quote(str(payload_path))} "$@"\n',
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)


def _fallback_downloader_env(fake_bin: Path) -> dict[str, str]:
    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "REPOSITORY": "xang1234/stock-screener",
        "CURRENT_RUN_ID": "999",
        "BRANCH_NAME": "main",
    }
    if pythonpath := os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = pythonpath
    return env


def test_static_fallback_downloader_import_does_not_initialize_database(tmp_path):
    env = _fallback_downloader_env(tmp_path)
    # Deliberately unsupported by app.database: artifact jobs must not load it.
    env["DATABASE_URL"] = "sqlite://"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.scripts.download_static_market_fallbacks; "
                "assert 'app.database' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT / "backend",
    )
    assert result.returncode == 0, result.stderr


def _build_market_job() -> str:
    content = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    return content.split("  build-market:\n", 1)[1].split(
        "\n  wake-publisher:",
        1,
    )[0]


def _build_cot_job() -> str:
    content = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    return content.split("  build-cot:\n", 1)[1].split(
        "\n  build-market:",
        1,
    )[0]


def test_static_site_workflow_publishes_and_combines_global_cot_artifact() -> None:
    workflow = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    cot_job = _build_cot_job()
    market_job = _build_market_job()

    assert "static-cot-global" in workflow
    assert "python -m app.scripts.export_static_cot" in cot_job
    assert "actions/upload-artifact@v7" in cot_job
    assert "static-cot-global" in cot_job
    assert "--skip-cot-refresh" in market_job
    assert "Upload current global COT artifact" not in market_job
    publisher = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    assert "--fallback-cot-dir /tmp/static-cot" in publisher
    assert "--fallback-cot-artifacts-dir /tmp/static-cot" in publisher


def test_fake_gh_launcher_handles_python_path_with_spaces(
    tmp_path, monkeypatch
) -> None:
    real_python = sys.executable
    interpreter = tmp_path / "interpreter dir" / "python"
    interpreter.parent.mkdir()
    interpreter.write_text(
        f'#!/bin/sh\nexec {shlex.quote(real_python)} "$@"\n',
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    fake_gh = tmp_path / "bin" / "gh"
    fake_gh.parent.mkdir()
    monkeypatch.setattr(sys, "executable", str(interpreter))

    _write_fake_gh(
        fake_gh,
        """\
        import sys

        print("|".join(sys.argv[1:]))
        """,
    )

    result = subprocess.run(
        [str(fake_gh), "api", "hello world"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "api|hello world"


def test_static_site_market_build_failures_are_not_marked_continue_on_error() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split(
        "      - name: Export market static data bundle\n", 1
    )[1].split(
        "\n      - name: Upload market status",
        1,
    )[0]

    assert "continue-on-error: true" not in export_step


def test_static_site_daily_price_seed_allows_stale_bootstrap() -> None:
    build_market_job = _build_market_job()
    seed_step = build_market_job.split(
        "      - name: Seed daily price bundle from GitHub\n", 1
    )[1].split(
        "\n      - name: Export market static data bundle",
        1,
    )[0]

    assert "--allow-stale" in seed_step


def test_static_site_market_export_preserves_price_bundle_after_soft_skip() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split(
        "      - name: Export market static data bundle\n", 1
    )[1].split(
        "\n      - name: Build daily price bundle",
        1,
    )[0]
    build_price_step = build_market_job.split(
        "      - name: Build daily price bundle\n", 1
    )[1].split(
        "\n      - name: Upload daily price assets",
        1,
    )[0]
    upload_price_step = build_market_job.split(
        "      - name: Upload daily price assets\n", 1
    )[1].split(
        "\n      - name: Upload market artifact",
        1,
    )[0]
    upload_market_step = build_market_job.split(
        "      - name: Upload market artifact\n", 1
    )[1].split(
        "\n\n  wake-publisher:",
        1,
    )[0]

    assert "id: export-market" in export_step
    assert 'status="${pipeline_status[0]}"' in export_step
    assert 'if [ "$status" -eq 78 ]; then' in export_step
    assert "has_artifact=false" in export_step
    assert "has_artifact=true" in export_step
    assert "has_price_bundle=false" in export_step
    assert "has_price_bundle=true" in export_step
    assert "steps.export-market.outputs.has_price_bundle == 'true'" in build_price_step
    assert "steps.export-market.outputs.has_price_bundle == 'true'" in upload_price_step
    assert "steps.export-market.outputs.has_artifact == 'true'" in upload_market_step


def test_static_site_market_export_soft_skips_no_current_artifact_exit_code() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split(
        "      - name: Export market static data bundle\n", 1
    )[1].split(
        "\n      - name: Upload market status",
        1,
    )[0]

    assert 'if [ "$status" -eq 79 ]; then' in export_step
    assert "has_artifact=false" in export_step
    assert "fallback artifacts" in export_step
    assert "no current market artifact will be uploaded" in export_step


def test_static_site_market_export_uses_status_price_bundle_signal_for_soft_skip() -> (
    None
):
    build_market_job = _build_market_job()
    export_step = build_market_job.split(
        "      - name: Export market static data bundle\n", 1
    )[1].split(
        "\n      - name: Upload market status",
        1,
    )[0]
    soft_skip_branch = export_step.split('if [ "$status" -eq 79 ]; then', 1)[1].split(
        "exit 0",
        1,
    )[0]

    assert (
        'STATUS_PATH="/tmp/static-data/status/${MARKET_LOWER}/status.json"'
        in export_step
    )
    assert ".has_price_bundle // false" in soft_skip_branch
    assert (
        'echo "has_price_bundle=$has_price_bundle" >> "$GITHUB_OUTPUT"'
        in soft_skip_branch
    )
    assert 'echo "has_price_bundle=true" >> "$GITHUB_OUTPUT"' not in soft_skip_branch


def test_static_site_uploads_canonical_market_status_after_export() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split(
        "      - name: Export market static data bundle\n", 1
    )[1].split(
        "\n      - name: Upload market status",
        1,
    )[0]
    status_step = build_market_job.split("      - name: Upload market status\n", 1)[
        1
    ].split(
        "\n      - name: Upload market diagnostics",
        1,
    )[0]

    assert "python -m app.scripts.export_static_market_artifact" in export_step
    assert "write_market_status" not in export_step
    assert "json_reason" not in export_step
    assert "cat >" not in export_step
    assert "if: ${{ always() }}" in status_step
    assert "uses: actions/upload-artifact@v7" in status_step
    assert "name: static-market-status-${{ matrix.market }}" in status_step
    assert (
        "path: /tmp/static-data/status/${{ env.MARKET_LOWER }}/status.json"
        in status_step
    )
    assert "if-no-files-found: error" in status_step


def test_static_site_uploads_market_diagnostics_after_export() -> None:
    build_market_job = _build_market_job()
    diagnostics_step = build_market_job.split(
        "      - name: Upload market diagnostics\n", 1
    )[1].split(
        "\n      - name: Build daily price bundle",
        1,
    )[0]

    assert "if: ${{ always() }}" in diagnostics_step
    assert "uses: actions/upload-artifact@v7" in diagnostics_step
    assert "name: static-market-diagnostics-${{ matrix.market }}" in diagnostics_step
    assert (
        "path: /tmp/static-data/diagnostics/${{ env.MARKET_LOWER }}" in diagnostics_step
    )
    assert "if-no-files-found: ignore" in diagnostics_step


def test_static_site_rrg_history_publish_skips_rewound_market_exports() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split(
        "      - name: Export market static data bundle\n", 1
    )[1].split(
        "\n      - name: Upload market status",
        1,
    )[0]
    publish_rrg_step = build_market_job.split(
        "      - name: Publish rolling RRG history\n", 1
    )[1].split(
        "\n\n  wake-publisher:",
        1,
    )[0]

    assert 'EXPORT_LOG="$(mktemp)"' in export_step
    assert '| tee "$EXPORT_LOG"' in export_step
    assert 'pipeline_status=("${PIPESTATUS[@]}")' in export_step
    assert 'status="${pipeline_status[0]}"' in export_step
    assert 'log_status="${pipeline_status[1]}"' in export_step
    assert 'if [ "$log_status" -ne 0 ]; then' in export_step
    assert 'exit "$log_status"' in export_step
    assert (
        'grep -qE "using (benchmark-backed|previous-session) as-of date"'
        in export_step
    )
    assert "rrg_history_publishable=false" in export_step
    assert "rrg_history_publishable=true" in export_step
    assert (
        "steps.export-market.outputs.rrg_history_publishable == 'true'"
        in publish_rrg_step
    )


def test_static_site_daily_price_build_requires_current_session_coverage() -> None:
    build_market_job = _build_market_job()
    build_price_step = build_market_job.split(
        "      - name: Build daily price bundle\n", 1
    )[1].split(
        "\n      - name: Upload daily price assets",
        1,
    )[0]
    upload_price_step = build_market_job.split(
        "      - name: Upload daily price assets\n", 1
    )[1].split(
        "\n      - name: Upload market artifact",
        1,
    )[0]

    assert "id: build-daily-price-bundle" in build_price_step
    assert "static_daily_price_bundle_min_coverage" in build_price_step
    assert 'BUILD_LOG="$(mktemp)"' in build_price_step
    assert '| tee "$BUILD_LOG"' in build_price_step
    assert 'build_pipeline_status=("${PIPESTATUS[@]}")' in build_price_step
    assert 'build_log_status="${build_pipeline_status[1]}"' in build_price_step
    assert "--require-complete" in build_price_step
    assert '--min-symbol-coverage "$MIN_SYMBOL_COVERAGE"' in build_price_step
    assert "Daily price bundle coverage .* is below required" in build_price_step
    assert "price_bundle_ready=false" in build_price_step
    assert "price_bundle_ready=true" in build_price_step
    assert 'exit "$status"' in build_price_step
    assert (
        "steps.build-daily-price-bundle.outputs.price_bundle_ready == 'true'"
        in upload_price_step
    )


def test_static_site_preserves_and_publishes_us_options_history() -> None:
    workflow = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    build_job = _build_market_job()
    publisher = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "options-analytics-data" in workflow
    assert "OPTIONS_ANALYTICS_ENABLED" in build_job
    assert "matrix.market == 'US'" in build_job
    assert "python -m app.scripts.import_options_history" in build_job
    assert "id: restore-options-history" in build_job
    assert "--allow-missing" not in build_job
    assert "python -m app.scripts.export_options_history" in build_job
    assert "--require-run-id" in build_job
    assert "name: static-options-US" in build_job
    assert "--fallback-options-dir /tmp/static-options" in publisher
    assert "--fallback-options-artifacts-dir /tmp/static-options" in publisher
    publish_history = build_job.split("      - name: Publish US options history\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert (
        "steps.restore-options-history.outputs.safe_to_publish == 'true'"
        in publish_history
    )
    assert "options-history-us-v1.previous.json.gz" in build_job
    assert "source_asset_name" in build_job
    assert "SOURCE_ASSET_NAME" in publish_history
    assert 'cp "$RESTORED_PATH" "$PREVIOUS_PATH"' in publish_history
    assert publish_history.index('$PREVIOUS_PATH" --clobber') < publish_history.index(
        "/tmp/options-history-us-v1.json.gz --clobber"
    )


def test_static_site_restores_breadth_history_before_export_and_publishes_after() -> None:
    build_job = _build_market_job()

    def step(name: str) -> str:
        return build_job.split(f"      - name: {name}\n", 1)[1].split("      - name:", 1)[0]

    restore, publish = step("Restore breadth history"), step("Publish breadth history")
    assert build_job.index("- name: Restore breadth history") < build_job.index(
        "- name: Export market static data bundle"
    )
    assert "static_breadth_history_cache import" in restore
    assert "continue-on-error: true" in restore
    assert "static_breadth_history_cache export" in publish
    assert "continue-on-error: true" in publish
    assert "steps.export-market.outputs.has_artifact == 'true'" in publish


def test_static_site_fallback_candidate_install_restores_incumbent_on_failure(
    tmp_path,
    monkeypatch,
) -> None:
    target_dir = tmp_path / "static-market-US"
    candidate_dir = tmp_path / ".static-market-US.candidate-222"
    target_dir.mkdir()
    candidate_dir.mkdir()
    (target_dir / "manifest.market.json").write_text(
        json.dumps(
            {
                "market": "US",
                "schema_version": "static-site-v3",
                "entry": {"as_of_date": "2026-07-31"},
            }
        ),
        encoding="utf-8",
    )
    (candidate_dir / "manifest.market.json").write_text(
        json.dumps(
            {
                "market": "US",
                "schema_version": "static-site-v3",
                "entry": {"as_of_date": "2026-08-03"},
            }
        ),
        encoding="utf-8",
    )
    original_rename = Path.rename

    def flaky_rename(self, target):
        if (
            self.parent == target_dir.parent
            and self.name.startswith(f".{target_dir.name}.stage-")
            and Path(target) == target_dir
        ):
            raise OSError("install failed")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    with pytest.raises(OSError, match="install failed"):
        fallback_script._install_market_candidate(
            target_dir=target_dir,
            candidate_dir=candidate_dir,
        )

    manifest = json.loads(
        (target_dir / "manifest.market.json").read_text(encoding="utf-8")
    )
    assert manifest["entry"]["as_of_date"] == "2026-07-31"


def test_static_site_fallback_run_bound_allows_next_day_session_dates() -> None:
    assert not fallback_script._run_cannot_beat_incumbent(
        run_upper_bound=date(2026, 8, 4),
        incumbent_date=date(2026, 8, 4),
    )
    assert fallback_script._run_cannot_beat_incumbent(
        run_upper_bound=date(2026, 8, 4),
        incumbent_date=date(2026, 8, 5),
    )


def _artifact(
    run_id,
    created,
    *,
    name="static-market-US",
    branch="main",
    repo_id=1,
    head_repo_id=1,
    expired=False,
):
    return {
        "name": name,
        "expired": expired,
        "created_at": created,
        "workflow_run": {
            "id": run_id,
            "head_branch": branch,
            "repository_id": repo_id,
            "head_repository_id": head_repo_id,
        },
    }


def test_list_artifact_runs_keeps_only_trusted_default_branch_runs(
    monkeypatch,
) -> None:
    calls = []

    def fake_gh_json(args):
        calls.append(args)
        return [
            {
                "artifacts": [
                    _artifact(500, "2026-09-01T00:00:00Z"),
                    _artifact(999, "2026-09-05T00:00:00Z"),  # current run
                    _artifact(501, "2026-09-04T00:00:00Z", expired=True),
                    _artifact(502, "2026-09-04T00:00:00Z", branch="feature"),
                    # A fork PR whose head branch is also called "main".
                    _artifact(503, "2026-09-04T00:00:00Z", head_repo_id=77),
                    {
                        **_artifact(504, "2026-09-04T00:00:00Z"),
                        "workflow_run": {"id": 504, "head_branch": "main"},
                    },
                    _artifact(505, "2026-09-03T00:00:00Z", name="static-market-USX"),
                    _artifact(506, "2026-09-03T00:00:00Z"),
                ]
            }
        ]

    monkeypatch.setattr(fallback_script, "gh_json", fake_gh_json)

    runs = fallback_script.list_artifact_runs(
        repo="xang1234/stock-screener",
        artifact_name="static-market-US",
        branch_name="main",
        current_run_id=999,
    )

    assert [run.run_id for run in runs] == [506, 500]
    assert "actions/artifacts?name=static-market-US" in calls[0][-1]


def test_market_fallback_prefers_newer_session_over_newer_upload(
    tmp_path,
    monkeypatch,
) -> None:
    listings = {
        "static-market-US": [
            _artifact(600, "2026-09-06T00:00:00Z"),  # rewound export: older session
            _artifact(500, "2026-09-05T00:00:00Z"),
            _artifact(400, "2026-08-20T00:00:00Z"),  # cannot beat 2026-09-05
        ]
    }
    sessions = {600: date(2026, 9, 4), 500: date(2026, 9, 5), 400: date(2026, 8, 19)}
    downloaded = []

    def fake_gh_json(args):
        name = args[-1].split("name=", 1)[1].split("&", 1)[0]
        return [{"artifacts": listings.get(name, [])}]

    def fake_download(*, run_id, parent_dir, **_kwargs):
        downloaded.append(run_id)
        wrapper = parent_dir / f"c-{run_id}"
        wrapper.mkdir(parents=True)
        return fallback_script._DownloadedCandidate(wrapper, wrapper, sessions[run_id])

    installed = {}
    monkeypatch.setattr(fallback_script, "gh_json", fake_gh_json)
    monkeypatch.setattr(fallback_script, "_download_candidate", fake_download)
    monkeypatch.setattr(
        fallback_script,
        "_install_market_candidate",
        lambda *, target_dir, candidate_dir: installed.__setitem__(
            target_dir.name, candidate_dir.name
        ),
    )

    markets = fallback_script.download_fallback_artifacts(
        repo="xang1234/stock-screener",
        current_run_id=999,
        branch_name="main",
        current_dir=tmp_path / "current",
        fallback_dir=tmp_path / "fallback",
    )

    assert markets == {"US"}
    assert downloaded == [600, 500]
    assert installed["static-market-US"] == "c-500"


def test_same_session_reruns_resolve_to_the_latest_upload(tmp_path, monkeypatch) -> None:
    # A corrected rerun or an RS rollback re-exports the same session on the
    # same day. The API order is not a contract, so the listing arrives
    # oldest-first here; the later upload must still win.
    listings = {
        "static-market-US": [
            _artifact(500, "2026-09-05T09:00:00Z"),
            _artifact(501, "2026-09-05T18:30:00Z"),
        ]
    }
    downloaded = []

    def fake_gh_json(args):
        name = args[-1].split("name=", 1)[1].split("&", 1)[0]
        return [{"artifacts": listings.get(name, [])}]

    def fake_download(*, run_id, parent_dir, **_kwargs):
        downloaded.append(run_id)
        wrapper = parent_dir / f"c-{run_id}"
        wrapper.mkdir(parents=True)
        return fallback_script._DownloadedCandidate(wrapper, wrapper, date(2026, 9, 5))

    installed = {}
    monkeypatch.setattr(fallback_script, "gh_json", fake_gh_json)
    monkeypatch.setattr(fallback_script, "_download_candidate", fake_download)
    monkeypatch.setattr(
        fallback_script,
        "_install_market_candidate",
        lambda *, target_dir, candidate_dir: installed.__setitem__(
            target_dir.name, candidate_dir.name
        ),
    )

    fallback_script.download_fallback_artifacts(
        repo="xang1234/stock-screener",
        current_run_id=999,
        branch_name="main",
        current_dir=tmp_path / "current",
        fallback_dir=tmp_path / "fallback",
    )

    assert downloaded[0] == 501
    assert installed["static-market-US"] == "c-501"


def test_options_fallback_skips_runs_that_cannot_beat_the_incumbent(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_gh_json(args):
        if "name=static-options-US" in args[-1]:
            return [
                {
                    "artifacts": [
                        _artifact(333, "2026-09-05T00:00:00Z", name="static-options-US"),
                        _artifact(222, "2026-09-01T00:00:00Z", name="static-options-US"),
                    ]
                }
            ]
        return [{"artifacts": []}]

    downloaded_runs = []

    def fake_download_candidate(*, run_id, artifact_name, parent_dir, **_kwargs):
        downloaded_runs.append(run_id)
        wrapper = parent_dir / f"candidate-{run_id}"
        artifact = wrapper / "options"
        artifact.mkdir(parents=True)
        return fallback_script._DownloadedCandidate(
            wrapper_dir=wrapper,
            artifact_dir=artifact,
            as_of_date=date(2026, 9, 4) if run_id == 333 else date(2026, 8, 29),
        )

    monkeypatch.setattr(fallback_script, "gh_json", fake_gh_json)
    monkeypatch.setattr(
        fallback_script,
        "_download_candidate",
        fake_download_candidate,
    )
    monkeypatch.setattr(
        fallback_script,
        "_install_market_candidate",
        lambda **_kwargs: None,
    )

    fallback_script.download_fallback_artifacts(
        repo="xang1234/stock-screener",
        current_run_id=999,
        branch_name="main",
        current_dir=tmp_path / "current",
        fallback_dir=tmp_path / "fallback",
        fallback_options_dir=tmp_path / "selected-options",
    )

    assert downloaded_runs == [333]


def test_options_fallback_keeps_a_newer_existing_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    fallback_options_dir = tmp_path / "selected-options"

    def fake_gh_json(args):
        if "name=static-options-US" in args[-1]:
            return [
                {
                    "artifacts": [
                        _artifact(333, "2026-09-05T00:00:00Z", name="static-options-US")
                    ]
                }
            ]
        return [{"artifacts": []}]

    downloaded_runs = []
    monkeypatch.setattr(fallback_script, "gh_json", fake_gh_json)
    monkeypatch.setattr(
        fallback_script,
        "global_artifact_as_of_date",
        lambda _spec, base: (
            date(2026, 9, 6) if base == fallback_options_dir else None
        ),
    )
    monkeypatch.setattr(
        fallback_script,
        "_download_candidate",
        lambda **kwargs: downloaded_runs.append(kwargs["run_id"]),
    )

    fallback_script.download_fallback_artifacts(
        repo="xang1234/stock-screener",
        current_run_id=999,
        branch_name="main",
        current_dir=tmp_path / "current",
        fallback_dir=tmp_path / "fallback",
        fallback_options_dir=fallback_options_dir,
    )

    assert downloaded_runs == []


def test_static_site_fallback_downloader_keeps_newest_candidate_for_current_market(
    tmp_path,
) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    current_us_dir = current_dir / "static-market-US" / "markets" / "us"
    current_us_dir.mkdir(parents=True)
    (current_us_dir / "manifest.market.json").write_text(
        json.dumps(
            {
                "market": "US",
                "schema_version": "static-site-v3",
                "entry": {"as_of_date": "2026-07-31"},
            }
        ),
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    downloads_log = tmp_path / "downloads.jsonl"
    _write_fake_gh(
        fake_gh,
        artifact_runs=[
            (999, '2026-08-05T00:00:00Z', ['static-market-HK', 'static-market-US', 'static-market-TW']),
            (333, '2026-08-05T00:00:00Z', ['static-market-diagnostics-CN', 'static-market-HK', 'static-market-status-CN', 'static-market-US', 'static-market-TW']),
            (222, '2026-08-04T00:00:00Z', ['static-market-US']),
            (111, '2026-08-03T00:00:00Z', ['static-market-US']),
        ],
        payload=f"""\
        import json
        import pathlib
        import sys

        downloads_log = pathlib.Path({str(downloads_log)!r})
        args = sys.argv[1:]
        if args[:2] == ["run", "download"]:
            run_id = args[2]
            artifact_name = args[args.index("--name") + 1]
            if artifact_name == "static-market-HK":
                target_dir = pathlib.Path(args[args.index("--dir") + 1])
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "partial.txt").write_text("partial")
                print("download denied for HK", file=sys.stderr)
                sys.exit(7)
            target_dir = pathlib.Path(args[args.index("--dir") + 1])
            target_dir.mkdir(parents=True, exist_ok=True)
            as_of_date_by_run = {{
                "333": "2026-07-31",
                "222": "2026-08-03",
                "111": "2026-07-30",
            }}
            (target_dir / "manifest.market.json").write_text(json.dumps({{
                "market": artifact_name.rsplit("-", 1)[1],
                "schema_version": "static-site-v3",
                "entry": {{"as_of_date": as_of_date_by_run.get(run_id, "2026-08-02")}},
            }}))
            with downloads_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({{"run": run_id, "artifact": artifact_name}}) + "\\n")
        else:
            print(f"unexpected gh args: {{args}}", file=sys.stderr)
            sys.exit(2)
        """,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.scripts.download_static_market_fallbacks",
            "--current-dir",
            str(current_dir),
            "--fallback-dir",
            str(fallback_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_fallback_downloader_env(fake_bin),
        cwd=ROOT / "backend",
    )

    assert downloads_log.exists(), f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    downloads = [
        json.loads(line)
        for line in downloads_log.read_text(encoding="utf-8").splitlines()
    ]
    assert downloads == [
        {"run": "333", "artifact": "static-market-TW"},
        {"run": "333", "artifact": "static-market-US"},
        {"run": "222", "artifact": "static-market-US"},
        {"run": "111", "artifact": "static-market-US"},
    ]
    us_manifest = json.loads(
        (fallback_dir / "static-market-US" / "manifest.market.json").read_text(
            encoding="utf-8"
        )
    )
    assert not (fallback_dir / "static-market-diagnostics-CN").exists()
    assert not (fallback_dir / "static-market-status-CN").exists()
    assert not (fallback_dir / "static-market-HK").exists()
    assert (fallback_dir / "static-market-TW" / "manifest.market.json").exists()
    assert us_manifest["entry"]["as_of_date"] == "2026-08-03"
    assert "exit 7. Details: stderr: download denied for HK" in result.stdout


def test_static_site_fallback_downloader_keeps_formula_compatible_candidate(
    tmp_path,
) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    downloads_log = tmp_path / "downloads.jsonl"
    _write_fake_gh(
        fake_gh,
        artifact_runs=[
            (999, '2026-08-05T00:00:00Z', ['static-market-US']),
            (333, '2026-08-04T00:00:00Z', ['static-market-US']),
            (222, '2026-08-03T00:00:00Z', ['static-market-US']),
        ],
        payload=f"""\
        import json
        import pathlib
        import sys

        downloads_log = pathlib.Path({str(downloads_log)!r})
        args = sys.argv[1:]
        if args[:2] == ["run", "download"]:
            run_id = args[2]
            artifact_name = args[args.index("--name") + 1]
            target_dir = pathlib.Path(args[args.index("--dir") + 1])
            market_dir = target_dir / "markets" / "us"
            (market_dir / "scan").mkdir(parents=True, exist_ok=True)
            formula_by_run = {{
                "333": "balanced-horizon-percentile-v2",
                "222": "legacy-linear-v1",
            }}
            date_by_run = {{
                "333": "2026-08-04",
                "222": "2026-08-03",
            }}
            formula = formula_by_run[run_id]
            (market_dir / "scan" / "manifest.json").write_text(
                json.dumps({{"rs_formula_version": formula}})
            )
            (market_dir / "manifest.market.json").write_text(json.dumps({{
                "market": "US",
                "schema_version": "static-site-v3",
                "entry": {{
                    "market": "US",
                    "as_of_date": date_by_run[run_id],
                    "rs_formula_version": formula,
                    "features": {{"scan": True}},
                    "pages": {{"scan": {{"path": "markets/us/scan/manifest.json"}}}},
                }},
            }}))
            with downloads_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({{"run": run_id, "artifact": artifact_name}}) + "\\n")
        else:
            print(f"unexpected gh args: {{args}}", file=sys.stderr)
            sys.exit(2)
        """,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.scripts.download_static_market_fallbacks",
            "--current-dir",
            str(current_dir),
            "--fallback-dir",
            str(fallback_dir),
            "--fallback-rs-formula-overrides-json",
            '{"US":"legacy-linear-v1"}',
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_fallback_downloader_env(fake_bin),
        cwd=ROOT / "backend",
    )

    assert downloads_log.exists(), f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    downloads = [
        json.loads(line)
        for line in downloads_log.read_text(encoding="utf-8").splitlines()
    ]
    assert downloads == [
        {"run": "333", "artifact": "static-market-US"},
        {"run": "222", "artifact": "static-market-US"},
    ]
    manifest = json.loads(
        (
            fallback_dir
            / "static-market-US"
            / "markets"
            / "us"
            / "manifest.market.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["entry"]["rs_formula_version"] == "legacy-linear-v1"


def test_static_site_fallback_downloader_skips_damaged_advertised_assets(
    tmp_path,
) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"
    current_us_dir = current_dir / "static-market-US" / "markets" / "us"
    current_us_dir.mkdir(parents=True)
    (current_us_dir / "manifest.market.json").write_text(
        json.dumps(
            {
                "market": "US",
                "schema_version": "static-site-v3",
                "entry": {"as_of_date": "2026-08-04"},
            }
        ),
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    _write_fake_gh(
        fake_gh,
        artifact_runs=[
            (999, '2026-08-05T00:00:00Z', ['static-market-US']),
            (333, '2026-08-04T00:00:00Z', ['static-market-US']),
        ],
        payload="""\
        import json
        import pathlib
        import sys

        args = sys.argv[1:]
        if args[:2] == ["run", "download"]:
            target_dir = pathlib.Path(args[args.index("--dir") + 1])
            market_dir = target_dir / "markets" / "us"
            market_dir.mkdir(parents=True, exist_ok=True)
            (market_dir / "manifest.market.json").write_text(json.dumps({
                "market": "US",
                "schema_version": "static-site-v3",
                "entry": {
                    "market": "US",
                    "as_of_date": "2026-08-04",
                    "features": {"groups": True},
                },
            }))
        else:
            print(f"unexpected gh args: {args}", file=sys.stderr)
            sys.exit(2)
        """,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.scripts.download_static_market_fallbacks",
            "--current-dir",
            str(current_dir),
            "--fallback-dir",
            str(fallback_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_fallback_downloader_env(fake_bin),
        cwd=ROOT / "backend",
    )

    assert not (fallback_dir / "static-market-US").exists()
    assert "advertises GROUPS but groups.json is absent" in result.stdout


def test_static_site_fallback_downloader_skips_incompatible_schema_and_keeps_searching(
    tmp_path,
) -> None:
    current_dir = tmp_path / "current"
    fallback_dir = tmp_path / "fallback"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    downloads_log = tmp_path / "downloads.jsonl"
    _write_fake_gh(
        fake_gh,
        artifact_runs=[
            (999, '2026-08-06T00:00:00Z', ['static-market-AU']),
            (333, '2026-08-05T00:00:00Z', ['static-market-AU']),
            (222, '2026-08-04T00:00:00Z', ['static-market-AU']),
        ],
        payload=f"""\
        import json
        import pathlib
        import sys

        downloads_log = pathlib.Path({str(downloads_log)!r})
        args = sys.argv[1:]
        if args[:2] == ["run", "download"]:
            run_id = args[2]
            artifact_name = args[args.index("--name") + 1]
            target_dir = pathlib.Path(args[args.index("--dir") + 1])
            target_dir.mkdir(parents=True, exist_ok=True)
            schema_version = "static-site-v2" if run_id == "333" else "static-site-v3"
            (target_dir / "manifest.market.json").write_text(
                json.dumps({{"market": "AU", "schema_version": schema_version}})
            )
            with downloads_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({{"run": run_id, "artifact": artifact_name}}) + "\\n")
        else:
            print(f"unexpected gh args: {{args}}", file=sys.stderr)
            sys.exit(2)
        """,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.scripts.download_static_market_fallbacks",
            "--current-dir",
            str(current_dir),
            "--fallback-dir",
            str(fallback_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_fallback_downloader_env(fake_bin),
        cwd=ROOT / "backend",
    )

    assert downloads_log.exists(), f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    downloads = [
        json.loads(line)
        for line in downloads_log.read_text(encoding="utf-8").splitlines()
    ]
    assert downloads == [
        {"run": "333", "artifact": "static-market-AU"},
        {"run": "222", "artifact": "static-market-AU"},
    ]
    manifest = json.loads(
        (fallback_dir / "static-market-AU" / "manifest.market.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == "static-site-v3"
    assert "static-site-v2" in result.stdout


def test_static_site_fallback_downloader_rejects_missing_manifest_market(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "static-market-AU"
    target_dir.mkdir()
    (target_dir / "manifest.market.json").write_text(
        json.dumps({"schema_version": "static-site-v3"}),
        encoding="utf-8",
    )

    assert not downloaded_market_is_compatible(
        target_dir,
        market="AU",
        artifact_name="static-market-AU",
        run_id=222,
    )


def test_static_site_fallback_downloader_rejects_multiple_market_manifests(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "static-market-AU"
    target_dir.mkdir()
    (target_dir / "manifest.market.json").write_text(
        json.dumps({"market": "AU", "schema_version": "static-site-v3"}),
        encoding="utf-8",
    )
    nested_dir = target_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "manifest.market.json").write_text(
        json.dumps({"market": "HK", "schema_version": "static-site-v3"}),
        encoding="utf-8",
    )

    assert not downloaded_market_is_compatible(
        target_dir,
        market="AU",
        artifact_name="static-market-AU",
        run_id=222,
    )


def test_static_site_current_market_collection_rejects_swapped_artifact_name(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current"
    market_dir = current_dir / "static-market-US"
    market_dir.mkdir(parents=True)
    (market_dir / "manifest.market.json").write_text(
        json.dumps({"market": "AU", "schema_version": "static-site-v3"}),
        encoding="utf-8",
    )

    assert collect_current_markets(current_dir) == set()



def _publish_workflow() -> dict:
    return yaml.safe_load(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))


def _publish_step(name: str) -> dict:
    steps = _publish_workflow()["jobs"]["build"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def test_publisher_runs_only_on_dispatch_and_same_repo_prs() -> None:
    workflow = _publish_workflow()
    triggers = workflow[True]  # PyYAML parses the `on:` key as True
    assert set(triggers) == {"workflow_dispatch", "pull_request"}
    # No publish-time RS override: a stateless publish cannot keep one (a
    # later wake-up would undo it). Rollbacks go through the export run.
    assert not triggers["workflow_dispatch"]
    build_if = workflow["jobs"]["build"]["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in build_if
    # Any-branch dispatch is a build-only rehearsal; only deploy needs main.
    assert "github.event_name == 'workflow_dispatch'" in build_if
    assert "default_branch" not in build_if


PRODUCTION_RUN = (
    "github.event_name == 'workflow_dispatch' && "
    "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
)


def test_publisher_scopes_token_permissions_per_job() -> None:
    # The build job runs PR code and third-party installs: read-only, so it
    # can neither mint an OIDC token nor delete the stored market artifacts.
    # Every write lives in the production-only deploy job.
    workflow = _publish_workflow()
    assert workflow["permissions"] == {}
    assert workflow["jobs"]["build"]["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert workflow["jobs"]["deploy"]["permissions"] == {
        "actions": "write",
        "pages": "write",
        "id-token": "write",
    }


def test_publisher_rehearsals_never_share_the_production_group() -> None:
    # Only a default-branch dispatch joins the production group; a PR or a
    # feature-branch dispatch would otherwise replace a pending production
    # publish and then deploy nothing.
    concurrency = _publish_workflow()["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    assert concurrency["group"] == (
        "${{ " + PRODUCTION_RUN + " && 'static-site-publisher' || "
        "format('static-site-publisher-{0}', github.ref) }}"
    )


def test_publisher_reads_default_branch_artifacts_into_one_selection() -> None:
    step = _publish_step("Download newest market artifacts")
    assert "continue-on-error" not in step
    assert step["env"]["BRANCH_NAME"] == "${{ github.event.repository.default_branch }}"
    assert step["env"]["CURRENT_RUN_ID"] == "${{ github.run_id }}"
    run = step["run"]
    assert "python -m app.scripts.download_static_market_fallbacks" in run
    assert "--fallback-dir /tmp/static-market-artifacts" in run
    assert "--fallback-options-dir /tmp/static-options" in run
    assert "--fallback-cot-dir /tmp/static-cot" in run


def test_publisher_validates_and_combines_one_selection() -> None:
    # Stored artifacts are last-good inputs: they go through the fallback path
    # (see test_publisher_combine_arguments_accept_last_good_legacy_artifacts).
    validate = _publish_step("Validate market artifacts")["run"]
    assert "python -m app.scripts.validate_static_market_artifacts" in validate
    assert "--current-dir /tmp/static-empty" in validate
    assert "--fallback-dir /tmp/static-market-artifacts" in validate
    assert "--selected-markets '[]'" in validate
    combine = _publish_step("Combine static data bundle")["run"]
    assert "--combine-artifacts-dir /tmp/static-empty" in combine
    assert "--fallback-artifacts-dir /tmp/static-market-artifacts" in combine
    assert "--fallback-options-artifacts-dir /tmp/static-options" in combine
    assert "--fallback-cot-artifacts-dir /tmp/static-cot" in combine
    report = _publish_step("Report market freshness")
    assert report["continue-on-error"] is True
    assert "app.scripts.report_static_market_freshness" in report["run"]


def test_publisher_deploys_only_from_default_branch_dispatch() -> None:
    jobs = _publish_workflow()["jobs"]
    deploy_if = jobs["deploy"]["if"]
    assert "github.event_name == 'workflow_dispatch'" in deploy_if
    assert "default_branch" in deploy_if
    assert jobs["deploy"]["environment"]["name"] == "github-pages"
    assert PRODUCTION_RUN in deploy_if
    build_steps = {step.get("name") for step in jobs["build"]["steps"]}
    deploy_steps = [step.get("name") or step.get("id") for step in jobs["deploy"]["steps"]]
    assert deploy_steps == [
        "Configure Pages",
        "Prune duplicate Pages artifacts",
        "deployment",
    ]
    assert not build_steps & {"Configure Pages", "Prune duplicate Pages artifacts"}
    prune = jobs["deploy"]["steps"][1]
    assert prune["env"]["KEEP_ARTIFACT_ID"] == "${{ needs.build.outputs.pages_artifact_id }}"
    assert jobs["build"]["outputs"]["pages_artifact_id"] == (
        "${{ steps.upload-pages-artifact.outputs.artifact_id }}"
    )


SITE_WORKFLOW = ROOT / ".github" / "workflows" / "static-site.yml"
WAKE_COMMAND = "gh workflow run static-site-publish.yml"


def _site_workflow() -> dict:
    return yaml.safe_load(SITE_WORKFLOW.read_text(encoding="utf-8"))


def _wake_step(job: str) -> dict:
    steps = _site_workflow()["jobs"][job]["steps"]
    return next(
        step for step in steps if step.get("name") == "Wake static-site publisher"
    )


def test_static_site_no_longer_combines_or_deploys() -> None:
    workflow = _site_workflow()
    assert "combine-and-build" not in workflow["jobs"]
    assert "deploy" not in workflow["jobs"]
    assert "pages" not in workflow["permissions"]
    assert "id-token" not in workflow["permissions"]
    # The calendar audit reports; it no longer gates every market's export.
    assert "needs" not in workflow["jobs"]["select-markets"]


def test_wake_steps_only_run_on_the_default_branch() -> None:
    for job in ("build-cot", "build-market"):
        step = _wake_step(job)
        assert step["continue-on-error"] is True
        assert "default_branch" in step["if"]
        assert WAKE_COMMAND in step["run"]
        assert "--ref" in step["run"]
    wake_job = _site_workflow()["jobs"]["wake-publisher"]
    assert "always()" in wake_job["if"] and "default_branch" in wake_job["if"]
    assert set(wake_job["needs"]) >= {"build-cot", "build-market"}
    assert WAKE_COMMAND in wake_job["steps"][0]["run"]


def test_market_wake_runs_after_any_later_failure() -> None:
    steps = _site_workflow()["jobs"]["build-market"]["steps"]
    assert steps[-1]["name"] == "Wake static-site publisher"
    wake_if = steps[-1]["if"]
    assert "always()" in wake_if
    assert "steps.upload-market-artifact.outcome == 'success'" in wake_if
    upload = next(step for step in steps if step.get("name") == "Upload market artifact")
    assert upload["id"] == "upload-market-artifact"


def test_cot_wake_follows_its_upload() -> None:
    steps = _site_workflow()["jobs"]["build-cot"]["steps"]
    names = [step.get("name") for step in steps]
    assert names.index("Wake static-site publisher") > names.index(
        "Upload current global COT artifact"
    )


def _workflow_script_args(run: str, module: str) -> list[str]:
    """Return the argv a workflow step passes to ``python -m <module>``."""
    command = run.split(f"python -m {module}", 1)[1].replace("\\\n", " ")
    return shlex.split(command.split("\n", 1)[0])


def _write_selected_market(root: Path, market: str, formula: str) -> None:
    market_dir = root / f"static-market-{market}"
    (market_dir / "scan" / "chunks").mkdir(parents=True)
    prefix = f"markets/{market.lower()}"
    (market_dir / "scan" / "chunks" / "chunk-0001.json").write_text(
        json.dumps({"rs_formula_version": formula, "rows": []}), encoding="utf-8"
    )
    (market_dir / "scan" / "manifest.json").write_text(
        json.dumps(
            {
                "rs_formula_version": formula,
                "chunks": [{"path": f"{prefix}/scan/chunks/chunk-0001.json"}],
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "market": market,
        "display_name": market,
        "as_of_date": "2026-10-06",
        "rs_formula_version": formula,
        "features": {"scan": True, "breadth": False, "groups": False, "charts": False},
        "pages": {"scan": {"path": f"{prefix}/scan/manifest.json"}},
        "assets": {},
    }
    (market_dir / "manifest.market.json").write_text(
        json.dumps(
            {
                "schema_version": "static-site-v3",
                "generated_at": "2026-10-06T22:00:00Z",
                "market": market,
                "entry": entry,
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_publisher_combine_arguments_accept_last_good_legacy_artifacts(
    tmp_path, monkeypatch
) -> None:
    """Run the publisher's own combine arguments end to end.

    The publisher selects stored artifacts, so they must get the last-good
    policy: an RS rollback (legacy formula) still publishes instead of failing
    every market's publish.
    """
    from app.domain.relative_strength import (
        BALANCED_RS_FORMULA_VERSION,
        LEGACY_RS_FORMULA_VERSION,
    )
    from app.scripts import export_static_site as export_script

    selection = tmp_path / "selection"
    _write_selected_market(selection, "US", BALANCED_RS_FORMULA_VERSION)
    _write_selected_market(selection, "HK", LEGACY_RS_FORMULA_VERSION)
    paths = {
        "/tmp/static-market-artifacts": selection,
        "/tmp/static-empty": tmp_path / "empty",
        "/tmp/static-options": tmp_path / "options",
        "/tmp/static-cot": tmp_path / "cot",
        "../frontend/public/static-data": tmp_path / "out",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    args = _workflow_script_args(
        _publish_step("Combine static data bundle")["run"], "app.scripts.export_static_site"
    )
    args = [
        str(paths[arg]) if arg in paths else arg
        for arg in args
    ]
    monkeypatch.setattr(sys, "argv", ["export_static_site.py", *args])

    assert export_script.main() == 0

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["supported_markets"] == ["US", "HK"]
    assert manifest["markets"]["HK"]["rs_formula_version"] == LEGACY_RS_FORMULA_VERSION


def test_price_history_repair_is_an_opt_in_for_one_market_job() -> None:
    workflow = _site_workflow()
    dispatch_input = workflow[True]["workflow_dispatch"]["inputs"]["repair_price_history_market"]
    assert dispatch_input["default"] == ""
    export = next(
        step
        for step in workflow["jobs"]["build-market"]["steps"]
        if step.get("id") == "export-market"
    )
    assert export["env"]["REPAIR_PRICE_HISTORY_MARKET"] == (
        "${{ github.event.inputs.repair_price_history_market || '' }}"
    )
    run = export["run"]
    # Only the named market's job gets the flag, compared case-insensitively.
    assert '= "${{ matrix.market }}" ]; then\n  repair_args=(--repair-price-history)' in run
    assert '"${breadth_metadata_args[@]}" "${repair_args[@]}" 2>&1' in run
