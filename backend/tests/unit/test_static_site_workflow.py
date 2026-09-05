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

from app.scripts import download_static_market_fallbacks as fallback_script
from app.scripts.download_static_market_fallbacks import (
    collect_current_markets,
    downloaded_market_is_compatible,
)

ROOT = Path(__file__).resolve().parents[3]


def _write_fake_gh(fake_gh: Path, payload: str) -> None:
    payload_path = fake_gh.with_suffix(".py")
    payload_path.write_text(textwrap.dedent(payload), encoding="utf-8")
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


def _build_market_job() -> str:
    content = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    return content.split("  build-market:\n", 1)[1].split(
        "\n  combine-and-build:",
        1,
    )[0]


def _combine_and_build_job() -> str:
    content = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    return content.split("  combine-and-build:\n", 1)[1].split(
        "\n  deploy:",
        1,
    )[0]


def _fallback_download_step() -> str:
    return (
        _combine_and_build_job()
        .split("      - name: Download per-market fallback artifacts\n", 1)[1]
        .split(
            "\n      - name: Validate market artifacts",
            1,
        )[0]
    )


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
        "\n\n  combine-and-build:",
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
    assert "uses: actions/upload-artifact@v4" in status_step
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
    assert "uses: actions/upload-artifact@v4" in diagnostics_step
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
        "\n\n  combine-and-build:",
        1,
    )[0]

    assert 'EXPORT_LOG="$(mktemp)"' in export_step
    assert '| tee "$EXPORT_LOG"' in export_step
    assert 'pipeline_status=("${PIPESTATUS[@]}")' in export_step
    assert 'status="${pipeline_status[0]}"' in export_step
    assert 'log_status="${pipeline_status[1]}"' in export_step
    assert 'if [ "$log_status" -ne 0 ]; then' in export_step
    assert 'exit "$log_status"' in export_step
    assert "using benchmark-backed as-of date" in export_step
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


def test_static_site_combine_downloads_current_and_per_market_fallback_artifacts() -> (
    None
):
    combine_job = _combine_and_build_job()
    fallback_step = _fallback_download_step()

    assert "needs: [select-markets, build-market]" in combine_job
    assert "needs.select-markets.outputs.markets" in combine_job
    assert "Download per-market fallback artifacts" in combine_job
    assert "Download current market artifacts" in combine_job
    assert "/tmp/static-market-artifacts-current" in combine_job
    assert "/tmp/static-market-artifacts-fallback" in combine_job
    assert (
        "--fallback-artifacts-dir /tmp/static-market-artifacts-fallback" in combine_job
    )
    assert "FALLBACK_MARKETS" not in combine_job
    assert "github.ref_name" in combine_job
    assert "python -m app.scripts.download_static_market_fallbacks" in fallback_step
    assert "--current-dir /tmp/static-market-artifacts-current" in fallback_step
    assert "--fallback-dir /tmp/static-market-artifacts-fallback" in fallback_step
    assert "python - <<'PY'" not in fallback_step
    assert "static-site-v3" not in fallback_step


def test_static_site_preserves_and_publishes_us_options_history() -> None:
    workflow = (ROOT / ".github" / "workflows" / "static-site.yml").read_text()
    build_job = _build_market_job()
    combine_job = _combine_and_build_job()

    assert "options-analytics-data" in workflow
    assert "OPTIONS_ANALYTICS_ENABLED" in build_job
    assert "matrix.market == 'US'" in build_job
    assert "python -m app.scripts.import_options_history" in build_job
    assert "id: restore-options-history" in build_job
    assert "--allow-missing" not in build_job
    assert "python -m app.scripts.export_options_history" in build_job
    assert "--require-run-id" in build_job
    assert "name: static-options-US" in build_job
    assert "--current-options-dir /tmp/static-options-current" in combine_job
    assert "--fallback-options-dir /tmp/static-options-fallback" in combine_job
    assert "--options-artifacts-dir /tmp/static-options-current" in combine_job
    assert (
        "--fallback-options-artifacts-dir /tmp/static-options-fallback" in combine_job
    )
    publish_history = build_job.split(
        "      - name: Publish US options history\n", 1
    )[1].split("      - name:", 1)[0]
    assert (
        "steps.restore-options-history.outputs.safe_to_publish == 'true'"
        in publish_history
    )


def test_static_site_validation_uses_python_module_not_inline_control_plane() -> None:
    combine_job = _combine_and_build_job()
    validation_step = combine_job.split("      - name: Validate market artifacts\n", 1)[
        1
    ].split(
        "\n      - name: Combine static data bundle",
        1,
    )[0]

    assert "python -m app.scripts.validate_static_market_artifacts" in validation_step
    assert "--current-dir /tmp/static-market-artifacts-current" in validation_step
    assert "--fallback-dir /tmp/static-market-artifacts-fallback" in validation_step
    assert '--selected-markets "${SELECTED_MARKETS}"' in validation_step
    assert "python - <<'PY'" not in validation_step
    assert "snapshot-failure.json" not in validation_step


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
        f"""\
        import json
        import pathlib
        import sys

        downloads_log = pathlib.Path({str(downloads_log)!r})
        args = sys.argv[1:]
        if args[:3] == ["api", "--paginate", "--slurp"] and "actions/workflows/static-site.yml/runs" in args[3]:
            print(json.dumps([{{"workflow_runs": [
                {{"id": 999, "conclusion": "failure", "created_at": "2026-08-05T00:00:00Z"}},
                {{"id": 333, "conclusion": "success", "created_at": "2026-08-05T00:00:00Z"}},
                {{"id": 222, "conclusion": "success", "created_at": "2026-08-04T00:00:00Z"}},
                {{"id": 111, "conclusion": "success", "created_at": "2026-08-03T00:00:00Z"}}
            ]}}]))
        elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/333/artifacts" in args[3]:
            print(json.dumps([{{"artifacts": [
                {{"name": "static-market-diagnostics-CN", "expired": False}},
                {{"name": "static-market-HK", "expired": False}},
                {{"name": "static-market-status-CN", "expired": False}},
                {{"name": "static-market-US", "expired": False}},
                {{"name": "static-market-TW", "expired": False}}
            ]}}]))
        elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/222/artifacts" in args[3]:
            print(json.dumps([{{"artifacts": [
                {{"name": "static-market-US", "expired": False}}
            ]}}]))
        elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/111/artifacts" in args[3]:
            print(json.dumps([{{"artifacts": [
                {{"name": "static-market-US", "expired": False}}
            ]}}]))
        elif args[:2] == ["run", "download"]:
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
        f"""\
        import json
        import pathlib
        import sys

        downloads_log = pathlib.Path({str(downloads_log)!r})
        args = sys.argv[1:]
        if args[:3] == ["api", "--paginate", "--slurp"] and "actions/workflows/static-site.yml/runs" in args[3]:
            print(json.dumps([{{"workflow_runs": [
                {{"id": 999, "created_at": "2026-08-05T00:00:00Z"}},
                {{"id": 333, "created_at": "2026-08-04T00:00:00Z"}},
                {{"id": 222, "created_at": "2026-08-03T00:00:00Z"}}
            ]}}]))
        elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/333/artifacts" in args[3]:
            print(json.dumps([{{"artifacts": [
                {{"name": "static-market-US", "expired": False}}
            ]}}]))
        elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/222/artifacts" in args[3]:
            print(json.dumps([{{"artifacts": [
                {{"name": "static-market-US", "expired": False}}
            ]}}]))
        elif args[:2] == ["run", "download"]:
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
        """\
        import json
        import pathlib
        import sys

        args = sys.argv[1:]
        if args[:3] == ["api", "--paginate", "--slurp"] and "actions/workflows/static-site.yml/runs" in args[3]:
            print(json.dumps([{"workflow_runs": [
                {"id": 999, "created_at": "2026-08-05T00:00:00Z"},
                {"id": 333, "created_at": "2026-08-04T00:00:00Z"}
            ]}]))
        elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/333/artifacts" in args[3]:
            print(json.dumps([{"artifacts": [
                {"name": "static-market-US", "expired": False}
            ]}]))
        elif args[:2] == ["run", "download"]:
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
        f"""\
        import json
        import pathlib
        import sys

        downloads_log = pathlib.Path({str(downloads_log)!r})
        args = sys.argv[1:]
        if args[:3] == ["api", "--paginate", "--slurp"] and "actions/workflows/static-site.yml/runs" in args[3]:
            print(json.dumps([{{"workflow_runs": [
                {{"id": 999}},
                {{"id": 333}},
                {{"id": 222}}
            ]}}]))
        elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/333/artifacts" in args[3]:
            print(json.dumps([{{"artifacts": [
                {{"name": "static-market-AU", "expired": False}}
            ]}}]))
        elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/222/artifacts" in args[3]:
            print(json.dumps([{{"artifacts": [
                {{"name": "static-market-AU", "expired": False}}
            ]}}]))
        elif args[:2] == ["run", "download"]:
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
