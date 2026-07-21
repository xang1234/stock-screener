from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[3]


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


def _fallback_download_script() -> str:
    step = _combine_and_build_job().split("      - name: Download per-market fallback artifacts\n", 1)[1].split(
        "\n      - name: Validate market artifacts",
        1,
    )[0]
    run = step.split("          python - <<'PY'\n", 1)[1].rsplit("\n          PY", 1)[0]
    return textwrap.dedent(run)


def test_static_site_market_build_failures_are_not_marked_continue_on_error() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split("      - name: Export market static data bundle\n", 1)[1].split(
        "\n      - name: Upload market status",
        1,
    )[0]

    assert "continue-on-error: true" not in export_step


def test_static_site_daily_price_seed_allows_stale_bootstrap() -> None:
    build_market_job = _build_market_job()
    seed_step = build_market_job.split("      - name: Seed daily price bundle from GitHub\n", 1)[1].split(
        "\n      - name: Export market static data bundle",
        1,
    )[0]

    assert "--allow-stale" in seed_step


def test_static_site_market_export_skips_artifact_steps_for_closed_market() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split("      - name: Export market static data bundle\n", 1)[1].split(
        "\n      - name: Build daily price bundle",
        1,
    )[0]
    build_price_step = build_market_job.split("      - name: Build daily price bundle\n", 1)[1].split(
        "\n      - name: Upload daily price assets",
        1,
    )[0]
    upload_price_step = build_market_job.split("      - name: Upload daily price assets\n", 1)[1].split(
        "\n      - name: Upload market artifact",
        1,
    )[0]
    upload_market_step = build_market_job.split("      - name: Upload market artifact\n", 1)[1].split(
        "\n\n  combine-and-build:",
        1,
    )[0]

    assert "id: export-market" in export_step
    assert "status=$?" in export_step
    assert 'if [ "$status" -eq 78 ]; then' in export_step
    assert "has_artifact=false" in export_step
    assert "has_artifact=true" in export_step
    assert "steps.export-market.outputs.has_artifact == 'true'" in build_price_step
    assert "steps.export-market.outputs.has_artifact == 'true'" in upload_price_step
    assert "steps.export-market.outputs.has_artifact == 'true'" in upload_market_step


def test_static_site_market_export_soft_skips_no_current_artifact_exit_code() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split("      - name: Export market static data bundle\n", 1)[1].split(
        "\n      - name: Upload market status",
        1,
    )[0]

    assert 'if [ "$status" -eq 79 ]; then' in export_step
    assert "has_artifact=false" in export_step
    assert "fallback artifacts" in export_step
    assert "no current market artifact will be uploaded" in export_step


def test_static_site_uploads_canonical_market_status_after_export() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split("      - name: Export market static data bundle\n", 1)[1].split(
        "\n      - name: Upload market status",
        1,
    )[0]
    status_step = build_market_job.split("      - name: Upload market status\n", 1)[1].split(
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
    assert "path: /tmp/static-data/status/${{ env.MARKET_LOWER }}/status.json" in status_step
    assert "if-no-files-found: error" in status_step


def test_static_site_uploads_market_diagnostics_after_export() -> None:
    build_market_job = _build_market_job()
    diagnostics_step = build_market_job.split("      - name: Upload market diagnostics\n", 1)[1].split(
        "\n      - name: Build daily price bundle",
        1,
    )[0]

    assert "if: ${{ always() }}" in diagnostics_step
    assert "uses: actions/upload-artifact@v4" in diagnostics_step
    assert "name: static-market-diagnostics-${{ matrix.market }}" in diagnostics_step
    assert "path: /tmp/static-data/diagnostics/${{ env.MARKET_LOWER }}" in diagnostics_step
    assert "if-no-files-found: ignore" in diagnostics_step


def test_static_site_combine_downloads_current_and_per_market_fallback_artifacts() -> None:
    combine_job = _combine_and_build_job()

    assert "needs: [select-markets, build-market]" in combine_job
    assert "needs.select-markets.outputs.markets" in combine_job
    assert "Download per-market fallback artifacts" in combine_job
    assert "Download current market artifacts" in combine_job
    assert "/tmp/static-market-artifacts-current" in combine_job
    assert "/tmp/static-market-artifacts-fallback" in combine_job
    assert "--fallback-artifacts-dir /tmp/static-market-artifacts-fallback" in combine_job
    assert "FALLBACK_MARKETS" not in combine_job
    assert "github.ref_name" in combine_job
    assert "--paginate" in combine_job
    assert "runs = extract_runs(pages)" in combine_job
    assert "market_from_artifact_name" in combine_job
    assert '"gh",' in combine_job
    assert '"run",' in combine_job
    assert '"download",' in combine_job
    assert "actions/runs/{run_id}/artifacts" in combine_job
    assert "artifact.get(\"expired\")" in combine_job
    assert "current_markets" in combine_job
    assert "command_error_detail" in combine_job
    assert "exc.stderr" in combine_job
    assert "run.get(\"conclusion\") == \"success\"" not in combine_job
    assert "subprocess.CalledProcessError" in combine_job
    assert "::warning::Unable to download fallback market artifact" in combine_job
    assert "isinstance(payload, dict)" in combine_job
    assert "isinstance(page, dict)" in combine_job
    assert "Unexpected GitHub API response shape" in combine_job


def test_static_site_validation_uses_python_module_not_inline_control_plane() -> None:
    combine_job = _combine_and_build_job()
    validation_step = combine_job.split("      - name: Validate market artifacts\n", 1)[1].split(
        "\n      - name: Combine static data bundle",
        1,
    )[0]

    assert "python -m app.scripts.validate_static_market_artifacts" in validation_step
    assert "--current-dir /tmp/static-market-artifacts-current" in validation_step
    assert "--fallback-dir /tmp/static-market-artifacts-fallback" in validation_step
    assert '--selected-markets "${SELECTED_MARKETS}"' in validation_step
    assert "python - <<'PY'" not in validation_step
    assert "snapshot-failure.json" not in validation_step


def test_static_site_fallback_downloader_only_fetches_missing_current_markets(tmp_path) -> None:
    current_dir = Path("/tmp/static-market-artifacts-current")
    fallback_dir = Path("/tmp/static-market-artifacts-fallback")
    shutil.rmtree(current_dir, ignore_errors=True)
    shutil.rmtree(fallback_dir, ignore_errors=True)
    current_us_dir = current_dir / "static-market-US" / "markets" / "us"
    current_us_dir.mkdir(parents=True)
    (current_us_dir / "manifest.market.json").write_text(
        json.dumps({"market": "US"}),
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    downloads_log = tmp_path / "downloads.jsonl"
    fake_gh.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json
            import pathlib
            import sys

            downloads_log = pathlib.Path({str(downloads_log)!r})
            args = sys.argv[1:]
            if args[:3] == ["api", "--paginate", "--slurp"] and "actions/workflows/static-site.yml/runs" in args[3]:
                print(json.dumps([{{"workflow_runs": [
                    {{"id": 999, "conclusion": "failure"}},
                    {{"id": 222, "conclusion": "failure"}}
                ]}}]))
            elif args[:3] == ["api", "--paginate", "--slurp"] and "actions/runs/222/artifacts" in args[3]:
                print(json.dumps([{{"artifacts": [
                    {{"name": "static-market-diagnostics-CN", "expired": False}},
                    {{"name": "static-market-HK", "expired": False}},
                    {{"name": "static-market-status-CN", "expired": False}},
                    {{"name": "static-market-US", "expired": False}},
                    {{"name": "static-market-TW", "expired": False}}
                ]}}]))
            elif args[:2] == ["run", "download"]:
                artifact_name = args[args.index("--name") + 1]
                if artifact_name == "static-market-HK":
                    print("download denied for HK", file=sys.stderr)
                    sys.exit(7)
                target_dir = pathlib.Path(args[args.index("--dir") + 1])
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "manifest.market.json").write_text(json.dumps({{"market": artifact_name.rsplit("-", 1)[1], "schema_version": "static-site-v3"}}))
                with downloads_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({{"artifact": artifact_name}}) + "\\n")
            else:
                print(f"unexpected gh args: {{args}}", file=sys.stderr)
                sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "REPOSITORY": "xang1234/stock-screener",
            "CURRENT_RUN_ID": "999",
            "BRANCH_NAME": "main",
        }
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", _fallback_download_script()],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        downloads = [
            json.loads(line)
            for line in downloads_log.read_text(encoding="utf-8").splitlines()
        ]
        assert downloads == [{"artifact": "static-market-TW"}]
        assert not (fallback_dir / "static-market-diagnostics-CN").exists()
        assert not (fallback_dir / "static-market-status-CN").exists()
        assert not (fallback_dir / "static-market-US").exists()
        assert not (fallback_dir / "static-market-HK").exists()
        assert (fallback_dir / "static-market-TW" / "manifest.market.json").exists()
        assert "exit 7. Details: stderr: download denied for HK" in result.stdout
    finally:
        shutil.rmtree(current_dir, ignore_errors=True)
        shutil.rmtree(fallback_dir, ignore_errors=True)


def test_static_site_fallback_downloader_skips_incompatible_schema_and_keeps_searching(tmp_path) -> None:
    current_dir = Path("/tmp/static-market-artifacts-current")
    fallback_dir = Path("/tmp/static-market-artifacts-fallback")
    shutil.rmtree(current_dir, ignore_errors=True)
    shutil.rmtree(fallback_dir, ignore_errors=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    downloads_log = tmp_path / "downloads.jsonl"
    fake_gh.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
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
            """
        ),
        encoding="utf-8",
    )
    fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "REPOSITORY": "xang1234/stock-screener",
            "CURRENT_RUN_ID": "999",
            "BRANCH_NAME": "main",
        }
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", _fallback_download_script()],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

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
    finally:
        shutil.rmtree(current_dir, ignore_errors=True)
        shutil.rmtree(fallback_dir, ignore_errors=True)
