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
    inspect_downloaded_market,
)

ROOT = Path(__file__).resolve().parents[3]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "static-site-publish.yml"
SITE_WORKFLOW = ROOT / ".github" / "workflows" / "static-site.yml"


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


def test_options_history_publishes_only_after_a_safe_restore() -> None:
    steps = _site_workflow()["jobs"]["build-market"]["steps"]
    publish = next(step for step in steps if step.get("name") == "Publish US options history")
    assert "steps.restore-options-history.outputs.safe_to_publish == 'true'" in publish["if"]


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
    # A corrected rerun re-exports the same session on the same day. The API
    # order is not a contract, so the listing arrives oldest-first here; the
    # later upload must still win.
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
                "entry": {{
                    "as_of_date": as_of_date_by_run.get(run_id, "2026-08-02"),
                    "rs_formula_version": "balanced-horizon-percentile-v2",
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


@pytest.mark.parametrize(
    ("newer_defect", "reason"),
    [
        (
            "legacy formula",
            "US fallback artifact uses incompatible RS formula: "
            "Scan manifest='legacy-linear-v1', market entry='legacy-linear-v1'; "
            "expected 'balanced-horizon-percentile-v2'",
        ),
        (
            "breadth r2",
            "US fallback artifact uses incompatible breadth revision: revision=2, "
            "source_revision='2026-08-04|breadth-r2'; expected 3 and marker '|breadth-r3'",
        ),
    ],
)
def test_static_site_fallback_downloader_keeps_newest_accepted_candidate(
    tmp_path, newer_defect, reason
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
            date_by_run = {{
                "333": "2026-08-04",
                "222": "2026-08-03",
            }}
            formula = "balanced-horizon-percentile-v2"
            features = {{"scan": True}}
            if run_id == "333" and {newer_defect!r} == "legacy formula":
                formula = "legacy-linear-v1"
            elif run_id == "333":
                features["breadth"] = True
                (market_dir / "breadth.json").write_text(json.dumps({{
                    "source_revision": "2026-08-04|breadth-r2",
                    "payload": {{"current": {{"calculation_revision": 2}}}},
                }}))
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
                    "features": features,
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
    assert (
        f"static-market-US from run 333 was rejected ({reason})." in result.stdout
    ), result.stdout
    manifest = json.loads(
        (
            fallback_dir
            / "static-market-US"
            / "markets"
            / "us"
            / "manifest.market.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["entry"]["rs_formula_version"] == "balanced-horizon-percentile-v2"
    assert manifest["entry"]["as_of_date"] == "2026-08-03"


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
    assert (
        "static-market-US from run 333 was rejected (US fallback artifact "
        "advertises GROUPS but groups.json is absent)." in result.stdout
    ), result.stdout


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
                json.dumps({{
                    "market": "AU",
                    "schema_version": schema_version,
                    "entry": {{
                        "as_of_date": "2026-08-03",
                        "rs_formula_version": "balanced-horizon-percentile-v2",
                    }},
                }})
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

    assert inspect_downloaded_market(
        target_dir,
        market="AU",
        artifact_name="static-market-AU",
        run_id=222,
    ) is None


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

    assert inspect_downloaded_market(
        target_dir,
        market="AU",
        artifact_name="static-market-AU",
        run_id=222,
    ) is None


@pytest.mark.parametrize(
    "relative_path, payload, reason",
    [
        pytest.param(
            "scan/manifest.json",
            {"rs_formula_version": "balanced-horizon-percentile-v2", "chunks": 7},
            "scan manifest 'markets/hk/scan/manifest.json' must list chunks as objects",
            id="chunks-as-number",
        ),
        pytest.param(
            "scan/manifest.json",
            {
                "rs_formula_version": "balanced-horizon-percentile-v2",
                "chunks": ["markets/hk/scan/chunks/chunk-0001.json"],
            },
            "scan manifest 'markets/hk/scan/manifest.json' must list chunks as objects",
            id="chunks-as-strings",
        ),
        pytest.param(
            "scan/manifest.json",
            [],
            "scan manifest 'markets/hk/scan/manifest.json' must list chunks as objects",
            id="manifest-is-list",
        ),
        pytest.param(
            "groups.json", [], "groups.json is not a JSON object", id="groups-is-list"
        ),
    ],
)
def test_static_site_fallback_downloader_rejects_a_malformed_payload(
    tmp_path: Path, capsys, relative_path, payload, reason
) -> None:
    _write_selected_market(tmp_path, "HK", "balanced-horizon-percentile-v2")
    target_dir = tmp_path / "static-market-HK"
    (target_dir / relative_path).write_text(json.dumps(payload), encoding="utf-8")

    assert inspect_downloaded_market(
        target_dir,
        market="HK",
        artifact_name="static-market-HK",
        run_id=333,
    ) is None
    assert capsys.readouterr().out == (
        "::warning::Unable to download fallback market artifact: static-market-HK "
        f"from run 333 was rejected (HK fallback {reason}).\n"
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


def _site_workflow() -> dict:
    return yaml.safe_load(SITE_WORKFLOW.read_text(encoding="utf-8"))


def _publish_step(name: str) -> dict:
    steps = _publish_workflow()["jobs"]["build"]["steps"]
    return next(step for step in steps if step.get("name") == name)


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


def test_publisher_reads_only_default_branch_artifacts() -> None:
    step = _publish_step("Download newest market artifacts")
    assert step["env"]["BRANCH_NAME"] == "${{ github.event.repository.default_branch }}"


def test_publisher_deploys_only_from_default_branch_dispatch() -> None:
    deploy = _publish_workflow()["jobs"]["deploy"]
    assert PRODUCTION_RUN in deploy["if"]
    assert deploy["environment"]["name"] == "github-pages"


def _step_names(steps: list[dict]) -> list[str | None]:
    return [step.get("name") for step in steps]


def test_record_ledger_runs_only_after_a_production_deploy() -> None:
    jobs = _publish_workflow()["jobs"]
    job = jobs["record-ledger"]
    assert job["if"] == jobs["deploy"]["if"]
    assert PRODUCTION_RUN in job["if"]
    assert job["needs"] == ["build", "deploy"]
    assert job["permissions"] == {"contents": "write"}


def test_record_ledger_installs_nothing_and_runs_only_the_ledger_script() -> None:
    steps = _publish_workflow()["jobs"]["record-ledger"]["steps"]
    assert [step["uses"].split("@")[0] for step in steps if "uses" in step] == [
        "actions/checkout",
        "actions/download-artifact",
    ]
    assert steps[0]["with"]["persist-credentials"] is False
    assert [step["run"].split() for step in steps if "run" in step] == [
        [
            "cd",
            "backend",
            "python3",
            "-m",
            "app.scripts.scoreboard_ledger",
            "record",
            "--days-dir",
            "/tmp/scoreboard-days",
        ]
    ]


def test_record_ledger_reads_the_artifact_the_build_uploads() -> None:
    upload = _publish_step("Upload published screen matches")
    steps = _publish_workflow()["jobs"]["record-ledger"]["steps"]
    download = next(step for step in steps if step.get("name") == "Download published screen matches")
    assert download["with"]["name"] == upload["with"]["name"] == "scoreboard-days"
    assert download["with"]["path"] == upload["with"]["path"]


def test_extraction_reads_the_built_site_and_never_blocks_the_deploy() -> None:
    names = _step_names(_publish_workflow()["jobs"]["build"]["steps"])
    extract = names.index("Extract published screen matches")
    upload = names.index("Upload published screen matches")
    assert names.index("Build static frontend") < extract < upload < names.index("Upload Pages artifact")
    assert _publish_step("Extract published screen matches")["continue-on-error"] is True
    assert _publish_step("Upload published screen matches")["continue-on-error"] is True
    assert "--site-dir ../frontend/dist/static-data" in _publish_step("Extract published screen matches")["run"]


def test_publisher_rehearses_a_ledger_script_change() -> None:
    # PyYAML reads the bare ``on`` key as True.
    paths = _publish_workflow()[True]["pull_request"]["paths"]
    assert "backend/app/scripts/scoreboard_ledger.py" in paths


def _market_step(name: str) -> dict:
    steps = _site_workflow()["jobs"]["build-market"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def test_market_scoreboard_runs_between_export_and_upload_and_cannot_lose_the_market() -> None:
    names = _step_names(_site_workflow()["jobs"]["build-market"]["steps"])
    download = names.index("Download scoreboard ledger")
    build = names.index("Build market scoreboard")
    assert names.index("Export market static data bundle") < download < build < names.index("Upload market artifact")

    has_artifact = "steps.export-market.outputs.has_artifact == 'true'"
    for step in (_market_step("Download scoreboard ledger"), _market_step("Build market scoreboard")):
        assert has_artifact in step["if"]
        assert step["continue-on-error"] is True
    scoreboard = _market_step("Build market scoreboard")
    assert scoreboard["timeout-minutes"] == 20
    assert "steps.download-scoreboard-ledger.outcome == 'success'" in scoreboard["if"]


def test_market_scoreboard_arguments_name_the_uploaded_market_directory() -> None:
    run = _market_step("Build market scoreboard")["run"]
    assert _workflow_script_args(run, "app.scripts.build_static_scoreboard") == [
        "--market",
        "${{ matrix.market }}",
        "--market-dir",
        "/tmp/static-data/markets/${MARKET_LOWER}",
        "--ledger-dir",
        "/tmp/scoreboard-ledger",
    ]
    assert _market_step("Upload market artifact")["with"]["path"] == "/tmp/static-data/markets/${{ env.MARKET_LOWER }}"


def test_static_site_workflow_never_writes_the_ledger_release() -> None:
    # The market job runs from any branch on dispatch, so only download may appear.
    ledger_steps = [
        step
        for job in _site_workflow()["jobs"].values()
        for step in job["steps"]
        if "scoreboard-ledger-data" in step.get("run", "") or "scoreboard_ledger" in step.get("run", "")
    ]
    assert [step["name"] for step in ledger_steps] == ["Download scoreboard ledger"]
    assert "gh release download scoreboard-ledger-data" in ledger_steps[0]["run"]
    assert "upload" not in ledger_steps[0]["run"]
    workflow_text = SITE_WORKFLOW.read_text(encoding="utf-8")
    for write in ("upload scoreboard-ledger-data", "create scoreboard-ledger-data", "scoreboard_ledger record"):
        assert write not in workflow_text


def test_every_upload_path_wakes_the_publisher() -> None:
    # The publisher has no schedule; these dispatches are its only trigger.
    jobs = _site_workflow()["jobs"]
    wake = "gh workflow run static-site-publish.yml"
    for job, upload in (
        ("build-cot", "Upload current global COT artifact"),
        ("build-market", "Upload market artifact"),
    ):
        names = [step.get("name") for step in jobs[job]["steps"]]
        assert names.index("Wake static-site publisher") > names.index(upload), job
        assert wake in jobs[job]["steps"][names.index("Wake static-site publisher")]["run"], job
    assert "always()" in jobs["build-market"]["steps"][names.index("Wake static-site publisher")]["if"]
    assert "always()" in jobs["wake-publisher"]["if"]
    assert wake in jobs["wake-publisher"]["steps"][0]["run"]


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


def test_publisher_combine_arguments_reject_a_stale_formula_artifact_with_a_reason(
    tmp_path, monkeypatch
) -> None:
    """Run the publisher's own combine arguments end to end.

    The downloader only hands over balanced artifacts, so a stale one here
    means every stored artifact for that market predates the cutover. An
    optional market is then left out with the reason in the manifest
    warnings; a required one fails the publish with the same reason.
    """
    from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
    from app.scripts import export_static_site as export_script

    selection = tmp_path / "selection"
    _write_selected_market(selection, "US", BALANCED_RS_FORMULA_VERSION)
    _write_selected_market(selection, "HK", "legacy-linear-v1")
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
    # The step's default (STATIC_DATA_GENERATIONS unset) passes --data-generation.
    args = [
        "--data-generation" if arg == "$GENERATION_ARGS" else str(paths[arg]) if arg in paths else arg
        for arg in args
    ]
    monkeypatch.setattr(sys, "argv", ["export_static_site.py", *args])

    assert export_script.main() == 0

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["supported_markets"] == ["US"]
    assert "HK" in manifest["unavailable_markets"]
    assert [w for w in manifest["warnings"] if w.startswith("Ignored static market")] == [
        "Ignored static market artifact: HK fallback artifact uses incompatible "
        "RS formula: Scan chunk chunk-0001.json='legacy-linear-v1', "
        "Scan manifest='legacy-linear-v1', market entry='legacy-linear-v1'; "
        f"expected {BALANCED_RS_FORMULA_VERSION!r}"
    ]


@pytest.mark.parametrize(
    ("repair_market", "group", "accepted"),
    [
        ("AU", "asia", True),
        ("au", "asia", True),
        ("AUS", "asia", False),
        ("DE", "asia", False),
        ("A.", "asia", False),
        ("", "us", True),
    ],
)
def test_repair_market_must_belong_to_the_selected_group(repair_market, group, accepted) -> None:
    step = next(
        step
        for step in _site_workflow()["jobs"]["select-markets"]["steps"]
        if step.get("id") == "pick"
    )
    env = {
        **os.environ,
        "EVENT_NAME": "workflow_dispatch",
        "SCHEDULE": "",
        "MARKET_GROUP": group,
        "REPAIR_PRICE_HISTORY_MARKET": repair_market,
        "GITHUB_OUTPUT": os.devnull,
    }
    result = subprocess.run(["bash", "-e", "-c", step["run"]], env=env, capture_output=True)
    assert (result.returncode == 0) is accepted


def test_search_console_verification_file_ships_with_the_site() -> None:
    # Google re-checks this file to keep the Search Console property verified;
    # Vite copies frontend/public/ to the site root.
    token = ROOT / "frontend" / "public" / "google2f3dbee9a723186e.html"
    assert token.read_text().strip() == "google-site-verification: google2f3dbee9a723186e.html"
