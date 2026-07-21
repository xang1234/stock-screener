"""Download compatible per-market fallback artifacts for the static-site build."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlencode

from app.services.static_market_artifact_contract import (
    STATIC_MARKET_METADATA_FILENAME,
    StaticMarketArtifactContractError,
    expected_market_from_static_market_manifest_path,
    market_from_static_market_artifact_name,
    read_static_market_manifest,
)


def warn(message: str) -> None:
    print(
        f"::warning::Unable to download fallback market artifact: {message}",
        flush=True,
    )


def command_error_detail(exc: subprocess.CalledProcessError, limit: int = 800) -> str:
    details = []
    for stream_name, stream_value in (("stderr", exc.stderr), ("stdout", exc.stdout)):
        text = (stream_value or "").strip()
        if not text:
            continue
        text = " | ".join(text.splitlines())
        if len(text) > limit:
            text = f"{text[:limit]}..."
        details.append(f"{stream_name}: {text}")
    return f" Details: {'; '.join(details)}" if details else ""


def extract_runs(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        workflow_runs = payload.get("workflow_runs")
        if not isinstance(workflow_runs, list):
            raise ValueError(
                "Unexpected GitHub API response shape: workflow_runs is not a list."
            )
        return [run for run in workflow_runs if isinstance(run, dict)]

    if isinstance(payload, list):
        runs = []
        for page in payload:
            if not isinstance(page, dict):
                raise ValueError(
                    "Unexpected GitHub API response shape: page is not an object."
                )
            workflow_runs = page.get("workflow_runs", [])
            if not isinstance(workflow_runs, list):
                raise ValueError(
                    "Unexpected GitHub API response shape: workflow_runs is not a list."
                )
            runs.extend(run for run in workflow_runs if isinstance(run, dict))
        return runs

    raise ValueError(
        "Unexpected GitHub API response shape: response is not an object or list."
    )


def extract_artifacts(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(
                "Unexpected GitHub API response shape: artifacts is not a list."
            )
        return [artifact for artifact in artifacts if isinstance(artifact, dict)]

    if isinstance(payload, list):
        artifacts = []
        for page in payload:
            if not isinstance(page, dict):
                raise ValueError(
                    "Unexpected GitHub API response shape: page is not an object."
                )
            page_artifacts = page.get("artifacts", [])
            if not isinstance(page_artifacts, list):
                raise ValueError(
                    "Unexpected GitHub API response shape: artifacts is not a list."
                )
            artifacts.extend(
                artifact for artifact in page_artifacts if isinstance(artifact, dict)
            )
        return artifacts

    raise ValueError(
        "Unexpected GitHub API response shape: response is not an object or list."
    )


def gh_json(args: Sequence[str]) -> Any:
    result = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def collect_current_markets(current_dir: Path) -> set[str]:
    current_markets = set()
    metadata_paths = (
        sorted(current_dir.rglob(STATIC_MARKET_METADATA_FILENAME))
        if current_dir.exists()
        else []
    )
    for metadata_path in metadata_paths:
        try:
            expected_market = expected_market_from_static_market_manifest_path(
                current_dir,
                metadata_path,
            )
            payload = read_static_market_manifest(
                metadata_path,
                expected_market=expected_market,
            )
            market = str(payload.get("market", "")).upper()
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            StaticMarketArtifactContractError,
        ) as exc:
            warn(f"Current artifact metadata at {metadata_path} could not be read ({exc}).")
            continue
        if market:
            current_markets.add(market)
    return current_markets


def downloaded_market_is_compatible(
    target_dir: Path,
    *,
    market: str,
    artifact_name: str,
    run_id: int,
) -> bool:
    metadata_paths = sorted(target_dir.rglob(STATIC_MARKET_METADATA_FILENAME))
    if not metadata_paths:
        warn(f"{artifact_name} from run {run_id} has no {STATIC_MARKET_METADATA_FILENAME}.")
        return False
    if len(metadata_paths) != 1:
        warn(
            f"{artifact_name} from run {run_id} has multiple "
            f"{STATIC_MARKET_METADATA_FILENAME} files."
        )
        return False

    metadata_path = metadata_paths[0]
    try:
        read_static_market_manifest(metadata_path, expected_market=market)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        warn(
            f"{artifact_name} metadata at {metadata_path} could not be read ({exc})."
        )
        return False
    except StaticMarketArtifactContractError as exc:
        warn(str(exc))
        return False
    return True


def download_fallback_artifacts(
    *,
    repo: str,
    current_run_id: int,
    branch_name: str,
    current_dir: Path,
    fallback_dir: Path,
) -> set[str]:
    fallback_dir.mkdir(parents=True, exist_ok=True)
    query = urlencode(
        {
            "branch": branch_name,
            "status": "completed",
            "per_page": "100",
        }
    )

    try:
        pages = gh_json(
            [
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repo}/actions/workflows/static-site.yml/runs?{query}",
            ]
        )
        runs = extract_runs(pages)
    except subprocess.CalledProcessError as exc:
        warn(
            "GitHub workflow runs API request failed "
            f"with exit {exc.returncode}.{command_error_detail(exc)}"
        )
        runs = []
    except json.JSONDecodeError as exc:
        warn(f"GitHub workflow runs API response was not valid JSON ({exc}).")
        runs = []
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        warn(str(exc))
        runs = []

    current_markets = collect_current_markets(current_dir)
    fallback_markets: set[str] = set()
    if current_markets:
        print(
            f"Current run already has market artifacts: {', '.join(sorted(current_markets))}.",
            flush=True,
        )

    for run in runs:
        run_id = run.get("id")
        if run_id == current_run_id:
            continue
        try:
            artifact_pages = gh_json(
                [
                    "api",
                    "--paginate",
                    "--slurp",
                    f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
                ]
            )
            artifacts = extract_artifacts(artifact_pages)
        except subprocess.CalledProcessError as exc:
            warn(
                f"Artifact list API request for run {run_id} failed "
                f"with exit {exc.returncode}.{command_error_detail(exc)}"
            )
            continue
        except json.JSONDecodeError as exc:
            warn(f"Artifact list API response for run {run_id} was not valid JSON ({exc}).")
            continue
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            warn(f"Artifact list API response for run {run_id} was invalid: {exc}")
            continue

        artifacts_by_name = {
            str(artifact.get("name")): artifact
            for artifact in artifacts
            if not artifact.get("expired")
        }

        for artifact_name in sorted(artifacts_by_name):
            market = market_from_static_market_artifact_name(artifact_name)
            if not market:
                continue
            if market in current_markets or market in fallback_markets:
                continue

            target_dir = fallback_dir / artifact_name
            shutil.rmtree(target_dir, ignore_errors=True)

            try:
                subprocess.run(
                    [
                        "gh",
                        "run",
                        "download",
                        str(run_id),
                        "--repo",
                        repo,
                        "--name",
                        artifact_name,
                        "--dir",
                        str(target_dir),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                warn(
                    f"{artifact_name} from run {run_id} failed to download "
                    f"with exit {exc.returncode}.{command_error_detail(exc)}"
                )
                shutil.rmtree(target_dir, ignore_errors=True)
                continue

            if not downloaded_market_is_compatible(
                target_dir,
                market=market,
                artifact_name=artifact_name,
                run_id=int(run_id),
            ):
                shutil.rmtree(target_dir, ignore_errors=True)
                continue

            fallback_markets.add(market)
            print(
                f"Using fallback artifact {artifact_name} from Static Site run {run_id} "
                f"on {branch_name}.",
                flush=True,
            )

    if fallback_markets:
        print(
            f"Downloaded fallback market artifacts: {', '.join(sorted(fallback_markets))}.",
            flush=True,
        )
    else:
        print(f"No fallback market artifacts found on {branch_name}.")
    return fallback_markets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--fallback-dir", type=Path, required=True)
    parser.add_argument("--repo", default=os.environ.get("REPOSITORY", ""))
    parser.add_argument("--current-run-id", default=os.environ.get("CURRENT_RUN_ID", "0"))
    parser.add_argument("--branch", default=os.environ.get("BRANCH_NAME", "main"))
    args = parser.parse_args(argv)

    if not args.repo:
        raise SystemExit("REPOSITORY is required.")

    download_fallback_artifacts(
        repo=args.repo,
        current_run_id=int(args.current_run_id),
        branch_name=args.branch,
        current_dir=args.current_dir,
        fallback_dir=args.fallback_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
