"""Download compatible per-market fallback artifacts for the static-site build."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from app.services.static_artifact_combiner import (
    StaticArtifactCombiner,
    StaticArtifactFormulaError,
)
from app.services.static_market_artifact_contract import (
    STATIC_MARKET_METADATA_FILENAME,
    StaticMarketArtifactContractError,
    expected_market_from_static_market_manifest_path,
    market_from_static_market_artifact_name,
    read_static_market_manifest,
)
from app.services.static_options_contract import (
    StaticOptionsArtifactError,
    validate_static_options_artifact,
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
            warn(
                f"Current artifact metadata at {metadata_path} could not be read ({exc})."
            )
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
        warn(
            f"{artifact_name} from run {run_id} has no {STATIC_MARKET_METADATA_FILENAME}."
        )
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
        warn(f"{artifact_name} metadata at {metadata_path} could not be read ({exc}).")
        return False
    except StaticMarketArtifactContractError as exc:
        warn(str(exc))
        return False
    return True


def downloaded_market_has_advertised_assets(
    target_dir: Path,
    *,
    market: str,
    artifact_name: str,
    run_id: int,
) -> bool:
    metadata_paths = sorted(target_dir.rglob(STATIC_MARKET_METADATA_FILENAME))
    if len(metadata_paths) != 1:
        warn(
            f"{artifact_name} from run {run_id} has no unique "
            f"{STATIC_MARKET_METADATA_FILENAME} for asset validation."
        )
        return False

    metadata_path = metadata_paths[0]
    try:
        metadata = read_static_market_manifest(metadata_path, expected_market=market)
        entry = metadata.get("entry")
        if not isinstance(entry, dict):
            return True
        StaticArtifactCombiner._validate_advertised_assets(
            market=market,
            source_label="fallback",
            entry=entry,
            market_dir=metadata_path.parent,
        )
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        StaticMarketArtifactContractError,
        StaticArtifactFormulaError,
    ) as exc:
        warn(
            f"{artifact_name} from run {run_id} has invalid advertised assets ({exc})."
        )
        return False
    return True


def downloaded_market_matches_required_formula(
    target_dir: Path,
    *,
    market: str,
    artifact_name: str,
    run_id: int,
    required_formula_by_market: Mapping[str, str],
) -> bool:
    expected_formula = required_formula_by_market.get(market)
    if expected_formula is None:
        return True
    metadata_paths = sorted(target_dir.rglob(STATIC_MARKET_METADATA_FILENAME))
    if len(metadata_paths) != 1:
        warn(
            f"{artifact_name} from run {run_id} has no unique "
            f"{STATIC_MARKET_METADATA_FILENAME} for formula validation."
        )
        return False
    metadata_path = metadata_paths[0]
    try:
        metadata = read_static_market_manifest(metadata_path, expected_market=market)
        StaticArtifactCombiner._validate_formula(
            market=market,
            source_label="fallback",
            metadata=metadata,
            market_dir=metadata_path.parent,
            expected_formula=expected_formula,
        )
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        RuntimeError,
        StaticMarketArtifactContractError,
        StaticArtifactFormulaError,
    ) as exc:
        warn(
            f"{artifact_name} from run {run_id} does not match the requested "
            f"RS formula {expected_formula!r} for {market} ({exc})."
        )
        return False
    return True


def _coerce_manifest_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text.split("T", 1)[0])
    except ValueError:
        return None


def downloaded_market_as_of_date(target_dir: Path) -> date | None:
    metadata_paths = sorted(target_dir.rglob(STATIC_MARKET_METADATA_FILENAME))
    if len(metadata_paths) != 1:
        return None
    try:
        metadata = read_static_market_manifest(metadata_paths[0])
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        StaticMarketArtifactContractError,
    ):
        return None
    entry = metadata.get("entry")
    value = entry.get("as_of_date") if isinstance(entry, dict) else None
    return _coerce_manifest_date(value)


def find_options_artifact_dir(base: Path) -> Path | None:
    candidates = [base]
    if base.exists():
        candidates.extend(path.parent for path in base.rglob("manifest.json"))
    for candidate in candidates:
        try:
            validate_static_options_artifact(candidate)
        except StaticOptionsArtifactError:
            continue
        return candidate
    return None


def downloaded_options_as_of_date(target_dir: Path) -> date | None:
    options_dir = find_options_artifact_dir(target_dir)
    if options_dir is None:
        return None
    try:
        manifest = validate_static_options_artifact(options_dir)
    except StaticOptionsArtifactError:
        return None
    return _coerce_manifest_date(manifest.get("source_as_of_date"))


def _candidate_is_newer(
    candidate_date: date | None,
    incumbent_date: date | None,
) -> bool:
    return candidate_date is not None and (
        incumbent_date is None or candidate_date > incumbent_date
    )


def _workflow_run_upper_bound_date(run: dict[str, Any]) -> date | None:
    for key in ("run_started_at", "created_at", "updated_at"):
        run_date = _coerce_manifest_date(run.get(key))
        if run_date is not None:
            return run_date
    return None


def _run_cannot_beat_incumbent(
    *,
    run_upper_bound: date | None,
    incumbent_date: date | None,
) -> bool:
    return (
        run_upper_bound is not None
        and incumbent_date is not None
        and run_upper_bound + timedelta(days=1) <= incumbent_date
    )


def _install_market_candidate(
    *,
    target_dir: Path,
    candidate_dir: Path,
) -> None:
    from app.services.atomic_directory_publisher import AtomicDirectoryPublisher

    AtomicDirectoryPublisher().publish(
        target_dir,
        lambda stage: shutil.copytree(candidate_dir, stage, dirs_exist_ok=True),
    )
    shutil.rmtree(candidate_dir)


@dataclass(frozen=True)
class _DownloadedCandidate:
    wrapper_dir: Path
    artifact_dir: Path
    as_of_date: date | None


def _download_candidate(
    *,
    repo: str,
    run_id: int,
    artifact_name: str,
    parent_dir: Path,
    finder: Callable[[Path], Path | None],
    date_reader: Callable[[Path], date | None],
    missing_warning: str | None = None,
) -> _DownloadedCandidate | None:
    wrapper = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_name}.candidate-{run_id}-",
            dir=parent_dir,
        )
    )
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
                str(wrapper),
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
        shutil.rmtree(wrapper, ignore_errors=True)
        return None

    artifact_dir = finder(wrapper)
    if artifact_dir is None:
        if missing_warning:
            warn(missing_warning)
        shutil.rmtree(wrapper, ignore_errors=True)
        return None
    return _DownloadedCandidate(
        wrapper_dir=wrapper,
        artifact_dir=artifact_dir,
        as_of_date=date_reader(wrapper),
    )


def _find_compatible_market_candidate(
    wrapper: Path,
    *,
    market: str,
    artifact_name: str,
    run_id: int,
    formula_requirements: Mapping[str, str],
) -> Path | None:
    if not downloaded_market_is_compatible(
        wrapper,
        market=market,
        artifact_name=artifact_name,
        run_id=run_id,
    ):
        return None
    if not downloaded_market_has_advertised_assets(
        wrapper,
        market=market,
        artifact_name=artifact_name,
        run_id=run_id,
    ):
        return None
    if not downloaded_market_matches_required_formula(
        wrapper,
        market=market,
        artifact_name=artifact_name,
        run_id=run_id,
        required_formula_by_market=formula_requirements,
    ):
        return None
    return wrapper


def download_fallback_artifacts(
    *,
    repo: str,
    current_run_id: int,
    branch_name: str,
    current_dir: Path,
    fallback_dir: Path,
    required_formula_by_market: Mapping[str, str] | None = None,
    current_options_dir: Path | None = None,
    fallback_options_dir: Path | None = None,
) -> set[str]:
    fallback_dir.mkdir(parents=True, exist_ok=True)
    formula_requirements = {
        str(market).strip().upper(): str(formula).strip()
        for market, formula in (required_formula_by_market or {}).items()
        if str(market).strip() and str(formula).strip()
    }
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
    fallback_dates_by_market: dict[str, date | None] = {}
    fallback_options_date: date | None = None
    if (
        current_options_dir is not None
        and find_options_artifact_dir(current_options_dir) is not None
    ):
        print("Current run already has a compatible static US options artifact.")
    if current_markets:
        print(
            f"Current run already has market artifacts: {', '.join(sorted(current_markets))}.",
            flush=True,
        )

    for run in runs:
        run_id = run.get("id")
        if run_id == current_run_id:
            continue
        run_upper_bound = _workflow_run_upper_bound_date(run)
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
            warn(
                f"Artifact list API response for run {run_id} was not valid JSON ({exc})."
            )
            continue
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            warn(f"Artifact list API response for run {run_id} was invalid: {exc}")
            continue

        artifacts_by_name = {
            str(artifact.get("name")): artifact
            for artifact in artifacts
            if not artifact.get("expired")
        }

        options_artifact = artifacts_by_name.get("static-options-US")
        if options_artifact is not None and fallback_options_dir is not None:
            artifact_name = "static-options-US"
            candidate = _download_candidate(
                repo=repo,
                run_id=int(run_id),
                artifact_name=artifact_name,
                parent_dir=fallback_dir,
                finder=find_options_artifact_dir,
                date_reader=downloaded_options_as_of_date,
                missing_warning=(
                    f"{artifact_name} from run {run_id} is not a compatible "
                    "static options artifact."
                ),
            )
            if candidate is not None:
                if _candidate_is_newer(candidate.as_of_date, fallback_options_date):
                    fallback_options_dir.parent.mkdir(parents=True, exist_ok=True)
                    _install_market_candidate(
                        target_dir=fallback_options_dir,
                        candidate_dir=candidate.artifact_dir,
                    )
                    fallback_options_date = candidate.as_of_date
                    print(
                        f"Using fallback artifact {artifact_name} from Static Site "
                        f"run {run_id} on {branch_name}.",
                        flush=True,
                    )
                shutil.rmtree(candidate.wrapper_dir, ignore_errors=True)

        for artifact_name in sorted(artifacts_by_name):
            market = market_from_static_market_artifact_name(artifact_name)
            if not market:
                continue
            # Download fallback artifacts for current markets too; the combiner
            # compares dates and keeps a newer last-known-good artifact when a
            # cache-only current run had to rewind.
            if market in fallback_markets and _run_cannot_beat_incumbent(
                run_upper_bound=run_upper_bound,
                incumbent_date=fallback_dates_by_market.get(market),
            ):
                continue
            target_dir = fallback_dir / artifact_name
            candidate = _download_candidate(
                repo=repo,
                run_id=int(run_id),
                artifact_name=artifact_name,
                parent_dir=fallback_dir,
                finder=partial(
                    _find_compatible_market_candidate,
                    market=market,
                    artifact_name=artifact_name,
                    run_id=int(run_id),
                    formula_requirements=formula_requirements,
                ),
                date_reader=downloaded_market_as_of_date,
            )
            if candidate is None:
                continue

            candidate_date = candidate.as_of_date
            if market in fallback_markets and not _candidate_is_newer(
                candidate_date,
                fallback_dates_by_market.get(market),
            ):
                shutil.rmtree(candidate.wrapper_dir, ignore_errors=True)
                continue

            try:
                _install_market_candidate(
                    target_dir=target_dir,
                    candidate_dir=candidate.artifact_dir,
                )
            except OSError as exc:
                warn(
                    f"{artifact_name} from run {run_id} could not replace "
                    f"the incumbent fallback artifact ({exc})."
                )
                shutil.rmtree(candidate.wrapper_dir, ignore_errors=True)
                continue
            fallback_markets.add(market)
            fallback_dates_by_market[market] = candidate_date
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


def parse_formula_requirements(raw: str) -> dict[str, str]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("expected a JSON object keyed by market")
    return {
        str(market).strip().upper(): str(formula).strip()
        for market, formula in payload.items()
        if str(market).strip() and str(formula).strip()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--fallback-dir", type=Path, required=True)
    parser.add_argument("--repo", default=os.environ.get("REPOSITORY", ""))
    parser.add_argument(
        "--current-run-id", default=os.environ.get("CURRENT_RUN_ID", "0")
    )
    parser.add_argument("--branch", default=os.environ.get("BRANCH_NAME", "main"))
    parser.add_argument(
        "--fallback-rs-formula-overrides-json",
        type=parse_formula_requirements,
        default={},
    )
    parser.add_argument("--current-options-dir", type=Path)
    parser.add_argument("--fallback-options-dir", type=Path)
    args = parser.parse_args(argv)

    if not args.repo:
        raise SystemExit("REPOSITORY is required.")

    download_fallback_artifacts(
        repo=args.repo,
        current_run_id=int(args.current_run_id),
        branch_name=args.branch,
        current_dir=args.current_dir,
        fallback_dir=args.fallback_dir,
        required_formula_by_market=args.fallback_rs_formula_overrides_json,
        current_options_dir=args.current_options_dir,
        fallback_options_dir=args.fallback_options_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
