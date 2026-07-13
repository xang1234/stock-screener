"""GitHub Release asset sync helpers for runtime bootstrap and refresh flows."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import requests


_ResultT = TypeVar("_ResultT")


class _ChecksumMismatchError(Exception):
    def __init__(self, *, actual: str, expected: str) -> None:
        super().__init__(f"Bundle checksum mismatch: {actual} != {expected}")
        self.actual = actual
        self.expected = expected


class _ReleaseNotFoundError(requests.HTTPError):
    pass


class NamedAssetFetchStatus(StrEnum):
    SUCCESS = "success"
    MISSING = "missing_asset"
    NETWORK_ERROR = "network_error"
    INVALID = "invalid_asset"


@dataclass(frozen=True)
class NamedAssetFetchResult:
    status: NamedAssetFetchStatus
    asset_name: str | None = None
    output_path: Path | None = None
    reason: str | None = None
    error: str | None = None

    @property
    def retryable(self) -> bool:
        return self.status is NamedAssetFetchStatus.NETWORK_ERROR


def retry_github_operation(
    operation: Callable[[], _ResultT],
    *,
    should_retry: Callable[[_ResultT], bool],
    attempts: int = 3,
    retry_delay_seconds: float = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> _ResultT:
    """Retry a typed GitHub operation under a caller-owned status policy."""
    total_attempts = max(1, int(attempts))
    result = operation()
    for retry_number in range(1, total_attempts):
        if not should_retry(result):
            break
        sleep(max(0, retry_delay_seconds) * retry_number)
        result = operation()
    return result


class GitHubReleaseSyncService:
    """Fetch manifests and bundle assets from a GitHub release tag."""

    def __init__(
        self,
        *,
        session: requests.Session | Any | None = None,
        api_base: str = "https://api.github.com",
    ) -> None:
        self._session = session or requests.Session()
        self._api_base = api_base.rstrip("/")

    @staticmethod
    def _headers(*, github_token: str | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        return headers

    def _release_assets(
        self,
        *,
        repository_full_name: str,
        release_tag: str,
        github_token: str | None,
        request_timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        release_url = (
            f"{self._api_base}/repos/{repository_full_name}/releases/tags/{release_tag}"
        )
        response = self._session.get(
            release_url,
            headers=self._headers(github_token=github_token),
            timeout=request_timeout_seconds,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code == 404:
            raise _ReleaseNotFoundError(f"Release {release_tag!r} was not found")
        if status_code and status_code >= 400:
            raise requests.HTTPError(
                f"GitHub release lookup failed with HTTP {status_code}"
            )
        payload = response.json() or {}
        return list(payload.get("assets") or [])

    @staticmethod
    def _asset_by_name(
        assets: Iterable[dict[str, Any]],
        asset_name: str | None,
    ) -> dict[str, Any] | None:
        return next(
            (
                asset
                for asset in assets
                if str(asset.get("name") or "").strip() == asset_name
            ),
            None,
        )

    @staticmethod
    def _asset_url(asset: dict[str, Any]) -> str:
        return str(
            asset.get("browser_download_url") or asset.get("url") or ""
        ).strip()

    @staticmethod
    def _valid_asset_name(asset_name: str | None) -> bool:
        return bool(
            asset_name
            and asset_name not in {".", ".."}
            and "/" not in asset_name
            and "\\" not in asset_name
        )

    def _download_bytes(
        self,
        url: str,
        *,
        github_token: str | None = None,
        request_timeout_seconds: int,
    ) -> bytes:
        response = self._session.get(
            url,
            headers=self._headers(github_token=github_token),
            timeout=request_timeout_seconds,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code and status_code >= 400:
            raise requests.HTTPError(
                f"GitHub download failed for {url}: HTTP {status_code}",
            )
        return bytes(getattr(response, "content", b""))

    def _download_to_path(
        self,
        url: str,
        output_path: Path,
        *,
        github_token: str | None = None,
        request_timeout_seconds: int,
        expected_sha256: str | None = None,
    ) -> str:
        response = self._session.get(
            url,
            headers=self._headers(github_token=github_token),
            timeout=request_timeout_seconds,
            stream=True,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code and status_code >= 400:
            raise requests.HTTPError(
                f"GitHub download failed for {url}: HTTP {status_code}",
            )

        digest = hashlib.sha256()
        temp_file = tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_file.name)
        iter_content = getattr(response, "iter_content", None)
        chunks = (
            iter_content(chunk_size=1024 * 1024)
            if callable(iter_content)
            else (bytes(getattr(response, "content", b"")),)
        )
        try:
            with temp_file as handle:
                for chunk in chunks:
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            expected = str(expected_sha256 or "").strip()
            if expected and actual_sha256 != expected:
                raise _ChecksumMismatchError(
                    actual=actual_sha256,
                    expected=expected,
                )
            temp_path.replace(output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return actual_sha256

    @staticmethod
    def _resolve_stale_validation(
        validation: Any,
    ) -> tuple[bool, str | None]:
        if validation is None:
            return False, None
        if isinstance(validation, tuple):
            return bool(validation[0]), str(validation[1]) if len(validation) > 1 else None
        if isinstance(validation, str):
            return True, validation
        return bool(validation), None

    @staticmethod
    def _result(
        status: str,
        *,
        manifest: dict[str, Any] | None = None,
        bundle_path: str | None = None,
        bundle_asset_name: str | None = None,
        source_revision: str | None = None,
        reason: str | None = None,
        stale_reason: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "manifest": manifest,
            "bundle_path": bundle_path,
            "bundle_asset_name": bundle_asset_name,
            "source_revision": source_revision,
            "reason": reason,
            "stale_reason": stale_reason,
            "error": error,
        }

    def fetch_named_asset(
        self,
        *,
        repository_full_name: str,
        release_tag: str,
        asset_name: str,
        output_path: str | Path,
        github_token: str | None = None,
        request_timeout_seconds: int = 60,
    ) -> NamedAssetFetchResult:
        """Atomically download one exact release asset to a caller-owned path."""
        normalized_asset_name = str(asset_name or "").strip()
        if not self._valid_asset_name(normalized_asset_name):
            return NamedAssetFetchResult(
                status=NamedAssetFetchStatus.INVALID,
                asset_name=normalized_asset_name or None,
                error=f"Invalid release asset name: {asset_name!r}",
            )

        try:
            assets = self._release_assets(
                repository_full_name=repository_full_name,
                release_tag=release_tag,
                github_token=github_token,
                request_timeout_seconds=request_timeout_seconds,
            )
        except _ReleaseNotFoundError as exc:
            return NamedAssetFetchResult(
                status=NamedAssetFetchStatus.MISSING,
                asset_name=normalized_asset_name,
                reason=str(exc),
            )
        except requests.RequestException as exc:
            return NamedAssetFetchResult(
                status=NamedAssetFetchStatus.NETWORK_ERROR,
                asset_name=normalized_asset_name,
                error=str(exc),
            )

        asset = self._asset_by_name(assets, normalized_asset_name)
        if asset is None:
            return NamedAssetFetchResult(
                status=NamedAssetFetchStatus.MISSING,
                asset_name=normalized_asset_name,
                reason=(
                    f"Asset {normalized_asset_name!r} is not present on "
                    f"release {release_tag!r}"
                ),
            )

        asset_url = self._asset_url(asset)
        if not asset_url:
            return NamedAssetFetchResult(
                status=NamedAssetFetchStatus.MISSING,
                asset_name=normalized_asset_name,
                reason=f"Asset {normalized_asset_name!r} has no download URL",
            )

        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._download_to_path(
                asset_url,
                resolved_output_path,
                github_token=github_token,
                request_timeout_seconds=request_timeout_seconds,
            )
        except requests.RequestException as exc:
            return NamedAssetFetchResult(
                status=NamedAssetFetchStatus.NETWORK_ERROR,
                asset_name=normalized_asset_name,
                error=str(exc),
            )

        return NamedAssetFetchResult(
            status=NamedAssetFetchStatus.SUCCESS,
            asset_name=normalized_asset_name,
            output_path=resolved_output_path,
        )

    def fetch_latest_bundle(
        self,
        *,
        repository_full_name: str,
        release_tag: str,
        manifest_asset_name: str,
        source_mode: str = "github_first",
        current_revision: str | None = None,
        expected_manifest_schema: str | None = None,
        required_manifest_keys: Iterable[str] = (),
        stale_validator: Callable[[dict[str, Any]], Any] | None = None,
        allow_stale: bool = False,
        github_token: str | None = None,
        request_timeout_seconds: int = 60,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Download the latest bundle referenced by ``manifest_asset_name``."""
        if str(source_mode or "").strip().lower() == "live_only":
            return self._result("live_only")

        try:
            assets = self._release_assets(
                repository_full_name=repository_full_name,
                release_tag=release_tag,
                github_token=github_token,
                request_timeout_seconds=request_timeout_seconds,
            )
        except _ReleaseNotFoundError as exc:
            return self._result("missing_manifest", reason=str(exc))
        except requests.RequestException as exc:
            return self._result("network_error", error=str(exc))

        manifest_asset = self._asset_by_name(assets, manifest_asset_name)
        if manifest_asset is None:
            return self._result(
                "missing_manifest",
                reason=f"Manifest asset {manifest_asset_name!r} is not present on release {release_tag!r}",
            )

        manifest_url = self._asset_url(manifest_asset)
        if not manifest_url:
            return self._result(
                "missing_manifest",
                reason=f"Manifest asset {manifest_asset_name!r} has no download URL",
            )

        try:
            manifest_bytes = self._download_bytes(
                manifest_url,
                github_token=github_token,
                request_timeout_seconds=request_timeout_seconds,
            )
            manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return self._result("invalid_manifest", error=str(exc))
        except requests.RequestException as exc:
            return self._result("network_error", error=str(exc))

        if not isinstance(manifest_payload, dict):
            return self._result(
                "invalid_manifest",
                error=(
                    "Manifest must be a JSON object, "
                    f"got {type(manifest_payload).__name__}"
                ),
            )
        manifest = manifest_payload

        if expected_manifest_schema and manifest.get("schema_version") != expected_manifest_schema:
            return self._result(
                "invalid_manifest",
                manifest=manifest,
                error=(
                    "Unexpected manifest schema version "
                    f"{manifest.get('schema_version')!r}; expected {expected_manifest_schema!r}"
                ),
            )

        missing_keys = [
            key for key in required_manifest_keys
            if manifest.get(key) in (None, "")
        ]
        if missing_keys:
            return self._result(
                "invalid_manifest",
                manifest=manifest,
                error=f"Manifest is missing required keys: {', '.join(sorted(missing_keys))}",
            )

        raw_bundle_asset_name = str(manifest.get("bundle_asset_name") or "").strip()
        bundle_asset_name = raw_bundle_asset_name or None
        source_revision = str(manifest.get("source_revision") or "").strip() or None

        if bundle_asset_name is not None and not self._valid_asset_name(
            bundle_asset_name
        ):
            return self._result(
                "invalid_manifest",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                error=f"Invalid bundle asset name: {bundle_asset_name!r}",
            )

        manifest_is_stale = False
        stale_reason: str | None = None
        if stale_validator is not None:
            try:
                is_stale, resolved_stale_reason = self._resolve_stale_validation(
                    stale_validator(manifest)
                )
            except ValueError as exc:
                return self._result("invalid_manifest", manifest=manifest, error=str(exc))
            if is_stale:
                manifest_is_stale = True
                stale_reason = resolved_stale_reason
                if not allow_stale:
                    return self._result(
                        "stale",
                        manifest=manifest,
                        bundle_asset_name=bundle_asset_name,
                        source_revision=source_revision,
                        reason=stale_reason,
                        stale_reason=stale_reason,
                    )

        if (
            not manifest_is_stale
            and current_revision
            and source_revision
            and current_revision == source_revision
        ):
            return self._result(
                "up_to_date",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                stale_reason=stale_reason,
            )

        bundle_asset = self._asset_by_name(assets, bundle_asset_name)
        if bundle_asset is None:
            return self._result(
                "missing_bundle",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                reason=f"Bundle asset {bundle_asset_name!r} is not present on release {release_tag!r}",
                stale_reason=stale_reason,
            )

        bundle_url = self._asset_url(bundle_asset)
        if not bundle_url:
            return self._result(
                "missing_bundle",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                reason=f"Bundle asset {bundle_asset_name!r} has no download URL",
                stale_reason=stale_reason,
            )

        output_root = Path(output_dir) if output_dir is not None else Path.cwd()
        output_root.mkdir(parents=True, exist_ok=True)
        bundle_path = output_root / str(bundle_asset_name)
        expected_sha = str(manifest.get("sha256") or "").strip()
        try:
            self._download_to_path(
                bundle_url,
                bundle_path,
                github_token=github_token,
                request_timeout_seconds=request_timeout_seconds,
                expected_sha256=expected_sha,
            )
        except _ChecksumMismatchError as exc:
            return self._result(
                "checksum_mismatch",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                reason=str(exc),
                stale_reason=stale_reason,
            )
        except requests.RequestException as exc:
            return self._result(
                "network_error",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                stale_reason=stale_reason,
                error=str(exc),
            )

        return self._result(
            "success",
            manifest=manifest,
            bundle_path=str(bundle_path),
            bundle_asset_name=bundle_asset_name,
            source_revision=source_revision,
            stale_reason=stale_reason,
        )
