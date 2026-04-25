"""GitHub Release asset sync helpers for runtime bootstrap and refresh flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import requests


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

        release_url = (
            f"{self._api_base}/repos/{repository_full_name}/releases/tags/{release_tag}"
        )
        try:
            release_response = self._session.get(
                release_url,
                headers=self._headers(github_token=github_token),
                timeout=request_timeout_seconds,
            )
        except requests.RequestException as exc:
            return self._result("network_error", error=str(exc))

        release_status = getattr(release_response, "status_code", 200)
        if release_status == 404:
            return self._result(
                "missing_manifest",
                reason=f"Release {release_tag!r} was not found",
            )
        if release_status and release_status >= 400:
            return self._result(
                "network_error",
                error=f"GitHub release lookup failed with HTTP {release_status}",
            )

        release_payload = release_response.json() or {}
        assets = release_payload.get("assets") or []
        manifest_asset = next(
            (
                asset
                for asset in assets
                if str(asset.get("name") or "").strip() == manifest_asset_name
            ),
            None,
        )
        if manifest_asset is None:
            return self._result(
                "missing_manifest",
                reason=f"Manifest asset {manifest_asset_name!r} is not present on release {release_tag!r}",
            )

        manifest_url = str(
            manifest_asset.get("browser_download_url") or manifest_asset.get("url") or ""
        ).strip()
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

        if (
            bundle_asset_name is not None
            and (
                bundle_asset_name in {".", ".."}
                or "/" in bundle_asset_name
                or "\\" in bundle_asset_name
            )
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

        bundle_asset = next(
            (
                asset
                for asset in assets
                if str(asset.get("name") or "").strip() == bundle_asset_name
            ),
            None,
        )
        if bundle_asset is None:
            return self._result(
                "missing_bundle",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                reason=f"Bundle asset {bundle_asset_name!r} is not present on release {release_tag!r}",
                stale_reason=stale_reason,
            )

        bundle_url = str(
            bundle_asset.get("browser_download_url") or bundle_asset.get("url") or ""
        ).strip()
        if not bundle_url:
            return self._result(
                "missing_bundle",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                reason=f"Bundle asset {bundle_asset_name!r} has no download URL",
                stale_reason=stale_reason,
            )

        try:
            bundle_bytes = self._download_bytes(
                bundle_url,
                github_token=github_token,
                request_timeout_seconds=request_timeout_seconds,
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

        output_root = Path(output_dir) if output_dir is not None else Path.cwd()
        output_root.mkdir(parents=True, exist_ok=True)
        bundle_path = output_root / str(bundle_asset_name)
        bundle_path.write_bytes(bundle_bytes)

        import hashlib

        expected_sha = str(manifest.get("sha256") or "").strip()
        digest = hashlib.sha256(bundle_bytes).hexdigest()
        if expected_sha and digest != expected_sha:
            bundle_path.unlink(missing_ok=True)
            return self._result(
                "checksum_mismatch",
                manifest=manifest,
                bundle_asset_name=bundle_asset_name,
                source_revision=source_revision,
                reason=f"Bundle checksum mismatch: {digest} != {expected_sha}",
                stale_reason=stale_reason,
            )

        return self._result(
            "success",
            manifest=manifest,
            bundle_path=str(bundle_path),
            bundle_asset_name=bundle_asset_name,
            source_revision=source_revision,
            stale_reason=stale_reason,
        )
