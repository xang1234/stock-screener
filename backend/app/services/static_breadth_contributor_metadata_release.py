"""Release restoration policy for rolling breadth-contributor metadata."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from app.services.github_release_sync_service import (
    NamedAssetFetchResult,
    NamedAssetFetchStatus,
    retry_github_operation,
)
from app.services.static_breadth_contributor_metadata_contract import (
    StaticBreadthContributorMetadataBundleError,
    read_static_breadth_contributor_metadata,
)


class NamedAssetFetcher(Protocol):
    def fetch_named_asset(
        self,
        *,
        repository_full_name: str,
        release_tag: str,
        asset_name: str,
        output_path: str | Path,
        github_token: str | None = None,
        request_timeout_seconds: int = 60,
    ) -> NamedAssetFetchResult: ...


class StaticBreadthContributorMetadataRestoreStatus(StrEnum):
    RESTORED = "restored"
    MISSING = "missing"
    FAILED = "failed"


@dataclass(frozen=True)
class StaticBreadthContributorMetadataRestoreResult:
    status: StaticBreadthContributorMetadataRestoreStatus
    asset_name: str
    output_path: Path
    detail: str | None = None
    source_asset_name: str | None = None

    @property
    def safe_to_publish(self) -> bool:
        return self.status in {
            StaticBreadthContributorMetadataRestoreStatus.RESTORED,
            StaticBreadthContributorMetadataRestoreStatus.MISSING,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "asset_name": self.asset_name,
            "output_path": str(self.output_path),
            "detail": self.detail,
            "source_asset_name": self.source_asset_name,
            "safe_to_publish": self.safe_to_publish,
        }


def _validate_bundle(path: Path, market: str) -> None:
    read_static_breadth_contributor_metadata(path, expected_market=market)


@dataclass(frozen=True)
class StaticBreadthContributorMetadataReleaseRestorer:
    sync_service: NamedAssetFetcher
    sleep: Callable[[float], None] = time.sleep
    bundle_validator: Callable[[Path, str], object] = _validate_bundle

    def _fetch(
        self,
        *,
        repository_full_name: str,
        release_tag: str,
        asset_name: str,
        output_path: Path,
        github_token: str | None,
        request_timeout_seconds: int,
        attempts: int,
        retry_delay_seconds: float,
    ) -> NamedAssetFetchResult:
        return retry_github_operation(
            lambda: self.sync_service.fetch_named_asset(
                repository_full_name=repository_full_name,
                release_tag=release_tag,
                asset_name=asset_name,
                output_path=output_path,
                github_token=github_token,
                request_timeout_seconds=request_timeout_seconds,
            ),
            should_retry=lambda result: result.retryable,
            attempts=attempts,
            retry_delay_seconds=retry_delay_seconds,
            sleep=self.sleep,
        )

    def restore(
        self,
        *,
        repository_full_name: str,
        release_tag: str,
        asset_name: str,
        previous_asset_name: str,
        output_path: Path,
        expected_market: str,
        github_token: str | None,
        request_timeout_seconds: int,
        attempts: int = 3,
        retry_delay_seconds: float = 5,
    ) -> StaticBreadthContributorMetadataRestoreResult:
        fetched = self._fetch(
            repository_full_name=repository_full_name,
            release_tag=release_tag,
            asset_name=asset_name,
            output_path=output_path,
            github_token=github_token,
            request_timeout_seconds=request_timeout_seconds,
            attempts=attempts,
            retry_delay_seconds=retry_delay_seconds,
        )

        source_asset_name: str | None = None
        if fetched.status is NamedAssetFetchStatus.MISSING:
            fetched = self._fetch(
                repository_full_name=repository_full_name,
                release_tag=release_tag,
                asset_name=previous_asset_name,
                output_path=output_path,
                github_token=github_token,
                request_timeout_seconds=request_timeout_seconds,
                attempts=attempts,
                retry_delay_seconds=retry_delay_seconds,
            )
            if fetched.status is NamedAssetFetchStatus.SUCCESS:
                source_asset_name = previous_asset_name

        if fetched.status is NamedAssetFetchStatus.SUCCESS:
            source_asset_name = source_asset_name or asset_name
            try:
                self.bundle_validator(output_path, expected_market)
            except (StaticBreadthContributorMetadataBundleError, OSError, ValueError) as exc:
                return StaticBreadthContributorMetadataRestoreResult(
                    status=StaticBreadthContributorMetadataRestoreStatus.FAILED,
                    asset_name=asset_name,
                    output_path=output_path,
                    detail=f"Downloaded metadata bundle is invalid: {exc}",
                    source_asset_name=source_asset_name,
                )
            status = StaticBreadthContributorMetadataRestoreStatus.RESTORED
        elif fetched.status is NamedAssetFetchStatus.MISSING:
            status = StaticBreadthContributorMetadataRestoreStatus.MISSING
        else:
            status = StaticBreadthContributorMetadataRestoreStatus.FAILED
        return StaticBreadthContributorMetadataRestoreResult(
            status=status,
            asset_name=asset_name,
            output_path=output_path,
            detail=fetched.reason or fetched.error,
            source_asset_name=source_asset_name,
        )


__all__ = [
    "StaticBreadthContributorMetadataReleaseRestorer",
    "StaticBreadthContributorMetadataRestoreResult",
    "StaticBreadthContributorMetadataRestoreStatus",
]
