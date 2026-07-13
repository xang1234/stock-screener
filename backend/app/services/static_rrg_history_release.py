"""Release restoration policy for rolling static RRG history."""

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


class StaticRRGHistoryRestoreStatus(StrEnum):
    RESTORED = "restored"
    MISSING = "missing"
    FAILED = "failed"


@dataclass(frozen=True)
class StaticRRGHistoryRestoreResult:
    status: StaticRRGHistoryRestoreStatus
    asset_name: str
    output_path: Path
    detail: str | None = None

    @property
    def safe_to_publish(self) -> bool:
        return self.status in {
            StaticRRGHistoryRestoreStatus.RESTORED,
            StaticRRGHistoryRestoreStatus.MISSING,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "asset_name": self.asset_name,
            "output_path": str(self.output_path),
            "detail": self.detail,
            "safe_to_publish": self.safe_to_publish,
        }


@dataclass(frozen=True)
class StaticRRGHistoryReleaseRestorer:
    sync_service: NamedAssetFetcher
    sleep: Callable[[float], None] = time.sleep

    def restore(
        self,
        *,
        repository_full_name: str,
        release_tag: str,
        asset_name: str,
        output_path: Path,
        github_token: str | None,
        request_timeout_seconds: int,
        attempts: int = 3,
        retry_delay_seconds: float = 5,
    ) -> StaticRRGHistoryRestoreResult:
        fetched = retry_github_operation(
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

        if fetched.status is NamedAssetFetchStatus.SUCCESS:
            status = StaticRRGHistoryRestoreStatus.RESTORED
        elif fetched.status is NamedAssetFetchStatus.MISSING:
            status = StaticRRGHistoryRestoreStatus.MISSING
        else:
            status = StaticRRGHistoryRestoreStatus.FAILED
        return StaticRRGHistoryRestoreResult(
            status=status,
            asset_name=asset_name,
            output_path=output_path,
            detail=fetched.reason or fetched.error,
        )


__all__ = [
    "StaticRRGHistoryReleaseRestorer",
    "StaticRRGHistoryRestoreResult",
    "StaticRRGHistoryRestoreStatus",
]
