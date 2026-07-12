"""Restore one exact GitHub Release asset with bounded retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from app.services.github_release_sync_service import (
    GitHubReleaseSyncService,
    retry_github_sync,
)


def restore_result_is_usable(
    result: dict[str, object],
    *,
    allow_missing: bool,
) -> bool:
    status = result.get("status")
    return status == "success" or (allow_missing and status == "missing_asset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=settings.github_data_repository)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--asset-name", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    service = GitHubReleaseSyncService(api_base=settings.github_data_api_base)
    result = retry_github_sync(
        lambda: service.fetch_named_asset(
            repository_full_name=args.repository,
            release_tag=args.release_tag,
            asset_name=args.asset_name,
            output_path=Path(args.output_path),
            github_token=settings.github_data_token,
            request_timeout_seconds=settings.github_data_timeout_seconds,
        ),
        attempts=args.attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    safe_to_publish = restore_result_is_usable(
        result,
        allow_missing=args.allow_missing,
    )
    print(
        json.dumps(
            {**result, "safe_to_publish": safe_to_publish},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if safe_to_publish else 1


if __name__ == "__main__":
    raise SystemExit(main())
