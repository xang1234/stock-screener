"""Restore rolling static RRG history from its GitHub Release asset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.config import settings
from app.services.github_release_sync_service import GitHubReleaseSyncService
from app.services.static_rrg_history_release import StaticRRGHistoryReleaseRestorer


def main(
    argv: Sequence[str] | None = None,
    *,
    restorer: StaticRRGHistoryReleaseRestorer | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=settings.github_data_repository)
    parser.add_argument("--release-tag", default="rrg-history-data")
    parser.add_argument("--asset-name", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5)
    args = parser.parse_args(argv)

    resolved_restorer = restorer or StaticRRGHistoryReleaseRestorer(
        sync_service=GitHubReleaseSyncService(api_base=settings.github_data_api_base)
    )
    result = resolved_restorer.restore(
        repository_full_name=args.repository,
        release_tag=args.release_tag,
        asset_name=args.asset_name,
        output_path=Path(args.output_path),
        github_token=settings.github_data_token,
        request_timeout_seconds=settings.github_data_timeout_seconds,
        attempts=args.attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0 if result.safe_to_publish else 1


if __name__ == "__main__":
    raise SystemExit(main())
