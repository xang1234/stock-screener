from __future__ import annotations

import hashlib
import json

from app.config import settings
from app.services.github_release_sync_service import GitHubReleaseSyncService
from app.services.weekly_reference_github_sync import (
    fetch_weekly_reference_bundle,
    retry_weekly_reference_sync,
)


class _Response:
    def __init__(self, *, payload=None, content: bytes = b""):
        self.status_code = 200
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1024 * 1024):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]


class _Session:
    def __init__(self, responses):
        self.responses = responses

    def get(self, url, **_kwargs):
        return self.responses[url]


def test_fetch_weekly_reference_bundle_accepts_authentic_legacy_us_manifest(
    tmp_path,
    monkeypatch,
):
    repository = "xang1234/stock-screener"
    release_tag = "weekly-reference-data"
    release_url = f"https://api.github.com/repos/{repository}/releases/tags/{release_tag}"
    manifest_url = "https://example.com/weekly-reference-latest.json"
    bundle_url = "https://example.com/weekly-reference-us.json.gz"
    bundle_name = "weekly-reference-20260418.json.gz"
    bundle = b"legacy-weekly-reference"
    legacy_manifest = {
        "schema_version": "weekly-reference-manifest-v1",
        "generated_at": "2026-04-18T12:00:00Z",
        "as_of_date": "2026-04-18",
        "source_revision": "fundamentals_v1:20260418120000",
        "coverage": {},
        "warnings": [],
        "bundle_asset_name": bundle_name,
        "sha256": hashlib.sha256(bundle).hexdigest(),
    }
    session = _Session(
        {
            release_url: _Response(
                payload={
                    "assets": [
                        {
                            "name": "weekly-reference-latest.json",
                            "browser_download_url": manifest_url,
                        },
                        {
                            "name": bundle_name,
                            "browser_download_url": bundle_url,
                        },
                    ]
                }
            ),
            manifest_url: _Response(content=json.dumps(legacy_manifest).encode()),
            bundle_url: _Response(content=bundle),
        }
    )
    monkeypatch.setattr(settings, "github_data_repository", repository)
    monkeypatch.setattr(settings, "github_weekly_reference_release_tag", release_tag)
    monkeypatch.setattr(settings, "market_data_source_mode", "github_first")

    result = fetch_weekly_reference_bundle(
        sync_service=GitHubReleaseSyncService(session=session),
        market="US",
        current_revision=None,
        output_dir=tmp_path,
        stale_validator=lambda _manifest: (False, None),
        allow_stale=False,
    )

    assert result["status"] == "success"
    assert result["manifest"] == legacy_manifest
    assert (tmp_path / bundle_name).read_bytes() == bundle


def test_retry_weekly_reference_sync_retries_network_errors_only():
    results = iter(
        [
            {"status": "network_error", "error": "temporary"},
            {"status": "success", "bundle_path": "/tmp/reference"},
        ]
    )
    sleeps: list[float] = []

    result = retry_weekly_reference_sync(
        lambda: next(results),
        attempts=3,
        retry_delay_seconds=2,
        sleep=sleeps.append,
    )

    assert result["status"] == "success"
    assert sleeps == [2]
