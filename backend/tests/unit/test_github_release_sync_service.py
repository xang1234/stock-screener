from __future__ import annotations

import hashlib
import json

import requests

from app.services.github_release_sync_service import GitHubReleaseSyncService


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_data=None, content: bytes = b""):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.text = content.decode("utf-8", errors="ignore")

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size: int = 1024 * 1024):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]


class _StreamingOnlyResponse:
    def __init__(self, *, content: bytes, status_code: int = 200):
        self.status_code = status_code
        self._content = content
        self.iter_content_called = False

    @property
    def content(self):
        raise AssertionError("bundle response content should not be materialized")

    def iter_content(self, chunk_size: int = 1024 * 1024):
        self.iter_content_called = True
        for offset in range(0, len(self._content), chunk_size):
            yield self._content[offset:offset + chunk_size]


class _FailingStreamingResponse:
    def __init__(self, *, status_code: int = 200):
        self.status_code = status_code

    @property
    def content(self):
        raise AssertionError("bundle response content should not be materialized")

    def iter_content(self, chunk_size: int = 1024 * 1024):
        _ = chunk_size
        yield b"partial-bundle"
        raise requests.ConnectionError("stream dropped")


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]):
        self._responses = responses
        self.calls: list[str] = []
        self.stream_by_url: dict[str, bool] = {}

    def get(self, url, headers=None, timeout=None, stream=False):  # noqa: ANN001 - requests-compatible stub
        _ = headers
        _ = timeout
        self.calls.append(url)
        self.stream_by_url[url] = bool(stream)
        response = self._responses.get(url)
        if response is None:
            raise AssertionError(f"Unexpected URL requested: {url}")
        return response


def test_fetch_latest_bundle_short_circuits_live_only(tmp_path):
    service = GitHubReleaseSyncService(session=_FakeSession({}))

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="weekly-reference-data",
        manifest_asset_name="weekly-reference-latest-us.json",
        source_mode="live_only",
        output_dir=tmp_path,
    )

    assert result["status"] == "live_only"
    assert result["bundle_path"] is None


def test_fetch_latest_bundle_returns_up_to_date_when_revision_matches(tmp_path):
    manifest = {
        "schema_version": "weekly-reference-manifest-v1",
        "market": "US",
        "as_of_date": "2026-04-18",
        "source_revision": "fundamentals_v1_us:20260418120000",
        "bundle_asset_name": "weekly-reference-us-20260418.json.gz",
        "sha256": "unused",
    }
    session = _FakeSession(
        {
            "https://api.github.com/repos/xang1234/stock-screener/releases/tags/weekly-reference-data": _FakeResponse(
                json_data={
                    "assets": [
                        {
                            "name": "weekly-reference-latest-us.json",
                            "browser_download_url": "https://example.com/manifest.json",
                        }
                    ]
                }
            ),
            "https://example.com/manifest.json": _FakeResponse(
                content=json.dumps(manifest).encode("utf-8")
            ),
        }
    )
    service = GitHubReleaseSyncService(session=session)

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="weekly-reference-data",
        manifest_asset_name="weekly-reference-latest-us.json",
        current_revision="fundamentals_v1_us:20260418120000",
        expected_manifest_schema="weekly-reference-manifest-v1",
        required_manifest_keys=("market", "as_of_date", "source_revision", "bundle_asset_name", "sha256"),
        output_dir=tmp_path,
    )

    assert result["status"] == "up_to_date"
    assert result["source_revision"] == "fundamentals_v1_us:20260418120000"
    assert result["bundle_path"] is None


def test_fetch_latest_bundle_rejects_checksum_mismatch(tmp_path):
    bundle_bytes = b"bundle-payload"
    manifest = {
        "schema_version": "weekly-reference-manifest-v1",
        "market": "US",
        "as_of_date": "2026-04-18",
        "source_revision": "fundamentals_v1_us:20260418120000",
        "bundle_asset_name": "weekly-reference-us-20260418.json.gz",
        "sha256": hashlib.sha256(b"other-payload").hexdigest(),
    }
    session = _FakeSession(
        {
            "https://api.github.com/repos/xang1234/stock-screener/releases/tags/weekly-reference-data": _FakeResponse(
                json_data={
                    "assets": [
                        {
                            "name": "weekly-reference-latest-us.json",
                            "browser_download_url": "https://example.com/manifest.json",
                        },
                        {
                            "name": "weekly-reference-us-20260418.json.gz",
                            "browser_download_url": "https://example.com/bundle.json.gz",
                        },
                    ]
                }
            ),
            "https://example.com/manifest.json": _FakeResponse(
                content=json.dumps(manifest).encode("utf-8")
            ),
            "https://example.com/bundle.json.gz": _FakeResponse(content=bundle_bytes),
        }
    )
    service = GitHubReleaseSyncService(session=session)

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="weekly-reference-data",
        manifest_asset_name="weekly-reference-latest-us.json",
        expected_manifest_schema="weekly-reference-manifest-v1",
        required_manifest_keys=("market", "as_of_date", "source_revision", "bundle_asset_name", "sha256"),
        output_dir=tmp_path,
    )

    assert result["status"] == "checksum_mismatch"
    assert result["bundle_asset_name"] == "weekly-reference-us-20260418.json.gz"
    assert not (tmp_path / "weekly-reference-us-20260418.json.gz").exists()
    assert not list(tmp_path.glob(".weekly-reference-us-20260418.json.gz.*.tmp"))


def test_fetch_latest_bundle_rejects_stale_manifest(tmp_path):
    manifest = {
        "schema_version": "daily-price-manifest-v1",
        "market": "US",
        "as_of_date": "2026-04-17",
        "source_revision": "daily_prices_us:20260418120000",
        "bundle_asset_name": "daily-price-us-20260417.json.gz",
        "sha256": "unused",
        "bar_period": "2y",
        "symbol_count": 10,
    }
    session = _FakeSession(
        {
            "https://api.github.com/repos/xang1234/stock-screener/releases/tags/daily-price-data": _FakeResponse(
                json_data={
                    "assets": [
                        {
                            "name": "daily-price-latest-us.json",
                            "browser_download_url": "https://example.com/manifest.json",
                        }
                    ]
                }
            ),
            "https://example.com/manifest.json": _FakeResponse(
                content=json.dumps(manifest).encode("utf-8")
            ),
        }
    )
    service = GitHubReleaseSyncService(session=session)

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="daily-price-data",
        manifest_asset_name="daily-price-latest-us.json",
        expected_manifest_schema="daily-price-manifest-v1",
        required_manifest_keys=(
            "market",
            "as_of_date",
            "source_revision",
            "bundle_asset_name",
            "sha256",
            "bar_period",
            "symbol_count",
        ),
        stale_validator=lambda parsed_manifest: (
            parsed_manifest["as_of_date"] != "2026-04-18",
            "bundle is behind the expected session",
        ),
        output_dir=tmp_path,
    )

    assert result["status"] == "stale"
    assert "expected session" in result["reason"]


def test_fetch_latest_bundle_can_bootstrap_from_stale_manifest(tmp_path):
    bundle_bytes = b"daily-price-bundle"
    manifest = {
        "schema_version": "daily-price-manifest-v1",
        "market": "US",
        "as_of_date": "2026-04-17",
        "source_revision": "daily_prices_us:20260418120000",
        "bundle_asset_name": "daily-price-us-20260417.json.gz",
        "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "bar_period": "2y",
        "symbol_count": 10,
    }
    session = _FakeSession(
        {
            "https://api.github.com/repos/xang1234/stock-screener/releases/tags/daily-price-data": _FakeResponse(
                json_data={
                    "assets": [
                        {
                            "name": "daily-price-latest-us.json",
                            "browser_download_url": "https://example.com/manifest.json",
                        },
                        {
                            "name": "daily-price-us-20260417.json.gz",
                            "browser_download_url": "https://example.com/bundle.json.gz",
                        },
                    ]
                }
            ),
            "https://example.com/manifest.json": _FakeResponse(
                content=json.dumps(manifest).encode("utf-8")
            ),
            "https://example.com/bundle.json.gz": _FakeResponse(content=bundle_bytes),
        }
    )
    service = GitHubReleaseSyncService(session=session)

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="daily-price-data",
        manifest_asset_name="daily-price-latest-us.json",
        current_revision="daily_prices_us:20260418120000",
        expected_manifest_schema="daily-price-manifest-v1",
        required_manifest_keys=(
            "market",
            "as_of_date",
            "source_revision",
            "bundle_asset_name",
            "sha256",
            "bar_period",
            "symbol_count",
        ),
        stale_validator=lambda parsed_manifest: (
            parsed_manifest["as_of_date"] != "2026-04-18",
            "bundle is behind the expected session",
        ),
        allow_stale=True,
        output_dir=tmp_path,
    )

    assert result["status"] == "success"
    assert result["stale_reason"] == "bundle is behind the expected session"
    assert result["bundle_asset_name"] == "daily-price-us-20260417.json.gz"
    assert (tmp_path / "daily-price-us-20260417.json.gz").read_bytes() == bundle_bytes


def test_fetch_latest_bundle_streams_bundle_download(tmp_path):
    bundle_bytes = b"daily-price-bundle"
    manifest = {
        "schema_version": "daily-price-manifest-v1",
        "market": "US",
        "as_of_date": "2026-04-18",
        "source_revision": "daily_prices_us:20260418120000",
        "bundle_asset_name": "daily-price-us-20260418.json.gz",
        "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "bar_period": "2y",
        "symbol_count": 10,
    }
    bundle_response = _StreamingOnlyResponse(content=bundle_bytes)
    session = _FakeSession(
        {
            "https://api.github.com/repos/xang1234/stock-screener/releases/tags/daily-price-data": _FakeResponse(
                json_data={
                    "assets": [
                        {
                            "name": "daily-price-latest-us.json",
                            "browser_download_url": "https://example.com/manifest.json",
                        },
                        {
                            "name": "daily-price-us-20260418.json.gz",
                            "browser_download_url": "https://example.com/bundle.json.gz",
                        },
                    ]
                }
            ),
            "https://example.com/manifest.json": _FakeResponse(
                content=json.dumps(manifest).encode("utf-8")
            ),
            "https://example.com/bundle.json.gz": bundle_response,
        }
    )
    service = GitHubReleaseSyncService(session=session)

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="daily-price-data",
        manifest_asset_name="daily-price-latest-us.json",
        expected_manifest_schema="daily-price-manifest-v1",
        required_manifest_keys=(
            "market",
            "as_of_date",
            "source_revision",
            "bundle_asset_name",
            "sha256",
            "bar_period",
            "symbol_count",
        ),
        output_dir=tmp_path,
    )

    assert result["status"] == "success"
    assert bundle_response.iter_content_called is True
    assert session.stream_by_url["https://example.com/bundle.json.gz"] is True
    assert (tmp_path / "daily-price-us-20260418.json.gz").read_bytes() == bundle_bytes


def test_fetch_latest_bundle_preserves_existing_file_on_stream_failure(tmp_path):
    existing_bundle = tmp_path / "daily-price-us-20260418.json.gz"
    existing_bundle.write_bytes(b"existing-good-bundle")
    manifest = {
        "schema_version": "daily-price-manifest-v1",
        "market": "US",
        "as_of_date": "2026-04-18",
        "source_revision": "daily_prices_us:20260418120000",
        "bundle_asset_name": existing_bundle.name,
        "sha256": hashlib.sha256(b"new-bundle").hexdigest(),
        "bar_period": "2y",
        "symbol_count": 10,
    }
    session = _FakeSession(
        {
            "https://api.github.com/repos/xang1234/stock-screener/releases/tags/daily-price-data": _FakeResponse(
                json_data={
                    "assets": [
                        {
                            "name": "daily-price-latest-us.json",
                            "browser_download_url": "https://example.com/manifest.json",
                        },
                        {
                            "name": existing_bundle.name,
                            "browser_download_url": "https://example.com/bundle.json.gz",
                        },
                    ]
                }
            ),
            "https://example.com/manifest.json": _FakeResponse(
                content=json.dumps(manifest).encode("utf-8")
            ),
            "https://example.com/bundle.json.gz": _FailingStreamingResponse(),
        }
    )
    service = GitHubReleaseSyncService(session=session)

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="daily-price-data",
        manifest_asset_name="daily-price-latest-us.json",
        expected_manifest_schema="daily-price-manifest-v1",
        required_manifest_keys=(
            "market",
            "as_of_date",
            "source_revision",
            "bundle_asset_name",
            "sha256",
            "bar_period",
            "symbol_count",
        ),
        output_dir=tmp_path,
    )

    assert result["status"] == "network_error"
    assert session.stream_by_url["https://example.com/bundle.json.gz"] is True
    assert existing_bundle.read_bytes() == b"existing-good-bundle"
    assert not list(tmp_path.glob(f".{existing_bundle.name}.*.tmp"))


def test_fetch_latest_bundle_checks_staleness_before_up_to_date(tmp_path):
    manifest = {
        "schema_version": "daily-price-manifest-v1",
        "market": "US",
        "as_of_date": "2026-04-17",
        "source_revision": "daily_prices_us:20260418120000",
        "bundle_asset_name": "daily-price-us-20260417.json.gz",
        "sha256": "unused",
        "bar_period": "2y",
        "symbol_count": 10,
    }
    session = _FakeSession(
        {
            "https://api.github.com/repos/xang1234/stock-screener/releases/tags/daily-price-data": _FakeResponse(
                json_data={
                    "assets": [
                        {
                            "name": "daily-price-latest-us.json",
                            "browser_download_url": "https://example.com/manifest.json",
                        }
                    ]
                }
            ),
            "https://example.com/manifest.json": _FakeResponse(
                content=json.dumps(manifest).encode("utf-8")
            ),
        }
    )
    service = GitHubReleaseSyncService(session=session)

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="daily-price-data",
        manifest_asset_name="daily-price-latest-us.json",
        current_revision="daily_prices_us:20260418120000",
        expected_manifest_schema="daily-price-manifest-v1",
        required_manifest_keys=(
            "market",
            "as_of_date",
            "source_revision",
            "bundle_asset_name",
            "sha256",
            "bar_period",
            "symbol_count",
        ),
        stale_validator=lambda parsed_manifest: (
            parsed_manifest["as_of_date"] != "2026-04-18",
            "bundle is behind the expected session",
        ),
        output_dir=tmp_path,
    )

    assert result["status"] == "stale"
    assert "expected session" in result["reason"]


def test_fetch_latest_bundle_rejects_bundle_asset_name_with_path_separator(tmp_path):
    bundle_bytes = b"bundle-payload"
    manifest = {
        "schema_version": "weekly-reference-manifest-v1",
        "market": "US",
        "as_of_date": "2026-04-18",
        "source_revision": "fundamentals_v1_us:20260418120000",
        "bundle_asset_name": "nested/weekly-reference-us-20260418.json.gz",
        "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
    }
    session = _FakeSession(
        {
            "https://api.github.com/repos/xang1234/stock-screener/releases/tags/weekly-reference-data": _FakeResponse(
                json_data={
                    "assets": [
                        {
                            "name": "weekly-reference-latest-us.json",
                            "browser_download_url": "https://example.com/manifest.json",
                        },
                        {
                            "name": "nested/weekly-reference-us-20260418.json.gz",
                            "browser_download_url": "https://example.com/bundle.json.gz",
                        },
                    ]
                }
            ),
            "https://example.com/manifest.json": _FakeResponse(
                content=json.dumps(manifest).encode("utf-8")
            ),
            "https://example.com/bundle.json.gz": _FakeResponse(content=bundle_bytes),
        }
    )
    service = GitHubReleaseSyncService(session=session)

    result = service.fetch_latest_bundle(
        repository_full_name="xang1234/stock-screener",
        release_tag="weekly-reference-data",
        manifest_asset_name="weekly-reference-latest-us.json",
        expected_manifest_schema="weekly-reference-manifest-v1",
        required_manifest_keys=("market", "as_of_date", "source_revision", "bundle_asset_name", "sha256"),
        output_dir=tmp_path,
    )

    assert result["status"] == "invalid_manifest"
    assert "bundle asset name" in str(result["error"]).lower()
