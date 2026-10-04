from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.models.app_settings import AppSetting
from app.models.stock import StockPrice
from app.services.daily_price_bundle_service import DailyPriceBundleService

from .daily_price_bundle_test_helpers import (
    make_session as _make_session,
)
from .daily_price_bundle_test_helpers import (
    stock_row as _stock_row,
)


def _make_service(session_factory):
    _ = session_factory
    return DailyPriceBundleService()


def test_sync_from_github_up_to_date_exposes_manifest_metadata():
    session_factory = _make_session()
    db = session_factory()
    service = _make_service(session_factory)

    result = service.sync_from_github(
        db,
        market="US",
        github_sync_service=SimpleNamespace(
            fetch_latest_bundle=lambda **kwargs: {
                "status": "up_to_date",
                "manifest": {
                    "market": "US",
                    "as_of_date": "2026-04-18",
                    "source_revision": "daily_prices_us:20260418120000",
                    "bundle_asset_name": "daily-price-us-20260418.json.gz",
                    "bar_period": "2y",
                    "symbol_count": 2,
                },
                "bundle_path": None,
                "bundle_asset_name": "daily-price-us-20260418.json.gz",
                "source_revision": "daily_prices_us:20260418120000",
            }
        ),
    )

    assert result["status"] == "up_to_date"
    assert result["as_of_date"] == "2026-04-18"
    assert result["bar_period"] == "2y"
    assert result["symbol_count"] == 2
    db.close()


def test_sync_from_github_passes_allow_stale_to_release_sync_service():
    session_factory = _make_session()
    db = session_factory()
    service = _make_service(session_factory)
    captured_kwargs = {}

    def _fetch_latest_bundle(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "status": "missing_manifest",
            "manifest": None,
            "bundle_path": None,
            "bundle_asset_name": None,
            "source_revision": None,
        }

    result = service.sync_from_github(
        db,
        market="US",
        allow_stale=True,
        github_sync_service=SimpleNamespace(fetch_latest_bundle=_fetch_latest_bundle),
    )

    assert result["status"] == "missing_manifest"
    assert captured_kwargs["allow_stale"] is True
    db.close()


def test_sync_from_github_rejects_manifest_market_mismatch():
    session_factory = _make_session()
    db = session_factory()
    service = _make_service(session_factory)

    result = service.sync_from_github(
        db,
        market="US",
        github_sync_service=SimpleNamespace(
            fetch_latest_bundle=lambda **kwargs: {
                "status": "up_to_date",
                "manifest": {
                    "market": "HK",
                    "as_of_date": "2026-04-18",
                    "source_revision": "daily_prices_hk:20260418120000",
                    "bundle_asset_name": "daily-price-hk-20260418.json.gz",
                    "bar_period": "2y",
                    "symbol_count": 2,
                },
                "bundle_path": None,
                "bundle_asset_name": "daily-price-hk-20260418.json.gz",
                "source_revision": "daily_prices_hk:20260418120000",
            }
        ),
    )

    assert result["status"] == "invalid_manifest"
    assert "does not match requested market" in str(result["error"])
    db.close()


def test_sync_from_github_import_uses_manifest_metadata_without_payload_prescan(
    monkeypatch,
    tmp_path,
):
    session_factory = _make_session()
    db = session_factory()
    db.add(_stock_row("AAPL", "US", "NASDAQ", 1000.0))
    db.commit()

    service = DailyPriceBundleService()
    monkeypatch.setattr(
        service,
        "_read_bundle_metadata",
        MagicMock(side_effect=AssertionError("payload metadata prescan")),
    )
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "symbol": "AAPL",
                        "prices": [
                            {
                                "date": "2026-04-18",
                                "open": 100.0,
                                "high": 101.0,
                                "low": 99.0,
                                "close": 100.5,
                                "adj_close": 100.0,
                                "volume": 1_000_000,
                            }
                        ],
                    }
                ],
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 1,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "market": "US",
        "as_of_date": "2026-04-18",
        "source_revision": "daily_prices_us:20260418120000",
        "bundle_asset_name": bundle_path.name,
        "bar_period": service.DAILY_PRICE_BAR_PERIOD,
        "symbol_count": 1,
    }

    result = service.sync_from_github(
        db,
        market="US",
        github_sync_service=SimpleNamespace(
            fetch_latest_bundle=lambda **kwargs: {
                "status": "success",
                "manifest": manifest,
                "bundle_path": str(bundle_path),
                "bundle_asset_name": bundle_path.name,
                "source_revision": manifest["source_revision"],
            }
        ),
    )

    assert result["status"] == "success"
    assert result["imported_symbols"] == 1
    assert db.query(StockPrice).filter(StockPrice.symbol == "AAPL").count() == 1
    service._read_bundle_metadata.assert_not_called()
    db.close()


def test_sync_from_github_rejects_bundle_metadata_mismatch_and_rolls_back(
    tmp_path,
):
    session_factory = _make_session()
    db = session_factory()

    service = DailyPriceBundleService()
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "symbol": "AAPL",
                        "prices": [
                            {
                                "date": "2026-04-18",
                                "open": 100.0,
                                "high": 101.0,
                                "low": 99.0,
                                "close": 100.5,
                                "adj_close": 100.0,
                                "volume": 1_000_000,
                            }
                        ],
                    }
                ],
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "HK",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_hk:20260418120000",
                "symbol_count": 1,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "market": "US",
        "as_of_date": "2026-04-18",
        "source_revision": "daily_prices_us:20260418120000",
        "bundle_asset_name": bundle_path.name,
        "bar_period": service.DAILY_PRICE_BAR_PERIOD,
        "symbol_count": 1,
    }

    with pytest.raises(ValueError, match="bundle market 'HK' does not match manifest 'US'"):
        service.sync_from_github(
            db,
            market="US",
            github_sync_service=SimpleNamespace(
                fetch_latest_bundle=lambda **kwargs: {
                    "status": "success",
                    "manifest": manifest,
                    "bundle_path": str(bundle_path),
                    "bundle_asset_name": bundle_path.name,
                    "source_revision": manifest["source_revision"],
                }
            ),
        )

    assert db.query(StockPrice).count() == 0
    assert db.query(AppSetting).count() == 0
    assert not bundle_path.exists()
    db.close()


_COMPLETED_REVISION = "daily_prices_us:20260417120000"


def _seed_import_state(service, db):
    service._upsert_import_state(
        db,
        market="US",
        source_revision=_COMPLETED_REVISION,
        as_of_date="2026-04-17",
        symbol_count=1,
        bar_period=service.DAILY_PRICE_BAR_PERIOD,
    )


def _capture_fetch(captured_kwargs):
    def _fetch_latest_bundle(**kwargs):
        captured_kwargs.update(kwargs)
        return {"status": "missing_manifest", "manifest": None, "bundle_path": None}

    return _fetch_latest_bundle


def test_sync_from_github_reuses_completed_revision_even_when_stale():
    session_factory = _make_session()
    db = session_factory()
    service = DailyPriceBundleService()
    _seed_import_state(service, db)
    captured_kwargs = {}

    service.sync_from_github(
        db,
        market="US",
        allow_stale=True,
        github_sync_service=SimpleNamespace(fetch_latest_bundle=_capture_fetch(captured_kwargs)),
    )

    assert captured_kwargs["current_revision"] == _COMPLETED_REVISION
    assert captured_kwargs["reuse_completed_stale_revision"] is True
    db.close()


def test_sync_from_github_reimport_bypasses_completed_revision():
    session_factory = _make_session()
    db = session_factory()
    service = DailyPriceBundleService()
    _seed_import_state(service, db)
    captured_kwargs = {}

    service.sync_from_github(
        db,
        market="US",
        allow_stale=True,
        reimport=True,
        github_sync_service=SimpleNamespace(fetch_latest_bundle=_capture_fetch(captured_kwargs)),
    )

    assert captured_kwargs["current_revision"] is None
    # A repair does not relax manifest validation.
    assert captured_kwargs["stale_validator"] is not None
    assert captured_kwargs["required_manifest_keys"]
    db.close()


def test_sync_from_github_failed_reimport_keeps_last_completed_revision(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    service = DailyPriceBundleService()
    _seed_import_state(service, db)
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "symbol": "AAPL",
                        "prices": [
                            {
                                "date": "2026-04-18",
                                "open": 100.0,
                                "high": 101.0,
                                "low": 99.0,
                                "close": 100.5,
                                "adj_close": 100.0,
                                "volume": 1_000_000,
                            }
                        ],
                    }
                ],
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "HK",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_hk:20260418120000",
                "symbol_count": 1,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "market": "US",
        "as_of_date": "2026-04-18",
        "source_revision": "daily_prices_us:20260418120000",
        "bundle_asset_name": bundle_path.name,
        "bar_period": service.DAILY_PRICE_BAR_PERIOD,
        "symbol_count": 1,
    }

    with pytest.raises(ValueError, match="does not match manifest"):
        service.sync_from_github(
            db,
            market="US",
            reimport=True,
            github_sync_service=SimpleNamespace(
                fetch_latest_bundle=lambda **kwargs: {
                    "status": "success",
                    "manifest": manifest,
                    "bundle_path": str(bundle_path),
                    "bundle_asset_name": bundle_path.name,
                    "source_revision": manifest["source_revision"],
                }
            ),
        )

    assert db.query(StockPrice).count() == 0
    assert service.get_import_state(db, "US")["source_revision"] == _COMPLETED_REVISION
    db.close()
