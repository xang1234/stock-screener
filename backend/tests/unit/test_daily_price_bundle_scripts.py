from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import app.scripts.build_daily_price_bundle as build_script
import app.scripts.bootstrap_cn_daily_price_shard as cn_shard_script
import app.scripts.import_daily_price_bundle as import_daily_script
import app.scripts.sync_daily_price_bundle_from_github as sync_daily_script
import app.scripts.sync_weekly_reference_from_github as sync_weekly_script
from app.domain.markets import market_registry
from app.services.daily_price_bundle_service import DailyPriceBundleService
from app.services.provider_snapshot_service import ProviderSnapshotService


def test_daily_and_weekly_reference_script_markets_match_market_registry():
    assert DailyPriceBundleService.DAILY_PRICE_SUPPORTED_MARKETS == market_registry.supported_market_codes()
    assert tuple(ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS_BY_MARKET) == (
        market_registry.supported_market_codes()
    )


@contextmanager
def _fake_session(db="db-session"):
    yield db


def test_build_daily_price_bundle_exports_market_bundle(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", _fake_session)

    def _fake_export(db, **kwargs):
        _ = db
        kwargs["latest_manifest_path"].write_text(
            json.dumps({"bundle_asset_name": kwargs["bundle_asset_name"]}),
            encoding="utf-8",
        )
        kwargs["output_path"].write_bytes(b"bundle")
        return {
            "bundle_path": str(kwargs["output_path"]),
            "manifest_path": str(kwargs["latest_manifest_path"]),
            "market": kwargs["market"],
            "as_of_date": kwargs["as_of_date"].isoformat(),
            "symbol_count": 2,
            "bar_period": "2y",
        }

    service = SimpleNamespace(
        market_calendar=SimpleNamespace(
            last_completed_trading_day=lambda market: date(2026, 4, 21)
        ),
        latest_manifest_name_for_market=lambda market: f"daily-price-latest-{market.lower()}.json",
        export_daily_price_bundle=_fake_export,
    )
    monkeypatch.setattr(build_script, "get_daily_price_bundle_service", lambda: service)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_daily_price_bundle",
            "--market",
            "US",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert build_script.main() == 0
    stdout = capsys.readouterr().out
    assert "Daily price bundle complete for US:" in stdout
    assert (tmp_path / "daily-price-us-20260421.json.gz").exists()
    assert (tmp_path / "daily-price-latest-us.json").exists()


def test_import_daily_price_bundle_script_calls_service(monkeypatch, tmp_path, capsys):
    bundle_path = tmp_path / "daily-price-cn-shard.json.gz"
    bundle_path.write_bytes(b"bundle")
    captured = {}

    monkeypatch.setattr(import_daily_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(import_daily_script, "SessionLocal", _fake_session)

    def _fake_import(db, **kwargs):
        captured["db"] = db
        captured.update(kwargs)
        return {
            "market": "CN",
            "as_of_date": "2026-05-08",
            "imported_symbols": 2,
            "imported_rows": 4,
        }

    monkeypatch.setattr(
        import_daily_script,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(import_daily_price_bundle=_fake_import),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_daily_price_bundle",
            "--input",
            str(bundle_path),
            "--warm-redis-symbols",
            "0",
        ],
    )

    assert import_daily_script.main() == 0
    assert captured["db"] == "db-session"
    assert captured["input_path"] == bundle_path
    assert captured["warm_redis_symbols"] == 0
    assert "Daily price bundle import complete:" in capsys.readouterr().out


def test_cn_daily_price_shard_selection_is_sorted_and_one_based():
    assert cn_shard_script.select_shard_symbols(
        ["600000.SS", "000001.SZ", "000002.SZ", "300001.SZ", "688001.SS"],
        shard_index=2,
        shard_count=3,
    ) == ["000002.SZ", "688001.SS"]


def test_cn_daily_price_shard_bootstrap_fetches_missing_symbols_and_exports_shard(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {"stored": [], "export": None}

    class FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def all(self):
            return [
                ("000001.SZ",),
                ("000002.SZ",),
                ("600000.SS",),
                ("688001.SS",),
            ]

    class FakeDb:
        def query(self, *args, **kwargs):
            return FakeQuery()

    class FakeService:
        DAILY_PRICE_BAR_PERIOD = "2y"

        def symbols_missing_as_of(self, db, *, symbols, as_of_date):
            assert symbols == ["000001.SZ", "600000.SS"]
            assert as_of_date == date(2026, 5, 8)
            return ["600000.SS"]

        def export_daily_price_bundle(self, db, **kwargs):
            captured["export"] = kwargs
            kwargs["output_path"].write_bytes(b"bundle")
            kwargs["latest_manifest_path"].write_text("{}", encoding="utf-8")
            return {
                "market": kwargs["market"],
                "symbol_count": len(kwargs["symbols"]),
                "rows": 1,
            }

        price_cache = SimpleNamespace(
            store_batch_in_cache=lambda batch, also_store_db=True: captured["stored"].append(
                (batch, also_store_db)
            )
            or len(batch)
        )

    class FakeFetcher:
        def fetch_prices_in_batches(self, symbols, *, period, market):
            assert symbols == ["600000.SS"]
            assert period == "2y"
            assert market == "CN"
            return {
                "600000.SS": {
                    "has_error": False,
                    "price_data": SimpleNamespace(empty=False),
                }
            }

    stats = cn_shard_script.bootstrap_cn_daily_price_shard(
        db=FakeDb(),
        service=FakeService(),
        fetcher=FakeFetcher(),
        shard_index=1,
        shard_count=2,
        as_of_date=date(2026, 5, 8),
        output_dir=tmp_path,
        batch_size=10,
    )

    assert stats["shard_symbols"] == 2
    assert stats["missing_symbols"] == 1
    assert stats["refreshed_symbols"] == 1
    assert captured["stored"] == [({"600000.SS": SimpleNamespace(empty=False)}, True)]
    assert captured["export"]["symbols"] == ["000001.SZ", "600000.SS"]
    assert captured["export"]["bundle_asset_name"] == "daily-price-cn-20260508-shard-1-of-2.json.gz"
    assert "CN daily price shard complete:" in capsys.readouterr().out


def test_sync_weekly_reference_from_github_script_calls_service(monkeypatch, capsys):
    monkeypatch.setattr(sync_weekly_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(sync_weekly_script, "SessionLocal", _fake_session)
    monkeypatch.setattr(
        sync_weekly_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            sync_weekly_reference_from_github=lambda db, **kwargs: {
                "status": "success",
                "market": kwargs["market"],
                "source_revision": "fundamentals_v1_us:20260418120000",
            }
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sync_weekly_reference_from_github", "--market", "US"],
    )

    assert sync_weekly_script.main() == 0
    assert "Weekly GitHub sync result:" in capsys.readouterr().out


def test_sync_daily_price_bundle_from_github_script_calls_service(monkeypatch, capsys):
    monkeypatch.setattr(sync_daily_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(sync_daily_script, "SessionLocal", _fake_session)
    monkeypatch.setattr(
        sync_daily_script,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(
            sync_from_github=lambda db, **kwargs: {
                "status": "success",
                "market": kwargs["market"],
                "source_revision": "daily_prices_us:20260421120000",
            }
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sync_daily_price_bundle_from_github", "--market", "US"],
    )

    assert sync_daily_script.main() == 0
    assert "Daily GitHub price sync result:" in capsys.readouterr().out


def test_sync_daily_price_bundle_from_github_script_passes_allow_stale(monkeypatch, capsys):
    monkeypatch.setattr(sync_daily_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(sync_daily_script, "SessionLocal", _fake_session)
    captured_kwargs = {}

    def _sync_from_github(db, **kwargs):
        _ = db
        captured_kwargs.update(kwargs)
        return {
            "status": "success",
            "market": kwargs["market"],
            "source_revision": "daily_prices_us:20260421120000",
        }

    monkeypatch.setattr(
        sync_daily_script,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(sync_from_github=_sync_from_github),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sync_daily_price_bundle_from_github", "--market", "US", "--allow-stale"],
    )

    assert sync_daily_script.main() == 0
    assert captured_kwargs["allow_stale"] is True
    assert "Daily GitHub price sync result:" in capsys.readouterr().out


def test_sync_daily_price_bundle_treats_live_only_as_non_fatal(monkeypatch, capsys):
    monkeypatch.setattr(sync_daily_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(sync_daily_script, "SessionLocal", _fake_session)
    monkeypatch.setattr(
        sync_daily_script,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(
            sync_from_github=lambda db, **kwargs: {
                "status": "live_only",
                "market": kwargs["market"],
                "reason": "No prior daily price bundle found",
            }
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sync_daily_price_bundle_from_github", "--market", "US"],
    )

    assert sync_daily_script.main() == 0
    stdout = capsys.readouterr().out
    assert "Daily GitHub price sync result:" in stdout
    assert "live_only" in stdout


def test_sync_daily_price_bundle_treats_missing_manifest_as_non_fatal(monkeypatch, capsys):
    monkeypatch.setattr(sync_daily_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(sync_daily_script, "SessionLocal", _fake_session)
    monkeypatch.setattr(
        sync_daily_script,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(
            sync_from_github=lambda db, **kwargs: {
                "status": "missing_manifest",
                "market": kwargs["market"],
                "reason": "daily-price-latest-us.json not found",
            }
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sync_daily_price_bundle_from_github", "--market", "US"],
    )

    assert sync_daily_script.main() == 0
    stdout = capsys.readouterr().out
    assert "Daily GitHub price sync result:" in stdout
    assert "missing_manifest" in stdout
