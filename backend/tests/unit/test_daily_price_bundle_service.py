from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.app_settings import AppSetting
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
from app.services.daily_price_bundle_reader import (
    StreamingJsonReader,
    read_daily_price_bundle_metadata,
)
from app.services.daily_price_bundle_service import DailyPriceBundleService


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _stock_row(symbol: str, market: str, exchange: str, market_cap: float) -> StockUniverse:
    return StockUniverse(
        symbol=symbol,
        market=market,
        exchange=exchange,
        is_active=True,
        status=UNIVERSE_STATUS_ACTIVE,
        status_reason="active",
        market_cap=market_cap,
    )


def _price_row(symbol: str, day: date, close: float) -> StockPrice:
    return StockPrice(
        symbol=symbol,
        date=day,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        adj_close=close - 0.5,
        volume=1_000_000,
    )


def _bundle_price(
    *,
    day: str = "2026-04-18",
    open_price: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
) -> dict[str, object]:
    return {
        "date": day,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": 1_000_000,
    }


def _make_service(session_factory):
    _ = session_factory
    return DailyPriceBundleService()


def test_daily_price_bundle_round_trips_and_preserves_other_market_rows(tmp_path):
    export_session_factory = _make_session()
    export_db = export_session_factory()
    export_db.add_all(
        [
            _stock_row("AAPL", "US", "NASDAQ", 1000.0),
            _stock_row("MSFT", "US", "NASDAQ", 900.0),
            _stock_row("0700.HK", "HK", "XHKG", 800.0),
            _price_row("AAPL", date(2026, 4, 17), 100.0),
            _price_row("AAPL", date(2026, 4, 18), 101.0),
            _price_row("MSFT", date(2026, 4, 18), 202.0),
            _price_row("0700.HK", date(2026, 4, 18), 300.0),
        ]
    )
    export_db.commit()

    service = _make_service(export_session_factory)
    bundle_path = tmp_path / "daily-price-us-20260418.json.gz"
    manifest_path = tmp_path / "daily-price-latest-us.json"

    export_stats = service.export_daily_price_bundle(
        export_db,
        market="US",
        output_path=bundle_path,
        bundle_asset_name=bundle_path.name,
        latest_manifest_path=manifest_path,
        as_of_date=date(2026, 4, 18),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert export_stats["market"] == "US"
    assert export_stats["bar_period"] == "2y"
    assert export_stats["symbol_count"] == 2
    assert manifest["market"] == "US"
    assert manifest["bar_period"] == "2y"
    assert manifest["symbol_count"] == 2

    import_session_factory = _make_session()
    import_db = import_session_factory()
    import_db.add_all(
        [
            _stock_row("AAPL", "US", "NASDAQ", 1000.0),
            _stock_row("MSFT", "US", "NASDAQ", 900.0),
            _stock_row("0700.HK", "HK", "XHKG", 800.0),
            _price_row("0700.HK", date(2026, 4, 18), 300.0),
        ]
    )
    import_db.commit()

    import_service = _make_service(import_session_factory)
    import_stats = import_service.import_daily_price_bundle(
        import_db,
        input_path=bundle_path,
        warm_redis_symbols=0,
    )

    assert import_stats["market"] == "US"
    assert import_stats["imported_symbols"] == 2
    assert import_stats["imported_rows"] == 3
    assert import_db.query(StockPrice).filter(StockPrice.symbol == "0700.HK").count() == 1
    assert import_db.query(StockPrice).filter(StockPrice.symbol == "AAPL").count() == 2
    assert import_db.query(StockPrice).filter(StockPrice.symbol == "MSFT").count() == 1

    sync_state = import_db.query(AppSetting).filter(
        AppSetting.key == import_service.sync_state_key("US")
    ).one()
    sync_payload = json.loads(sync_state.value)
    assert sync_payload["market"] == "US"
    assert sync_payload["as_of_date"] == "2026-04-18"
    assert sync_payload["bar_period"] == "2y"

    row_count_before = import_db.query(StockPrice).count()
    import_service.import_daily_price_bundle(
        import_db,
        input_path=bundle_path,
        warm_redis_symbols=0,
    )
    assert import_db.query(StockPrice).count() == row_count_before

    export_db.close()
    import_db.close()


def test_export_daily_price_bundle_can_filter_to_shard_symbols(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    db.add_all(
        [
            _stock_row("000001.SZ", "CN", "SZSE", 1000.0),
            _stock_row("000002.SZ", "CN", "SZSE", 900.0),
            _stock_row("600000.SS", "CN", "SSE", 800.0),
            _price_row("000001.SZ", date(2026, 5, 8), 10.0),
            _price_row("000002.SZ", date(2026, 5, 8), 20.0),
            _price_row("600000.SS", date(2026, 5, 8), 30.0),
        ]
    )
    db.commit()

    service = _make_service(session_factory)
    bundle_path = tmp_path / "daily-price-cn-20260508-shard-2-of-2.json.gz"
    manifest_path = tmp_path / "daily-price-cn-20260508-shard-2-of-2.manifest.json"

    stats = service.export_daily_price_bundle(
        db,
        market="CN",
        output_path=bundle_path,
        bundle_asset_name=bundle_path.name,
        latest_manifest_path=manifest_path,
        as_of_date=date(2026, 5, 8),
        symbols=["000002.SZ", "600000.SS"],
    )

    payload = service._read_bundle_payload(bundle_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert stats["market"] == "CN"
    assert stats["symbol_count"] == 2
    assert [row["symbol"] for row in payload["rows"]] == ["000002.SZ", "600000.SS"]
    assert manifest["symbol_count"] == 2

    db.close()


def test_export_daily_price_bundle_can_require_complete_symbol_coverage(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    db.add_all(
        [
            _stock_row("000001.SZ", "CN", "SZSE", 1000.0),
            _stock_row("000002.SZ", "CN", "SZSE", 900.0),
            _price_row("000001.SZ", date(2026, 5, 8), 10.0),
            _price_row("000002.SZ", date(2026, 5, 7), 20.0),
        ]
    )
    db.commit()

    service = _make_service(session_factory)

    with pytest.raises(ValueError, match="Missing 1 CN symbols"):
        service.export_daily_price_bundle(
            db,
            market="CN",
            output_path=tmp_path / "daily-price-cn-20260508.json.gz",
            bundle_asset_name="daily-price-cn-20260508.json.gz",
            latest_manifest_path=tmp_path / "daily-price-latest-cn.json",
            as_of_date=date(2026, 5, 8),
            require_complete=True,
        )

    db.close()


def test_export_daily_price_bundle_can_allow_stale_rows_when_complete_required(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    db.add_all(
        [
            _stock_row("000001.SZ", "CN", "SZSE", 1000.0),
            _stock_row("000004.SZ", "CN", "SZSE", 900.0),
            _price_row("000001.SZ", date(2026, 5, 8), 10.0),
            _price_row("000004.SZ", date(2026, 4, 27), 20.0),
        ]
    )
    db.commit()

    service = _make_service(session_factory)
    bundle_path = tmp_path / "daily-price-cn-20260508.json.gz"

    stats = service.export_daily_price_bundle(
        db,
        market="CN",
        output_path=bundle_path,
        bundle_asset_name=bundle_path.name,
        latest_manifest_path=tmp_path / "daily-price-latest-cn.json",
        as_of_date=date(2026, 5, 8),
        require_complete=True,
        allow_stale_complete=True,
    )

    payload = service._read_bundle_payload(bundle_path)

    assert stats["symbol_count"] == 2
    assert stats["stale_symbol_count"] == 1
    assert [row["symbol"] for row in payload["rows"]] == ["000001.SZ", "000004.SZ"]

    db.close()


def test_export_daily_price_bundle_allows_missing_rows_at_coverage_threshold(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    symbols = [f"00000{index}.SZ" for index in range(10)]
    db.add_all(
        [_stock_row(symbol, "CN", "SZSE", 1000.0 - index) for index, symbol in enumerate(symbols)]
        + [
            _price_row(symbol, date(2026, 5, 8), 10.0 + index)
            for index, symbol in enumerate(symbols[:-1])
        ]
    )
    db.commit()

    service = _make_service(session_factory)
    bundle_path = tmp_path / "daily-price-cn-20260508.json.gz"
    manifest_path = tmp_path / "daily-price-latest-cn.json"

    stats = service.export_daily_price_bundle(
        db,
        market="CN",
        output_path=bundle_path,
        bundle_asset_name=bundle_path.name,
        latest_manifest_path=manifest_path,
        as_of_date=date(2026, 5, 8),
        require_complete=True,
        allow_stale_complete=True,
        min_symbol_coverage=0.9,
    )

    payload = service._read_bundle_payload(bundle_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert stats["symbol_universe_count"] == 10
    assert stats["covered_symbol_count"] == 9
    assert stats["symbol_coverage"] == 0.9
    assert stats["min_symbol_coverage"] == 0.9
    assert payload["symbol_coverage"] == 0.9
    assert manifest["symbol_coverage"] == 0.9

    db.close()


def test_export_daily_price_bundle_fails_below_coverage_threshold(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    symbols = [f"00000{index}.SZ" for index in range(10)]
    db.add_all(
        [_stock_row(symbol, "CN", "SZSE", 1000.0 - index) for index, symbol in enumerate(symbols)]
        + [
            _price_row(symbol, date(2026, 5, 8), 10.0 + index)
            for index, symbol in enumerate(symbols[:8])
        ]
    )
    db.commit()

    service = _make_service(session_factory)

    with pytest.raises(ValueError, match="coverage 80.00% is below required 90.00%"):
        service.export_daily_price_bundle(
            db,
            market="CN",
            output_path=tmp_path / "daily-price-cn-20260508.json.gz",
            bundle_asset_name="daily-price-cn-20260508.json.gz",
            latest_manifest_path=tmp_path / "daily-price-latest-cn.json",
            as_of_date=date(2026, 5, 8),
            require_complete=True,
            allow_stale_complete=True,
            min_symbol_coverage=0.9,
        )

    db.close()


def test_export_daily_price_bundle_rejects_non_finite_coverage_threshold(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    db.add_all(
        [
            _stock_row("000001.SZ", "CN", "SZSE", 1000.0),
            _price_row("000001.SZ", date(2026, 5, 8), 10.0),
        ]
    )
    db.commit()

    service = _make_service(session_factory)

    with pytest.raises(ValueError, match="min_symbol_coverage must be between 0 and 1"):
        service.export_daily_price_bundle(
            db,
            market="CN",
            output_path=tmp_path / "daily-price-cn-20260508.json.gz",
            bundle_asset_name="daily-price-cn-20260508.json.gz",
            latest_manifest_path=tmp_path / "daily-price-latest-cn.json",
            as_of_date=date(2026, 5, 8),
            min_symbol_coverage=float("nan"),
        )

    db.close()


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


def test_daily_price_bundle_metadata_reads_scalar_across_small_chunks(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(StreamingJsonReader, "_CHUNK_SIZE", 3)
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": DailyPriceBundleService.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": DailyPriceBundleService.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 123456,
                "rows": [],
            }
        ),
        encoding="utf-8",
    )

    metadata = read_daily_price_bundle_metadata(bundle_path)

    assert metadata["symbol_count"] == 123456


def test_import_daily_price_bundle_skips_redis_warm_for_two_year_bundle(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    db.add(_stock_row("AAPL", "US", "NASDAQ", 1000.0))
    db.commit()

    service = DailyPriceBundleService()
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "generated_at": "2026-04-18T12:00:00Z",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 1,
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
            }
        ),
        encoding="utf-8",
    )

    result = service.import_daily_price_bundle(
        db,
        input_path=bundle_path,
        warm_redis_symbols=1,
    )

    assert result["redis_warmed_symbols"] == 0
    assert db.query(StockPrice).filter(StockPrice.symbol == "AAPL").count() == 1
    db.close()


def test_import_daily_price_bundle_streams_rows_without_full_payload_read(
    monkeypatch,
    tmp_path,
):
    session_factory = _make_session()
    db = session_factory()
    db.add(_stock_row("AAPL", "US", "NASDAQ", 1000.0))
    db.commit()

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
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        service,
        "_read_bundle_payload",
        MagicMock(side_effect=AssertionError("full payload read")),
    )

    result = service.import_daily_price_bundle(
        db,
        input_path=bundle_path,
        warm_redis_symbols=0,
    )

    assert result["imported_symbols"] == 1
    assert result["imported_rows"] == 1
    assert db.query(StockPrice).filter(StockPrice.symbol == "AAPL").count() == 1
    service._read_bundle_payload.assert_not_called()
    db.close()


def test_import_daily_price_bundle_flushes_streamed_rows_in_bounded_chunks(
    monkeypatch,
    tmp_path,
):
    session_factory = _make_session()
    db = session_factory()
    symbol_count = 205
    db.add_all(
        [
            _stock_row(f"T{index:04d}", "US", "NYSE", 1000.0 - index)
            for index in range(symbol_count)
        ]
    )
    db.commit()

    service = DailyPriceBundleService()
    persist_spy = MagicMock(wraps=service._persist_bundle_price_batch)
    monkeypatch.setattr(service, "_persist_bundle_price_batch", persist_spy)
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": symbol_count,
                "rows": [
                    {
                        "symbol": f"T{index:04d}",
                        "prices": [
                            {
                                "date": "2026-04-18",
                                "open": float(index),
                                "high": float(index + 1),
                                "low": float(index - 1),
                                "close": float(index) + 0.5,
                                "adj_close": float(index),
                                "volume": 1_000_000 + index,
                            }
                        ],
                    }
                    for index in range(symbol_count)
                ],
            }
        ),
        encoding="utf-8",
    )

    result = service.import_daily_price_bundle(
        db,
        input_path=bundle_path,
        warm_redis_symbols=0,
    )

    assert result["imported_symbols"] == symbol_count
    assert result["imported_rows"] == symbol_count
    assert persist_spy.call_count == 3
    chunk_sizes = [
        len(call.args[1])
        for call in persist_spy.call_args_list
    ]
    assert chunk_sizes == [100, 100, 5]
    db.close()


def test_import_daily_price_bundle_rejects_malformed_streamed_row(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    service = DailyPriceBundleService()
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 1,
                "rows": [42],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row 1 must be an object"):
        service.import_daily_price_bundle(
            db,
            input_path=bundle_path,
            warm_redis_symbols=0,
        )

    assert db.query(AppSetting).count() == 0
    db.close()


def test_import_daily_price_bundle_rejects_invalid_ohlcv_and_rolls_back(tmp_path):
    session_factory = _make_session()
    db = session_factory()
    service = DailyPriceBundleService()
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 1,
                "rows": [
                    {
                        "symbol": "AAPL",
                        "prices": [
                            {
                                "date": "2026-04-18",
                                "open": 100.0,
                                "high": 101.0,
                                "low": 99.0,
                                "close": None,
                                "adj_close": 100.0,
                                "volume": 1_000_000,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid OHLCV"):
        service.import_daily_price_bundle(
            db,
            input_path=bundle_path,
            warm_redis_symbols=0,
        )

    assert db.query(AppSetting).count() == 0
    assert db.query(StockPrice).filter(StockPrice.symbol == "AAPL").count() == 0
    db.close()


def test_import_daily_price_bundle_rejects_duplicate_symbols_and_rolls_back(
    monkeypatch,
    tmp_path,
):
    session_factory = _make_session()
    db = session_factory()
    service = DailyPriceBundleService()
    monkeypatch.setattr(service, "DAILY_PRICE_IMPORT_CHUNK_SIZE", 1)
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 2,
                "rows": [
                    {"symbol": "AAPL", "prices": [_bundle_price()]},
                    {"symbol": "aapl", "prices": [_bundle_price(day="2026-04-19")]},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate symbol AAPL"):
        service.import_daily_price_bundle(
            db,
            input_path=bundle_path,
            warm_redis_symbols=0,
        )

    assert db.query(AppSetting).count() == 0
    assert db.query(StockPrice).filter(StockPrice.symbol == "AAPL").count() == 0
    db.close()


def test_import_daily_price_bundle_rejects_symbol_count_mismatch_and_rolls_back(
    tmp_path,
):
    session_factory = _make_session()
    db = session_factory()
    service = DailyPriceBundleService()
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 2,
                "rows": [
                    {"symbol": "AAPL", "prices": [_bundle_price()]},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match manifest symbol_count 2"):
        service.import_daily_price_bundle(
            db,
            input_path=bundle_path,
            warm_redis_symbols=0,
        )

    assert db.query(AppSetting).count() == 0
    assert db.query(StockPrice).filter(StockPrice.symbol == "AAPL").count() == 0
    db.close()


def test_import_daily_price_bundle_rolls_back_state_on_persist_error(
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
        "_persist_bundle_price_batch",
        MagicMock(side_effect=RuntimeError("db failed")),
    )
    bundle_path = tmp_path / "daily-price-us.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": service.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
                "market": "US",
                "as_of_date": "2026-04-18",
                "bar_period": service.DAILY_PRICE_BAR_PERIOD,
                "source_revision": "daily_prices_us:20260418120000",
                "symbol_count": 1,
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
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="db failed"):
        service.import_daily_price_bundle(
            db,
            input_path=bundle_path,
            warm_redis_symbols=0,
        )

    assert db.query(AppSetting).count() == 0
    assert db.query(StockPrice).filter(StockPrice.symbol == "AAPL").count() == 0
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
