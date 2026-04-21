from __future__ import annotations

import json
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.app_settings import AppSetting
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
from app.services.daily_price_bundle_service import DailyPriceBundleService
from app.services.price_cache_service import PriceCacheService


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


def _make_service(session_factory):
    return DailyPriceBundleService(
        price_cache=PriceCacheService(redis_client=None, session_factory=session_factory),
    )


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
