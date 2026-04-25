from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
from app.services.breadth_calculator_service import BreadthCalculatorService


def _make_price_df(end_date: date, base_close: float = 100.0) -> pd.DataFrame:
    index = pd.bdate_range(end=end_date, periods=80)
    closes = [base_close + i for i in range(len(index))]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(index),
        },
        index=index,
    )


def _flat_price_df(end_date: date, close: float = 100.0, periods: int = 80) -> pd.DataFrame:
    index = pd.bdate_range(end=end_date, periods=periods)
    closes = [close] * len(index)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * len(index),
        },
        index=index,
    )


def _make_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[StockUniverse.__table__, MarketBreadth.__table__])
    testing_session_local = sessionmaker(bind=engine)
    return testing_session_local()


def _add_breadth_row(
    db,
    row_date: date,
    *,
    up: int,
    down: int,
    total: int = 2,
) -> None:
    db.add(MarketBreadth(
        date=row_date,
        stocks_up_4pct=up,
        stocks_down_4pct=down,
        ratio_5day=None,
        ratio_10day=None,
        stocks_up_25pct_quarter=0,
        stocks_down_25pct_quarter=0,
        stocks_up_25pct_month=0,
        stocks_down_25pct_month=0,
        stocks_up_50pct_month=0,
        stocks_down_50pct_month=0,
        stocks_up_13pct_34days=0,
        stocks_down_13pct_34days=0,
        total_stocks_scanned=total,
    ))


def test_calculate_daily_breadth_uses_bulk_cached_prices(monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAA"),
        SimpleNamespace(symbol="BBB"),
    ]

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(date(2026, 3, 20), 100.0),
        "BBB": _make_price_df(date(2026, 3, 20), 200.0),
    }
    price_cache.get_historical_data.side_effect = AssertionError(
        "breadth should not use per-symbol historical fetches"
    )
    calculator = BreadthCalculatorService(db, price_cache)
    monkeypatch.setattr(
        calculator,
        "_calculate_ratios",
        lambda calculation_date: {"ratio_5day": 1.5, "ratio_10day": 2.5},
    )

    metrics = calculator.calculate_daily_breadth(date(2026, 3, 20), cache_only=True)

    assert metrics["total_stocks_scanned"] == 2
    assert metrics["skipped_stocks"] == 0
    assert metrics["ratio_5day"] == 1.5
    assert metrics["ratio_10day"] == 2.5
    assert metrics["cache_miss_stocks"] == 0
    price_cache.get_many_cached_only_fresh.assert_called_once_with(["AAA", "BBB"], period="2y")
    price_cache.get_historical_data.assert_not_called()


def test_calculate_daily_breadth_counts_fresh_cache_misses(monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAA"),
        SimpleNamespace(symbol="BBB"),
    ]

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(date(2026, 3, 20), 100.0),
        "BBB": None,
    }
    calculator = BreadthCalculatorService(db, price_cache)
    monkeypatch.setattr(
        calculator,
        "_calculate_ratios",
        lambda calculation_date: {"ratio_5day": 1.5, "ratio_10day": 2.5},
    )

    metrics = calculator.calculate_daily_breadth(date(2026, 3, 20), cache_only=True)

    assert metrics["total_stocks_scanned"] == 1
    assert metrics["cache_miss_stocks"] == 1
    assert metrics["skipped_stocks"] == 1


def test_calculate_daily_breadth_preserves_historical_fetch_fallback(monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAA"),
        SimpleNamespace(symbol="BBB"),
    ]

    price_cache = MagicMock()
    price_cache.get_many_cached_only.return_value = {
        "AAA": _make_price_df(date(2026, 3, 19), 100.0),
        "BBB": None,
    }
    calculator = BreadthCalculatorService(db, price_cache)
    price_cache.get_historical_data.return_value = _make_price_df(date(2026, 3, 19), 150.0)
    monkeypatch.setattr(
        calculator,
        "_calculate_ratios",
        lambda calculation_date: {"ratio_5day": 1.0, "ratio_10day": 1.2},
    )

    metrics = calculator.calculate_daily_breadth(date(2026, 3, 19), cache_only=False)

    assert metrics["total_stocks_scanned"] == 2
    assert metrics["cache_miss_stocks"] == 1
    assert metrics["skipped_stocks"] == 0
    price_cache.get_historical_data.assert_called_once_with(symbol="BBB", period="2y")


def test_calculate_stock_metrics_reads_cached_only(monkeypatch):
    db = MagicMock(spec=[])
    price_cache = MagicMock()
    price_cache.get_historical_data.return_value = _make_price_df(date(2026, 3, 20), 150.0)
    calculator = BreadthCalculatorService(db, price_cache)
    metrics = calculator._calculate_stock_metrics("AAA", date(2026, 3, 20))

    assert metrics is not None
    assert set(metrics) == {"pct_change_1d", "pct_change_21d", "pct_change_34d", "pct_change_63d"}
    price_cache.get_historical_data.assert_called_once_with(symbol="AAA", period="2y")


def test_backfill_range_reuses_loaded_histories_and_computes_chronological_ratios(monkeypatch):
    db = _make_db_session()
    db.add_all([
        StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        StockUniverse(symbol="BBB", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
    ])

    prior_dates = [date(2026, 3, day) for day in range(2, 12)]
    for prior_date in prior_dates:
        db.add(MarketBreadth(
            date=prior_date,
            stocks_up_4pct=2,
            stocks_down_4pct=1,
            ratio_5day=2.0,
            ratio_10day=2.0,
            stocks_up_25pct_quarter=0,
            stocks_down_25pct_quarter=0,
            stocks_up_25pct_month=0,
            stocks_down_25pct_month=0,
            stocks_up_50pct_month=0,
            stocks_down_50pct_month=0,
            stocks_up_13pct_34days=0,
            stocks_down_13pct_34days=0,
            total_stocks_scanned=2,
        ))
    db.commit()

    aaa_df = _flat_price_df(date(2026, 3, 13))
    bbb_df = _flat_price_df(date(2026, 3, 13))
    aaa_df.loc[pd.Timestamp(date(2026, 3, 12)), "Close"] = 105.0
    aaa_df.loc[pd.Timestamp(date(2026, 3, 13)), "Close"] = 110.0
    bbb_df.loc[pd.Timestamp(date(2026, 3, 12)), "Close"] = 95.0
    bbb_df.loc[pd.Timestamp(date(2026, 3, 13)), "Close"] = 100.0

    price_cache = MagicMock()
    price_cache.get_many_cached_only.return_value = {"AAA": aaa_df, "BBB": None}
    price_cache.get_historical_data.return_value = bbb_df
    service = BreadthCalculatorService(db, price_cache)
    trading_dates = [date(2026, 3, 12), date(2026, 3, 13)]

    result = service.backfill_range(trading_dates[0], trading_dates[-1], trading_dates=trading_dates)

    assert result == {
        "total_dates": 2,
        "processed": 2,
        "errors": 0,
        "error_dates": [],
    }
    price_cache.get_many_cached_only.assert_called_once_with(["AAA", "BBB"], period="2y")
    price_cache.get_historical_data.assert_called_once_with(symbol="BBB", period="2y")

    stored = db.query(MarketBreadth).filter(
        MarketBreadth.date.in_(trading_dates)
    ).order_by(MarketBreadth.date.asc()).all()

    assert len(stored) == 2
    assert stored[0].stocks_up_4pct == 1
    assert stored[0].stocks_down_4pct == 1
    assert stored[0].ratio_5day == 2.0
    assert stored[0].ratio_10day == 2.0
    assert stored[1].stocks_up_4pct == 2
    assert stored[1].stocks_down_4pct == 0
    assert stored[1].ratio_5day == 1.8
    assert stored[1].ratio_10day == 1.9


def test_backfill_range_fallback_uses_market_calendar(monkeypatch):
    db = _make_db_session()
    price_cache = MagicMock()
    service = BreadthCalculatorService(db, price_cache, market="HK")

    class _FakeCalendarService:
        def is_trading_day(self, market, current_date):
            assert market == "HK"
            return current_date == date(2026, 3, 13)

    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _FakeCalendarService(),
    )

    result = service.backfill_range(date(2026, 3, 12), date(2026, 3, 14))

    assert result == {
        "total_dates": 1,
        "processed": 0,
        "errors": 1,
        "error_dates": ["2026-03-13"],
    }
    price_cache.get_many_cached_only.assert_not_called()


def test_backfill_range_is_idempotent_for_existing_records(monkeypatch):
    db = _make_db_session()
    db.add(StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE))
    db.commit()

    up_df = _flat_price_df(date(2026, 3, 12))
    up_df.loc[pd.Timestamp(date(2026, 3, 12)), "Close"] = 105.0
    down_df = _flat_price_df(date(2026, 3, 12))
    down_df.loc[pd.Timestamp(date(2026, 3, 12)), "Close"] = 95.0

    price_cache = MagicMock()
    price_cache.get_many_cached_only.side_effect = [
        {"AAA": up_df},
        {"AAA": down_df},
    ]
    service = BreadthCalculatorService(db, price_cache)
    trading_date = date(2026, 3, 12)
    service.backfill_range(trading_date, trading_date, trading_dates=[trading_date])
    service.backfill_range(trading_date, trading_date, trading_dates=[trading_date])

    records = db.query(MarketBreadth).filter(MarketBreadth.date == trading_date).all()

    assert len(records) == 1
    assert records[0].stocks_up_4pct == 0
    assert records[0].stocks_down_4pct == 1


def test_backfill_range_cache_only_skips_historical_fetch_fallback():
    db = _make_db_session()
    db.add_all([
        StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        StockUniverse(symbol="BBB", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
    ])
    db.commit()

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(date(2026, 3, 20), 100.0),
        "BBB": None,
    }
    price_cache.get_historical_data.side_effect = AssertionError(
        "cache-only backfill must not fetch per-symbol history"
    )
    service = BreadthCalculatorService(db, price_cache)
    trading_date = date(2026, 3, 12)

    result = service.backfill_range(
        trading_date,
        trading_date,
        trading_dates=[trading_date],
        cache_only=True,
    )

    assert result["total_dates"] == 1
    assert result["processed"] == 1
    assert result["errors"] == 0
    price_cache.get_many_cached_only_fresh.assert_called_once_with(["AAA", "BBB"], period="2y")
    price_cache.get_historical_data.assert_not_called()


def test_vectorized_stock_metrics_preserve_invalid_close_semantics():
    service = BreadthCalculatorService(MagicMock(), MagicMock())
    latest_date = date(2026, 3, 20)

    previous_nan = _flat_price_df(latest_date)
    previous_nan.iloc[-2, previous_nan.columns.get_loc("Close")] = float("nan")
    previous_nan.loc[pd.Timestamp(latest_date), "Close"] = 105.0

    current_nan = _flat_price_df(latest_date)
    current_nan.iloc[-2, current_nan.columns.get_loc("Close")] = 125.0
    current_nan.loc[pd.Timestamp(latest_date), "Close"] = float("nan")

    previous_zero = _flat_price_df(latest_date)
    previous_zero.iloc[-2, previous_zero.columns.get_loc("Close")] = 0.0
    previous_zero.loc[pd.Timestamp(latest_date), "Close"] = 105.0

    for prices in (previous_nan, current_nan, previous_zero):
        metrics = service._calculate_stock_metrics_by_date_from_prices(
            prices,
            [latest_date],
        )
        assert metrics[latest_date]["pct_change_1d"] == 0.0


def test_fill_gaps_delegates_to_single_backfill_range_call(monkeypatch):
    db = _make_db_session()
    price_cache = MagicMock()
    service = BreadthCalculatorService(db, price_cache)
    expected = {
        "total_dates": 2,
        "processed": 2,
        "errors": 0,
        "error_dates": [],
    }
    backfill_range = MagicMock(return_value=expected)
    monkeypatch.setattr(service, "backfill_range", backfill_range)
    monkeypatch.setattr(
        service,
        "calculate_daily_breadth",
        MagicMock(side_effect=AssertionError("fill_gaps should use range backfill")),
    )

    result = service.fill_gaps([date(2026, 3, 16), date(2026, 3, 12)])

    assert result == expected
    backfill_range.assert_called_once_with(
        date(2026, 3, 12),
        date(2026, 3, 16),
        trading_dates=[date(2026, 3, 12), date(2026, 3, 16)],
    )


def test_backfill_range_sparse_dates_include_existing_intervening_counts_in_ratios():
    db = _make_db_session()
    db.add_all([
        StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        StockUniverse(symbol="BBB", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
    ])

    for prior_date in [
        date(2026, 2, 26),
        date(2026, 2, 27),
        date(2026, 3, 2),
        date(2026, 3, 3),
        date(2026, 3, 4),
        date(2026, 3, 5),
        date(2026, 3, 6),
        date(2026, 3, 9),
        date(2026, 3, 10),
        date(2026, 3, 11),
    ]:
        _add_breadth_row(db, prior_date, up=1, down=1)
    _add_breadth_row(db, date(2026, 3, 13), up=10, down=1)
    db.commit()

    aaa_df = _flat_price_df(date(2026, 3, 16))
    bbb_df = _flat_price_df(date(2026, 3, 16))
    for item_date, aaa_close, bbb_close in (
        (date(2026, 3, 12), 105.0, 95.0),
        (date(2026, 3, 13), 100.0, 100.0),
        (date(2026, 3, 16), 105.0, 95.0),
    ):
        aaa_df.loc[pd.Timestamp(item_date), "Close"] = aaa_close
        bbb_df.loc[pd.Timestamp(item_date), "Close"] = bbb_close

    price_cache = MagicMock()
    price_cache.get_many_cached_only.return_value = {"AAA": aaa_df, "BBB": bbb_df}
    service = BreadthCalculatorService(db, price_cache)

    result = service.backfill_range(
        date(2026, 3, 12),
        date(2026, 3, 16),
        trading_dates=[date(2026, 3, 12), date(2026, 3, 16)],
    )

    assert result["processed"] == 2
    rows = {
        row.date: row
        for row in db.query(MarketBreadth)
        .filter(MarketBreadth.date.in_([date(2026, 3, 12), date(2026, 3, 16)]))
        .all()
    }
    assert rows[date(2026, 3, 12)].ratio_5day == 1.0
    assert rows[date(2026, 3, 16)].ratio_5day == 2.8


def test_backfill_range_vectorized_changes_preserve_rounded_thresholds():
    db = _make_db_session()
    symbols = ["UP4", "UP13", "UP25", "UP50"]
    db.add_all([
        StockUniverse(symbol=symbol, is_active=True, status=UNIVERSE_STATUS_ACTIVE)
        for symbol in symbols
    ])
    db.commit()

    latest_date = date(2026, 3, 20)
    latest_closes = {
        "UP4": 103.995,
        "UP13": 112.995,
        "UP25": 124.995,
        "UP50": 149.995,
    }
    price_data = {}
    for symbol, latest_close in latest_closes.items():
        frame = _flat_price_df(latest_date)
        frame.loc[pd.Timestamp(latest_date), "Close"] = latest_close
        price_data[symbol] = frame

    price_cache = MagicMock()
    price_cache.get_many_cached_only.return_value = price_data
    service = BreadthCalculatorService(db, price_cache)

    result = service.backfill_range(latest_date, latest_date, trading_dates=[latest_date])

    assert result["processed"] == 1
    row = db.query(MarketBreadth).filter(MarketBreadth.date == latest_date).one()
    assert row.total_stocks_scanned == 4
    assert row.stocks_up_4pct == 4
    assert row.stocks_up_13pct_34days == 3
    assert row.stocks_up_25pct_month == 2
    assert row.stocks_up_25pct_quarter == 2
    assert row.stocks_up_50pct_month == 1
