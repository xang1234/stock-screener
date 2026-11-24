from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import app.services.breadth_backfill as breadth_backfill_module
import app.services.breadth_calculator_service as breadth_calculator_module
import pandas as pd
import pytest
from app.database import Base
from app.models.breadth_contributor import (
    MarketBreadthContributor,
    MarketBreadthContributorSnapshot,
)
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import UNIVERSE_STATUS_ACTIVE, StockUniverse
from app.services.breadth.types import (
    BreadthContributorMetadata,
    BreadthIndicatorValues,
    BreadthUniverseMember,
    BreadthUniverseSnapshot,
)
from app.services.breadth.universe import breadth_eligibility_signature
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    DerivedDataExecutionPolicy,
    DerivedDataTargetKind,
    resolve_derived_data_execution_policy,
)
from app.services.fx_service import default_currency_for_market
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _policy(mode: str, target: date):
    return resolve_derived_data_execution_policy(
        execution_policy=mode,
        target_date=target,
        current_date=date(2026, 3, 20),
    )


def _make_price_df(end_date: date, base_close: float = 100.0) -> pd.DataFrame:
    index = pd.bdate_range(end=end_date, periods=80)
    closes = [base_close + i for i in range(len(index))]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1_000_000] * len(index),
        },
        index=index,
    )


def _flat_price_df(
    end_date: date, close: float = 100.0, periods: int = 80
) -> pd.DataFrame:
    index = pd.bdate_range(end=end_date, periods=periods)
    closes = [close] * len(index)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1_000_000] * len(index),
        },
        index=index,
    )


def _make_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            StockUniverse.__table__,
            MarketBreadth.__table__,
            MarketBreadthContributorSnapshot.__table__,
            MarketBreadthContributor.__table__,
        ],
    )
    testing_session_local = sessionmaker(bind=engine)
    return testing_session_local()


@pytest.fixture(autouse=True)
def _use_current_rows_as_default_backfill_universe(monkeypatch):
    """Keep calculator tests independent of lifecycle-event reconstruction."""

    def snapshots(db, market, dates):
        rows = sorted(
            db.query(StockUniverse)
            .filter(
                StockUniverse.market == market,
                StockUniverse.is_active == True,
                StockUniverse.is_common_stock.is_(True),
            )
            .all(),
            key=lambda row: row.symbol,
        )
        members = tuple(
            BreadthUniverseMember(
                row.symbol,
                getattr(row, "currency", None) or default_currency_for_market(market),
            )
            for row in rows
        )
        signature = breadth_eligibility_signature(member.symbol for member in members)
        return {
            calculation_date: BreadthUniverseSnapshot(
                calculation_date=calculation_date,
                members=members,
                broad_signature=signature,
            )
            for calculation_date in dates
        }

    monkeypatch.setattr(
        breadth_backfill_module,
        "build_breadth_universe_snapshots",
        snapshots,
    )
    monkeypatch.setattr(
        breadth_calculator_module,
        "build_breadth_universe_snapshots",
        snapshots,
    )


def _add_breadth_row(
    db,
    row_date: date,
    *,
    up: int,
    down: int,
    total: int = 2,
) -> None:
    db.add(
        MarketBreadth(
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
            broad_universe_count=total,
            calculation_revision=3,
        )
    )


@pytest.mark.parametrize("existing", [False, True])
def test_store_daily_breadth_upserts_in_market_partition(existing):
    db = _make_db_session()
    calc_date = date(2026, 3, 20)
    if existing:
        db.add(
            MarketBreadth(
                market="HK",
                date=calc_date,
                stocks_up_4pct=1,
                stocks_down_4pct=1,
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
                total_stocks_scanned=2,
            )
        )
        db.commit()

    service = BreadthCalculatorService(db, MagicMock(), market="HK")
    metrics = {
        **{
            name: getattr(BreadthIndicatorValues(), name)
            for name in BreadthIndicatorValues.__dataclass_fields__
        },
        "stocks_up_4pct": 12,
        "stocks_down_4pct": 4,
        "ratio_5day": 2.0,
        "ratio_10day": 1.5,
        "total_stocks_scanned": 100,
    }

    service.store_daily_breadth(
        calc_date,
        metrics,
        duration_seconds=1.25,
    )

    rows = (
        db.query(MarketBreadth)
        .filter(
            MarketBreadth.market == "HK",
            MarketBreadth.date == calc_date,
        )
        .all()
    )
    assert len(rows) == 1
    assert rows[0].stocks_up_4pct == 12
    assert rows[0].total_stocks_scanned == 100
    assert rows[0].calculation_duration_seconds == 1.25


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
        breadth_calculator_module.BreadthContributorMetadataLoader,
        "current",
        lambda _db, _market, symbols: {
            symbol: BreadthContributorMetadata(
                company_name=symbol,
                ibd_industry_group="Test Group",
            )
            for symbol in symbols
        },
    )

    result = calculator.calculate_daily_breadth(
        date(2026, 3, 20),
        policy=_policy("refresh_guarded", date(2026, 3, 20)),
    )
    metrics = result.to_metrics_dict()

    assert metrics["total_stocks_scanned"] == 2
    assert metrics["skipped_stocks"] == 0
    assert metrics["ratio_5day"] is None
    assert metrics["ratio_10day"] is None
    assert metrics["cache_miss_stocks"] == 0
    assert result.contributor_snapshot is not None
    assert result.contributor_snapshot.calculation_date == date(2026, 3, 20)
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA", "BBB"],
        period="2y",
        required_as_of_date=date(2026, 3, 20),
        minimum_rows=1,
    )
    price_cache.get_historical_data.assert_not_called()


def test_calculate_daily_breadth_counts_fresh_cache_misses():
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

    result = calculator.calculate_daily_breadth(
        date(2026, 3, 20),
        policy=_policy("refresh_guarded", date(2026, 3, 20)),
    )
    metrics = result.to_metrics_dict()

    assert metrics["total_stocks_scanned"] == 2
    assert metrics["broad_universe_count"] == 2
    assert metrics["cache_miss_stocks"] == 1
    assert metrics["skipped_stocks"] == 1
    assert metrics["candidate_stocks"] == 2
    assert metrics["symbols_with_cached_history"] == 1
    assert metrics["cache_coverage_ratio"] == 0.5
    assert metrics["cache_miss_symbols_sample"] == ["BBB"]
    assert result.coverage.cache_miss_symbols_sample == ("BBB",)


def test_calculate_daily_omits_snapshot_when_metadata_source_fails(monkeypatch):
    calculation_date = date(2026, 3, 20)
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="AAA",
            name="Alpha",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _flat_price_df(calculation_date),
    }
    transaction = {"aborted": False, "rollbacks": 0}
    original_rollback = db.rollback

    def fail_metadata(*_args, **_kwargs):
        transaction["aborted"] = True
        raise RuntimeError("metadata database unavailable")

    def recover_session():
        transaction["rollbacks"] += 1
        transaction["aborted"] = False
        original_rollback()

    monkeypatch.setattr(
        breadth_calculator_module.BreadthContributorMetadataLoader,
        "current",
        fail_metadata,
    )
    monkeypatch.setattr(db, "rollback", recover_session)

    calculator = BreadthCalculatorService(db, price_cache)
    result = calculator.calculate_daily_breadth(
        calculation_date,
        policy=_policy("refresh_guarded", calculation_date),
    )

    assert result.daily_result is not None
    assert result.contributor_snapshot is None
    assert transaction == {"aborted": False, "rollbacks": 1}

    calculator.store_daily_result(
        result.daily_result,
        contributor_snapshot=result.contributor_snapshot,
        duration_seconds=1.0,
    )
    assert db.query(MarketBreadth).count() == 1


def test_historical_daily_breadth_uses_metadata_frozen_for_requested_date(
    monkeypatch,
):
    """Catches current group classifications leaking into historical snapshots."""
    calculation_date = date(2026, 3, 20)
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="AAA",
            name="Alpha",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            is_common_stock=True,
        )
    )
    db.commit()
    prices = _flat_price_df(calculation_date)
    prices.loc[prices.index[-2], ["Close", "Adj Close"]] = 90.0
    prices.loc[prices.index[-1], "Volume"] = 1_100_000
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": prices,
    }
    historical = MagicMock(
        return_value={
            calculation_date: {
                "AAA": BreadthContributorMetadata(
                    company_name="Alpha then",
                    ibd_industry_group="Historical Group",
                )
            }
        }
    )
    current = MagicMock(side_effect=AssertionError("current metadata is look-ahead"))
    monkeypatch.setattr(
        breadth_calculator_module.BreadthContributorMetadataLoader,
        "historical",
        historical,
    )
    monkeypatch.setattr(
        breadth_calculator_module.BreadthContributorMetadataLoader,
        "current",
        current,
    )

    result = BreadthCalculatorService(db, price_cache).calculate_daily_breadth(
        calculation_date,
        policy=DerivedDataExecutionPolicy(
            mode=DerivedDataExecutionMode.STRICT_CACHE_ONLY,
            target_kind=DerivedDataTargetKind.HISTORICAL,
        ),
    )

    assert result.contributor_snapshot is not None
    assert result.contributor_snapshot.contributors[0].ibd_industry_group == (
        "Historical Group"
    )
    historical.assert_called_once_with(db, "US", {calculation_date: ["AAA"]})
    current.assert_not_called()


def test_historical_daily_breadth_uses_the_requested_dates_universe(monkeypatch):
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="RECENT",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()
    calculation_date = date(2026, 3, 20)

    def point_in_time_snapshots(_db, market, dates):
        assert (market, tuple(dates)) == ("US", (calculation_date,))
        return {
            calculation_date: BreadthUniverseSnapshot(
                calculation_date=calculation_date,
                members=(BreadthUniverseMember("HISTORICAL", "USD"),),
                broad_signature="historical",
            )
        }

    monkeypatch.setattr(
        "app.services.breadth_calculator_service.build_breadth_universe_snapshots",
        point_in_time_snapshots,
        raising=False,
    )
    prices = _flat_price_df(calculation_date)
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "HISTORICAL": prices,
    }
    service = BreadthCalculatorService(db, price_cache)

    result = service.calculate_daily_breadth(calculation_date)

    assert result.indicators["broad_universe_count"] == 1
    assert result.indicators["eligibility_signature"] == breadth_eligibility_signature(
        ("HISTORICAL",)
    )
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["HISTORICAL"],
        period="2y",
    )


def test_daily_breadth_preserves_month_eligibility_without_prior_close(monkeypatch):
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="MONTH",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            is_common_stock=True,
        )
    )
    db.commit()
    calculation_date = date(2026, 3, 20)
    prices = _flat_price_df(calculation_date)
    prices.loc[prices.index[-21], ["Close", "Adj Close"]] = 80.0
    prices.loc[prices.index[-2], "Adj Close"] = None
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"MONTH": prices}
    service = BreadthCalculatorService(db, price_cache)
    monkeypatch.setattr(
        breadth_calculator_module.BreadthContributorMetadataLoader,
        "historical",
        lambda _db, _market, symbols_by_date: {
            snapshot_date: {
                symbol: BreadthContributorMetadata(
                    company_name=symbol,
                    ibd_industry_group="Test Group",
                )
                for symbol in symbols
            }
            for snapshot_date, symbols in symbols_by_date.items()
        },
    )

    result = service.calculate_daily_breadth(calculation_date)

    assert result.coverage.total_stocks_scanned == 1
    assert result.indicators["advance_decline_eligible_count"] == 0
    assert result.indicators["stockbee_month_eligible_count"] == 1
    assert result.indicators["stocks_up_25pct_month"] == 1
    assert result.contributor_snapshot is not None
    assert result.contributor_snapshot.contributors[0].daily_change_pct is None


def test_daily_breadth_rejects_an_unusable_target_adjusted_close():
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="NULL_TARGET",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            is_common_stock=True,
        )
    )
    db.commit()
    calculation_date = date(2026, 3, 20)
    prices = _flat_price_df(calculation_date)
    prices.loc[prices.index[-1], "Adj Close"] = None
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"NULL_TARGET": prices}
    service = BreadthCalculatorService(db, price_cache)

    result = service.calculate_daily_breadth(calculation_date)

    assert result.coverage.total_stocks_scanned == 0
    assert result.coverage.insufficient_data_stocks == 1
    assert result.indicators["advance_decline_eligible_count"] == 0
    assert result.indicators["stockbee_month_eligible_count"] == 0


def test_cache_only_daily_breadth_requires_calculation_session():
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAA")
    ]
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": None}
    service = BreadthCalculatorService(db, price_cache)
    calculation_date = date(2026, 3, 20)

    result = service.calculate_daily_breadth(
        calculation_date,
        policy=_policy("refresh_guarded", calculation_date),
    )

    assert result.coverage.cache_miss_stocks == 1
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA"],
        period="2y",
        required_as_of_date=calculation_date,
        minimum_rows=1,
    )


def test_cache_only_daily_breadth_admits_short_history_for_formula_eligibility():
    db = _make_db_session()
    calculation_date = date(2026, 3, 20)
    db.add(
        StockUniverse(
            symbol="NEW",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            is_common_stock=True,
        )
    )
    db.commit()
    short_history = _flat_price_df(calculation_date, periods=2)

    class ShortHistoryCache:
        def get_many_cached_only_fresh(
            self,
            symbols,
            period="2y",
            *,
            required_as_of_date=None,
            minimum_rows=50,
        ):
            assert symbols == ["NEW"]
            assert period == "2y"
            assert required_as_of_date == calculation_date
            return {
                "NEW": short_history if minimum_rows <= len(short_history) else None
            }

    service = BreadthCalculatorService(db, ShortHistoryCache())

    result = service.calculate_daily_breadth(
        calculation_date,
        policy=_policy("refresh_guarded", calculation_date),
    )

    assert result.coverage.total_stocks_scanned == 1
    assert result.indicators["advance_decline_eligible_count"] == 1
    assert result.indicators["stockbee_daily_eligible_count"] == 0


def test_calculate_daily_breadth_preserves_historical_fetch_fallback():
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAA"),
        SimpleNamespace(symbol="BBB"),
    ]

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(date(2026, 3, 19), 100.0),
        "BBB": None,
    }
    calculator = BreadthCalculatorService(db, price_cache)
    price_cache.get_historical_data.return_value = _make_price_df(
        date(2026, 3, 19), 150.0
    )

    result = calculator.calculate_daily_breadth(
        date(2026, 3, 19),
        policy=_policy("auto", date(2026, 3, 19)),
    )
    metrics = result.to_metrics_dict()

    assert metrics["total_stocks_scanned"] == 2
    assert metrics["cache_miss_stocks"] == 1
    assert metrics["skipped_stocks"] == 0
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA", "BBB"],
        period="2y",
    )
    price_cache.get_many_cached_only.assert_not_called()
    price_cache.get_historical_data.assert_called_once_with(symbol="BBB", period="2y")


def test_backfill_allocates_symbol_coverage_once(monkeypatch):
    created = 0
    real_accumulator = breadth_backfill_module.BreadthPriceCoverageAccumulator

    class CountingAccumulator(real_accumulator):
        def __init__(self):
            nonlocal created
            created += 1
            super().__init__()

    monkeypatch.setattr(
        breadth_backfill_module,
        "BreadthPriceCoverageAccumulator",
        CountingAccumulator,
    )
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(
                symbol="AAA",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
            StockUniverse(
                symbol="BBB",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
        ]
    )
    db.commit()
    trading_dates = [date(2026, 3, 19), date(2026, 3, 20)]
    history = _make_price_df(trading_dates[-1])
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": history,
        "BBB": history,
    }
    service = BreadthCalculatorService(db, price_cache)

    service.backfill_range(
        trading_dates[0],
        trading_dates[-1],
        trading_dates=trading_dates,
        policy=_policy("refresh_guarded", trading_dates[-1]),
    )

    assert created == 1


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

    # The HK universe has no rows, so the calendar-selected date has no
    # point-in-time membership: unavailable, not a failed calculation.
    assert result == {
        "total_dates": 1,
        "processed": 0,
        "errors": 0,
        "error_dates": [],
        "unavailable": 1,
        "unavailable_dates": ["2026-03-13"],
    }
    price_cache.get_many_cached_only_fresh.assert_not_called()
    price_cache.get_many_cached_only.assert_not_called()


def test_backfill_range_is_idempotent_for_existing_records(monkeypatch):
    db = _make_db_session()
    db.add(StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE))
    db.commit()

    up_df = _flat_price_df(date(2026, 3, 12))
    up_df.loc[pd.Timestamp(date(2026, 3, 12)), ["Close", "Adj Close"]] = 105.0
    up_df.loc[pd.Timestamp(date(2026, 3, 12)), "Volume"] = 1_100_000
    down_df = _flat_price_df(date(2026, 3, 12))
    down_df.loc[pd.Timestamp(date(2026, 3, 12)), ["Close", "Adj Close"]] = 95.0
    down_df.loc[pd.Timestamp(date(2026, 3, 12)), "Volume"] = 1_100_000

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.side_effect = [
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
    db.add_all(
        [
            StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
            StockUniverse(symbol="BBB", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        ]
    )
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
        policy=_policy("refresh_guarded", trading_date),
    )

    assert result["total_dates"] == 1
    assert result["processed"] == 1
    assert result["errors"] == 0
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA", "BBB"],
        period="2y",
        required_as_of_date=trading_date,
        minimum_rows=1,
    )
    price_cache.get_historical_data.assert_not_called()


def test_backfill_range_accepts_legacy_cache_only_keyword():
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="AAA",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()
    target = date(2026, 3, 20)
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(target)
    }
    price_cache.get_historical_data.side_effect = AssertionError(
        "legacy cache-only backfill must not call a provider"
    )

    result = BreadthCalculatorService(db, price_cache).backfill_range(
        target,
        target,
        trading_dates=[target],
        cache_only=True,
    )

    assert result["processed"] == 1
    assert result["cache_miss_stocks"] == 0
    price_cache.get_historical_data.assert_not_called()


def test_backfill_range_cache_only_reports_gaps_without_provider_fallback():
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
            StockUniverse(symbol="BBB", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
            StockUniverse(symbol="NEW", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        ]
    )
    db.commit()

    trading_date = date(2026, 3, 20)
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(trading_date),
        "BBB": None,
        "NEW": _flat_price_df(trading_date, periods=20),
    }
    price_cache.get_historical_data.side_effect = AssertionError(
        "guarded breadth gap-fill must not call a provider"
    )
    service = BreadthCalculatorService(db, price_cache)

    result = service.backfill_range(
        trading_date,
        trading_date,
        trading_dates=[trading_date],
        policy=_policy("refresh_guarded", trading_date),
    )

    assert result == {
        "total_dates": 1,
        "processed": 1,
        "errors": 0,
        "error_dates": [],
        "target_symbols": 3,
        "symbols_with_cached_history": 2,
        "cache_miss_stocks": 1,
        "error_stocks": 0,
        "cache_miss_symbols_sample": ["BBB"],
        "cache_coverage_ratio": pytest.approx(2 / 3),
        "insufficient_history_observations": 0,
    }
    price_cache.get_historical_data.assert_not_called()


def test_backfill_range_can_exclude_unsupported_yahoo_symbols(monkeypatch):
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(
                symbol="7203.T",
                market="JP",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
            StockUniverse(
                symbol="0335.T",
                market="JP",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
        ]
    )
    db.commit()

    trading_date = date(2026, 3, 20)
    supported_df = _make_price_df(trading_date)
    service = BreadthCalculatorService(db, MagicMock(), market="JP")
    load_prices = MagicMock(return_value=({"7203.T": supported_df}, set()))
    monkeypatch.setattr(service, "_load_price_data_for_batch", load_prices)

    result = service.backfill_range(
        trading_date,
        trading_date,
        trading_dates=[trading_date],
        policy=_policy("refresh_guarded", trading_date),
        exclude_unsupported_price_symbols=True,
    )

    load_prices.assert_called_once_with(
        batch_symbols=["7203.T"],
        cache_only=True,
        required_as_of_date=trading_date,
    )
    assert result["processed"] == 1
    assert result["target_symbols"] == 1
    assert result["cache_miss_stocks"] == 0
    assert result["skipped_unsupported_symbols"] == 1
    assert result["unsupported_symbols_sample"] == ["0335.T"]
    stored = (
        db.query(MarketBreadth)
        .filter_by(
            market="JP",
            date=trading_date,
        )
        .one()
    )
    assert stored.broad_universe_count == 2
    assert stored.eligibility_signature == breadth_eligibility_signature(
        ("0335.T", "7203.T")
    )


def test_backfill_range_can_require_target_date_cached_prices(monkeypatch):
    db = _make_db_session()
    db.add(StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE))
    db.commit()

    trading_date = date(2026, 3, 20)
    service = BreadthCalculatorService(db, MagicMock())
    load_prices = MagicMock(return_value=({"AAA": _make_price_df(trading_date)}, set()))
    monkeypatch.setattr(service, "_load_price_data_for_batch", load_prices)

    result = service.backfill_range(
        trading_date,
        trading_date,
        trading_dates=[trading_date],
        policy=_policy("refresh_guarded", trading_date),
        required_as_of_date=trading_date,
    )

    assert result["processed"] == 1
    load_prices.assert_called_once_with(
        batch_symbols=["AAA"],
        cache_only=True,
        required_as_of_date=trading_date,
    )


def test_backfill_range_requests_history_for_full_interval_and_warmup(monkeypatch):
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="AAA",
            market="US",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()
    service = BreadthCalculatorService(db, MagicMock())
    load_prices = MagicMock(return_value=({"AAA": None}, {"AAA"}))
    monkeypatch.setattr(service, "_load_price_data_for_batch", load_prices)
    start_date = date(2024, 1, 2)
    end_date = date(2026, 8, 25)

    service.backfill_range(
        start_date,
        end_date,
        trading_dates=[start_date, end_date],
        policy=_policy("refresh_guarded", end_date),
    )

    load_prices.assert_called_once_with(
        batch_symbols=["AAA"],
        cache_only=True,
        required_as_of_date=end_date,
        period="5y",
    )


def test_history_period_uses_the_252nd_prior_market_session(monkeypatch):
    calculation_date = date(2025, 9, 1)
    cache_anchor = date(2026, 8, 25)

    class _Calendar:
        def is_trading_day(self, market, day):
            assert (market, day) == ("US", calculation_date)
            return True

        def session_anchors(self, market, as_of_date, *, offsets):
            assert (market, as_of_date, offsets) == (
                "US",
                calculation_date,
                (252,),
            )
            return {0: calculation_date, 252: date(2024, 8, 20)}

    import app.wiring.bootstrap as bootstrap_module

    monkeypatch.setattr(
        bootstrap_module,
        "get_market_calendar_service",
        lambda: _Calendar(),
    )
    service = BreadthCalculatorService(_make_db_session(), MagicMock())

    assert (
        service._history_period_for_dates(
            (calculation_date,),
            cache_anchor_date=cache_anchor,
        )
        == "5y"
    )


def test_historical_daily_breadth_loads_full_52_week_warmup():
    db = _make_db_session()
    calculation_date = date(2022, 1, 3)
    db.add(
        StockUniverse(
            symbol="HISTORICAL",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            is_common_stock=True,
        )
    )
    db.commit()
    full_history = _flat_price_df(calculation_date, periods=252)

    class DateAwareCache:
        def get_many_cached_only_fresh(self, symbols, period="2y"):
            assert symbols == ["HISTORICAL"]
            return {
                "HISTORICAL": (
                    full_history if period == "max" else full_history.tail(1)
                )
            }

    result = BreadthCalculatorService(db, DateAwareCache()).calculate_daily_breadth(
        calculation_date
    )

    assert result.indicators["high_low_52week_eligible_count"] == 1


def test_backfill_range_requests_history_for_old_narrow_interval(monkeypatch):
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="AAA",
            market="US",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()
    service = BreadthCalculatorService(db, MagicMock())
    load_prices = MagicMock(return_value=({"AAA": None}, {"AAA"}))
    monkeypatch.setattr(service, "_load_price_data_for_batch", load_prices)
    old_date = date(2022, 1, 3)

    service.backfill_range(
        old_date,
        old_date,
        trading_dates=[old_date],
        policy=_policy("refresh_guarded", old_date),
    )

    load_prices.assert_called_once_with(
        batch_symbols=["AAA"],
        cache_only=True,
        required_as_of_date=old_date,
        period="max",
    )


def test_daily_breadth_does_not_load_historical_fx(monkeypatch):
    db = _make_db_session()
    calculation_date = date(2026, 8, 25)
    db.add(
        StockUniverse(
            symbol="0700.HK",
            market="HK",
            currency="HKD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()
    recent_index = pd.bdate_range(end=calculation_date, periods=253)
    index = pd.DatetimeIndex([pd.Timestamp("2020-01-02"), *recent_index])
    prices = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Adj Close": 100.0,
            "Volume": 1_000_000,
        },
        index=index,
    )
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"0700.HK": prices}
    monkeypatch.setattr(
        breadth_calculator_module,
        "get_fx_service",
        MagicMock(side_effect=AssertionError("breadth must not request FX")),
        raising=False,
    )
    service = BreadthCalculatorService(
        db,
        price_cache,
        market="HK",
    )

    result = service.calculate_daily_breadth(calculation_date)

    assert result.indicators["advance_decline_eligible_count"] == 1


def test_backfill_range_cache_only_reports_calculation_errors(monkeypatch):
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
            StockUniverse(symbol="BAD", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        ]
    )
    db.commit()

    trading_date = date(2026, 3, 20)
    aaa_df = _make_price_df(trading_date)
    bad_df = _make_price_df(trading_date, 200.0).drop(columns=["Adj Close"])
    service = BreadthCalculatorService(db, MagicMock())
    monkeypatch.setattr(
        service,
        "_load_price_data_for_batch",
        MagicMock(return_value=({"AAA": aaa_df, "BAD": bad_df}, set())),
    )

    result = service.backfill_range(
        trading_date,
        trading_date,
        trading_dates=[trading_date],
        policy=_policy("refresh_guarded", trading_date),
    )

    assert result["processed"] == 1
    assert result["errors"] == 0
    assert result["target_symbols"] == 2
    assert result["cache_miss_stocks"] == 0
    assert result["error_stocks"] == 1
    assert result["insufficient_history_observations"] == 0


def test_backfill_range_reports_duplicate_sessions_without_aborting(monkeypatch):
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
            StockUniverse(symbol="BAD", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        ]
    )
    db.commit()

    trading_date = date(2026, 3, 20)
    aaa_df = _make_price_df(trading_date)
    duplicate = aaa_df.iloc[[-1]].copy()
    duplicate.index = pd.DatetimeIndex([aaa_df.index[-1] + pd.Timedelta(hours=1)])
    bad_df = pd.concat([aaa_df, duplicate])
    service = BreadthCalculatorService(db, MagicMock())
    monkeypatch.setattr(
        service,
        "_load_price_data_for_batch",
        MagicMock(return_value=({"AAA": aaa_df, "BAD": bad_df}, set())),
    )

    result = service.backfill_range(
        trading_date,
        trading_date,
        trading_dates=[trading_date],
        policy=_policy("refresh_guarded", trading_date),
    )

    assert result["processed"] == 1
    assert result["error_stocks"] == 1
    row = db.query(MarketBreadth).filter(MarketBreadth.date == trading_date).one()
    assert row.advance_decline_eligible_count == 1


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
        policy=_policy("auto", date(2026, 3, 16)),
    )


def test_fill_gaps_propagates_cache_only_to_backfill_range(monkeypatch):
    service = BreadthCalculatorService(_make_db_session(), MagicMock())
    expected = {
        "total_dates": 1,
        "processed": 1,
        "errors": 0,
        "error_dates": [],
        "target_symbols": 2,
        "symbols_with_cached_history": 1,
        "cache_miss_stocks": 1,
        "error_stocks": 0,
        "cache_miss_symbols_sample": ["BBB"],
        "cache_coverage_ratio": 0.5,
        "insufficient_history_observations": 0,
    }
    backfill_range = MagicMock(return_value=expected)
    monkeypatch.setattr(service, "backfill_range", backfill_range)

    guarded_policy = _policy("refresh_guarded", date(2026, 3, 12))
    result = service.fill_gaps(
        [date(2026, 3, 12)],
        policy=guarded_policy,
    )

    assert result == expected
    backfill_range.assert_called_once_with(
        date(2026, 3, 12),
        date(2026, 3, 12),
        trading_dates=[date(2026, 3, 12)],
        policy=guarded_policy,
    )


def test_backfill_range_cache_sample_is_sorted_across_batches(monkeypatch):
    db = _make_db_session()
    active_stocks = [
        StockUniverse(
            symbol=f"S{index:03d}",
            market="US",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
        for index in range(501)
    ]
    db.add_all(active_stocks)
    db.commit()
    service = BreadthCalculatorService(db, MagicMock())
    full_history = _make_price_df(date(2026, 3, 20))

    responses = [
        (
            {
                stock.symbol: (None if stock.symbol == "S499" else full_history)
                for stock in active_stocks[:500]
            },
            {"S499"},
        ),
        (
            {"S500": None},
            {"S500"},
        ),
    ]
    monkeypatch.setattr(
        service,
        "_load_price_data_for_batch",
        MagicMock(side_effect=responses),
    )

    result = service.backfill_range(
        date(2026, 3, 20),
        date(2026, 3, 20),
        trading_dates=[date(2026, 3, 20)],
        policy=_policy("refresh_guarded", date(2026, 3, 20)),
    )

    assert result["cache_miss_symbols_sample"] == ["S499", "S500"]


def test_fill_gaps_refreshes_stale_cached_prices_before_counting():
    db = _make_db_session()
    db.add(StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE))
    db.commit()

    trading_date = date(2026, 3, 12)
    fresh_df = _flat_price_df(trading_date)
    fresh_df.loc[pd.Timestamp(trading_date), ["Close", "Adj Close"]] = 105.0
    fresh_df.loc[pd.Timestamp(trading_date), "Volume"] = 1_100_000

    price_cache = MagicMock()
    price_cache.get_many_cached_only.side_effect = AssertionError(
        "gap fill must not use stale cache-only price data"
    )
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": None}
    price_cache.get_historical_data.return_value = fresh_df
    service = BreadthCalculatorService(db, price_cache)

    result = service.fill_gaps([trading_date])

    assert result == {
        "total_dates": 1,
        "processed": 1,
        "errors": 0,
        "error_dates": [],
    }
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA"],
        period="2y",
        required_as_of_date=trading_date,
    )
    price_cache.get_historical_data.assert_called_once_with(symbol="AAA", period="2y")
    row = db.query(MarketBreadth).filter(MarketBreadth.date == trading_date).one()
    assert row.stocks_up_4pct == 1
    assert row.total_stocks_scanned == 1


def test_backfill_range_sparse_dates_include_existing_intervening_counts_in_ratios():
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
            StockUniverse(symbol="BBB", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        ]
    )

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
        aaa_df.loc[pd.Timestamp(item_date), ["Close", "Adj Close"]] = aaa_close
        bbb_df.loc[pd.Timestamp(item_date), ["Close", "Adj Close"]] = bbb_close
    for item_date, volume in (
        (date(2026, 3, 12), 1_100_000),
        (date(2026, 3, 13), 1_000_000),
        (date(2026, 3, 16), 1_100_000),
    ):
        aaa_df.loc[pd.Timestamp(item_date), "Volume"] = volume
        bbb_df.loc[pd.Timestamp(item_date), "Volume"] = volume

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": aaa_df, "BBB": bbb_df}
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


def test_backfill_range_requires_exact_bar_for_each_requested_date():
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
            StockUniverse(symbol="GAP", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        ]
    )
    db.commit()

    first_date = date(2026, 3, 12)
    latest_date = date(2026, 3, 16)
    aaa_df = _flat_price_df(latest_date)
    gap_df = _flat_price_df(latest_date)
    gap_df = gap_df.drop(index=pd.Timestamp(first_date))
    gap_df.loc[pd.Timestamp(date(2026, 3, 11)), ["Close", "Adj Close"]] = 95.0
    gap_df.loc[pd.Timestamp(latest_date), ["Close", "Adj Close"]] = 105.0

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": aaa_df,
        "GAP": gap_df,
    }
    service = BreadthCalculatorService(db, price_cache)

    result = service.backfill_range(
        first_date,
        latest_date,
        trading_dates=[first_date, latest_date],
        policy=_policy("refresh_guarded", latest_date),
    )

    rows = {
        row.date: row
        for row in db.query(MarketBreadth)
        .filter(MarketBreadth.date.in_([first_date, latest_date]))
        .all()
    }
    assert result["processed"] == 2
    assert result["insufficient_history_observations"] == 1
    assert rows[first_date].total_stocks_scanned == 2
    assert rows[first_date].advance_decline_eligible_count == 1
    assert rows[latest_date].total_stocks_scanned == 2
    assert rows[latest_date].advance_decline_eligible_count == 2


def test_backfill_range_uses_exact_unrounded_canonical_thresholds():
    db = _make_db_session()
    symbols = ["UP4", "UP13", "UP25", "UP50"]
    db.add_all(
        [
            StockUniverse(symbol=symbol, is_active=True, status=UNIVERSE_STATUS_ACTIVE)
            for symbol in symbols
        ]
    )
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
        frame.loc[pd.Timestamp(latest_date), ["Close", "Adj Close"]] = latest_close
        frame.loc[pd.Timestamp(latest_date), "Volume"] = 1_100_000
        price_data[symbol] = frame

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = price_data
    service = BreadthCalculatorService(db, price_cache)

    result = service.backfill_range(
        latest_date, latest_date, trading_dates=[latest_date]
    )

    assert result["processed"] == 1
    row = db.query(MarketBreadth).filter(MarketBreadth.date == latest_date).one()
    assert row.total_stocks_scanned == 4
    assert row.stocks_up_4pct == 3
    assert row.stocks_up_13pct_34days == 2
    assert row.stocks_up_25pct_month == 1
    assert row.stocks_up_25pct_quarter == 1
    assert row.stocks_up_50pct_month == 0


def test_live_and_backfill_use_identical_canonical_counts():
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(
                symbol="AAA",
                market="US",
                currency="USD",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                is_common_stock=True,
            ),
            StockUniverse(
                symbol="SPY",
                market="US",
                currency="USD",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                is_common_stock=False,
            ),
        ]
    )
    db.commit()
    calculation_date = date(2026, 3, 20)
    prices = _flat_price_df(calculation_date, periods=252)
    prices.loc[
        pd.Timestamp(calculation_date),
        ["Close", "Adj Close"],
    ] = 105.0
    prices.loc[pd.Timestamp(calculation_date), "Volume"] = 1_100_000
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": prices}
    service = BreadthCalculatorService(db, price_cache)

    live = service.calculate_daily_breadth(calculation_date)
    backfill = service.backfill_range(
        calculation_date,
        calculation_date,
        trading_dates=[calculation_date],
    )

    assert backfill["processed"] == 1
    assert live.indicators["broad_universe_count"] == 1
    assert price_cache.get_many_cached_only_fresh.call_count == 2
    assert all(
        call.args[0] == ["AAA"]
        for call in price_cache.get_many_cached_only_fresh.call_args_list
    )
    stored = (
        db.query(MarketBreadth)
        .filter(
            MarketBreadth.date == calculation_date,
            MarketBreadth.market == "US",
        )
        .one()
    )
    for field in BreadthIndicatorValues.__dataclass_fields__:
        assert getattr(stored, field) == live.indicators[field]
    assert stored.calculation_revision == live.indicators["calculation_revision"] == 3


def test_calculate_daily_breadth_reduces_each_price_batch_before_the_next(
    monkeypatch,
):
    calculation_date = date(2026, 3, 20)
    frames = {
        symbol: _make_price_df(calculation_date, base)
        for symbol, base in (("AAA", 100.0), ("BBB", 200.0), ("CCC", 50.0))
    }
    monkeypatch.setattr(
        breadth_calculator_module.BreadthContributorMetadataLoader,
        "current",
        lambda _db, _market, symbols: {},
    )

    def run(batch_size):
        monkeypatch.setattr(breadth_calculator_module, "PRICE_BATCH_SIZE", batch_size)
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [
            SimpleNamespace(symbol=symbol) for symbol in frames
        ]
        price_cache = MagicMock()
        price_cache.get_many_cached_only_fresh.side_effect = lambda symbols, **_: {
            symbol: frames[symbol] for symbol in symbols
        }
        calculator = BreadthCalculatorService(db, price_cache)
        make_accumulator = calculator.engine.accumulator
        sizes = []

        def recording_accumulator(**kwargs):
            accumulator = make_accumulator(**kwargs)
            add_prices = accumulator.add_prices

            def add(batch):
                sizes.append(len(batch))
                add_prices(batch)

            accumulator.add_prices = add
            return accumulator

        monkeypatch.setattr(calculator.engine, "accumulator", recording_accumulator)
        result = calculator.calculate_daily_breadth(
            calculation_date,
            policy=_policy("refresh_guarded", calculation_date),
        )
        return result, sizes, price_cache.get_many_cached_only_fresh.call_count

    streamed, streamed_sizes, streamed_loads = run(1)
    whole, whole_sizes, whole_loads = run(500)

    assert streamed_sizes == [1, 1, 1]
    assert streamed_loads == 3
    assert (whole_sizes, whole_loads) == ([3], 1)
    assert streamed.to_metrics_dict() == whole.to_metrics_dict()
    assert streamed.daily_result == whole.daily_result
    assert streamed.to_metrics_dict()["total_stocks_scanned"] == 3


def test_find_missing_dates_skips_dates_before_the_universe_existed(monkeypatch):
    from datetime import UTC, datetime

    class _WeekdayCalendar:
        def is_trading_day(self, market, current_date):
            return current_date.weekday() < 5

    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _WeekdayCalendar(),
    )
    db = _make_db_session()
    service = BreadthCalculatorService(db, MagicMock())

    assert service.find_missing_dates(14, end_date=date(2026, 3, 20)) == []

    db.add(
        StockUniverse(
            symbol="AAA",
            market="US",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            first_seen_at=datetime(2026, 3, 16, 15, tzinfo=UTC),
        )
    )
    db.commit()

    assert service.find_missing_dates(14, end_date=date(2026, 3, 20)) == [
        date(2026, 3, 16),
        date(2026, 3, 17),
        date(2026, 3, 18),
        date(2026, 3, 19),
    ]


def test_backfill_reports_unreconstructable_dates_unavailable_not_failed(
    monkeypatch,
):
    from app.services.point_in_time_universe_service import (
        PointInTimeUniverseUnavailable,
    )

    db = _make_db_session()
    good_date = date(2026, 3, 20)
    broken_date = date(2026, 3, 19)
    empty_date = date(2026, 3, 18)

    def snapshots(_db, _market, dates):
        dates = tuple(dates)
        if broken_date in dates:
            raise PointInTimeUniverseUnavailable("missing lifecycle events")
        return {
            value: BreadthUniverseSnapshot(
                calculation_date=value,
                members=(
                    ()
                    if value == empty_date
                    else (BreadthUniverseMember("AAA", "USD"),)
                ),
                broad_signature="sig",
            )
            for value in dates
        }

    monkeypatch.setattr(
        breadth_backfill_module, "build_breadth_universe_snapshots", snapshots
    )
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(good_date)
    }

    result = BreadthCalculatorService(db, price_cache).backfill_range(
        empty_date,
        good_date,
        trading_dates=[empty_date, broken_date, good_date],
    )

    assert result == {
        "total_dates": 3,
        "processed": 1,
        "errors": 0,
        "error_dates": [],
        "unavailable": 2,
        "unavailable_dates": [empty_date.isoformat(), broken_date.isoformat()],
    }
    assert [row.date for row in db.query(MarketBreadth).all()] == [good_date]


def test_refresh_ratios_between_repairs_rows_written_across_a_gap():
    from app.services.breadth.ratios import calculate_inclusive_ratios
    from app.services.breadth.types import BreadthDailyCount

    db = _make_db_session()
    sessions = [date(2026, 3, day) for day in (9, 10, 11, 12, 13, 16, 17)]
    gap = date(2026, 3, 12)
    counts = {day: (index + 1, 2) for index, day in enumerate(sessions)}
    for day in sessions:
        if day != gap:
            _add_breadth_row(db, day, up=counts[day][0], down=counts[day][1])
    db.commit()
    service = BreadthCalculatorService(db, MagicMock())
    # 3/13 and 3/16 were computed while 3/12 was missing: their windows
    # skipped it. Store what that produced, then fill the gap.
    stale = service.refresh_ratios_between(date(2026, 3, 8), date(2026, 3, 17))
    assert stale > 0
    _add_breadth_row(db, gap, up=counts[gap][0], down=counts[gap][1])
    db.commit()

    changed = service.refresh_ratios_between(gap, date(2026, 3, 17))

    expected = calculate_inclusive_ratios(
        BreadthDailyCount(day, up, down, market="US", calculation_revision=3)
        for day, (up, down) in counts.items()
    )
    stored = {
        row.date: (row.ratio_5day, row.ratio_10day)
        for row in db.query(MarketBreadth).filter(MarketBreadth.date > gap)
    }
    assert changed == 2
    assert {day: stored[day] for day in (date(2026, 3, 13), date(2026, 3, 16))} == {
        day: (expected[day].ratio_5day, expected[day].ratio_10day)
        for day in (date(2026, 3, 13), date(2026, 3, 16))
    }
    # The target date itself is excluded: it is calculated next, with the
    # repaired rows as its seeds.
    assert stored[date(2026, 3, 17)] == (None, None)
