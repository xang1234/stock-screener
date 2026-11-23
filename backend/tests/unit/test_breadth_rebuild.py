from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import app.services.breadth.rebuild as rebuild_module
from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.breadth_contributor import (
    MarketBreadthContributor,
    MarketBreadthContributorSnapshot,
)
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.scripts.rebuild_market_breadth import (
    EXIT_CONFIRMATION_REQUIRED,
    EXIT_VALIDATION_REQUIRED,
    main,
)
from app.services.breadth.rebuild import BreadthRebuildService
from app.services.point_in_time_universe_service import (
    PointInTimeUniverse,
    PointInTimeUniverseMember,
    hash_point_in_time_universe_symbols,
)


class _FakeRebuildService:
    def __init__(self, *, valid: bool = False) -> None:
        self.valid = valid
        self.calls: list[tuple] = []

    def build(self, **kwargs):
        self.calls.append(("build", kwargs))
        return {"processed": 1}

    def validate(self, **kwargs):
        self.calls.append(("validate", kwargs))
        return {"valid": self.valid, "errors": [] if self.valid else ["invalid"]}

    def activate(self):
        self.calls.append(("activate", {}))
        return {"activated": 1}

    def cleanup(self):
        self.calls.append(("cleanup", {}))


def test_table_inspection_uses_the_session_connection(monkeypatch):
    db = MagicMock()
    session_connection = object()
    db.connection.return_value = session_connection
    inspector = MagicMock()
    inspector.has_table.return_value = True
    inspect_mock = MagicMock(return_value=inspector)
    monkeypatch.setattr(rebuild_module, "inspect", inspect_mock)
    service = BreadthRebuildService(db, required_markets=("US",))

    assert service._has_table("market_breadth_rebuild") is True
    inspect_mock.assert_called_once_with(session_connection)


def test_activate_requires_explicit_confirmation():
    service = _FakeRebuildService(valid=True)

    result = main(["activate"], service_factory=lambda: service)

    assert result == EXIT_CONFIRMATION_REQUIRED
    assert service.calls == []


def test_activate_refuses_unvalidated_staging_data():
    service = _FakeRebuildService(valid=False)

    assert (
        main(
            ["activate", "--confirm-replace"],
            service_factory=lambda: service,
        )
        == EXIT_VALIDATION_REQUIRED
    )
    assert [call[0] for call in service.calls] == ["validate"]


def test_build_dispatches_market_and_date_range():
    service = _FakeRebuildService()

    result = main(
        ["build", "--market", "US", "--start-date", "2026-01-01"],
        service_factory=lambda: service,
    )

    assert result == 0
    assert service.calls[0][0] == "build"
    assert service.calls[0][1]["markets"] == ("US",)
    assert service.calls[0][1]["start_date"] == date(2026, 1, 1)


def test_rebuild_does_not_stage_a_date_with_partial_supported_cache_coverage():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            MarketBreadth.__table__,
            MarketBreadthContributorSnapshot.__table__,
            MarketBreadthContributor.__table__,
            StockUniverse.__table__,
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
        ],
    )
    db = sessionmaker(bind=engine)()
    calculation_date = date(2026, 8, 21)
    index = pd.bdate_range(end=calculation_date, periods=2)
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

    class _UniverseService:
        def resolve(self, _db, *, market, as_of_date):
            symbols = ("AAA", "MISSING")
            return PointInTimeUniverse(
                market=market,
                as_of_date=as_of_date,
                symbols=symbols,
                universe_hash=hash_point_in_time_universe_symbols(symbols),
                members=tuple(
                    PointInTimeUniverseMember(symbol=symbol, currency="USD")
                    for symbol in symbols
                ),
            )

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": prices,
        "MISSING": None,
    }
    calendar = SimpleNamespace(
        is_trading_day=lambda _market, day: day == calculation_date,
    )
    service = BreadthRebuildService(
        db,
        price_cache=price_cache,
        universe_service=_UniverseService(),
        calendar_service=calendar,
        required_markets=("US",),
    )

    result = service.build(
        markets=("US",),
        start_date=calculation_date,
        end_date=calculation_date,
    )

    staged_count = db.execute(
        text("SELECT COUNT(*) FROM market_breadth_rebuild")
    ).scalar_one()
    assert result["processed"] == 0
    assert result["markets"]["US"]["error_dates"] == ["2026-08-21"]
    assert staged_count == 0


def test_rebuild_reads_delisted_history_at_its_last_membership_date():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            MarketBreadth.__table__,
            MarketBreadthContributorSnapshot.__table__,
            MarketBreadthContributor.__table__,
            StockUniverse.__table__,
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
        ],
    )
    db = sessionmaker(bind=engine)()
    calculation_date = date(2026, 8, 21)
    prices = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Adj Close": 100.0,
            "Volume": 1_000_000,
        },
        index=pd.bdate_range(end=calculation_date, periods=2),
    )

    class _UniverseService:
        def resolve(self, _db, *, market, as_of_date):
            symbols = ("DELISTED",)
            return PointInTimeUniverse(
                market=market,
                as_of_date=as_of_date,
                symbols=symbols,
                universe_hash=hash_point_in_time_universe_symbols(symbols),
                members=(
                    PointInTimeUniverseMember(
                        symbol="DELISTED",
                        currency="USD",
                    ),
                ),
            )

    price_cache = MagicMock()

    def cached_histories(
        symbols,
        period,
        *,
        required_as_of_date=None,
        minimum_rows,
    ):
        assert symbols == ["DELISTED"]
        assert period == "2y"
        assert minimum_rows == 1
        return {
            "DELISTED": (prices if required_as_of_date == calculation_date else None)
        }

    price_cache.get_many_cached_only_fresh.side_effect = cached_histories
    service = BreadthRebuildService(
        db,
        price_cache=price_cache,
        universe_service=_UniverseService(),
        calendar_service=SimpleNamespace(
            is_trading_day=lambda _market, day: day == calculation_date,
        ),
        required_markets=("US",),
    )

    result = service.build(
        markets=("US",),
        start_date=calculation_date,
        end_date=calculation_date,
    )

    assert result["processed"] == 1
    assert (
        db.execute(text("SELECT COUNT(*) FROM market_breadth_rebuild")).scalar_one()
        == 1
    )
    assert len(
        db.execute(
            text(
                "SELECT contributor_calculation_signature "
                "FROM market_breadth_rebuild"
            )
        ).scalar_one()
    ) == 64
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["DELISTED"],
        period="2y",
        required_as_of_date=calculation_date,
        minimum_rows=1,
    )


def _staged_one_symbol_rebuild():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            MarketBreadth.__table__,
            MarketBreadthContributorSnapshot.__table__,
            MarketBreadthContributor.__table__,
            StockUniverse.__table__,
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
        ],
    )
    db = sessionmaker(bind=engine)()
    calculation_date = date(2026, 8, 21)
    prices = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Adj Close": 100.0,
            "Volume": 1_000_000,
        },
        index=pd.bdate_range(end=calculation_date, periods=2),
    )

    class _UniverseService:
        def resolve(self, _db, *, market, as_of_date):
            return PointInTimeUniverse(
                market=market,
                as_of_date=as_of_date,
                symbols=("AAA",),
                universe_hash=hash_point_in_time_universe_symbols(("AAA",)),
                members=(PointInTimeUniverseMember(symbol="AAA", currency="USD"),),
            )

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": prices}
    service = BreadthRebuildService(
        db,
        price_cache=price_cache,
        universe_service=_UniverseService(),
        calendar_service=SimpleNamespace(
            is_trading_day=lambda _market, day: day == calculation_date,
        ),
        required_markets=("US",),
    )
    service.build(markets=("US",), start_date=calculation_date, end_date=calculation_date)
    return db, service


def test_rebuild_stages_participation_and_accepts_an_empty_sma200_denominator():
    db, service = _staged_one_symbol_rebuild()

    row = db.execute(
        text(
            "SELECT sma200_eligible_count, above_sma200_pct, "
            "off_high_eligible_count, down_20pct_from_high_pct "
            "FROM market_breadth_rebuild"
        )
    ).one()

    assert tuple(row) == (0, None, 1, 0.0)
    assert service.validate()["valid"] is True


def test_rebuild_validation_rejects_a_participation_pct_that_does_not_reconcile():
    db, service = _staged_one_symbol_rebuild()
    db.execute(text("UPDATE market_breadth_rebuild SET down_20pct_from_high_pct = 50.0"))

    result = service.validate()

    assert result["valid"] is False
    assert any(
        error.startswith("pct_reconciliation:down_20pct_from_high_pct")
        for error in result["errors"]
    )
