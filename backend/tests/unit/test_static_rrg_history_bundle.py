from __future__ import annotations

import gzip
import json
from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.industry import IBDGroupRank
from app.services.group_rank_history_backfill_service import (
    GroupRankHistoryBackfillService,
)
from app.services.group_rank_input_loader import GroupRankInputLoader
from app.services.ibd_group_rank_service import IBDGroupRankService
from app.services.market_calendar_service import MarketCalendarService
from app.services.rrg_service import MIN_TAIL_WEEKS, RRGService
from app.services.static_rrg_history_bundle import (
    StaticRRGHistoryBundleService,
    StaticRRGHistoryProvider,
)
from app.services.static_rrg_history_contract import (
    STATIC_RRG_HISTORY_SCHEMA_VERSION,
    StaticRRGHistoryBundleError,
)


def _session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[IBDGroupRank.__table__])
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _rank(*, market: str, group: str, row_date: date, rank: int, avg_rs: float):
    return IBDGroupRank(
        market=market,
        industry_group=group,
        date=row_date,
        rank=rank,
        avg_rs_rating=avg_rs,
        median_rs_rating=avg_rs - 1,
        num_stocks=12,
        num_stocks_rs_above_80=3,
        top_symbol="AAA",
        top_rs_rating=95,
    )


def _seed_weeks(db, *, latest: date, weeks: int, market: str = "HK") -> None:
    for week in range(weeks):
        row_date = latest - timedelta(weeks=weeks - week - 1)
        db.add_all(
            [
                _rank(
                    market=market,
                    group="Semiconductors",
                    row_date=row_date,
                    rank=1,
                    avg_rs=60 + week * 1.5,
                ),
                _rank(
                    market=market,
                    group="Banks",
                    row_date=row_date,
                    rank=2,
                    avg_rs=85 - week,
                ),
            ]
        )
    db.commit()


def test_first_static_build_bootstraps_via_production_gap_fill(
    tmp_path, monkeypatch
):
    engine, factory = _session_factory()
    latest = date(2026, 3, 20)
    groups = {
        "Semiconductors": ["CHIP1", "CHIP2", "CHIP3"],
        "Banks": ["BANK1", "BANK2", "BANK3"],
    }
    symbols = [symbol for members in groups.values() for symbol in members]

    def _prices(daily_return: float) -> pd.DataFrame:
        dates = pd.bdate_range(end=latest, periods=420)
        closes = [100 * ((1 + daily_return) ** index) for index in range(len(dates))]
        return pd.DataFrame({"Close": closes}, index=dates)

    benchmark_prices = _prices(0.0003)
    prices_by_symbol = {
        symbol: _prices(0.0008 + index * 0.0001)
        for index, symbol in enumerate(symbols)
    }

    class _BenchmarkCache:
        @staticmethod
        def get_benchmark_symbol(market):
            assert market == "HK"
            return "^HSI"

        @staticmethod
        def get_benchmark_data(*, market, period):
            assert (market, period) == ("HK", "2y")
            return benchmark_prices

    class _PriceCache:
        @staticmethod
        def get_many(requested_symbols, *, period):
            assert period == "2y"
            return {symbol: prices_by_symbol[symbol] for symbol in requested_symbols}

    class _UniverseSource:
        @staticmethod
        def active_symbols(_db, market):
            assert market == "HK"
            return frozenset(symbols)

    class _TaxonomySource:
        @staticmethod
        def groups(_db, market):
            return tuple(groups) if market == "HK" else ()

        @staticmethod
        def symbols_for_group(_db, group, market):
            return tuple(groups[group]) if market == "HK" else ()

    class _MarketCapSource:
        @staticmethod
        def market_caps(_db, requested_symbols):
            return {
                symbol: 1_000_000_000
                for symbol in requested_symbols
            }

    input_loader = GroupRankInputLoader(
        price_cache=_PriceCache(),
        benchmark_cache=_BenchmarkCache(),
        universe_source=_UniverseSource(),
        taxonomy_source=_TaxonomySource(),
        market_cap_source=_MarketCapSource(),
    )
    group_rank_service = IBDGroupRankService(
        price_cache=_PriceCache(),
        benchmark_cache=_BenchmarkCache(),
        input_loader=input_loader,
    )

    class _Taxonomy:
        @staticmethod
        def sector_map_for_market(_market):
            return {"Semiconductors": "Technology", "Banks": "Financials"}

    try:
        backfill = GroupRankHistoryBackfillService(
            session_factory=factory,
            calendar_service=MarketCalendarService(),
            group_rank_service=group_rank_service,
        ).backfill(
            as_of_date=latest,
            market="HK",
        )
        with factory() as db:
            preparation = StaticRRGHistoryBundleService().prepare(
                db,
                market="HK",
                through_date=latest,
                directory=tmp_path,
            )
            assert preparation.state is not None
            payload = RRGService(
                history_provider=StaticRRGHistoryProvider(preparation.state),
                taxonomy_service=_Taxonomy(),
            ).get_rrg_scopes(
                db,
                market="HK",
                scopes=("groups", "sectors"),
                as_of_date=latest,
            )
        persisted = StaticRRGHistoryBundleService().persist(
            preparation,
            exported_as_of_date=latest,
        )

        assert backfill.status.value == "completed"
        assert backfill.processed == backfill.missing_dates
        assert len(preparation.state.weeks) >= MIN_TAIL_WEEKS
        assert len(payload["groups"]["groups"]) == 2
        assert persisted is not None
        assert preparation.plan.output_path.exists()
    finally:
        engine.dispose()


def test_weekly_state_round_trips_only_rrg_inputs(tmp_path):
    engine, factory = _session_factory()
    latest = date(2026, 7, 10)
    path = tmp_path / "rrg-history-hk.json.gz"
    try:
        with factory() as db:
            _seed_weeks(db, latest=latest, weeks=14)
            state = StaticRRGHistoryBundleService().build(
                db,
                market="HK",
                through_date=latest,
            )
        stats = StaticRRGHistoryBundleService().write(state, path)
        restored = StaticRRGHistoryBundleService().load(path, expected_market="HK")

        assert stats["weeks"] == 14
        assert restored == state
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["schema_version"] == STATIC_RRG_HISTORY_SCHEMA_VERSION
        assert set(payload) == {"schema_version", "market", "weeks"}
        assert set(payload["weeks"][0]) == {"source_date", "groups"}
        assert set(payload["weeks"][0]["groups"][0]) == {
            "industry_group",
            "rank",
            "avg_rs_rating",
            "num_stocks",
        }
    finally:
        engine.dispose()


def test_merge_keeps_prior_weeks_without_mutating_database(tmp_path):
    source_engine, source_factory = _session_factory()
    target_engine, target_factory = _session_factory()
    latest = date(2026, 7, 10)
    try:
        with source_factory() as db:
            _seed_weeks(db, latest=latest - timedelta(weeks=1), weeks=13)
            previous = StaticRRGHistoryBundleService().build(
                db,
                market="HK",
                through_date=latest - timedelta(weeks=1),
            )
        with target_factory() as db:
            db.add(_rank(market="HK", group="Semiconductors", row_date=latest, rank=1, avg_rs=91))
            db.commit()
            merged = StaticRRGHistoryBundleService().build(
                db,
                market="HK",
                through_date=latest,
                previous=previous,
            )
            database_groups = [row.industry_group for row in db.query(IBDGroupRank).all()]

        assert len(merged.weeks) == 14
        assert database_groups == ["Semiconductors"]
        assert merged.weeks[-2] == previous.weeks[-1]
    finally:
        source_engine.dispose()
        target_engine.dispose()


def test_merge_discards_restored_weeks_after_requested_export_date():
    source_engine, source_factory = _session_factory()
    target_engine, target_factory = _session_factory()
    latest = date(2026, 7, 10)
    future = latest + timedelta(weeks=1)
    try:
        with source_factory() as db:
            _seed_weeks(db, latest=future, weeks=14)
            previous = StaticRRGHistoryBundleService().build(
                db,
                market="HK",
                through_date=future,
            )
        with target_factory() as db:
            db.add(
                _rank(
                    market="HK",
                    group="Semiconductors",
                    row_date=latest,
                    rank=1,
                    avg_rs=91,
                )
            )
            db.commit()
            merged = StaticRRGHistoryBundleService().build(
                db,
                market="HK",
                through_date=latest,
                previous=previous,
            )

        assert merged.weeks[-1].source_date == latest
        assert all(week.source_date <= latest for week in merged.weeks)
    finally:
        source_engine.dispose()
        target_engine.dispose()


def test_invalid_row_is_normalized_to_bundle_error(tmp_path):
    path = tmp_path / "rrg-history-hk.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": STATIC_RRG_HISTORY_SCHEMA_VERSION,
                "market": "HK",
                "weeks": [
                    {
                        "source_date": "2026-07-10",
                        "groups": [
                            {
                                "industry_group": "Semiconductors",
                                "rank": "bad",
                                "avg_rs_rating": 88,
                                "num_stocks": 12,
                            }
                        ],
                    }
                ],
            },
            handle,
        )

    with pytest.raises(StaticRRGHistoryBundleError, match="Unable to load"):
        StaticRRGHistoryBundleService().load(path, expected_market="HK")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("schema_version"),
        lambda payload: payload.update({"schema_version": "static-rrg-history-v2"}),
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["weeks"][0].update({"unexpected": True}),
        lambda payload: payload["weeks"][0]["groups"][0].update({"unexpected": True}),
        lambda payload: payload["weeks"][0]["groups"][0].update({"rank": "1"}),
        lambda payload: payload["weeks"][0]["groups"][0].update(
            {"avg_rs_rating": "80.0"}
        ),
        lambda payload: payload["weeks"][0]["groups"][0].update(
            {"num_stocks": "10"}
        ),
        lambda payload: payload["weeks"][0]["groups"][0].update(
            {"avg_rs_rating": float("nan")}
        ),
    ],
)
def test_history_contract_rejects_unversioned_and_unknown_fields(tmp_path, mutation):
    path = tmp_path / "rrg-history-hk.json.gz"
    payload = {
        "schema_version": STATIC_RRG_HISTORY_SCHEMA_VERSION,
        "market": "HK",
        "weeks": [
            {
                "source_date": "2026-07-10",
                "groups": [
                    {
                        "industry_group": "Banks",
                        "rank": 1,
                        "avg_rs_rating": 80,
                        "num_stocks": 10,
                    }
                ],
            }
        ],
    }
    mutation(payload)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(StaticRRGHistoryBundleError, match="Unable to load"):
        StaticRRGHistoryBundleService().load(path, expected_market="HK")


def test_prepare_bootstraps_invalid_prior_state_and_persists_current_state(tmp_path):
    engine, factory = _session_factory()
    latest = date(2026, 7, 10)
    path = tmp_path / "rrg-history-hk.json.gz"
    path.write_bytes(b"not gzip")
    try:
        with factory() as db:
            _seed_weeks(db, latest=latest, weeks=14)
            preparation = StaticRRGHistoryBundleService().prepare(
                db,
                market="HK",
                through_date=latest,
                directory=tmp_path,
            )

        assert preparation.state is not None
        assert preparation.warnings
        assert path.read_bytes() == b"not gzip"
        stats = StaticRRGHistoryBundleService().persist(
            preparation,
            exported_as_of_date=latest,
        )
        assert stats is not None
        assert stats["weeks"] == 14
        assert preparation.plan.output_path.exists()
    finally:
        engine.dispose()


def test_weekly_state_computes_non_us_group_and_sector_rrg():
    engine, factory = _session_factory()
    latest = date(2026, 7, 10)

    class _Taxonomy:
        def sector_map_for_market(self, market):
            assert market == "HK"
            return {"Semiconductors": "Technology", "Banks": "Financials"}

    try:
        with factory() as db:
            _seed_weeks(db, latest=latest, weeks=14)
            state = StaticRRGHistoryBundleService().build(
                db,
                market="HK",
                through_date=latest,
            )
            payload = RRGService(
                history_provider=StaticRRGHistoryProvider(state),
                taxonomy_service=_Taxonomy(),
            ).get_rrg_scopes(
                db,
                market="HK",
                scopes=("groups", "sectors"),
                as_of_date=latest,
            )

        assert len(payload["groups"]["groups"]) == 2
        assert {row["industry_group"] for row in payload["sectors"]["groups"]} == {
            "Technology",
            "Financials",
        }
    finally:
        engine.dispose()
