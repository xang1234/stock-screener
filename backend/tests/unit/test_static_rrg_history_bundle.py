from __future__ import annotations

import gzip
import json
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.industry import IBDGroupRank
from app.scripts.export_static_site import STATIC_GROUP_HISTORY_LOOKBACK_DAYS
from app.services.static_rrg_history_bundle import (
    STATIC_RRG_HISTORY_SCHEMA_VERSION,
    StaticRRGHistoryBundleError,
    StaticRRGHistoryBundleService,
)
from app.services.rrg_history_provider import PersistedGroupRankHistoryProvider
from app.services.rrg_service import MIN_TAIL_WEEKS, RRGService


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


def test_first_static_build_lookback_can_bootstrap_minimum_rrg_tail():
    assert STATIC_GROUP_HISTORY_LOOKBACK_DAYS >= MIN_TAIL_WEEKS * 7


def test_static_rrg_history_bundle_round_trips_market_rows(tmp_path):
    source_engine, source_factory = _session_factory()
    target_engine, target_factory = _session_factory()
    through_date = date(2026, 7, 10)
    try:
        with source_factory() as db:
            db.add_all(
                [
                    _rank(
                        market="HK",
                        group="Semiconductors",
                        row_date=through_date - timedelta(days=offset),
                        rank=1,
                        avg_rs=80 + offset,
                    )
                    for offset in (0, 7, 14)
                ]
                + [
                    _rank(
                        market="US",
                        group="Software",
                        row_date=through_date,
                        rank=1,
                        avg_rs=90,
                    )
                ]
            )
            db.commit()
            bundle_path = tmp_path / "rrg-history-hk.json.gz"
            exported = StaticRRGHistoryBundleService().export_bundle(
                db,
                market="HK",
                output_path=bundle_path,
                through_date=through_date,
            )

        with target_factory() as db:
            imported = StaticRRGHistoryBundleService().import_bundle(
                db,
                market="HK",
                input_path=bundle_path,
            )
            rows = db.query(IBDGroupRank).order_by(IBDGroupRank.date).all()

        assert exported["rows"] == imported["rows"] == 3
        assert {row.market for row in rows} == {"HK"}
        assert [row.avg_rs_rating for row in rows] == [94, 87, 80]
        with gzip.open(bundle_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["schema_version"] == STATIC_RRG_HISTORY_SCHEMA_VERSION
        assert payload["market"] == "HK"
    finally:
        source_engine.dispose()
        target_engine.dispose()


def test_static_rrg_history_bundle_prunes_rows_outside_retention(tmp_path):
    engine, factory = _session_factory()
    through_date = date(2026, 7, 10)
    try:
        with factory() as db:
            db.add_all(
                [
                    _rank(
                        market="HK",
                        group="Semiconductors",
                        row_date=through_date - timedelta(days=offset),
                        rank=1,
                        avg_rs=80,
                    )
                    for offset in (0, 31)
                ]
            )
            db.commit()
            path = tmp_path / "history.json.gz"
            stats = StaticRRGHistoryBundleService(retention_days=30).export_bundle(
                db,
                market="HK",
                output_path=path,
                through_date=through_date,
            )

        assert stats["rows"] == 1
    finally:
        engine.dispose()


def test_static_rrg_history_bundle_rejects_wrong_market_before_mutating(tmp_path):
    engine, factory = _session_factory()
    path = tmp_path / "history.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": STATIC_RRG_HISTORY_SCHEMA_VERSION,
                "market": "JP",
                "rows": [
                    {
                        "industry_group": "Software",
                        "date": "2026-07-10",
                        "rank": 1,
                        "avg_rs_rating": 88,
                    }
                ],
            },
            handle,
        )
    try:
        with factory() as db:
            db.add(
                _rank(
                    market="HK",
                    group="Existing",
                    row_date=date(2026, 7, 10),
                    rank=1,
                    avg_rs=80,
                )
            )
            db.commit()
            with pytest.raises(StaticRRGHistoryBundleError, match="does not match"):
                StaticRRGHistoryBundleService().import_bundle(
                    db,
                    market="HK",
                    input_path=path,
                )
            assert db.query(IBDGroupRank).one().industry_group == "Existing"
    finally:
        engine.dispose()


def test_persisted_history_computes_non_us_group_and_sector_rrg():
    engine, factory = _session_factory()
    latest = date(2026, 7, 10)

    class _GroupRankService:
        def get_current_rankings(self, db, *, limit, market, calculation_date):
            assert limit == 197
            assert market == "HK"
            rows = (
                db.query(IBDGroupRank)
                .filter(
                    IBDGroupRank.market == market,
                    IBDGroupRank.date == calculation_date,
                )
                .order_by(IBDGroupRank.rank)
                .all()
            )
            return [
                {
                    "date": row.date.isoformat(),
                    "industry_group": row.industry_group,
                    "rank": row.rank,
                    "avg_rs_rating": row.avg_rs_rating,
                    "num_stocks": row.num_stocks,
                }
                for row in rows
            ]

    class _Taxonomy:
        def sector_map_for_market(self, market):
            assert market == "HK"
            return {"Semiconductors": "Technology", "Banks": "Financials"}

    try:
        with factory() as db:
            for week in range(14):
                row_date = latest - timedelta(weeks=13 - week)
                db.add_all(
                    [
                        _rank(
                            market="HK",
                            group="Semiconductors",
                            row_date=row_date,
                            rank=1,
                            avg_rs=60 + week * 1.5,
                        ),
                        _rank(
                            market="HK",
                            group="Banks",
                            row_date=row_date,
                            rank=2,
                            avg_rs=85 - week,
                        ),
                    ]
                )
            db.commit()
            service = RRGService(
                history_provider=PersistedGroupRankHistoryProvider(_GroupRankService()),
                taxonomy_service=_Taxonomy(),
            )
            payload = service.get_rrg_scopes(
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
