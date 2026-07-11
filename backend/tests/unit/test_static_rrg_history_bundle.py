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
from app.services.rrg_service import MIN_TAIL_WEEKS, RRGService
from app.services.static_rrg_history_bundle import (
    STATIC_RRG_HISTORY_SCHEMA_VERSION,
    StaticRRGHistoryBundleError,
    StaticRRGHistoryBundleService,
    StaticRRGHistoryProvider,
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


def test_first_static_build_lookback_can_bootstrap_minimum_rrg_tail():
    assert STATIC_GROUP_HISTORY_LOOKBACK_DAYS >= MIN_TAIL_WEEKS * 7


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


def test_invalid_row_is_normalized_to_bundle_error(tmp_path):
    path = tmp_path / "rrg-history-hk.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": STATIC_RRG_HISTORY_SCHEMA_VERSION,
                "market": "HK",
                "through_date": "2026-07-10",
                "weeks": [
                    {
                        "week_start": "2026-07-05",
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
        assert (tmp_path / "current" / path.name).exists()
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
