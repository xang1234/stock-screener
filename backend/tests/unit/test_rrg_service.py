"""Unit tests for the RRG DB orchestrator (``RRGService``).

Uses an in-memory SQLite database seeded with ``IBDGroupRank`` history and a
minimal stub for the current-rankings lookup, so it exercises the single
batched query, group omission, and the sector roll-up without the full app.

(The pure RRG math is covered separately by
``tests/unit/golden/test_golden_rrg.py``.)
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.stock_universe import StockUniverse
from app.services.rrg_service import RRGService


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _weekly_fridays(n_weeks: int, start: date = date(2024, 1, 5)):
    """n_weeks consecutive Fridays, ascending."""
    return [start + timedelta(weeks=k) for k in range(n_weeks)]


def _seed_group(session, *, market, group, dates, rs_fn, num_stocks=10, rank=1):
    for i, d in enumerate(dates):
        session.add(
            IBDGroupRank(
                market=market,
                industry_group=group,
                date=d,
                rank=rank,
                avg_rs_rating=float(rs_fn(i)),
                num_stocks=num_stocks,
                num_stocks_rs_above_80=0,
            )
        )
    session.commit()


class _StubRankService:
    """Minimal stand-in for IBDGroupRankService.get_current_rankings."""

    def __init__(self, rows):
        self._rows = rows

    def get_current_rankings(self, db, limit=197, market="US"):
        return [r for r in self._rows if r["market"] == market][:limit]


class _ExplodingRankService:
    def get_current_rankings(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("US IBDGroupRankService path should not be used")


def test_get_rrg_groups_scope_returns_quadrants_and_tails():
    session = _session()
    dates = _weekly_fridays(40)
    _seed_group(session, market="US", group="AlphaTech", dates=dates,
                rs_fn=lambda i: 40 + 0.6 * i, num_stocks=12, rank=1)
    _seed_group(session, market="US", group="BetaMetals", dates=dates,
                rs_fn=lambda i: 60 - 0.6 * i, num_stocks=8, rank=2)

    latest = dates[-1].isoformat()
    stub = _StubRankService([
        {"market": "US", "industry_group": "AlphaTech", "date": latest,
         "rank": 1, "num_stocks": 12, "avg_rs_rating": 64.0},
        {"market": "US", "industry_group": "BetaMetals", "date": latest,
         "rank": 2, "num_stocks": 8, "avg_rs_rating": 36.0},
    ])
    payload = RRGService(group_rank_service=stub).get_rrg(session, market="US")

    assert payload["date"] == latest
    assert payload["scope"] == "groups"
    by_name = {g["industry_group"]: g for g in payload["groups"]}
    assert set(by_name) == {"AlphaTech", "BetaMetals"}

    alpha = by_name["AlphaTech"]
    assert alpha["quadrant"] == "Leading"      # rising RS
    assert alpha["current"]["x"] > 100 and alpha["current"]["y"] > 100
    assert alpha["current"] == alpha["tail"][-1]
    assert by_name["BetaMetals"]["quadrant"] == "Lagging"  # falling RS

    # Sorted by rank ascending.
    assert [g["industry_group"] for g in payload["groups"]] == ["AlphaTech", "BetaMetals"]


def test_get_rrg_omits_groups_with_too_little_history():
    session = _session()
    dates = _weekly_fridays(40)
    _seed_group(session, market="US", group="AlphaTech", dates=dates,
                rs_fn=lambda i: 40 + 0.6 * i)
    # Only 5 weekly points -> below MIN_TAIL_WEEKS -> omitted.
    _seed_group(session, market="US", group="TinyNew", dates=_weekly_fridays(5),
                rs_fn=lambda i: 50.0)

    latest = dates[-1].isoformat()
    stub = _StubRankService([
        {"market": "US", "industry_group": "AlphaTech", "date": latest,
         "rank": 1, "num_stocks": 10, "avg_rs_rating": 64.0},
        {"market": "US", "industry_group": "TinyNew", "date": latest,
         "rank": 2, "num_stocks": 3, "avg_rs_rating": 50.0},
    ])
    payload = RRGService(group_rank_service=stub).get_rrg(session, market="US")
    assert [g["industry_group"] for g in payload["groups"]] == ["AlphaTech"]


def test_get_rrg_sectors_scope_rolls_groups_into_gics_sectors():
    session = _session()
    dates = _weekly_fridays(40)
    _seed_group(session, market="US", group="AlphaTech", dates=dates,
                rs_fn=lambda i: 40 + 0.6 * i, num_stocks=12)
    _seed_group(session, market="US", group="GammaChips", dates=dates,
                rs_fn=lambda i: 42 + 0.5 * i, num_stocks=8)
    _seed_group(session, market="US", group="BetaMetals", dates=dates,
                rs_fn=lambda i: 60 - 0.6 * i, num_stocks=10)

    # Group -> dominant GICS sector via constituents.
    for sym, grp, sector in [
        ("AAA", "AlphaTech", "Technology"),
        ("CCC", "GammaChips", "Technology"),
        ("MMM", "BetaMetals", "Basic Materials"),
    ]:
        session.add(IBDIndustryGroup(symbol=sym, industry_group=grp, market="US"))
        session.add(StockUniverse(symbol=sym, market="US", sector=sector))
    session.commit()

    latest = dates[-1].isoformat()
    stub = _StubRankService([
        {"market": "US", "industry_group": "AlphaTech", "date": latest,
         "rank": 1, "num_stocks": 12, "avg_rs_rating": 64.0},
        {"market": "US", "industry_group": "GammaChips", "date": latest,
         "rank": 2, "num_stocks": 8, "avg_rs_rating": 62.0},
        {"market": "US", "industry_group": "BetaMetals", "date": latest,
         "rank": 3, "num_stocks": 10, "avg_rs_rating": 36.0},
    ])
    payload = RRGService(group_rank_service=stub).get_rrg(
        session, market="US", scope="sectors"
    )

    assert payload["scope"] == "sectors"
    sectors = {g["industry_group"]: g for g in payload["groups"]}
    assert set(sectors) == {"Technology", "Basic Materials"}
    # Technology aggregates two rising groups -> Leading; num_stocks summed.
    assert sectors["Technology"]["quadrant"] == "Leading"
    assert sectors["Technology"]["num_stocks"] == 20
    assert sectors["Basic Materials"]["quadrant"] == "Lagging"


def test_get_group_sector_map_majority_vote():
    session = _session()
    for sym, sector in [("A1", "Technology"), ("A2", "Technology"), ("A3", "Healthcare")]:
        session.add(IBDIndustryGroup(symbol=sym, industry_group="AlphaTech", market="US"))
        session.add(StockUniverse(symbol=sym, market="US", sector=sector))
    session.commit()

    mapping = RRGService(group_rank_service=_StubRankService([])).get_group_sector_map(
        session, market="US"
    )
    assert mapping["AlphaTech"] == "Technology"  # 2 vs 1


def test_get_rrg_non_us_uses_feature_run_history_provider():
    session = _session()
    dates = _weekly_fridays(40)

    class _FakeMarketHistoryService:
        def get_all_groups_history(self, db, *, market, days):  # noqa: ARG002
            assert market == "HK"
            latest = dates[-1].isoformat()
            return (
                latest,
                {
                    "Internet Services": {
                        "industry_group": "Internet Services",
                        "date": latest,
                        "rank": 1,
                        "num_stocks": 9,
                        "avg_rs_rating": 80.0,
                    }
                },
                {
                    "Internet Services": [
                        (d, 40.0 + index, 9)
                        for index, d in enumerate(dates)
                    ]
                },
            )

    payload = RRGService(
        group_rank_service=_ExplodingRankService(),
        market_group_ranking_service=_FakeMarketHistoryService(),
    ).get_rrg(session, market="HK")

    assert payload["market"] == "HK"
    assert [g["industry_group"] for g in payload["groups"]] == ["Internet Services"]


def test_get_rrg_disabled_non_us_market_returns_empty_without_history_lookup():
    session = _session()

    class _FakeMarketHistoryService:
        def get_all_groups_history(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("disabled RRG markets should not load history")

    payload = RRGService(
        group_rank_service=_ExplodingRankService(),
        market_group_ranking_service=_FakeMarketHistoryService(),
    ).get_rrg_scopes(session, market="KR", scopes=("groups", "sectors"))

    assert payload["groups"]["groups"] == []
    assert payload["sectors"]["groups"] == []


def test_get_group_sector_map_uses_taxonomy_for_curated_rrg_market():
    session = _session()

    class _FakeTaxonomyService:
        def sector_map_for_market(self, market):
            assert market == "HK"
            return {"Internet Services": "Information Technology"}

    mapping = RRGService(
        group_rank_service=_StubRankService([]),
        taxonomy_service=_FakeTaxonomyService(),
    ).get_group_sector_map(session, market="HK")

    assert mapping == {"Internet Services": "Information Technology"}
