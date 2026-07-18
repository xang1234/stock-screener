"""Unit tests for the RRG DB orchestrator (``RRGService``).

Uses an in-memory SQLite database seeded with ``IBDGroupRank`` history and a
minimal stub for the current-rankings lookup, so it exercises the single
batched query, group omission, and the sector roll-up without the full app.

(The pure RRG math is covered separately by
``tests/unit/golden/test_golden_rrg.py``.)
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.infra.db.models.relative_strength import MarketRsFormulaPointer
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.stock_universe import StockUniverse
from app.services.rrg_history_provider import (
    StoredGroupRankHistoryProvider,
    USGroupRankHistoryProvider,
)
from app.services.rrg_service import RRGService
from app.services.static_rrg_history_bundle import (
    StaticRRGHistoryBundleService,
    StaticRRGHistoryProvider,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _weekly_fridays(n_weeks: int, start: date = date(2024, 1, 5)):
    """n_weeks consecutive Fridays, ascending."""
    return [start + timedelta(weeks=k) for k in range(n_weeks)]


def _seed_group(
    session,
    *,
    market,
    group,
    dates,
    rs_fn,
    num_stocks=10,
    rank=1,
    formula_version=LEGACY_RS_FORMULA_VERSION,
):
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
                rs_formula_version=formula_version,
            )
        )
    session.commit()


class _StubRankService:
    """Minimal stand-in for IBDGroupRankService.get_current_rankings."""

    def __init__(self, rows):
        self._rows = rows

    def get_current_rankings(
        self,
        db,
        limit=197,
        calculation_date=None,
        market="US",
        formula_version=None,
    ):
        rows = [r for r in self._rows if r["market"] == market]
        if formula_version is not None:
            rows = [
                r for r in rows
                if r.get("rs_formula_version", LEGACY_RS_FORMULA_VERSION)
                == formula_version
            ]
        if calculation_date is not None:
            expected = calculation_date.isoformat()
            rows = [r for r in rows if r["date"] == expected]
        return rows[:limit]


class _ExplodingRankService:
    def get_current_rankings(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("US IBDGroupRankService path should not be used")


def _service_for_rank_rows(rows, **kwargs):
    return RRGService(
        history_provider=USGroupRankHistoryProvider(_StubRankService(rows)),
        **kwargs,
    )


def test_stored_rrg_history_provider_isolates_the_active_formula():
    session = _session()
    dates = _weekly_fridays(14)
    _seed_group(
        session,
        market="HK",
        group="Internet Services",
        dates=dates,
        rs_fn=lambda i: 10 + i,
        formula_version=LEGACY_RS_FORMULA_VERSION,
    )
    _seed_group(
        session,
        market="HK",
        group="Internet Services",
        dates=dates,
        rs_fn=lambda i: 70 + i,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    session.add(
        MarketRsFormulaPointer(
            market="HK",
            formula_version=BALANCED_RS_FORMULA_VERSION,
        )
    )
    session.commit()

    class _StoredRankService:
        def get_current_rankings(
            self,
            db,
            limit=197,
            calculation_date=None,
            *,
            market="US",
            formula_version=None,
        ):
            query = db.query(IBDGroupRank).filter(
                IBDGroupRank.market == market,
                IBDGroupRank.rs_formula_version == formula_version,
            )
            if calculation_date is not None:
                query = query.filter(IBDGroupRank.date == calculation_date)
            latest = query.order_by(IBDGroupRank.date.desc()).first()
            if latest is None:
                return []
            return [
                {
                    "industry_group": latest.industry_group,
                    "date": latest.date.isoformat(),
                    "rank": latest.rank,
                    "avg_rs_rating": latest.avg_rs_rating,
                    "num_stocks": latest.num_stocks,
                    "rs_formula_version": latest.rs_formula_version,
                }
            ]

    provider = StoredGroupRankHistoryProvider(
        _StoredRankService(),
        MarketRsRunRepository(),
    )
    latest, _meta, balanced_series = provider.get_all_groups_history(
        session,
        market="hk",
        days=365,
    )
    assert latest == dates[-1].isoformat()
    assert [point[1] for point in balanced_series["Internet Services"]] == [
        70.0 + index for index in range(14)
    ]

    pointer = session.get(MarketRsFormulaPointer, "HK")
    assert pointer is not None
    pointer.formula_version = LEGACY_RS_FORMULA_VERSION
    session.commit()
    _latest, _meta, legacy_series = provider.get_all_groups_history(
        session,
        market="HK",
        days=365,
    )
    assert [point[1] for point in legacy_series["Internet Services"]] == [
        10.0 + index for index in range(14)
    ]


def test_live_and_static_rrg_coordinates_match_for_formula_isolated_history():
    session = _session()
    dates = _weekly_fridays(40)
    _seed_group(
        session,
        market="HK",
        group="Internet Services",
        dates=dates,
        rs_fn=lambda i: 40 + 0.6 * i,
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    session.add(
        MarketRsFormulaPointer(
            market="HK",
            formula_version=BALANCED_RS_FORMULA_VERSION,
        )
    )
    session.commit()

    latest = dates[-1].isoformat()
    current = _StubRankService(
        [
            {
                "market": "HK",
                "industry_group": "Internet Services",
                "date": latest,
                "rank": 1,
                "num_stocks": 10,
                "avg_rs_rating": 63.4,
                "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
            }
        ]
    )
    live = RRGService(
        history_provider=StoredGroupRankHistoryProvider(
            current,
            MarketRsRunRepository(),
        )
    ).get_rrg(session, market="HK")
    state = StaticRRGHistoryBundleService().build(
        session,
        market="HK",
        through_date=dates[-1],
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    static = RRGService(
        history_provider=StaticRRGHistoryProvider(state)
    ).get_rrg(session, market="HK")

    assert state.rs_formula_version == BALANCED_RS_FORMULA_VERSION
    assert live["groups"] == static["groups"]


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
    payload = RRGService(
        history_provider=USGroupRankHistoryProvider(stub)
    ).get_rrg(session, market="US")

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


def test_get_rrg_passes_as_of_date_to_history_provider():
    session = _session()
    expected_date = date(2026, 6, 11)

    class _FakeHistoryProvider:
        def get_all_groups_history(self, db, *, market, days, as_of_date=None):
            assert market == "US"
            assert as_of_date == expected_date
            return (
                expected_date.isoformat(),
                {
                    "Software": {
                        "industry_group": "Software",
                        "date": expected_date.isoformat(),
                        "rank": 1,
                        "num_stocks": 10,
                        "avg_rs_rating": 80.0,
                    }
                },
                {
                    "Software": [
                        (expected_date - timedelta(weeks=i), 80.0 - i, 10)
                        for i in range(40)
                    ]
                },
            )

    payload = RRGService(history_provider=_FakeHistoryProvider()).get_rrg(
        session,
        market="US",
        as_of_date=expected_date,
    )

    assert payload["date"] == "2026-06-11"


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
    payload = RRGService(
        history_provider=USGroupRankHistoryProvider(stub)
    ).get_rrg(session, market="US")
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
    payload = RRGService(history_provider=USGroupRankHistoryProvider(stub)).get_rrg(
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

    mapping = _service_for_rank_rows([]).get_group_sector_map(
        session, market="US"
    )
    assert mapping["AlphaTech"] == "Technology"  # 2 vs 1


def test_get_rrg_non_us_uses_feature_run_history_provider():
    session = _session()
    dates = _weekly_fridays(40)

    class _FakeHistoryProvider:
        def get_all_groups_history(self, db, *, market, days, as_of_date=None):  # noqa: ARG002
            assert market == "HK"
            assert as_of_date is None
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

    payload = RRGService(history_provider=_FakeHistoryProvider()).get_rrg(
        session, market="HK"
    )

    assert payload["market"] == "HK"
    assert [g["industry_group"] for g in payload["groups"]] == ["Internet Services"]


def test_get_rrg_uses_injected_history_provider_for_all_markets():
    session = _session()
    dates = _weekly_fridays(40)
    calls: list[tuple[str, int]] = []

    class _FakeHistoryProvider:
        def get_all_groups_history(self, db, *, market, days, as_of_date=None):  # noqa: ARG002
            assert as_of_date is None
            calls.append((market, days))
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
        history_provider=_FakeHistoryProvider(),
    ).get_rrg(session, market="HK", lookback_days=123)

    assert calls == [("HK", 123)]
    assert [group["industry_group"] for group in payload["groups"]] == ["Internet Services"]


def test_get_rrg_as_of_date_caps_us_history_series():
    session = _session()
    dates = _weekly_fridays(41)
    expected_dates = dates[:-1]
    future_date = dates[-1]
    _seed_group(
        session,
        market="US",
        group="AlphaTech",
        dates=expected_dates,
        rs_fn=lambda i: 40 + 0.6 * i,
    )
    _seed_group(
        session,
        market="US",
        group="AlphaTech",
        dates=[future_date],
        rs_fn=lambda _i: 5.0,
    )

    expected_latest = expected_dates[-1]
    future_week = (future_date - timedelta(days=(future_date.weekday() + 1) % 7)).isoformat()
    stub = _StubRankService([
        {"market": "US", "industry_group": "AlphaTech", "date": expected_latest.isoformat(),
         "rank": 1, "num_stocks": 10, "avg_rs_rating": 64.0},
        {"market": "US", "industry_group": "AlphaTech", "date": future_date.isoformat(),
         "rank": 1, "num_stocks": 10, "avg_rs_rating": 5.0},
    ])

    payload = RRGService(
        history_provider=USGroupRankHistoryProvider(stub),
    ).get_rrg(session, market="US", as_of_date=expected_latest)

    alpha = payload["groups"][0]
    assert payload["date"] == expected_latest.isoformat()
    assert alpha["current"]["date"] != future_week
    assert all(point["date"] != future_week for point in alpha["tail"])


def test_get_rrg_rejects_unknown_scope_at_service_boundary():
    session = _session()

    class _FakeHistoryProvider:
        def get_all_groups_history(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("unknown scopes should fail before loading history")

    with pytest.raises(ValueError, match="Unsupported RRG scope"):
        RRGService(history_provider=_FakeHistoryProvider()).get_rrg(
            session,
            market="US",
            scope="bogus",
        )


def test_get_rrg_disabled_non_us_market_returns_empty_without_history_lookup():
    session = _session()

    class _FakeHistoryProvider:
        def get_all_groups_history(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("disabled RRG markets should not load history")

    payload = RRGService(history_provider=_FakeHistoryProvider()).get_rrg_scopes(
        session, market="KR", scopes=("groups", "sectors")
    )

    assert payload["groups"]["groups"] == []
    assert payload["sectors"]["groups"] == []


def test_get_group_sector_map_uses_taxonomy_for_curated_rrg_market():
    session = _session()

    class _FakeTaxonomyService:
        def sector_map_for_market(self, market):
            assert market == "HK"
            return {"Internet Services": "Information Technology"}

    mapping = RRGService(
        history_provider=USGroupRankHistoryProvider(_StubRankService([])),
        taxonomy_service=_FakeTaxonomyService(),
    ).get_group_sector_map(session, market="HK")

    assert mapping == {"Internet Services": "Information Technology"}


def test_non_us_sector_map_does_not_fall_back_to_us_stock_universe_join():
    session = _session()
    session.add(
        IBDIndustryGroup(
            symbol="0700.HK",
            industry_group="Internet Services",
            market="HK",
        )
    )
    session.add(StockUniverse(symbol="0700.HK", market="HK", sector="Technology"))
    session.commit()

    class _EmptyTaxonomyService:
        def sector_map_for_market(self, market):
            assert market == "HK"
            return {}

    mapping = RRGService(
        history_provider=USGroupRankHistoryProvider(_StubRankService([])),
        taxonomy_service=_EmptyTaxonomyService(),
    ).get_group_sector_map(session, market="HK")

    assert mapping == {}
