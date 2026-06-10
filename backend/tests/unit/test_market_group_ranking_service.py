from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun
from app.services.market_group_ranking_service import MarketGroupRankingService


def test_get_rank_movers_separates_gainers_and_losers(monkeypatch):
    service = MarketGroupRankingService()

    monkeypatch.setattr(
        service,
        "get_current_rankings",
        lambda db, *, market, limit=10_000, calculation_date=None: [  # noqa: ARG005
            {"industry_group": "Positive A", "rank_change_1w": 4},
            {"industry_group": "Positive B", "rank_change_1w": 1},
            {"industry_group": "Negative A", "rank_change_1w": -2},
            {"industry_group": "Negative B", "rank_change_1w": -5},
            {"industry_group": "Flat", "rank_change_1w": 0},
        ],
    )

    movers = service.get_rank_movers(Session(), market="HK", period="1w", limit=2)

    assert [row["industry_group"] for row in movers["gainers"]] == ["Positive A", "Positive B"]
    assert [row["industry_group"] for row in movers["losers"]] == ["Negative B", "Negative A"]


def test_get_market_run_series_honors_min_runs_without_cutoff():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[FeatureRun.__table__])
    service = MarketGroupRankingService()

    with Session(engine) as db:
        runs = [
            FeatureRun(
                id=10,
                as_of_date=date(2026, 4, 4),
                run_type="daily_snapshot",
                status="published",
                published_at=datetime(2026, 4, 4, 21, 30, 0),
                config_json={"universe": {"market": "HK"}},
            ),
            FeatureRun(
                id=9,
                as_of_date=date(2026, 4, 3),
                run_type="daily_snapshot",
                status="published",
                published_at=datetime(2026, 4, 3, 21, 30, 0),
                config_json={"universe": {"market": "HK"}},
            ),
            FeatureRun(
                id=8,
                as_of_date=date(2026, 4, 2),
                run_type="daily_snapshot",
                status="published",
                published_at=datetime(2026, 4, 2, 21, 30, 0),
                config_json={"universe": {"market": "HK"}},
            ),
            FeatureRun(
                id=7,
                as_of_date=date(2026, 4, 1),
                run_type="daily_snapshot",
                status="published",
                published_at=datetime(2026, 4, 1, 21, 30, 0),
                config_json={"universe": {"market": "JP"}},
            ),
        ]
        db.add_all(runs)
        db.commit()

        market_runs = service._get_market_run_series(  # noqa: SLF001
            db,
            market="HK",
            latest_run=runs[0],
            cutoff_date=None,
            min_runs=2,
        )

    assert [run.id for run in market_runs] == [10, 9]


def test_get_current_rank_map_skips_historical_rank_change_work(monkeypatch):
    service = MarketGroupRankingService()
    latest_run = SimpleNamespace(id=42, as_of_date=date(2026, 4, 4))

    monkeypatch.setattr(
        service,
        "_get_latest_published_run",
        lambda db, *, market, calculation_date=None: latest_run,  # noqa: ARG005
    )
    monkeypatch.setattr(
        service,
        "_load_run_rows",
        lambda db, run_id: ["placeholder"],  # noqa: ARG005
    )
    monkeypatch.setattr(
        service,
        "compute_group_rankings_from_rows",
        lambda rows, *, ranking_date: [  # noqa: ARG005
            {"industry_group": "Internet Services", "rank": 4},
            {"industry_group": "Software", "rank": 7},
        ],
    )

    def _unexpected_historical_call(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("historical rank-change work should be skipped")

    monkeypatch.setattr(service, "_get_market_run_series", _unexpected_historical_call)
    monkeypatch.setattr(service, "apply_group_rank_changes", _unexpected_historical_call)

    rank_map = service.get_current_rank_map(Session(), market="HK")

    assert rank_map == {"Internet Services": 4, "Software": 7}


def test_get_all_groups_history_loads_each_run_once_and_returns_ascending_series(monkeypatch):
    service = MarketGroupRankingService()
    latest_run = SimpleNamespace(id=3, as_of_date=date(2026, 4, 3))
    middle_run = SimpleNamespace(id=2, as_of_date=date(2026, 4, 2))
    oldest_run = SimpleNamespace(id=1, as_of_date=date(2026, 4, 1))
    load_calls: list[int] = []

    monkeypatch.setattr(
        service,
        "_get_latest_published_run",
        lambda db, *, market, calculation_date=None: latest_run,  # noqa: ARG005
    )
    monkeypatch.setattr(
        service,
        "_get_market_run_series",
        lambda db, *, market, latest_run, cutoff_date, min_runs=0: [  # noqa: ARG005
            latest_run,
            middle_run,
            oldest_run,
        ],
    )

    def _load_rows(db, run_id, *, include_sparklines=True):  # noqa: ANN001, ARG001
        assert include_sparklines is False
        load_calls.append(run_id)
        return [f"rows-{run_id}"]

    def _rankings(rows, *, ranking_date):  # noqa: ANN001
        run_id = int(str(rows[0]).split("-")[-1])
        return [
            {
                "industry_group": "Internet Services",
                "date": ranking_date.isoformat(),
                "rank": 1,
                "avg_rs_rating": 70.0 + run_id,
                "num_stocks": 10 + run_id,
            },
            {
                "industry_group": "Banks",
                "date": ranking_date.isoformat(),
                "rank": 2,
                "avg_rs_rating": 55.0 + run_id,
                "num_stocks": 5 + run_id,
            },
        ]

    monkeypatch.setattr(service, "_load_run_rows", _load_rows)
    monkeypatch.setattr(service, "compute_group_rankings_from_rows", _rankings)

    latest_date, meta, series = service.get_all_groups_history(
        Session(), market="HK", days=30
    )

    assert load_calls == [3, 2, 1]
    assert latest_date == "2026-04-03"
    assert meta["Internet Services"]["rank"] == 1
    assert series["Internet Services"] == [
        (date(2026, 4, 1), 71.0, 11),
        (date(2026, 4, 2), 72.0, 12),
        (date(2026, 4, 3), 73.0, 13),
    ]
    assert series["Banks"][-1] == (date(2026, 4, 3), 58.0, 8)
