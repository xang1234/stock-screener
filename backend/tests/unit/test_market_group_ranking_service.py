from __future__ import annotations

from datetime import date, datetime
import json
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.services.market_group_ranking_service as market_group_module
from app.database import Base
from app.infra.db.models.feature_store import FeatureRun
from app.services.market_group_ranking_service import MarketGroupRankingService
from app.services.group_ranking_history import select_market_run_series
from app.services.rrg_history_provider import (
    MarketDispatchRRGHistoryProvider,
    build_rrg_history_provider,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.set_calls: list[tuple[str, int]] = []

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def setex(self, key: str, ttl_seconds: int, value: bytes) -> bool:
        self.set_calls.append((key, ttl_seconds))
        self.values[key] = value
        return True


class _FakeStoredGroupRankService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get_current_rankings(
        self,
        db,
        limit=197,
        calculation_date=None,
        *,
        market="US",
    ):  # noqa: ANN001
        self.calls.append(("current", db, market, limit, calculation_date))
        return [
            {
                "industry_group": "Internet Services",
                "date": "2026-04-04",
                "rank": 4,
                "avg_rs_rating": 81.0,
                "avg_rs_rating_1m": 44.0,
                "avg_rs_rating_3m": 72.0,
            },
            {
                "industry_group": "Software",
                "date": "2026-04-04",
                "rank": 7,
                "avg_rs_rating": 76.0,
                "avg_rs_rating_1m": 51.0,
                "avg_rs_rating_3m": 68.0,
            },
        ]

    def get_rank_movers(
        self,
        db,
        period="1w",
        limit=20,
        calculation_date=None,
        *,
        market="US",
    ):  # noqa: ANN001
        self.calls.append(("movers", db, market, period, limit, calculation_date))
        return {
            "period": period,
            "gainers": [{"industry_group": "Internet Services"}],
            "losers": [{"industry_group": "Software"}],
        }

    def get_group_history(
        self,
        db,
        industry_group,
        days=180,
        *,
        market="US",
    ):  # noqa: ANN001
        self.calls.append(("history", db, market, industry_group, days))
        return {
            "industry_group": industry_group,
            "current_rank": 4,
            "current_avg_rs": 81.0,
            "current_avg_rs_1m": 44.0,
            "current_avg_rs_3m": 72.0,
            "history": [],
            "stocks": [],
        }


def test_get_rank_movers_delegates_to_stored_versioned_service():
    stored = _FakeStoredGroupRankService()
    service = MarketGroupRankingService(group_rank_service=stored)
    db = Session()
    as_of = date(2026, 4, 4)

    movers = service.get_rank_movers(
        db,
        market="hk",
        period="1m",
        limit=2,
        calculation_date=as_of,
    )

    assert movers["gainers"] == [{"industry_group": "Internet Services"}]
    assert stored.calls == [("movers", db, "HK", "1m", 2, as_of)]


def test_select_market_run_series_honors_min_runs_without_cutoff():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[FeatureRun.__table__])

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

        market_runs = select_market_run_series(
            db,
            market="HK",
            latest_run=runs[0],
            cutoff_date=None,
            min_runs=2,
        )

    assert [run.id for run in market_runs] == [10, 9]


def test_current_rank_snapshot_uses_stored_versioned_rankings():
    stored = _FakeStoredGroupRankService()
    service = MarketGroupRankingService(group_rank_service=stored)
    db = Session()

    snapshot = service.get_current_rank_snapshot(db, market="hk")

    assert snapshot.date == "2026-04-04"
    assert snapshot.ranks_by_group == {"Internet Services": 4, "Software": 7}
    assert stored.calls == [("current", db, "HK", 10_000, None)]


def test_group_history_delegates_to_stored_versioned_service():
    stored = _FakeStoredGroupRankService()
    service = MarketGroupRankingService(group_rank_service=stored)
    db = Session()

    detail = service.get_group_history(
        db,
        market="hk",
        industry_group="Internet Services",
        days=90,
    )

    assert detail["current_avg_rs_1m"] == 44.0
    assert detail["current_avg_rs_3m"] == 72.0
    assert stored.calls == [("history", db, "HK", "Internet Services", 90)]


def test_market_group_ranking_service_loads_rrg_runs_once_and_returns_ascending_series(monkeypatch):
    fake_redis = _FakeRedis()
    service = MarketGroupRankingService(redis_client=fake_redis)
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
        market_group_module,
        "select_market_run_series",
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

    db = Session()
    latest_date, meta, series = service.get_all_groups_history(
        db, market="HK", days=30
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

    assert service.get_all_groups_history(db, market="HK", days=30) == (
        latest_date,
        meta,
        series,
    )
    assert load_calls == [3, 2, 1]
    assert len(fake_redis.set_calls) == 1
    cache_key, ttl_seconds = fake_redis.set_calls[0]
    assert cache_key.startswith("rrg_history:v1:")
    assert ttl_seconds == 604800
    stored_payload = json.loads(fake_redis.values[cache_key])
    assert stored_payload["schema_version"] == 1
    assert stored_payload["series"]["Internet Services"] == [
        ["2026-04-01", 71.0, 11],
        ["2026-04-02", 72.0, 12],
        ["2026-04-03", 73.0, 13],
    ]


def test_market_group_ranking_service_recomputes_malformed_rrg_cache(monkeypatch):
    fake_redis = _FakeRedis()
    service = MarketGroupRankingService(redis_client=fake_redis)
    latest_run = SimpleNamespace(id=3, as_of_date=date(2026, 4, 3))
    expected = (
        "2026-04-03",
        {"Banks": {"rank": 1}},
        {"Banks": [(date(2026, 4, 3), 88.0, 2)]},
    )
    build_calls: list[tuple[str, int, int]] = []

    monkeypatch.setattr(
        service,
        "_get_latest_published_run",
        lambda db, *, market, calculation_date=None: latest_run,  # noqa: ARG005
    )

    def _build_result(db, *, market, days, latest_run):  # noqa: ANN001, ARG001
        build_calls.append((market, days, latest_run.id))
        return expected

    monkeypatch.setattr(service, "_build_rrg_history_result", _build_result)

    db = Session()
    cache_key = service._rrg_history_cache_key(  # noqa: SLF001
        db,
        market="HK",
        days=30,
        latest_run_id=latest_run.id,
    )
    fake_redis.values[cache_key] = (
        b'{"schema_version":1,"latest_date":"2026-04-03",'
        b'"meta":{},"series":{"Banks":[["2026-04-03"]]}}'
    )

    assert service.get_all_groups_history(db, market="hk", days=30) == expected
    assert build_calls == [("HK", 30, 3)]
    assert json.loads(fake_redis.values[cache_key])["series"]["Banks"] == [
        ["2026-04-03", 88.0, 2],
    ]


def test_rrg_history_dispatcher_uses_market_group_service_directly_for_non_us():
    calls: list[tuple[str, int, date | None]] = []

    class _GroupRankService:
        def get_current_rankings(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("US history source should not handle HK")

    class _MarketGroupRankingService:
        def get_all_groups_history(self, db, *, market, days, as_of_date=None):  # noqa: ANN001
            calls.append((market, days, as_of_date))
            return "2026-04-03", {}, {}

    provider = build_rrg_history_provider(
        group_rank_service=_GroupRankService(),
        market_group_ranking_service=_MarketGroupRankingService(),
    )

    as_of = date(2026, 4, 3)
    assert provider.get_all_groups_history(
        Session(),
        market="HK",
        days=400,
        as_of_date=as_of,
    ) == (
        "2026-04-03",
        {},
        {},
    )
    assert calls == [("HK", 400, as_of)]


def test_rrg_history_dispatcher_normalizes_configured_us_market():
    calls: list[str] = []

    class _Provider:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_all_groups_history(self, db, *, market, days, as_of_date=None):  # noqa: ANN001, ARG002
            assert as_of_date is None
            calls.append(self.name)
            return "2026-04-03", {}, {}

    provider = MarketDispatchRRGHistoryProvider(
        us_provider=_Provider("us"),
        non_us_provider=_Provider("non_us"),
        us_market="us",
    )

    assert provider.get_all_groups_history(Session(), market="US", days=400) == (
        "2026-04-03",
        {},
        {},
    )
    assert calls == ["us"]
