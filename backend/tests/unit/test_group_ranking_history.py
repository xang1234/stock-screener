from __future__ import annotations

from datetime import date, datetime
from inspect import signature
from importlib import import_module

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun
from app.services import group_ranking_history as history_module
from app.services.group_ranking_history import (
    GROUP_RANK_CHANGE_OFFSETS,
    apply_group_rank_changes,
    build_group_detail_payload,
    build_group_details,
    select_group_history_runs,
    select_market_run_series,
)


def _run(run_id: int, as_of: date, market: str, published_hour: int = 21) -> FeatureRun:
    return FeatureRun(
        id=run_id,
        as_of_date=as_of,
        run_type="daily_snapshot",
        status="published",
        published_at=datetime(as_of.year, as_of.month, as_of.day, published_hour, 30, 0),
        config_json={"universe": {"market": market}},
    )


def test_select_market_run_series_filters_market_dedupes_dates_and_honors_min_runs():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[FeatureRun.__table__])
    try:
        with Session(engine) as db:
            latest = _run(5, date(2026, 4, 5), "HK")
            db.add_all(
                [
                    latest,
                    _run(4, date(2026, 4, 4), "HK"),
                    _run(3, date(2026, 4, 4), "HK", published_hour=20),
                    _run(2, date(2026, 4, 3), "JP"),
                    _run(1, date(2026, 4, 2), "HK"),
                ]
            )
            db.commit()

            runs = select_market_run_series(
                db,
                market="hk",
                latest_run=latest,
                cutoff_date=date(2026, 4, 4),
                min_runs=3,
            )

        assert [run.id for run in runs] == [5, 4, 1]
    finally:
        engine.dispose()


def test_select_market_run_series_without_cutoff_returns_all_market_runs():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[FeatureRun.__table__])
    try:
        with Session(engine) as db:
            latest = _run(5, date(2026, 4, 5), "HK")
            db.add_all(
                [
                    latest,
                    _run(4, date(2026, 4, 4), "HK"),
                    _run(3, date(2026, 4, 4), "HK", published_hour=20),
                    _run(2, date(2026, 4, 3), "JP"),
                    _run(1, date(2026, 4, 2), "HK"),
                ]
            )
            db.commit()

            runs = select_market_run_series(
                db,
                market="HK",
                latest_run=latest,
            )

        assert [run.id for run in runs] == [5, 4, 1]
    finally:
        engine.dispose()


def test_feature_run_market_is_not_owned_by_group_history_module():
    module = import_module("app.domain.feature_store.run_metadata")

    assert not hasattr(history_module, "feature_run_market")
    assert module.feature_run_market(_run(1, date(2026, 4, 4), "HK")) == "HK"
    assert module.feature_run_market(_run(2, date(2026, 4, 4), " hk ")) == "HK"
    assert module.feature_run_market(_run(3, date(2026, 4, 4), "   ")) is None


def test_select_group_history_runs_includes_visible_history_and_change_offsets():
    runs = [
        _run(index, date(2026, 4, 1), "HK")
        for index in range(max(GROUP_RANK_CHANGE_OFFSETS.values()) + 1)
    ]

    selected = select_group_history_runs(
        runs,
        history_runs=2,
        offsets=GROUP_RANK_CHANGE_OFFSETS,
    )

    selected_indexes = [runs.index(run) for run in selected]
    assert selected_indexes[:2] == [0, 1]
    assert set(GROUP_RANK_CHANGE_OFFSETS.values()).issubset(selected_indexes)


def test_apply_group_rank_changes_has_no_caller_specific_fallback_hook():
    assert "fallback" not in signature(apply_group_rank_changes).parameters


def test_apply_group_rank_changes_only_uses_supplied_feature_run_history():
    rankings = [
        {
            "industry_group": "Semiconductors",
            "rank": 3,
            "rank_change_1w": None,
            "rank_change_1m": None,
        }
    ]
    market_runs = [
        _run(10, date(2026, 4, 10), "HK"),
        _run(9, date(2026, 4, 9), "HK"),
    ]
    historical_rankings = {
        9: [{"industry_group": "Semiconductors", "rank": 7}],
    }

    apply_group_rank_changes(
        rankings,
        market_runs,
        historical_rankings,
        offsets={"1w": 1, "1m": 2},
    )

    assert rankings[0]["rank_change_1w"] == 4
    assert rankings[0]["rank_change_1m"] is None


def test_build_group_detail_payload_uses_shared_schema_history_and_stock_sorting():
    current = _run(10, date(2026, 4, 10), "HK")
    prior = _run(9, date(2026, 4, 9), "HK")
    payload = build_group_detail_payload(
        "Semiconductors",
        ranking={
            "industry_group": "Semiconductors",
            "rank": 2,
            "avg_rs_rating": 91.5,
            "median_rs_rating": 92.0,
            "weighted_avg_rs_rating": 90.0,
            "rs_std_dev": 3.1,
            "num_stocks": 2,
            "pct_rs_above_80": 100.0,
            "top_symbol": "AAA",
            "top_symbol_name": "AAA Corp",
            "top_rs_rating": 98,
            "rank_change_1w": 3,
            "rank_change_1m": None,
            "rank_change_3m": None,
            "rank_change_6m": None,
        },
        current_rows=[
            {"symbol": "BBB", "rs_rating": 83, "composite_score": 80},
            {"symbol": "AAA", "rs_rating": 98, "composite_score": 70},
        ],
        market_runs=[current, prior],
        historical_rankings={
            10: [
                {
                    "industry_group": "Semiconductors",
                    "date": "2026-04-10",
                    "rank": 2,
                    "avg_rs_rating": 91.5,
                    "num_stocks": 2,
                }
            ],
            9: [
                {
                    "industry_group": "Semiconductors",
                    "date": "2026-04-09",
                    "rank": 5,
                    "avg_rs_rating": 88.0,
                    "num_stocks": 2,
                }
            ],
        },
    )

    assert payload["industry_group"] == "Semiconductors"
    assert payload["history"] == [
        {"date": "2026-04-10", "rank": 2, "avg_rs_rating": 91.5, "num_stocks": 2},
        {"date": "2026-04-09", "rank": 5, "avg_rs_rating": 88.0, "num_stocks": 2},
    ]
    assert [stock["symbol"] for stock in payload["stocks"]] == ["AAA", "BBB"]
    module = import_module("app.domain.feature_store.run_metadata")
    assert module.feature_run_market(current) == "HK"


def test_build_group_details_indexes_history_once_per_run(monkeypatch):
    runs = [
        _run(10, date(2026, 4, 10), "HK"),
        _run(9, date(2026, 4, 9), "HK"),
    ]
    rankings = [
        {
            "industry_group": "Semiconductors",
            "rank": 1,
            "avg_rs_rating": 95.0,
            "num_stocks": 1,
        },
        {
            "industry_group": "Software",
            "rank": 2,
            "avg_rs_rating": 90.0,
            "num_stocks": 1,
        },
    ]
    historical_rankings = {
        10: [
            {
                "industry_group": "Semiconductors",
                "date": "2026-04-10",
                "rank": 1,
                "avg_rs_rating": 95.0,
                "num_stocks": 1,
            },
            {
                "industry_group": "Software",
                "date": "2026-04-10",
                "rank": 2,
                "avg_rs_rating": 90.0,
                "num_stocks": 1,
            },
        ],
        9: [
            {
                "industry_group": "Semiconductors",
                "date": "2026-04-09",
                "rank": 2,
                "avg_rs_rating": 85.0,
                "num_stocks": 1,
            },
            {
                "industry_group": "Software",
                "date": "2026-04-09",
                "rank": 1,
                "avg_rs_rating": 96.0,
                "num_stocks": 1,
            },
        ],
    }
    original_group_rank_map = history_module.group_rank_map
    calls = 0

    def counting_group_rank_map(rows):
        nonlocal calls
        calls += 1
        return original_group_rank_map(rows)

    monkeypatch.setattr(history_module, "group_rank_map", counting_group_rank_map)

    details = build_group_details(
        rankings,
        serialized_rows=[
            {"symbol": "AAA", "ibd_industry_group": "Semiconductors", "rs_rating": 95},
            {"symbol": "BBB", "ibd_industry_group": "Software", "rs_rating": 90},
        ],
        market_runs=runs,
        historical_rankings=historical_rankings,
        history_limit=2,
    )

    assert calls == len(runs)
    assert details["Semiconductors"]["history"][1]["rank"] == 2
    assert details["Software"]["history"][1]["rank"] == 1


def test_build_group_detail_payload_from_parts_uses_the_shared_response_shape():
    builder = getattr(history_module, "build_group_detail_payload_from_parts", None)
    assert builder is not None, "shared detail builder should accept prebuilt history/stocks"

    payload = builder(
        "Semiconductors",
        ranking={
            "industry_group": "Semiconductors",
            "rank": 4,
            "avg_rs_rating": 82.5,
            "median_rs_rating": 81.0,
            "weighted_avg_rs_rating": 83.0,
            "rs_std_dev": 4.0,
            "num_stocks": 3,
            "pct_rs_above_80": 66.7,
            "top_symbol": "AAA",
            "top_symbol_name": "AAA Corp",
            "top_rs_rating": 95,
            "rank_change_1w": 2,
            "rank_change_1m": None,
            "rank_change_3m": None,
            "rank_change_6m": None,
        },
        history=[
            {"date": "2026-04-10", "rank": 4, "avg_rs_rating": 82.5, "num_stocks": 3}
        ],
        stocks=[{"symbol": "AAA", "rs_rating": 95}],
    )

    assert payload["industry_group"] == "Semiconductors"
    assert payload["current_rank"] == 4
    assert payload["history"] == [
        {"date": "2026-04-10", "rank": 4, "avg_rs_rating": 82.5, "num_stocks": 3}
    ]
    assert payload["stocks"][0]["symbol"] == "AAA"
    assert payload["stocks"][0]["rs_rating"] == 95
