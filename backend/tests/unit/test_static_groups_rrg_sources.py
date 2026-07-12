"""Tests for explicit static RRG history source modes."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import static_groups_rrg_export as rrg_export
from app.services.static_groups_rrg_export import (
    StaticGroupsRRGDatabasePayloadSource,
    StaticGroupsRRGRollingHistoryPayloadSource,
    StaticGroupsRRGUnavailableError,
)
from app.services.static_rrg_history_bundle import StaticRRGHistoryPreparation
from app.services.static_rrg_history_contract import (
    STATIC_RRG_HISTORY_SCHEMA_VERSION,
    StaticRRGGroupPoint,
    StaticRRGHistoryPlan,
    StaticRRGHistoryState,
    StaticRRGWeek,
)


def _state(*, market: str, source_date: date) -> StaticRRGHistoryState:
    return StaticRRGHistoryState(
        schema_version=STATIC_RRG_HISTORY_SCHEMA_VERSION,
        market=market,
        weeks=(
            StaticRRGWeek(
                source_date=source_date,
                groups=(
                    StaticRRGGroupPoint(
                        industry_group="Semiconductors",
                        rank=1,
                        avg_rs_rating=88.0,
                        num_stocks=12,
                    ),
                ),
            ),
        ),
    )


def _plan(tmp_path: Path, market: str) -> StaticRRGHistoryPlan:
    asset_name = f"rrg-history-{market.lower()}.json.gz"
    return StaticRRGHistoryPlan(
        enabled=True,
        market=market,
        asset_name=asset_name,
        source_path=tmp_path / asset_name,
        output_path=tmp_path / "current" / asset_name,
    )


def test_rolling_source_uses_export_date_for_prepare_and_persist(monkeypatch, tmp_path):
    export_date = date(2026, 4, 18)
    state = _state(market="HK", source_date=export_date)
    preparation = StaticRRGHistoryPreparation(
        plan=_plan(tmp_path, "HK"),
        state=state,
    )
    calls = []

    class _HistoryService:
        def prepare(self, db, *, market, through_date, directory):
            calls.append(("prepare", db, market, through_date, directory))
            return preparation

        def persist(self, prepared, *, exported_as_of_date):
            calls.append(("persist", prepared, exported_as_of_date))
            return {"weeks": 1}

    monkeypatch.setattr(
        rrg_export,
        "_build_payload_from_state",
        lambda **kwargs: {"as_of_date": kwargs["expected_as_of_date"].isoformat()},
    )
    source = StaticGroupsRRGRollingHistoryPayloadSource(
        schema_version="static-site-v2",
        market="HK",
        directory=tmp_path,
        history_service=_HistoryService(),
    )
    db = object()

    payload = source.build(
        db=db,
        generated_at="2026-04-18T22:00:00Z",
        expected_as_of_date=export_date,
        market="HK",
    )
    persisted = source.persist(exported_as_of_date=export_date)

    assert payload == {"as_of_date": "2026-04-18"}
    assert persisted == {"weeks": 1}
    assert calls == [
        ("prepare", db, "HK", export_date, tmp_path),
        ("persist", preparation, export_date),
    ]


def test_rolling_source_does_not_retry_failed_preparation(tmp_path):
    calls = []

    class _HistoryService:
        def prepare(self, *_args, **_kwargs):
            calls.append("prepare")
            return StaticRRGHistoryPreparation(
                plan=_plan(tmp_path, "HK"),
                state=None,
                warnings=("history unavailable",),
            )

        def build(self, *_args, **_kwargs):
            raise AssertionError("failed preparation must not fall back to a DB rebuild")

    source = StaticGroupsRRGRollingHistoryPayloadSource(
        schema_version="static-site-v2",
        market="HK",
        directory=tmp_path,
        history_service=_HistoryService(),
    )

    with pytest.raises(StaticGroupsRRGUnavailableError, match="history unavailable"):
        source.build(
            db=object(),
            generated_at="2026-04-18T22:00:00Z",
            expected_as_of_date=date(2026, 4, 18),
            market="HK",
        )

    assert calls == ["prepare"]
    assert source.warnings == ("history unavailable",)


def test_rolling_source_rejects_wrong_market_before_preparation(tmp_path):
    history_service = SimpleNamespace(
        prepare=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wrong-market source must fail before preparation")
        )
    )
    source = StaticGroupsRRGRollingHistoryPayloadSource(
        schema_version="static-site-v2",
        market="HK",
        directory=tmp_path,
        history_service=history_service,
    )

    with pytest.raises(ValueError, match="cannot build US"):
        source.build(
            db=object(),
            generated_at="2026-04-18T22:00:00Z",
            expected_as_of_date=date(2026, 4, 18),
            market="US",
        )


def test_database_source_has_one_explicit_build_path(monkeypatch):
    export_date = date(2026, 4, 18)
    state = _state(market="HK", source_date=export_date)
    calls = []
    history_service = SimpleNamespace(
        build=lambda db, *, market, through_date: (
            calls.append((db, market, through_date)) or state
        )
    )
    monkeypatch.setattr(
        rrg_export,
        "_build_payload_from_state",
        lambda **_kwargs: {"available": True},
    )
    source = StaticGroupsRRGDatabasePayloadSource(
        schema_version="static-site-v2",
        history_service=history_service,
    )
    db = object()

    assert source.build(
        db=db,
        generated_at="2026-04-18T22:00:00Z",
        expected_as_of_date=export_date,
        market="HK",
    ) == {"available": True}
    assert calls == [(db, "HK", export_date)]
