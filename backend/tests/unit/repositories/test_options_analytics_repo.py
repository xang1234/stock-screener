from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from app.database import Base
from app.domain.options_analytics.models import (
    CandidateKind,
    ChainObservation,
    HistoryReadiness,
    NormalizedOptionContract,
    OptionCandidate,
    OptionSide,
    OptionsRunStatus,
    OptionsRunSummary,
)
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.infra.db.models.options_analytics import (
    OptionsAnalyticsPointer,
    OptionsAnalyticsRun,
    OptionsAnalyticsRunItem,
    OptionsAnalyticsStrikePoint,
)
from app.infra.db.repositories.options_analytics_repo import (
    SqlOptionsAnalyticsRepository,
)
from app.infra.db.uow import SqlUnitOfWork
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from ..test_options_history_transfer import _observation as _transfer_observation


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            FeatureRunPointer.__table__,
            OptionsAnalyticsRun.__table__,
            OptionsAnalyticsRunItem.__table__,
            OptionsAnalyticsStrikePoint.__table__,
            OptionsAnalyticsPointer.__table__,
        ],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()
    db.add(
        FeatureRun(
            id=1,
            as_of_date=date(2026, 9, 4),
            run_type="daily_snapshot",
            status="published",
        )
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _start(repo, signature="sig", *, version="v1", as_of=date(2026, 9, 4), force=False):
    return repo.start_or_reuse(
        market="US",
        source_feature_run_id=1,
        calculation_version=version,
        schema_version="v1",
        provider="yahoo",
        input_signature=signature,
        as_of_date=as_of,
        force=force,
    )


def _candidate(symbol: str, kind=CandidateKind.CURRENT) -> OptionCandidate:
    return OptionCandidate(
        symbol=symbol,
        kind=kind,
        composite_score=90,
        daily_dollar_volume=200_000_000,
        spot_price=100,
        candidate_rank=1 if kind is CandidateKind.CURRENT else None,
    )


def _observation(symbol: str, fetched_day=4) -> ChainObservation:
    return ChainObservation(
        symbol=symbol,
        expiration=date(2026, 9, 18),
        source_spot_price=100,
        fetched_at=datetime(2026, 9, fetched_day, tzinfo=timezone.utc),
        contracts=(
            NormalizedOptionContract(
                side=OptionSide.CALL,
                strike=100,
                bid=1,
                ask=2,
                last_price=1.5,
                volume=100,
                open_interest=200,
                implied_volatility=0.25,
                last_trade_at=None,
                contract_size="REGULAR",
                multiplier=100,
            ),
        ),
    )


def _published_summary() -> OptionsRunSummary:
    return OptionsRunSummary(
        status=OptionsRunStatus.PUBLISHED,
        expected_count=1,
        completed_count=1,
        core_valid_current_count=1,
        failed_count=0,
        retried_count=0,
        coverage=1.0,
    )


def test_start_reuses_signature_unless_forced_attempt_is_requested(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)

    first = _start(repo)
    reused = _start(repo)
    forced = _start(repo, force=True)

    assert reused.id == first.id
    assert reused.attempt_number == 1
    assert forced.id != first.id
    assert forced.attempt_number == 2


def test_staging_and_strike_save_are_idempotent_per_symbol(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    run = _start(repo)
    repo.stage_candidates(run.id, [_candidate("AAPL"), _candidate("AAPL")])

    repo.save_item_result(
        run.id,
        "AAPL",
        observation=_observation("AAPL"),
        metric_values={"atm_iv": 0.25},
        core_valid=True,
        strike_points=[{"strike": 100, "call_open_interest": 200}],
    )
    repo.save_item_result(
        run.id,
        "AAPL",
        observation=_observation("AAPL"),
        metric_values={"atm_iv": 0.26},
        strike_points=[{"strike": 100, "call_open_interest": 250}],
    )

    assert session.query(OptionsAnalyticsRunItem).count() == 1
    assert session.query(OptionsAnalyticsStrikePoint).count() == 1
    item = session.query(OptionsAnalyticsRunItem).one()
    assert item.atm_iv == 0.26
    assert item.core_valid is True
    assert item.observation_at.replace(tzinfo=timezone.utc) == datetime(
        2026, 9, 4, tzinfo=timezone.utc
    )
    assert item.strike_points[0].call_open_interest == 250


def test_resume_returns_only_nonterminal_items(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    run = _start(repo)
    repo.stage_candidates(run.id, [_candidate("AAPL"), _candidate("MSFT"), _candidate("NVDA")])
    repo.save_item_result(run.id, "AAPL", observation=_observation("AAPL"))
    repo.save_unavailable(run.id, "MSFT", reason_codes=("expiration_unavailable",))

    assert repo.incomplete_symbols(run.id) == ("NVDA",)
    assert [item.security_symbol for item in repo.items_for_run(run.id)] == [
        "AAPL",
        "MSFT",
        "NVDA",
    ]


def test_activity_ranks_are_persisted_for_available_values_only(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    run = _start(repo)
    repo.stage_candidates(run.id, [_candidate("AAPL"), _candidate("MSFT")])

    repo.save_activity_ranks(run.id, {"AAPL": 1, "MSFT": None})

    items = {
        item.security_symbol: item
        for item in session.query(OptionsAnalyticsRunItem).all()
    }
    assert items["AAPL"].activity_rank == 1
    assert items["MSFT"].activity_rank is None


def test_run_level_assumptions_are_persisted_once(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    run = _start(repo)

    repo.save_run_assumptions(
        run.id,
        risk_free_rate=0.041,
        assumptions={"risk_free_source": "Yahoo ^IRX close"},
    )

    assert run.risk_free_rate == 0.041
    assert run.assumptions_json == {"risk_free_source": "Yahoo ^IRX close"}


def test_latest_source_run_identity_is_market_scoped(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    session.add(FeatureRunPointer(key="latest_published_market:US", run_id=1))
    session.flush()

    assert repo.latest_source_feature_run_id("US") == 1
    assert repo.latest_source_feature_run_id("HK") is None


def test_save_records_history_readiness_counts_without_filling_gaps(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    run = _start(repo)
    repo.stage_candidates(run.id, [_candidate("AAPL")])

    repo.save_item_result(
        run.id,
        "AAPL",
        observation=_observation("AAPL"),
        history_readiness=HistoryReadiness(
            short_history_available=True,
            iv_history_available=False,
            short_observation_count=5,
            iv_observation_count=12,
            lifetime_observation_count=14,
            reason_codes=("building_history",),
        ),
    )

    item = session.query(OptionsAnalyticsRunItem).one()
    assert item.short_history_observation_count == 5
    assert item.iv_history_observation_count == 12
    assert item.lifetime_observation_count == 14


def test_publish_advances_pointer_atomically_and_failed_quality_does_not(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    first = _start(repo, "first")
    repo.stage_candidates(first.id, [_candidate("AAPL")])
    repo.publish(first.id, _published_summary())
    session.commit()

    second = _start(repo, "second")
    repo.mark_failed_quality(second.id, reason_codes=("insufficient_core_coverage",))
    session.commit()

    pointer = session.get(OptionsAnalyticsPointer, ("US", "v1"))
    assert pointer.run_id == first.id
    assert repo.get_published_run("US", "v1").id == first.id


def test_published_symbol_detail_and_run_diagnostics_use_repository_state(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    run = _start(repo)
    repo.stage_candidates(run.id, [_candidate("AAPL")])
    repo.save_item_result(
        run.id,
        "AAPL",
        observation=_observation("AAPL"),
        strike_points=[{"strike": 100, "call_open_interest": 200}],
    )
    repo.publish(run.id, _published_summary())
    session.commit()

    detail = repo.get_published_symbol_detail("AAPL", "US", "v1")
    diagnostics = repo.get_run_diagnostics(run.id)

    assert detail.security_symbol == "AAPL"
    assert [point.strike for point in detail.strike_points] == [100]
    assert diagnostics.id == run.id
    assert diagnostics.status == "published"


def test_history_crosses_absent_cohort_gaps_and_ignores_other_versions(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    for index, (day, version, present) in enumerate(
        ((1, "v1", True), (2, "v1", False), (3, "v1", True), (4, "v2", True)),
        start=1,
    ):
        run = _start(
            repo,
            f"history-{index}",
            version=version,
            as_of=date(2026, 9, day),
        )
        if present:
            repo.stage_candidates(run.id, [_candidate("AAPL")])
            repo.save_item_result(run.id, "AAPL", observation=_observation("AAPL", day))
        repo.publish(run.id, _published_summary())
    session.commit()

    history = repo.symbol_history("AAPL", market="US", calculation_version="v1")

    assert [row.run.as_of_date for row in history] == [date(2026, 9, 1), date(2026, 9, 3)]


def test_last_current_membership_ignores_later_continuity_only_rows(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    current_run = _start(repo, "current", as_of=date(2026, 9, 1))
    repo.stage_candidates(current_run.id, [_candidate("AAPL")])
    repo.publish(current_run.id, _published_summary())
    continuity_run = _start(repo, "continuity", as_of=date(2026, 9, 2))
    repo.stage_candidates(continuity_run.id, [_candidate("AAPL", CandidateKind.CONTINUITY)])
    repo.publish(continuity_run.id, _published_summary())
    session.commit()

    memberships = repo.last_current_memberships("US", "v1")

    assert memberships["AAPL"].as_of_date == date(2026, 9, 1)
    assert memberships["AAPL"].prior_best_rank == 1


def test_rollback_removes_uncommitted_pointer_and_observation(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    run = _start(repo)
    repo.stage_candidates(run.id, [_candidate("AAPL")])
    repo.save_item_result(run.id, "AAPL", observation=_observation("AAPL"))
    repo.publish(run.id, _published_summary())

    session.rollback()

    assert session.query(OptionsAnalyticsRun).count() == 0
    assert session.query(OptionsAnalyticsRunItem).count() == 0
    assert session.query(OptionsAnalyticsPointer).count() == 0


def test_retention_prunes_old_aggregates_and_keeps_only_30_runs_of_strikes(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    first_date = date(2026, 1, 5)
    for index in range(32):
        run = _start(
            repo,
            f"retention-{index}",
            as_of=first_date + timedelta(days=index),
        )
        repo.stage_candidates(run.id, [_candidate("AAPL")])
        repo.save_item_result(
            run.id,
            "AAPL",
            observation=_observation("AAPL"),
            strike_points=[{"strike": 100, "call_open_interest": index + 1}],
        )
        repo.publish(run.id, _published_summary())
    session.commit()

    repo.prune(
        aggregate_before=first_date + timedelta(days=1),
        strike_history_run_limit=30,
    )
    session.commit()

    pointer = session.get(OptionsAnalyticsPointer, ("US", "v1"))
    assert session.get(OptionsAnalyticsRun, pointer.run_id).as_of_date == first_date + timedelta(days=31)
    assert session.query(OptionsAnalyticsRunItem).count() == 31
    assert session.query(OptionsAnalyticsStrikePoint).count() == 30


def test_unit_of_work_exposes_options_repository(session) -> None:
    factory = sessionmaker(bind=session.bind, expire_on_commit=False)

    with SqlUnitOfWork(factory) as uow:
        assert isinstance(uow.options_analytics, SqlOptionsAnalyticsRepository)


def test_history_transfer_import_is_idempotent_and_never_moves_pointer(session) -> None:
    repo = SqlOptionsAnalyticsRepository(session)
    local = _start(repo, "local-published")
    repo.stage_candidates(local.id, [_candidate("MSFT")])
    repo.save_item_result(local.id, "MSFT", observation=_observation("MSFT"))
    repo.publish(local.id, _published_summary())
    session.commit()

    row = _transfer_observation(as_of_date="2026-09-01")
    first = repo.import_history_transfer(
        (row,),
        market="US",
        calculation_version="v1",
        schema_version="options-history-transfer-v1",
    )
    second = repo.import_history_transfer(
        (row,),
        market="US",
        calculation_version="v1",
        schema_version="options-history-transfer-v1",
    )
    session.commit()

    pointer = session.get(OptionsAnalyticsPointer, ("US", "v1"))
    transferred = (
        session.query(OptionsAnalyticsRun)
        .filter(OptionsAnalyticsRun.origin == "history_transfer")
        .one()
    )
    assert first["imported_observations"] == 1
    assert second["imported_observations"] == 0
    assert pointer.run_id == local.id
    assert transferred.source_feature_run_id is None
    assert transferred.external_source_feature_run_key == row[
        "external_source_feature_run_key"
    ]
    assert transferred.items[0].security_symbol == "AAPL"
    assert transferred.items[0].strike_points == []
