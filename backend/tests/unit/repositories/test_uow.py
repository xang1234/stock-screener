"""Tests for SqlUnitOfWork using in-memory SQLite.

Verifies that all repositories share the same session and that
cross-repo transactions are visible within a single UoW.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import sessionmaker

from app.domain.feature_store.models import FeatureRowWrite, RunType
from app.infra.db.uow import SqlUnitOfWork


class TestSqlUnitOfWork:
    def test_nested_use_shares_the_session_and_closes_it_once(self, engine):
        """A use case given an entered UoW re-enters it (``with uow:``). It must not
        swap in a second session and orphan the first one mid-transaction: that
        session stays idle in transaction until garbage collection and blocked the
        PostgreSQL integration sweep's TRUNCATE."""
        sessions = []
        base = sessionmaker(bind=engine)

        def factory():
            sessions.append(base())
            return sessions[-1]

        uow = SqlUnitOfWork(factory)
        with uow:
            outer = uow.session
            uow.scans.list_recent(limit=1)  # begins the outer transaction
            with uow:
                assert uow.session is outer
                uow.scans.list_recent(limit=1)
            assert uow.session is outer
            assert outer.in_transaction()  # the inner exit leaves it open

        assert sessions == [outer]
        assert not outer.in_transaction()

    def test_nested_failure_rolls_back_but_keeps_the_outer_session(self, engine):
        uow = SqlUnitOfWork(sessionmaker(bind=engine))
        with uow:
            outer = uow.session
            try:
                with uow:
                    uow.scans.list_recent(limit=1)
                    raise RuntimeError("use case failed")
            except RuntimeError:
                pass
            assert uow.session is outer
            assert not outer.in_transaction()  # rolled back, still usable
            uow.scans.list_recent(limit=1)
        assert not outer.in_transaction()

    def test_failed_enter_closes_its_session_and_allows_a_fresh_entry(self, engine, monkeypatch):
        import app.infra.db.uow as uow_module

        sessions = []
        base = sessionmaker(bind=engine)

        def factory():
            sessions.append(base())
            return sessions[-1]

        class Broken:
            def __init__(self, session):
                raise RuntimeError("repository setup failed")

        uow = SqlUnitOfWork(factory)
        with monkeypatch.context() as patch:
            patch.setattr(uow_module, "SqlScanResultRepository", Broken)
            try:
                with uow:
                    pass
            except RuntimeError:
                pass
        assert len(sessions) == 1 and not sessions[0].in_transaction()

        with uow:  # a fresh, top-level entry, not a nested one
            assert uow.session is sessions[1]
        assert len(sessions) == 2

    def test_outermost_exit_closes_even_when_rollback_fails(self, engine, monkeypatch):
        uow = SqlUnitOfWork(sessionmaker(bind=engine))
        closed = []
        try:
            with uow:
                monkeypatch.setattr(uow.session, "rollback", lambda: (_ for _ in ()).throw(OSError("dead connection")))
                monkeypatch.setattr(uow.session, "close", lambda: closed.append(True))
                raise RuntimeError("use case failed")
        except OSError:
            pass
        assert closed == [True]

    def test_repos_share_session(self, engine):
        factory = sessionmaker(bind=engine)
        uow = SqlUnitOfWork(factory)

        with uow:
            sessions = {
                id(uow.scans._session),
                id(uow.scan_results._session),
                id(uow.universe._session),
                id(uow.feature_runs._session),
                id(uow.feature_store._session),
            }
            # All 5 repos must reference the exact same session object
            assert len(sessions) == 1

    def test_cross_repo_transaction(self, engine):
        """Write via feature_runs, upsert via feature_store, read back."""
        factory = sessionmaker(bind=engine)

        # Write data in one UoW
        with SqlUnitOfWork(factory) as uow:
            run = uow.feature_runs.start_run(
                as_of_date=date(2026, 2, 17),
                run_type=RunType.DAILY_SNAPSHOT,
            )
            row = FeatureRowWrite(
                symbol="AAPL",
                as_of_date=date(2026, 2, 17),
                composite_score=85.0,
                overall_rating=4,
                passes_count=3,
                details={"minervini_score": 85.0},
            )
            count = uow.feature_store.upsert_snapshot_rows(run.id, [row])
            assert count == 1
            uow.commit()

        # Verify in a fresh UoW
        with SqlUnitOfWork(factory) as uow:
            stored_count = uow.feature_store.count_by_run_id(run.id)
            assert stored_count == 1
