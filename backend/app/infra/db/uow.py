"""SQLAlchemy Unit of Work — concrete implementation of the domain UoW port.

Wraps a SQLAlchemy Session and exposes repository instances that share
the same session, so a use case can read/write across multiple repos
within one transaction.
"""

from __future__ import annotations

from typing import Self

from sqlalchemy.orm import Session, sessionmaker

from app.domain.common.uow import UnitOfWork
from app.infra.db.repositories.feature_run_repo import SqlFeatureRunRepository
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.infra.db.repositories.opportunity_summary_repo import (
    SqlOpportunityStateSummaryRepository,
)
from app.infra.db.repositories.options_history_repository import (
    SqlOptionsHistoryRepository,
)
from app.infra.db.repositories.options_retention import (
    SqlOptionsRetentionRepository,
)
from app.infra.db.repositories.options_run_writer import SqlOptionsRunWriter
from app.infra.db.repositories.published_options_reader import (
    SqlPublishedOptionsReader,
)
from app.infra.db.repositories.scan_repo import SqlScanRepository
from app.infra.db.repositories.scan_result_repo import SqlScanResultRepository
from app.infra.db.repositories.universe_repo import SqlUniverseRepository


class SqlUnitOfWork(UnitOfWork):
    """Transactional boundary backed by a SQLAlchemy Session."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        self._depth = 0

    def __enter__(self) -> Self:
        # Re-entrant: a use case handed an entered UoW (``with uow:``) shares its
        # session. Swapping in a new one would orphan the first mid-transaction.
        if self._depth:
            self._depth += 1
            return self
        self.session: Session = self._session_factory()
        try:
            self.scans = SqlScanRepository(self.session)
            self.scan_results = SqlScanResultRepository(self.session)
            self.opportunity_summaries = SqlOpportunityStateSummaryRepository(self.session)
            self.universe = SqlUniverseRepository(self.session)
            self.feature_runs = SqlFeatureRunRepository(self.session)
            self.feature_store = SqlFeatureStoreRepository(self.session)
            self.options_run_writer = SqlOptionsRunWriter(self.session)
            self.published_options = SqlPublishedOptionsReader(self.session)
            self.options_history = SqlOptionsHistoryRepository(self.session)
            self.options_retention = SqlOptionsRetentionRepository(self.session)
        except BaseException:
            # __exit__ never runs when __enter__ raises.
            self.session.close()
            raise
        self._depth = 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._depth -= 1
        try:
            if exc_type is not None:
                self.rollback()
        finally:
            if self._depth == 0:
                self.session.close()

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()
