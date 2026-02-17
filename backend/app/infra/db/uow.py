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
from app.infra.db.repositories.scan_repo import SqlScanRepository
from app.infra.db.repositories.scan_result_repo import SqlScanResultRepository
from app.infra.db.repositories.universe_repo import SqlUniverseRepository


class SqlUnitOfWork(UnitOfWork):
    """Transactional boundary backed by a SQLAlchemy Session."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def __enter__(self) -> Self:
        self.session: Session = self._session_factory()
        self.scans = SqlScanRepository(self.session)
        self.scan_results = SqlScanResultRepository(self.session)
        self.universe = SqlUniverseRepository(self.session)
        self.feature_runs = SqlFeatureRunRepository(self.session)
        self.feature_store = SqlFeatureStoreRepository(self.session)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.rollback()
        self.session.close()

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()
