"""SQLAlchemy implementation of FeatureStoreRepository."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.query import FilterSpec, PageSpec, SortSpec
from app.domain.feature_store.models import FeaturePage, FeatureRow, FeatureRowWrite
from app.domain.feature_store.ports import FeatureStoreRepository
from app.infra.db.models.feature_store import (
    FeatureRun,
    FeatureRunPointer,
    FeatureRunUniverseSymbol,
    StockFeatureDaily,
)
from app.infra.query.feature_store_query import apply_filters, apply_sort_and_paginate

_BATCH_SIZE = 500


class SqlFeatureStoreRepository(FeatureStoreRepository):
    """Persist and query per-symbol feature rows via SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_snapshot_rows(
        self,
        run_id: int,
        rows: Sequence[FeatureRowWrite],
    ) -> int:
        if not rows:
            return 0

        count = 0
        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            values = [
                {
                    "run_id": run_id,
                    "symbol": row.symbol,
                    "as_of_date": row.as_of_date,
                    "composite_score": row.composite_score,
                    "overall_rating": row.overall_rating,
                    "passes_count": row.passes_count,
                    "details_json": row.details,
                }
                for row in batch
            ]
            stmt = sqlite_insert(StockFeatureDaily).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["run_id", "symbol"],
                set_={
                    "as_of_date": stmt.excluded.as_of_date,
                    "composite_score": stmt.excluded.composite_score,
                    "overall_rating": stmt.excluded.overall_rating,
                    "passes_count": stmt.excluded.passes_count,
                    "details_json": stmt.excluded.details_json,
                },
            )
            self._session.execute(stmt)
            count += len(batch)

        self._session.flush()
        return count

    def save_run_universe_symbols(
        self,
        run_id: int,
        symbols: Sequence[str],
    ) -> None:
        if not symbols:
            return

        for i in range(0, len(symbols), _BATCH_SIZE):
            batch = symbols[i : i + _BATCH_SIZE]
            objects = [
                FeatureRunUniverseSymbol(run_id=run_id, symbol=s) for s in batch
            ]
            self._session.bulk_save_objects(objects)

        self._session.flush()

    def count_by_run_id(self, run_id: int) -> int:
        return (
            self._session.query(func.count(StockFeatureDaily.symbol))
            .filter(StockFeatureDaily.run_id == run_id)
            .scalar()
            or 0
        )

    def query_latest(
        self,
        filters=None,
        sort=None,
        page=None,
    ) -> FeaturePage:
        p = page or PageSpec()
        pointer = (
            self._session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        if pointer is None:
            return FeaturePage(items=(), total=0, page=p.page, per_page=p.per_page)

        return self._query_features(
            pointer.run_id,
            filters or FilterSpec(),
            sort or SortSpec(),
            p,
        )

    def query_run(
        self,
        run_id: int,
        filters=None,
        sort=None,
        page=None,
    ) -> FeaturePage:
        run = self._session.get(FeatureRun, run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)

        return self._query_features(
            run_id,
            filters or FilterSpec(),
            sort or SortSpec(),
            page or PageSpec(),
        )

    # -- Private helpers ---------------------------------------------------

    def _query_features(
        self,
        run_id: int,
        filters: FilterSpec,
        sort: SortSpec,
        page: PageSpec,
    ) -> FeaturePage:
        """Build and execute a filtered, sorted, paginated feature query."""
        q = (
            self._session.query(StockFeatureDaily)
            .filter(StockFeatureDaily.run_id == run_id)
        )
        q = apply_filters(q, filters)
        rows, total = apply_sort_and_paginate(q, sort, page)

        items = tuple(self._to_domain_row(r) for r in rows)
        return FeaturePage(
            items=items,
            total=total,
            page=page.page,
            per_page=page.per_page,
        )

    @staticmethod
    def _to_domain_row(row: StockFeatureDaily) -> FeatureRow:
        """Map ORM model to domain value object."""
        return FeatureRow(
            run_id=row.run_id,
            symbol=row.symbol,
            as_of_date=row.as_of_date,
            composite_score=row.composite_score,
            overall_rating=row.overall_rating,
            passes_count=row.passes_count,
            details=row.details_json,
        )
