"""Transactional persistence operations for canonical Market RS runs."""

from __future__ import annotations

from datetime import date, datetime, timezone
import math
from typing import Mapping

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, object_session, selectinload

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    StockRsScore,
)
from app.infra.db.models.relative_strength import (
    MarketRsFormulaPointer,
    MarketRsRun,
    StockRsSnapshot,
)


SUPPORTED_FORMULAS = frozenset(
    {BALANCED_RS_FORMULA_VERSION, LEGACY_RS_FORMULA_VERSION}
)


class MarketRsFormulaNotConfigured(LookupError):
    pass


class MarketRsFormulaUnsupported(ValueError):
    pass


class MarketRsRunRepository:
    @staticmethod
    def _run_query(
        db: Session,
        *,
        market: str,
        as_of_date: date,
        formula_version: str,
    ):
        return db.query(MarketRsRun).filter(
            MarketRsRun.market == market.upper(),
            MarketRsRun.as_of_date == as_of_date,
            MarketRsRun.formula_version == formula_version,
        )

    def start_or_restart(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date,
        formula_version: str,
        benchmark_symbol: str,
        benchmark_as_of_date: date,
        universe_hash: str,
        expected_symbol_count: int,
    ) -> MarketRsRun:
        normalized = market.upper()
        run = (
            self._run_query(
                db,
                market=normalized,
                as_of_date=as_of_date,
                formula_version=formula_version,
            )
            .with_for_update()
            .one_or_none()
        )
        if run is None:
            try:
                with db.begin_nested():
                    run = MarketRsRun(
                        market=normalized,
                        as_of_date=as_of_date,
                        formula_version=formula_version,
                        status="running",
                        benchmark_symbol=benchmark_symbol,
                        benchmark_as_of_date=benchmark_as_of_date,
                        universe_hash=universe_hash,
                        expected_symbol_count=expected_symbol_count,
                        eligible_symbol_count=0,
                        excluded_symbol_count=0,
                        diagnostics_json={},
                    )
                    db.add(run)
                    db.flush()
            except IntegrityError:
                run = (
                    self._run_query(
                        db,
                        market=normalized,
                        as_of_date=as_of_date,
                        formula_version=formula_version,
                    )
                    .with_for_update()
                    .one()
                )

        if run.status == "completed":
            return run

        run.rows.clear()
        run.status = "running"
        run.benchmark_symbol = benchmark_symbol
        run.benchmark_as_of_date = benchmark_as_of_date
        run.universe_hash = universe_hash
        run.expected_symbol_count = expected_symbol_count
        run.eligible_symbol_count = 0
        run.excluded_symbol_count = 0
        run.diagnostics_json = {}
        run.completed_at = None
        db.flush()
        return run

    def replace_rows(
        self,
        db: Session,
        run: MarketRsRun,
        scores: Mapping[str, StockRsScore],
    ) -> None:
        if run.status == "completed":
            raise ValueError("completed Market RS runs are immutable")
        run.rows.clear()
        run.rows.extend(
            StockRsSnapshot(
                symbol=symbol,
                overall_rs=score.overall_rs,
                rs_1m=score.rs_1m,
                rs_3m=score.rs_3m,
                rs_6m=score.rs_6m,
                rs_9m=score.rs_9m,
                rs_12m=score.rs_12m,
                weighted_composite=score.weighted_composite,
                excess_return_1m=score.excess_return_1m,
                excess_return_3m=score.excess_return_3m,
                excess_return_6m=score.excess_return_6m,
                excess_return_9m=score.excess_return_9m,
                excess_return_12m=score.excess_return_12m,
            )
            for symbol, score in sorted(scores.items())
        )
        run.eligible_symbol_count = len(scores)
        db.flush()

    @staticmethod
    def _flush(run: MarketRsRun) -> None:
        session = object_session(run)
        if session is None:
            raise ValueError("Market RS run is not attached to a session")
        session.flush()

    def mark_completed(
        self,
        run: MarketRsRun,
        *,
        excluded_symbol_count: int,
        diagnostics: dict[str, object],
    ) -> MarketRsRun:
        if len(run.rows) != run.eligible_symbol_count:
            raise ValueError("eligible symbol count does not match persisted rows")
        if run.expected_symbol_count != (
            run.eligible_symbol_count + excluded_symbol_count
        ):
            raise ValueError("expected symbol count does not reconcile")
        for row in run.rows:
            ratings = (
                row.overall_rs,
                row.rs_1m,
                row.rs_3m,
                row.rs_6m,
                row.rs_9m,
                row.rs_12m,
            )
            if any(value < 1 or value > 99 for value in ratings):
                raise ValueError(f"{row.symbol} has an out-of-range RS rating")
            if not math.isfinite(float(row.weighted_composite)):
                raise ValueError(f"{row.symbol} has a non-finite RS composite")
        run.excluded_symbol_count = excluded_symbol_count
        run.diagnostics_json = diagnostics
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        self._flush(run)
        return run

    def mark_failed(
        self,
        run: MarketRsRun,
        *,
        diagnostics: dict[str, object],
    ) -> MarketRsRun:
        if run.status == "completed":
            raise ValueError("completed Market RS runs cannot be marked failed")
        run.status = "failed"
        run.diagnostics_json = diagnostics
        run.completed_at = datetime.now(timezone.utc)
        self._flush(run)
        return run

    def get_completed_exact(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date,
        formula_version: str,
    ) -> MarketRsRun | None:
        return (
            self._run_query(
                db,
                market=market,
                as_of_date=as_of_date,
                formula_version=formula_version,
            )
            .options(selectinload(MarketRsRun.rows))
            .filter(MarketRsRun.status == "completed")
            .one_or_none()
        )

    def get_latest_completed(
        self,
        db: Session,
        *,
        market: str,
        formula_version: str,
        through_date: date | None = None,
    ) -> MarketRsRun | None:
        query = (
            db.query(MarketRsRun)
            .options(selectinload(MarketRsRun.rows))
            .filter(
                MarketRsRun.market == market.upper(),
                MarketRsRun.formula_version == formula_version,
                MarketRsRun.status == "completed",
            )
        )
        if through_date is not None:
            query = query.filter(MarketRsRun.as_of_date <= through_date)
        return query.order_by(
            MarketRsRun.as_of_date.desc(), MarketRsRun.id.desc()
        ).first()

    def active_formula(self, db: Session, *, market: str) -> str:
        pointer = db.get(MarketRsFormulaPointer, market.upper())
        if pointer is None:
            raise MarketRsFormulaNotConfigured(
                f"Market RS formula is not configured for {market.upper()}"
            )
        return str(pointer.formula_version)

    def activate_formula(
        self,
        db: Session,
        *,
        market: str,
        formula_version: str,
    ) -> None:
        if formula_version not in SUPPORTED_FORMULAS:
            raise MarketRsFormulaUnsupported(
                f"Unsupported Market RS formula: {formula_version}"
            )
        normalized = market.upper()
        pointer = (
            db.query(MarketRsFormulaPointer)
            .filter(MarketRsFormulaPointer.market == normalized)
            .with_for_update()
            .one_or_none()
        )
        if pointer is None:
            raise MarketRsFormulaNotConfigured(
                f"Market RS formula is not configured for {normalized}"
            )
        if formula_version == BALANCED_RS_FORMULA_VERSION:
            completed = (
                db.query(MarketRsRun.id)
                .filter(
                    MarketRsRun.market == normalized,
                    MarketRsRun.formula_version == formula_version,
                    MarketRsRun.status == "completed",
                )
                .first()
            )
            if completed is None:
                raise ValueError(
                    f"Cannot activate {formula_version} for {normalized} without "
                    "a completed run"
                )
        pointer.formula_version = formula_version
        db.flush()
