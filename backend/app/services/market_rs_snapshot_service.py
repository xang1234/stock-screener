"""Atomic calculation and publication of canonical Market RS snapshots."""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    BALANCED_RS_PRICE_BASIS,
    balanced_run_has_required_price_basis,
    calculate_balanced_rs,
)
from app.infra.db.models.relative_strength import MarketRsRun
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.services.market_rs_inputs import (
    MarketRsInputLoader,
    MarketRsInputUnavailable,
)


class MarketRsSnapshotService:
    def __init__(
        self,
        *,
        input_loader: MarketRsInputLoader,
        repository: MarketRsRunRepository,
    ) -> None:
        self.input_loader = input_loader
        self.repository = repository

    def calculate(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date,
        formula_version: str = BALANCED_RS_FORMULA_VERSION,
        rebuild_incompatible: bool = False,
    ) -> MarketRsRun:
        if formula_version != BALANCED_RS_FORMULA_VERSION:
            raise ValueError(
                "Unsupported Market RS formula for snapshot calculation: "
                f"{formula_version}"
            )

        existing = self.repository.get_completed_exact(
            db,
            market=market,
            as_of_date=as_of_date,
            formula_version=formula_version,
        )
        if existing is not None and balanced_run_has_required_price_basis(existing):
            return existing
        if existing is not None and not rebuild_incompatible:
            raise MarketRsSnapshotIncompatible(
                f"Completed Market RS run {existing.id} has an incompatible price basis"
            )

        try:
            inputs = self.input_loader.load(
                db,
                market=market,
                as_of_date=as_of_date,
            )
        except MarketRsInputUnavailable as exc:
            db.rollback()
            failed = self.repository.start_or_restart(
                db,
                market=market,
                as_of_date=as_of_date,
                formula_version=formula_version,
                benchmark_symbol=exc.benchmark_symbol,
                benchmark_as_of_date=as_of_date,
                universe_hash=exc.universe_hash,
                expected_symbol_count=exc.expected_symbol_count,
            )
            if failed.status != "completed":
                self.repository.mark_failed(
                    failed,
                    diagnostics={"reason_code": exc.reason_code, **exc.diagnostics},
                )
                db.commit()
            else:
                db.rollback()
            raise

        run = self.repository.start_or_restart(
            db,
            market=market,
            as_of_date=as_of_date,
            formula_version=formula_version,
            benchmark_symbol=inputs.benchmark_symbol,
            benchmark_as_of_date=inputs.benchmark_as_of_date,
            universe_hash=inputs.universe_hash,
            expected_symbol_count=len(inputs.expected_symbols),
            rebuild_completed=rebuild_incompatible,
        )
        if run.status == "completed":
            return run

        try:
            scores = calculate_balanced_rs(inputs.excess_returns_by_symbol)
            self.repository.replace_rows(db, run, scores)
            self.repository.mark_completed(
                run,
                excluded_symbol_count=len(inputs.exclusions),
                diagnostics={
                    "current_price_coverage": inputs.current_price_coverage,
                    "exclusions": inputs.exclusions,
                    "price_basis": BALANCED_RS_PRICE_BASIS,
                },
            )
            db.commit()
            db.refresh(run)
            return run
        except Exception as exc:
            db.rollback()
            failed = self.repository.start_or_restart(
                db,
                market=market,
                as_of_date=as_of_date,
                formula_version=formula_version,
                benchmark_symbol=inputs.benchmark_symbol,
                benchmark_as_of_date=inputs.benchmark_as_of_date,
                universe_hash=inputs.universe_hash,
                expected_symbol_count=len(inputs.expected_symbols),
            )
            if failed.status != "completed":
                self.repository.mark_failed(
                    failed,
                    diagnostics={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                db.commit()
            else:
                db.rollback()
            raise


class MarketRsSnapshotIncompatible(RuntimeError):
    """A completed run predates the required canonical input policy."""
