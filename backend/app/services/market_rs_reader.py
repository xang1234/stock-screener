"""Read canonical Market RS values without constructing local universes."""

from __future__ import annotations

from datetime import date
from typing import Sequence

from sqlalchemy.orm import Session

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
    balanced_run_has_required_price_basis,
)
from app.domain.scanning.ports import MarketRsResolution
from app.infra.db.models.relative_strength import MarketRsRun, StockRsSnapshot
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository


class CanonicalMarketRsUnavailable(LookupError):
    pass


class SqlMarketRsReader:
    def __init__(
        self,
        session_factory,
        *,
        repository: MarketRsRunRepository | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._repository = repository or MarketRsRunRepository()

    def get(
        self,
        *,
        market: str,
        symbols: Sequence[str],
        as_of_date: date | None,
        formula_version: str | None = None,
        run_id: int | None = None,
    ) -> MarketRsResolution:
        normalized_market = market.strip().upper()
        normalized_symbols = tuple(
            dict.fromkeys(str(symbol).strip().upper() for symbol in symbols)
        )
        db: Session = self._session_factory()
        try:
            requested_run = (
                db.get(MarketRsRun, int(run_id)) if run_id is not None else None
            )
            if run_id is not None and requested_run is None:
                raise CanonicalMarketRsUnavailable(
                    f"Canonical Market RS run {run_id} is unavailable"
                )
            resolved_formula = (
                formula_version
                or (
                    requested_run.formula_version
                    if requested_run is not None
                    else self._repository.active_formula(
                        db, market=normalized_market
                    )
                )
            )
            if resolved_formula == LEGACY_RS_FORMULA_VERSION:
                if run_id is not None:
                    raise CanonicalMarketRsUnavailable(
                        "Legacy Market RS does not have a canonical run ID"
                    )
                return MarketRsResolution.legacy(
                    market=normalized_market,
                    as_of_date=as_of_date,
                    formula_version=resolved_formula,
                )
            if resolved_formula != BALANCED_RS_FORMULA_VERSION:
                raise CanonicalMarketRsUnavailable(
                    f"Unsupported active Market RS formula for {normalized_market}: "
                    f"{resolved_formula}"
                )

            if requested_run is not None:
                run = requested_run
                run_matches_request = (
                    run.status == "completed"
                    and run.market == normalized_market
                    and run.formula_version == resolved_formula
                    and (as_of_date is None or run.as_of_date == as_of_date)
                )
                if not run_matches_request:
                    raise CanonicalMarketRsUnavailable(
                        f"Canonical Market RS run {run_id} does not match the "
                        f"requested publication for {normalized_market}"
                    )
            elif as_of_date is None:
                run = self._repository.get_latest_completed(
                    db,
                    market=normalized_market,
                    formula_version=resolved_formula,
                )
            else:
                run = self._repository.get_completed_exact(
                    db,
                    market=normalized_market,
                    as_of_date=as_of_date,
                    formula_version=resolved_formula,
                )
            if run is None:
                requested_date = as_of_date.isoformat() if as_of_date else "latest"
                raise CanonicalMarketRsUnavailable(
                    f"Canonical Market RS is unavailable for {normalized_market} "
                    f"at {requested_date} ({resolved_formula})"
                )
            if not balanced_run_has_required_price_basis(run):
                raise CanonicalMarketRsUnavailable(
                    f"Canonical Market RS run {run.id} has an incompatible price basis"
                )

            rows = []
            if normalized_symbols:
                rows = (
                    db.query(StockRsSnapshot)
                    .filter(
                        StockRsSnapshot.run_id == run.id,
                        StockRsSnapshot.symbol.in_(normalized_symbols),
                    )
                    .all()
                )
            ratings = {
                row.symbol: {
                    "rs_rating": int(row.overall_rs),
                    "rs_rating_1m": int(row.rs_1m),
                    "rs_rating_3m": int(row.rs_3m),
                    "rs_rating_12m": int(row.rs_12m),
                }
                for row in rows
            }
            return MarketRsResolution.canonical(
                market=normalized_market,
                as_of_date=run.as_of_date,
                formula_version=resolved_formula,
                run_id=run.id,
                universe_size=run.eligible_symbol_count,
                ratings_by_symbol=ratings,
            )
        finally:
            db.close()
