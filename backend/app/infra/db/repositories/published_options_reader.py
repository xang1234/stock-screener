"""Read-only SQL queries for published options analytics."""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session, selectinload

from app.domain.options_analytics.history import HistoricalObservation
from app.domain.options_analytics.models import (
    CandidateKind,
    DividendSource,
    ObservationState,
    OptionsRunStatus,
)
from app.domain.options_analytics.ports import LastCurrentMembership
from app.infra.db.models.feature_store import FeatureRunPointer
from app.infra.db.models.options_analytics import (
    OptionsAnalyticsPointer,
    OptionsAnalyticsRun,
    OptionsAnalyticsRunItem,
)


class SqlPublishedOptionsReader:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_published_run(
        self,
        market: str,
        calculation_version: str,
    ) -> OptionsAnalyticsRun | None:
        pointer = self._session.get(
            OptionsAnalyticsPointer,
            (market.strip().upper(), calculation_version),
        )
        if pointer is None:
            return None
        return (
            self._session.query(OptionsAnalyticsRun)
            .options(
                selectinload(OptionsAnalyticsRun.items).selectinload(
                    OptionsAnalyticsRunItem.strike_points
                )
            )
            .filter(
                OptionsAnalyticsRun.id == pointer.run_id,
                OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value,
            )
            .one_or_none()
        )

    def get_published_symbol_detail(
        self,
        symbol: str,
        market: str,
        calculation_version: str,
    ) -> OptionsAnalyticsRunItem | None:
        run = self.get_published_run(market, calculation_version)
        if run is None:
            return None
        canonical = symbol.strip().upper()
        return next(
            (
                item
                for item in run.items
                if item.security_symbol == canonical
                and item.candidate_kind == CandidateKind.CURRENT.value
            ),
            None,
        )

    def get_run_diagnostics(self, run_id: int) -> OptionsAnalyticsRun | None:
        return self._session.get(OptionsAnalyticsRun, run_id)

    def latest_source_feature_run_id(self, market: str) -> int | None:
        pointer = self._session.get(
            FeatureRunPointer,
            f"latest_published_market:{market.strip().upper()}",
        )
        return pointer.run_id if pointer is not None else None

    def symbol_history(
        self,
        symbol: str,
        *,
        market: str,
        calculation_version: str,
    ) -> tuple[OptionsAnalyticsRunItem, ...]:
        rows = (
            self._session.query(OptionsAnalyticsRunItem)
            .join(OptionsAnalyticsRun)
            .options(selectinload(OptionsAnalyticsRunItem.run))
            .filter(
                OptionsAnalyticsRunItem.security_symbol == symbol.strip().upper(),
                OptionsAnalyticsRun.market == market.strip().upper(),
                OptionsAnalyticsRun.calculation_version == calculation_version,
                OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value,
            )
            .order_by(
                OptionsAnalyticsRun.as_of_date.asc(),
                OptionsAnalyticsRun.attempt_number.desc(),
                OptionsAnalyticsRun.id.desc(),
            )
            .all()
        )
        history: list[OptionsAnalyticsRunItem] = []
        seen_sessions: set[date] = set()
        for item in rows:
            run = item.run
            if run.as_of_date in seen_sessions:
                continue
            seen_sessions.add(run.as_of_date)
            history.append(item)
        return tuple(history)

    def analysis_history(
        self,
        symbol: str,
        *,
        market: str,
        calculation_version: str,
    ) -> tuple[HistoricalObservation, ...]:
        return tuple(
            HistoricalObservation(
                session=item.run.as_of_date,
                calculation_version=item.run.calculation_version,
                state=ObservationState(item.observation_state),
                max_pain=item.max_pain,
                net_gex=item.net_gex,
                gamma_flip=item.gamma_flip,
                atm_iv=item.atm_iv,
                skew_25_delta=item.skew_25_delta,
                realized_volatility=item.realized_volatility,
                vrp=item.vrp,
                activity_intensity=item.activity_intensity,
            )
            for item in self.symbol_history(
                symbol,
                market=market,
                calculation_version=calculation_version,
            )
            if item.observation_state == ObservationState.AVAILABLE.value
            and item.core_valid
        )

    def last_current_memberships(
        self,
        market: str,
        calculation_version: str,
    ) -> dict[str, LastCurrentMembership]:
        runs = (
            self._session.query(OptionsAnalyticsRun)
            .options(selectinload(OptionsAnalyticsRun.items))
            .filter(
                OptionsAnalyticsRun.market == market.strip().upper(),
                OptionsAnalyticsRun.calculation_version == calculation_version,
                OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value,
            )
            .order_by(
                OptionsAnalyticsRun.as_of_date.desc(),
                OptionsAnalyticsRun.attempt_number.desc(),
                OptionsAnalyticsRun.id.desc(),
            )
            .all()
        )
        memberships: dict[str, LastCurrentMembership] = {}
        seen_sessions: set[date] = set()
        for run in runs:
            if run.as_of_date in seen_sessions:
                continue
            seen_sessions.add(run.as_of_date)
            for item in run.items:
                if item.candidate_kind != CandidateKind.CURRENT.value:
                    continue
                if item.security_symbol in memberships:
                    continue
                ranks = [
                    rank for rank in (item.candidate_rank, item.leader_rank) if rank
                ]
                memberships[item.security_symbol] = LastCurrentMembership(
                    symbol=item.security_symbol,
                    as_of_date=run.as_of_date,
                    prior_best_rank=min(ranks) if ranks else 10_000,
                    dividend_yield=(item.assumptions_json or {}).get("dividend_yield"),
                    dividend_source=(
                        DividendSource(source)
                        if (
                            source := (item.assumptions_json or {}).get(
                                "dividend_source"
                            )
                        )
                        else None
                    ),
                )
        return memberships


__all__ = ["SqlPublishedOptionsReader"]
