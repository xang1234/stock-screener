"""Retention policy persistence for options aggregates and strike history."""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from app.domain.options_analytics.models import OptionsRunStatus
from app.infra.db.models.options_analytics import (
    OptionsAnalyticsPointer,
    OptionsAnalyticsRun,
    OptionsAnalyticsRunItem,
    OptionsAnalyticsStrikePoint,
)


class SqlOptionsRetentionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def prune(
        self,
        *,
        aggregate_before: date,
        strike_history_run_limit: int = 30,
    ) -> None:
        pointed_run_ids = {
            run_id for (run_id,) in self._session.query(OptionsAnalyticsPointer.run_id)
        }
        old_run_ids = {
            run_id
            for (run_id,) in self._session.query(OptionsAnalyticsRun.id).filter(
                OptionsAnalyticsRun.as_of_date < aggregate_before,
                ~OptionsAnalyticsRun.id.in_(pointed_run_ids or {-1}),
            )
        }
        if old_run_ids:
            self._session.query(OptionsAnalyticsRunItem).filter(
                OptionsAnalyticsRunItem.run_id.in_(old_run_ids)
            ).delete(synchronize_session=False)

        published_ids = [
            run_id
            for (run_id,) in self._session.query(OptionsAnalyticsRun.id)
            .filter(OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value)
            .order_by(
                OptionsAnalyticsRun.as_of_date.desc(),
                OptionsAnalyticsRun.id.desc(),
            )
            .all()
        ]
        retained_run_ids = (
            set(published_ids[:strike_history_run_limit]) | pointed_run_ids
        )
        stale_run_ids = set(published_ids) - retained_run_ids
        if stale_run_ids:
            stale_item_ids = self._session.query(OptionsAnalyticsRunItem.id).filter(
                OptionsAnalyticsRunItem.run_id.in_(stale_run_ids)
            )
            self._session.query(OptionsAnalyticsStrikePoint).filter(
                OptionsAnalyticsStrikePoint.item_id.in_(stale_item_ids)
            ).delete(synchronize_session=False)
        self._session.commit()


__all__ = ["SqlOptionsRetentionRepository"]
