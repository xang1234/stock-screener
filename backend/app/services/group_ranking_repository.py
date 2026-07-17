"""Persistence and query operations for group-ranking rows."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Mapping, Sequence

from sqlalchemy import and_, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..models.industry import IBDGroupRank


class GroupRankingRepository:
    def store_rankings(
        self,
        db: Session,
        *,
        calculation_date: date,
        rankings: Sequence[Mapping[str, Any]],
        market: str,
    ) -> None:
        values = [
            self._ranking_values(
                calculation_date,
                metrics,
                market=market,
            )
            for metrics in rankings
        ]
        if not values:
            return

        bind = db.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            stmt = pg_insert(IBDGroupRank).values(values)
            db.execute(
                stmt.on_conflict_do_update(
                    index_elements=[
                        "industry_group",
                        "date",
                        "market",
                    ],
                    set_={
                        "rank": stmt.excluded.rank,
                        "avg_rs_rating": stmt.excluded.avg_rs_rating,
                        "median_rs_rating": (
                            stmt.excluded.median_rs_rating
                        ),
                        "weighted_avg_rs_rating": (
                            stmt.excluded.weighted_avg_rs_rating
                        ),
                        "rs_std_dev": stmt.excluded.rs_std_dev,
                        "num_stocks": stmt.excluded.num_stocks,
                        "num_stocks_rs_above_80": (
                            stmt.excluded.num_stocks_rs_above_80
                        ),
                        "top_symbol": stmt.excluded.top_symbol,
                        "top_rs_rating": (
                            stmt.excluded.top_rs_rating
                        ),
                    },
                )
            )
            return

        self._store_rankings_sqlalchemy_fallback(
            db,
            calculation_date,
            values,
            market=market,
        )

    def delete_range(
        self,
        db: Session,
        *,
        start_date: date,
        end_date: date,
        market: str,
    ) -> int:
        normalized_market = (market or "US").upper()
        return (
            db.query(IBDGroupRank)
            .filter(
                and_(
                    IBDGroupRank.date >= start_date,
                    IBDGroupRank.date <= end_date,
                    IBDGroupRank.market == normalized_market,
                )
            )
            .delete(synchronize_session=False)
        )

    def current_rank_rows(
        self,
        db: Session,
        *,
        limit: int,
        market: str,
        calculation_date: date | None,
    ) -> list[IBDGroupRank]:
        normalized_market = (market or "US").upper()
        if calculation_date is not None:
            latest_date = calculation_date
        else:
            latest_record = (
                db.query(IBDGroupRank)
                .filter(
                    IBDGroupRank.market == normalized_market,
                )
                .order_by(desc(IBDGroupRank.date))
                .first()
            )
            if not latest_record:
                return []
            latest_date = latest_record.date

        return (
            db.query(IBDGroupRank)
            .filter(
                IBDGroupRank.date == latest_date,
                IBDGroupRank.market == normalized_market,
            )
            .order_by(IBDGroupRank.rank)
            .limit(limit)
            .all()
        )

    def existing_dates(
        self,
        db: Session,
        *,
        start_date: date,
        end_date: date,
        market: str,
    ) -> frozenset[date]:
        rows = (
            db.query(IBDGroupRank.date)
            .distinct()
            .filter(
                and_(
                    IBDGroupRank.date >= start_date,
                    IBDGroupRank.date <= end_date,
                    IBDGroupRank.market
                    == (market or "US").upper(),
                )
            )
            .all()
        )
        return frozenset(
            item_date
            for item_date, in rows
        )

    def historical_ranks_batch(
        self,
        db: Session,
        *,
        group_names: Sequence[str],
        current_date: date,
        period_days: Mapping[str, int],
        market: str,
    ) -> dict[tuple[str, str], int]:
        if not group_names or not period_days:
            return {}

        max_days = max(period_days.values())
        earliest_date = current_date - timedelta(
            days=max_days + 7
        )
        all_records = (
            db.query(
                IBDGroupRank.industry_group,
                IBDGroupRank.date,
                IBDGroupRank.rank,
            )
            .filter(
                and_(
                    IBDGroupRank.industry_group.in_(group_names),
                    IBDGroupRank.date >= earliest_date,
                    IBDGroupRank.date < current_date,
                    IBDGroupRank.market
                    == (market or "US").upper(),
                )
            )
            .all()
        )

        group_history: dict[str, list[tuple[date, int]]] = {}
        for record in all_records:
            group_history.setdefault(
                record.industry_group,
                [],
            ).append((record.date, record.rank))

        result: dict[tuple[str, str], int] = {}
        for group_name in group_names:
            history = group_history.get(group_name, [])
            if not history:
                continue

            for period_name, days in period_days.items():
                target_date = current_date - timedelta(days=days)
                candidates = [
                    (item_date, rank)
                    for item_date, rank in history
                    if abs((item_date - target_date).days) <= 7
                ]
                if not candidates:
                    continue

                def distance_key(
                    item: tuple[date, int],
                ) -> tuple[int, bool]:
                    item_date, _ = item
                    delta = item_date - target_date
                    return (
                        abs(delta.days),
                        delta.days > 0,
                    )

                _, closest_rank = min(
                    candidates,
                    key=distance_key,
                )
                result[(group_name, period_name)] = closest_rank

        return result

    def group_rank_rows(
        self,
        db: Session,
        *,
        industry_group: str,
        start_date: date,
        market: str,
    ) -> list[IBDGroupRank]:
        return (
            db.query(IBDGroupRank)
            .filter(
                and_(
                    IBDGroupRank.industry_group
                    == industry_group,
                    IBDGroupRank.date >= start_date,
                    IBDGroupRank.market
                    == (market or "US").upper(),
                )
            )
            .order_by(IBDGroupRank.date.desc())
            .all()
        )

    @staticmethod
    def _ranking_values(
        calculation_date: date,
        metrics: Mapping[str, Any],
        *,
        market: str,
    ) -> dict[str, Any]:
        return {
            "market": market,
            "industry_group": metrics["industry_group"],
            "date": calculation_date,
            "rank": metrics["rank"],
            "avg_rs_rating": metrics["avg_rs_rating"],
            "median_rs_rating": metrics.get(
                "median_rs_rating"
            ),
            "weighted_avg_rs_rating": metrics.get(
                "weighted_avg_rs_rating"
            ),
            "rs_std_dev": metrics.get("rs_std_dev"),
            "num_stocks": metrics["num_stocks"],
            "num_stocks_rs_above_80": metrics[
                "num_stocks_rs_above_80"
            ],
            "top_symbol": metrics["top_symbol"],
            "top_rs_rating": metrics["top_rs_rating"],
        }

    @staticmethod
    def _store_rankings_sqlalchemy_fallback(
        db: Session,
        calculation_date: date,
        values: Sequence[Mapping[str, Any]],
        *,
        market: str,
    ) -> None:
        group_names = [
            value["industry_group"]
            for value in values
        ]
        existing_records = (
            db.query(IBDGroupRank)
            .filter(
                and_(
                    IBDGroupRank.industry_group.in_(
                        group_names
                    ),
                    IBDGroupRank.date == calculation_date,
                    IBDGroupRank.market == market,
                )
            )
            .all()
        )
        existing_by_group = {
            record.industry_group: record
            for record in existing_records
        }

        for value in values:
            existing = existing_by_group.get(
                value["industry_group"]
            )
            if existing:
                existing.rank = value["rank"]
                existing.avg_rs_rating = value[
                    "avg_rs_rating"
                ]
                existing.num_stocks = value["num_stocks"]
                existing.num_stocks_rs_above_80 = value[
                    "num_stocks_rs_above_80"
                ]
                existing.top_symbol = value["top_symbol"]
                existing.top_rs_rating = value[
                    "top_rs_rating"
                ]
                existing.median_rs_rating = value.get(
                    "median_rs_rating"
                )
                existing.weighted_avg_rs_rating = value.get(
                    "weighted_avg_rs_rating"
                )
                existing.rs_std_dev = value.get("rs_std_dev")
            else:
                db.add(IBDGroupRank(**dict(value)))
