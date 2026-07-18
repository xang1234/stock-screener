"""Aggregate versioned Group rankings from one canonical Market RS run."""

from __future__ import annotations

from datetime import date
import statistics
from typing import Any

from sqlalchemy import and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.industry import IBDGroupRank
from app.models.stock_universe import StockUniverse
from app.services.ibd_industry_service import IBDIndustryService


class CanonicalGroupRankingUnavailable(LookupError):
    pass


class CanonicalGroupRankingService:
    """Compute all Group metrics from one immutable canonical stock snapshot."""

    def __init__(
        self,
        *,
        repository: MarketRsRunRepository | None = None,
    ) -> None:
        self.repository = repository or MarketRsRunRepository()

    def calculate_and_store(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date,
        formula_version: str,
    ) -> list[dict[str, Any]]:
        normalized_market = market.strip().upper()
        run = self.repository.get_completed_exact(
            db,
            market=normalized_market,
            as_of_date=as_of_date,
            formula_version=formula_version,
        )
        if run is None:
            raise CanonicalGroupRankingUnavailable(
                f"Completed Market RS run is unavailable for {normalized_market} "
                f"on {as_of_date.isoformat()} ({formula_version})"
            )

        stock_rows = {row.symbol: row for row in run.rows}
        market_caps = dict(
            db.query(StockUniverse.symbol, StockUniverse.market_cap)
            .filter(
                StockUniverse.market == normalized_market,
                StockUniverse.symbol.in_(tuple(stock_rows)),
            )
            .all()
        )

        metrics_by_group: list[dict[str, Any]] = []
        for group_name in IBDIndustryService.get_all_groups(
            db, market=normalized_market
        ):
            symbols = IBDIndustryService.get_group_symbols(
                db,
                group_name,
                market=normalized_market,
            )
            eligible_rows = [
                stock_rows[symbol]
                for symbol in dict.fromkeys(symbols)
                if symbol in stock_rows
            ]
            if len(eligible_rows) < 3:
                continue

            overall = [float(row.overall_rs) for row in eligible_rows]
            one_month = [float(row.rs_1m) for row in eligible_rows]
            three_month = [float(row.rs_3m) for row in eligible_rows]
            caps = [
                float(market_caps.get(row.symbol) or 0.0)
                for row in eligible_rows
            ]
            positive_cap_total = sum(cap for cap in caps if cap > 0)
            weighted_avg = (
                sum(
                    value * cap
                    for value, cap in zip(overall, caps)
                    if cap > 0
                )
                / positive_cap_total
                if positive_cap_total > 0
                else None
            )
            above_80 = sum(value >= 80 for value in overall)
            top_row = min(
                eligible_rows,
                key=lambda row: (
                    -int(row.overall_rs),
                    -int(row.rs_1m),
                    -float(market_caps.get(row.symbol) or 0.0),
                    row.symbol,
                ),
            )
            avg_rs = statistics.fmean(overall)
            metrics_by_group.append(
                {
                    "market": normalized_market,
                    "industry_group": group_name,
                    "date": as_of_date,
                    "rank": 0,
                    "avg_rs_rating": avg_rs,
                    "avg_rs_rating_1m": statistics.fmean(one_month),
                    "avg_rs_rating_3m": statistics.fmean(three_month),
                    "median_rs_rating": statistics.median(overall),
                    "weighted_avg_rs_rating": weighted_avg,
                    "rs_std_dev": statistics.pstdev(overall),
                    "num_stocks": len(eligible_rows),
                    "num_stocks_rs_above_80": above_80,
                    "pct_rs_above_80": above_80 / len(eligible_rows) * 100.0,
                    "top_symbol": top_row.symbol,
                    "top_rs_rating": float(top_row.overall_rs),
                    "rs_formula_version": formula_version,
                    "market_rs_run_id": run.id,
                    "_unrounded_avg_rs": avg_rs,
                }
            )

        metrics_by_group.sort(
            key=lambda row: (
                -row["_unrounded_avg_rs"],
                row["industry_group"],
            )
        )
        for rank, row in enumerate(metrics_by_group, start=1):
            row["rank"] = rank

        self._store(
            db,
            market=normalized_market,
            as_of_date=as_of_date,
            formula_version=formula_version,
            metrics=metrics_by_group,
        )
        for row in metrics_by_group:
            row.pop("_unrounded_avg_rs", None)
        return metrics_by_group

    @staticmethod
    def _store(
        db: Session,
        *,
        market: str,
        as_of_date: date,
        formula_version: str,
        metrics: list[dict[str, Any]],
    ) -> None:
        persisted_fields = {
            column.name for column in IBDGroupRank.__table__.columns
        } - {"id", "created_at"}
        values = [
            {key: value for key, value in row.items() if key in persisted_fields}
            for row in metrics
        ]
        group_names = [row["industry_group"] for row in values]

        stale_query = db.query(IBDGroupRank).filter(
            IBDGroupRank.market == market,
            IBDGroupRank.date == as_of_date,
            IBDGroupRank.rs_formula_version == formula_version,
        )
        if group_names:
            stale_query = stale_query.filter(
                ~IBDGroupRank.industry_group.in_(group_names)
            )
        stale_query.delete(synchronize_session=False)

        if not values:
            db.commit()
            return

        bind = db.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            statement = pg_insert(IBDGroupRank).values(values)
            excluded = statement.excluded
            update_fields = {
                key: getattr(excluded, key)
                for key in persisted_fields
                if key
                not in {
                    "market",
                    "industry_group",
                    "date",
                    "rs_formula_version",
                }
            }
            db.execute(
                statement.on_conflict_do_update(
                    index_elements=[
                        "industry_group",
                        "date",
                        "market",
                        "rs_formula_version",
                    ],
                    set_=update_fields,
                )
            )
        else:
            existing = (
                db.query(IBDGroupRank)
                .filter(
                    and_(
                        IBDGroupRank.market == market,
                        IBDGroupRank.date == as_of_date,
                        IBDGroupRank.rs_formula_version == formula_version,
                        IBDGroupRank.industry_group.in_(group_names),
                    )
                )
                .all()
            )
            by_group = {row.industry_group: row for row in existing}
            for value in values:
                record = by_group.get(value["industry_group"])
                if record is None:
                    db.add(IBDGroupRank(**value))
                    continue
                for key, field_value in value.items():
                    setattr(record, key, field_value)
        db.commit()
