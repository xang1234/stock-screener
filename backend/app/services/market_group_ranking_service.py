"""Market-aware group rankings derived from published feature runs."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.domain.common.query import FilterSpec, SortOrder, SortSpec
from app.infra.db.models.feature_store import FeatureRun
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository

GROUP_CHANGE_OFFSETS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}


class MarketGroupRankingService:
    """Read-only group ranking service for non-US markets."""

    def get_current_rankings(
        self,
        db: Session,
        *,
        market: str,
        limit: int = 197,
        calculation_date: date | None = None,
        include_rank_changes: bool = True,
    ) -> list[dict[str, Any]]:
        latest_run = self._get_latest_published_run(db, market=market, calculation_date=calculation_date)
        if latest_run is None:
            return []

        rows = self._load_run_rows(db, latest_run.id)
        rankings = self.compute_group_rankings_from_rows(rows, ranking_date=latest_run.as_of_date)
        if not rankings:
            return []
        if not include_rank_changes:
            return rankings[:limit]

        market_runs = self._get_market_run_series(
            db,
            market=market,
            latest_run=latest_run,
            min_runs=max(GROUP_CHANGE_OFFSETS.values()) + 1,
        )
        historical_rankings = {
            run.id: self.compute_group_rankings_from_rows(
                self._load_run_rows(db, run.id),
                ranking_date=run.as_of_date,
            )
            for index, run in enumerate(market_runs)
            if index in GROUP_CHANGE_OFFSETS.values()
        }
        self.apply_group_rank_changes(rankings, market_runs, historical_rankings)
        return rankings[:limit]

    def get_current_rank_map(
        self,
        db: Session,
        *,
        market: str,
        calculation_date: date | None = None,
    ) -> dict[str, int]:
        rankings = self.get_current_rankings(
            db,
            market=market,
            limit=10_000,
            calculation_date=calculation_date,
            include_rank_changes=False,
        )
        return {
            str(row["industry_group"]): int(row["rank"])
            for row in rankings
            if row.get("industry_group") and row.get("rank") is not None
        }

    def get_rank_movers(
        self,
        db: Session,
        *,
        market: str,
        period: str = "1w",
        limit: int = 20,
        calculation_date: date | None = None,
    ) -> dict[str, Any]:
        current_rankings = self.get_current_rankings(
            db,
            market=market,
            limit=10_000,
            calculation_date=calculation_date,
        )
        if not current_rankings:
            return {"period": period, "gainers": [], "losers": []}

        change_key = f"rank_change_{period}"
        groups_with_change = [
            row for row in current_rankings
            if row.get(change_key) is not None
        ]
        gainers = [row for row in groups_with_change if row[change_key] > 0]
        losers = [row for row in groups_with_change if row[change_key] < 0]
        gainers.sort(key=lambda row: row[change_key], reverse=True)
        losers.sort(key=lambda row: row[change_key])
        return {
            "period": period,
            "gainers": gainers[:limit],
            "losers": losers[:limit],
        }

    def get_group_history(
        self,
        db: Session,
        *,
        market: str,
        industry_group: str,
        days: int = 180,
    ) -> dict[str, Any]:
        latest_run = self._get_latest_published_run(db, market=market)
        if latest_run is None:
            return {"industry_group": industry_group, "history": []}

        rows = self._load_run_rows(db, latest_run.id)
        rankings = self.compute_group_rankings_from_rows(rows, ranking_date=latest_run.as_of_date)
        current = next((row for row in rankings if row["industry_group"] == industry_group), None)
        if current is None:
            return {"industry_group": industry_group, "history": []}

        cutoff_date = latest_run.as_of_date - timedelta(days=days)
        market_runs = self._get_market_run_series(
            db,
            market=market,
            latest_run=latest_run,
            cutoff_date=cutoff_date,
            min_runs=max(GROUP_CHANGE_OFFSETS.values()) + 1,
        )
        historical_rankings = {
            run.id: self.compute_group_rankings_from_rows(
                self._load_run_rows(db, run.id),
                ranking_date=run.as_of_date,
            )
            for run in market_runs
        }
        self.apply_group_rank_changes([current], market_runs, historical_rankings)

        history = []
        for run in market_runs:
            if run.as_of_date < cutoff_date:
                continue
            historical = self._group_rank_map(historical_rankings.get(run.id, [])).get(industry_group)
            if historical is None:
                continue
            history.append(
                {
                    "date": historical["date"],
                    "rank": historical["rank"],
                    "avg_rs_rating": historical["avg_rs_rating"],
                    "num_stocks": historical["num_stocks"],
                }
            )

        current_rows = [
            payload
            for payload in (self.extract_group_row_payload(row) for row in rows)
            if payload.get("ibd_industry_group") == industry_group
        ]
        current_rows.sort(
            key=lambda row: (
                row.get("rs_rating") if row.get("rs_rating") is not None else float("-inf"),
                row.get("composite_score") if row.get("composite_score") is not None else float("-inf"),
            ),
            reverse=True,
        )
        stocks = [
            {
                "symbol": row["symbol"],
                "price": row.get("current_price"),
                "rs_rating": row.get("rs_rating"),
                "rs_rating_1m": row.get("rs_rating_1m"),
                "rs_rating_3m": row.get("rs_rating_3m"),
                "rs_rating_12m": row.get("rs_rating_12m"),
                "eps_growth_qq": row.get("eps_growth_qq"),
                "eps_growth_yy": row.get("eps_growth_yy"),
                "sales_growth_qq": row.get("sales_growth_qq"),
                "sales_growth_yy": row.get("sales_growth_yy"),
                "composite_score": row.get("composite_score"),
                "stage": row.get("stage"),
                "price_sparkline_data": row.get("price_sparkline_data"),
                "price_trend": row.get("price_trend"),
                "price_change_1d": row.get("price_change_1d"),
                "rs_sparkline_data": row.get("rs_sparkline_data"),
                "rs_trend": row.get("rs_trend"),
            }
            for row in current_rows
        ]

        return {
            "industry_group": industry_group,
            "current_rank": current["rank"],
            "current_avg_rs": current["avg_rs_rating"],
            "current_median_rs": current.get("median_rs_rating"),
            "current_weighted_avg_rs": current.get("weighted_avg_rs_rating"),
            "current_rs_std_dev": current.get("rs_std_dev"),
            "num_stocks": current["num_stocks"],
            "pct_rs_above_80": current.get("pct_rs_above_80"),
            "top_symbol": current.get("top_symbol"),
            "top_rs_rating": current.get("top_rs_rating"),
            "rank_change_1w": current.get("rank_change_1w"),
            "rank_change_1m": current.get("rank_change_1m"),
            "rank_change_3m": current.get("rank_change_3m"),
            "rank_change_6m": current.get("rank_change_6m"),
            "history": history,
            "stocks": stocks,
        }

    @staticmethod
    def extract_group_row_payload(row: Any) -> dict[str, Any]:
        extended = getattr(row, "extended_fields", {}) or {}
        return {
            "symbol": getattr(row, "symbol", None),
            "composite_score": getattr(row, "composite_score", None),
            "current_price": getattr(row, "current_price", None),
            "rs_rating": extended.get("rs_rating"),
            "rs_rating_1m": extended.get("rs_rating_1m"),
            "rs_rating_3m": extended.get("rs_rating_3m"),
            "rs_rating_12m": extended.get("rs_rating_12m"),
            "eps_growth_qq": extended.get("eps_growth_qq"),
            "eps_growth_yy": extended.get("eps_growth_yy"),
            "sales_growth_qq": extended.get("sales_growth_qq"),
            "sales_growth_yy": extended.get("sales_growth_yy"),
            "stage": extended.get("stage"),
            "market_cap": extended.get("market_cap"),
            "market_cap_usd": extended.get("market_cap_usd"),
            "ibd_industry_group": extended.get("ibd_industry_group"),
            "price_sparkline_data": extended.get("price_sparkline_data"),
            "price_trend": extended.get("price_trend"),
            "price_change_1d": extended.get("price_change_1d"),
            "rs_sparkline_data": extended.get("rs_sparkline_data"),
            "rs_trend": extended.get("rs_trend"),
        }

    def compute_group_rankings_from_rows(
        self,
        rows: list[Any],
        *,
        ranking_date: date,
    ) -> list[dict[str, Any]]:
        normalized_rows = [self.extract_group_row_payload(row) for row in rows]
        return self.compute_group_rankings_from_serialized_rows(
            normalized_rows,
            ranking_date=ranking_date,
        )

    @staticmethod
    def compute_group_rankings_from_serialized_rows(
        rows: list[dict[str, Any]],
        *,
        ranking_date: date,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_name = row.get("ibd_industry_group")
            rs_rating = row.get("rs_rating")
            if not group_name or rs_rating is None:
                continue
            grouped[str(group_name)].append(row)

        rankings: list[dict[str, Any]] = []
        for group_name, group_rows in grouped.items():
            rs_values = [float(row["rs_rating"]) for row in group_rows if row.get("rs_rating") is not None]
            if not rs_values:
                continue
            avg_rs = round(sum(rs_values) / len(rs_values), 2)
            median_rs = round(float(pd.Series(rs_values).median()), 2)
            std_dev = round(float(pd.Series(rs_values).std(ddof=0)), 2) if len(rs_values) > 1 else 0.0
            weight_pairs = [
                (
                    float(row.get("market_cap_usd") or row.get("market_cap") or 0),
                    float(row["rs_rating"]),
                )
                for row in group_rows
                if row.get("rs_rating") is not None
            ]
            total_weight = sum(weight for weight, _ in weight_pairs if weight > 0)
            weighted_avg = (
                round(sum(weight * value for weight, value in weight_pairs if weight > 0) / total_weight, 2)
                if total_weight > 0
                else None
            )
            top_row = max(
                group_rows,
                key=lambda row: (
                    row.get("rs_rating") if row.get("rs_rating") is not None else float("-inf"),
                    row.get("composite_score") if row.get("composite_score") is not None else float("-inf"),
                ),
            )
            above_80 = sum(1 for value in rs_values if value >= 80)
            rankings.append(
                {
                    "industry_group": group_name,
                    "date": ranking_date.isoformat(),
                    "rank": 0,
                    "avg_rs_rating": avg_rs,
                    "median_rs_rating": median_rs,
                    "weighted_avg_rs_rating": weighted_avg,
                    "rs_std_dev": std_dev,
                    "num_stocks": len(rs_values),
                    "num_stocks_rs_above_80": above_80,
                    "pct_rs_above_80": round((above_80 / len(rs_values)) * 100, 2) if rs_values else None,
                    "top_symbol": top_row.get("symbol"),
                    "top_rs_rating": top_row.get("rs_rating"),
                    "rank_change_1w": None,
                    "rank_change_1m": None,
                    "rank_change_3m": None,
                    "rank_change_6m": None,
                }
            )

        rankings.sort(
            key=lambda row: (
                -(row.get("avg_rs_rating") or 0),
                -(row.get("weighted_avg_rs_rating") or 0),
                -(row.get("num_stocks") or 0),
                row["industry_group"],
            )
        )
        for index, row in enumerate(rankings, start=1):
            row["rank"] = index
        return rankings

    def apply_group_rank_changes(
        self,
        rankings: list[dict[str, Any]],
        market_runs: list[FeatureRun],
        historical_rankings: dict[int, list[dict[str, Any]]],
    ) -> None:
        for period, offset in GROUP_CHANGE_OFFSETS.items():
            key = f"rank_change_{period}"
            if offset >= len(market_runs):
                for ranking in rankings:
                    ranking[key] = None
                continue
            reference_run = market_runs[offset]
            reference_map = self._group_rank_map(historical_rankings.get(reference_run.id, []))
            for ranking in rankings:
                historical = reference_map.get(ranking["industry_group"])
                ranking[key] = historical["rank"] - ranking["rank"] if historical is not None else None

    def _load_run_rows(self, db: Session, run_id: int) -> list[Any]:
        repo = SqlFeatureStoreRepository(db)
        return repo.query_all_as_scan_results(
            run_id,
            FilterSpec(),
            SortSpec(field="composite_score", order=SortOrder.DESC),
            include_sparklines=True,
        )

    def _get_latest_published_run(
        self,
        db: Session,
        *,
        market: str,
        calculation_date: date | None = None,
    ) -> FeatureRun | None:
        normalized_market = str(market or "").strip().upper()
        query = (
            db.query(FeatureRun)
            .filter(FeatureRun.status == "published")
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.published_at.desc(), FeatureRun.id.desc())
        )
        if calculation_date is not None:
            query = query.filter(FeatureRun.as_of_date <= calculation_date)
        for run in query.all():
            if self._run_market(run) == normalized_market:
                return run
        return None

    def _get_market_run_series(
        self,
        db: Session,
        *,
        market: str,
        latest_run: FeatureRun,
        cutoff_date: date | None = None,
        min_runs: int = 0,
    ) -> list[FeatureRun]:
        normalized_market = str(market or "").strip().upper()
        query = (
            db.query(FeatureRun)
            .filter(
                FeatureRun.status == "published",
                FeatureRun.as_of_date <= latest_run.as_of_date,
            )
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.published_at.desc(), FeatureRun.id.desc())
        )
        market_runs: list[FeatureRun] = []
        seen_dates: set[date] = set()
        for run in query.all():
            if self._run_market(run) != normalized_market:
                continue
            if run.as_of_date in seen_dates:
                continue
            should_include = len(market_runs) < min_runs or (
                cutoff_date is not None and run.as_of_date >= cutoff_date
            )
            if not should_include:
                break
            market_runs.append(run)
            seen_dates.add(run.as_of_date)
        return market_runs

    @staticmethod
    def _run_market(run: FeatureRun) -> str | None:
        config = run.config_json or {}
        if not isinstance(config, dict):
            return None
        universe = config.get("universe")
        if isinstance(universe, dict):
            market = universe.get("market")
            if market:
                return str(market).upper()
        return None

    @staticmethod
    def _group_rank_map(rankings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(row["industry_group"]): row for row in rankings if row.get("industry_group")}


_market_group_ranking_service: MarketGroupRankingService | None = None


def get_market_group_ranking_service() -> MarketGroupRankingService:
    global _market_group_ranking_service
    if _market_group_ranking_service is None:
        _market_group_ranking_service = MarketGroupRankingService()
    return _market_group_ranking_service


__all__ = [
    "GROUP_CHANGE_OFFSETS",
    "MarketGroupRankingService",
    "get_market_group_ranking_service",
]
