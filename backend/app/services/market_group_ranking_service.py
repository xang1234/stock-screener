"""Market-aware group rankings derived from published feature runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import json
import logging
from typing import Any, TypeAlias

try:
    from redis.exceptions import RedisError
except ModuleNotFoundError:  # pragma: no cover - exercised in desktop packaging
    RedisError = RuntimeError  # type: ignore[assignment]
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.domain.common.query import FilterSpec, SortOrder, SortSpec
from app.infra.db.models.feature_store import FeatureRun
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.services.group_detail_payloads import (
    constituent_stock_payloads_from_group_rows,
    scan_result_item_to_group_row,
)
from app.services.group_ranking_payloads import (
    compute_group_rankings_from_serialized_rows as _compute_group_rankings_from_serialized_rows,
)
from app.services.redis_pool import get_redis_client

logger = logging.getLogger(__name__)

GROUP_CHANGE_OFFSETS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}

GroupRankingHistoryResult: TypeAlias = tuple[
    str | None,
    dict[str, dict[str, Any]],
    dict[str, list[tuple[date, float, int]]],
]
RRG_HISTORY_CACHE_TTL_SECONDS = 604800
RRG_HISTORY_CACHE_SCHEMA_VERSION = 1
_REDIS_CLIENT_UNSET = object()


@dataclass(frozen=True)
class GroupRankSnapshot:
    date: str | None
    ranks_by_group: dict[str, int]


class MarketGroupRankingService:
    """Read-only group ranking service for non-US markets."""

    def __init__(
        self,
        *,
        redis_client: Any = _REDIS_CLIENT_UNSET,
        rrg_history_cache_ttl_seconds: int = RRG_HISTORY_CACHE_TTL_SECONDS,
    ) -> None:
        self._redis_client = (
            get_redis_client() if redis_client is _REDIS_CLIENT_UNSET else redis_client
        )
        self._rrg_history_cache_ttl_seconds = int(rrg_history_cache_ttl_seconds)

    def get_all_groups_history(
        self,
        db: Session,
        *,
        market: str,
        days: int,
    ) -> GroupRankingHistoryResult:
        """Return RRG-ready group-rank history from published feature runs."""
        normalized_market = str(market or "").strip().upper()
        latest_run = self._get_latest_published_run(db, market=normalized_market)
        if latest_run is None:
            return None, {}, {}

        cache_key = self._rrg_history_cache_key(
            db,
            market=normalized_market,
            days=days,
            latest_run_id=int(latest_run.id),
        )
        cached = self._get_cached_rrg_history(cache_key)
        if cached is not None:
            return cached

        result = self._build_rrg_history_result(
            db,
            market=normalized_market,
            days=days,
            latest_run=latest_run,
        )
        self._store_cached_rrg_history(cache_key, result)
        return result

    def _get_cached_rrg_history(self, cache_key: str) -> GroupRankingHistoryResult | None:
        if not self._redis_client:
            return None
        try:
            cached_bytes = self._redis_client.get(cache_key)
            if not cached_bytes:
                return None
            return self._deserialize_rrg_history(cached_bytes)
        except (RedisError, TypeError, ValueError, IndexError, OSError) as exc:
            logger.warning("Error reading RRG history cache from Redis: %s", exc)
            return None

    def _store_cached_rrg_history(
        self,
        cache_key: str,
        result: GroupRankingHistoryResult,
    ) -> None:
        if not self._redis_client or self._rrg_history_cache_ttl_seconds <= 0:
            return
        try:
            self._redis_client.setex(
                cache_key,
                self._rrg_history_cache_ttl_seconds,
                self._serialize_rrg_history(result),
            )
        except (RedisError, TypeError, ValueError, OSError) as exc:
            logger.warning("Error storing RRG history cache in Redis: %s", exc)

    @staticmethod
    def _serialize_rrg_history(result: GroupRankingHistoryResult) -> bytes:
        latest_date, meta, series = result
        payload = {
            "schema_version": RRG_HISTORY_CACHE_SCHEMA_VERSION,
            "latest_date": latest_date,
            "meta": meta,
            "series": {
                group: [
                    [point_date.isoformat(), avg_rs, num_stocks]
                    for point_date, avg_rs, num_stocks in points
                ]
                for group, points in series.items()
            },
        }
        return json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @staticmethod
    def _deserialize_rrg_history(cached_bytes: bytes | str) -> GroupRankingHistoryResult:
        payload = json.loads(cached_bytes)
        if not isinstance(payload, dict):
            raise ValueError("RRG history cache payload must be a JSON object")
        if payload.get("schema_version") != RRG_HISTORY_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported RRG history cache schema version")

        meta = payload.get("meta") or {}
        series_payload = payload.get("series") or {}
        if not isinstance(meta, dict) or not isinstance(series_payload, dict):
            raise ValueError("Malformed RRG history cache payload")

        series: dict[str, list[tuple[date, float, int]]] = {}
        for group, points in series_payload.items():
            if not isinstance(points, list):
                raise ValueError("Malformed RRG history cache series")
            series_points: list[tuple[date, float, int]] = []
            for point in points:
                if not isinstance(point, list) or len(point) != 3:
                    raise ValueError("Malformed RRG history cache point")
                series_points.append(
                    (
                        date.fromisoformat(str(point[0])),
                        float(point[1]),
                        int(point[2]),
                    )
                )
            series[str(group)] = series_points

        return payload.get("latest_date"), dict(meta), series

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
        return self.get_current_rank_snapshot(
            db,
            market=market,
            calculation_date=calculation_date,
        ).ranks_by_group

    def get_current_rank_snapshot(
        self,
        db: Session,
        *,
        market: str,
        calculation_date: date | None = None,
    ) -> GroupRankSnapshot:
        rankings = self.get_current_rankings(
            db,
            market=market,
            limit=10_000,
            calculation_date=calculation_date,
            include_rank_changes=False,
        )
        ranking_date = next(
            (str(row["date"]) for row in rankings if row.get("date")),
            None,
        )
        return GroupRankSnapshot(
            date=ranking_date,
            ranks_by_group={
                str(row["industry_group"]): int(row["rank"])
                for row in rankings
                if row.get("industry_group") and row.get("rank") is not None
            },
        )

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
                self._load_run_rows(db, run.id, include_sparklines=False),
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
        stocks = constituent_stock_payloads_from_group_rows(current_rows)

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
            "top_symbol_name": current.get("top_symbol_name"),
            "top_rs_rating": current.get("top_rs_rating"),
            "rank_change_1w": current.get("rank_change_1w"),
            "rank_change_1m": current.get("rank_change_1m"),
            "rank_change_3m": current.get("rank_change_3m"),
            "rank_change_6m": current.get("rank_change_6m"),
            "history": history,
            "stocks": stocks,
        }

    def _build_rrg_history_result(
        self,
        db: Session,
        *,
        market: str,
        days: int,
        latest_run: FeatureRun,
    ) -> GroupRankingHistoryResult:
        cutoff_date = latest_run.as_of_date - timedelta(days=days)
        market_runs = self._get_market_run_series(
            db,
            market=market,
            latest_run=latest_run,
            cutoff_date=cutoff_date,
            min_runs=0,
        )

        rankings_by_run: dict[int, list[dict[str, Any]]] = {}
        for run in market_runs:
            rows = self._load_run_rows(
                db,
                run.id,
                include_sparklines=False,
            )
            rankings_by_run[run.id] = self.compute_group_rankings_from_rows(
                rows,
                ranking_date=run.as_of_date,
            )

        latest_rankings = rankings_by_run.get(latest_run.id, [])
        meta = self._group_rank_map(latest_rankings)

        series: dict[str, list[tuple[date, float, int]]] = defaultdict(list)
        for run in reversed(market_runs):
            for ranking in rankings_by_run.get(run.id, []):
                group = ranking.get("industry_group")
                avg_rs = ranking.get("avg_rs_rating")
                if not group or avg_rs is None:
                    continue
                series[str(group)].append(
                    (
                        run.as_of_date,
                        float(avg_rs),
                        int(ranking.get("num_stocks") or 0),
                    )
                )

        return latest_run.as_of_date.isoformat(), meta, dict(series)

    @staticmethod
    def extract_group_row_payload(row: Any) -> dict[str, Any]:
        return scan_result_item_to_group_row(row)

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
        return _compute_group_rankings_from_serialized_rows(
            rows,
            ranking_date=ranking_date,
        )

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

    def _load_run_rows(
        self,
        db: Session,
        run_id: int,
        *,
        include_sparklines: bool = True,
    ) -> list[Any]:
        repo = SqlFeatureStoreRepository(db)
        return repo.query_all_as_scan_results(
            run_id,
            FilterSpec(),
            SortSpec(field="composite_score", order=SortOrder.DESC),
            include_sparklines=include_sparklines,
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

    def _rrg_history_cache_key(
        self,
        db: Session,
        *,
        market: str,
        days: int,
        latest_run_id: int,
    ) -> str:
        return (
            "rrg_history:v1:"
            f"{self._db_bind_identity(db)}:"
            f"{market}:"
            f"{int(days)}:"
            f"{int(latest_run_id)}"
        )

    @staticmethod
    def _db_bind_identity(db: Session) -> str:
        get_bind = getattr(db, "get_bind", None)
        if callable(get_bind):
            try:
                bind = get_bind()
                url = getattr(bind, "url", None)
                if url is not None:
                    render = getattr(url, "render_as_string", None)
                    if callable(render):
                        try:
                            return str(render(hide_password=True))
                        except TypeError:
                            return str(render())
                    return str(url)
                return f"bind:{id(bind)}"
            except SQLAlchemyError:
                return f"session:{id(db)}"
        return f"session:{id(db)}"


_market_group_ranking_service: MarketGroupRankingService | None = None


def get_market_group_ranking_service() -> MarketGroupRankingService:
    global _market_group_ranking_service
    if _market_group_ranking_service is None:
        _market_group_ranking_service = MarketGroupRankingService()
    return _market_group_ranking_service


__all__ = [
    "GROUP_CHANGE_OFFSETS",
    "GroupRankingHistoryResult",
    "MarketGroupRankingService",
    "get_market_group_ranking_service",
]
