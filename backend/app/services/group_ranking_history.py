"""Shared group-ranking history helpers for live and static group views."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from app.domain.relative_strength import LEGACY_RS_FORMULA_VERSION
from app.domain.feature_store.run_metadata import (
    feature_run_market as _feature_run_market,
)
from app.infra.db.models.feature_store import FeatureRun
from app.schemas.groups import GroupDetailResponse, HistoricalDataPoint
from app.services.group_detail_payloads import constituent_stock_payloads_from_group_rows

GROUP_RANK_CHANGE_OFFSETS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}


def group_rank_map(rankings: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["industry_group"]): row
        for row in rankings
        if row.get("industry_group")
    }


def select_market_run_series(
    db: Session,
    *,
    market: str,
    latest_run: FeatureRun,
    cutoff_date: date | None = None,
    min_runs: int = 0,
    max_runs: int | None = None,
) -> list[FeatureRun]:
    normalized_market = str(market or "").strip().upper()
    query = (
        db.query(FeatureRun)
        .filter(
            FeatureRun.status == "published",
            FeatureRun.as_of_date <= latest_run.as_of_date,
        )
        .order_by(
            FeatureRun.as_of_date.desc(),
            FeatureRun.published_at.desc(),
            FeatureRun.id.desc(),
        )
    )
    market_runs: list[FeatureRun] = []
    seen_dates: set[date] = set()
    for run in query.all():
        if max_runs is not None and len(market_runs) >= max_runs:
            break
        if _feature_run_market(run) != normalized_market:
            continue
        if run.as_of_date in seen_dates:
            continue
        if cutoff_date is None:
            should_include = min_runs <= 0 or len(market_runs) < min_runs
        else:
            should_include = len(market_runs) < min_runs or run.as_of_date >= cutoff_date
        if not should_include:
            break
        market_runs.append(run)
        seen_dates.add(run.as_of_date)
    return market_runs


def select_group_history_runs(
    market_runs: list[FeatureRun],
    *,
    history_runs: int,
    offsets: Mapping[str, int] = GROUP_RANK_CHANGE_OFFSETS,
) -> list[FeatureRun]:
    selected_indexes = set(range(min(history_runs, len(market_runs))))
    selected_indexes.update(
        offset
        for offset in offsets.values()
        if offset < len(market_runs)
    )
    return [market_runs[index] for index in sorted(selected_indexes)]


def apply_group_rank_changes(
    rankings: list[dict[str, Any]],
    market_runs: list[FeatureRun],
    historical_rankings: Mapping[int, list[dict[str, Any]]],
    *,
    offsets: Mapping[str, int] = GROUP_RANK_CHANGE_OFFSETS,
) -> None:
    for period, offset in offsets.items():
        key = f"rank_change_{period}"
        if offset >= len(market_runs):
            for ranking in rankings:
                ranking[key] = None
            continue
        reference_run = market_runs[offset]
        reference_map = group_rank_map(historical_rankings.get(reference_run.id, []))
        for ranking in rankings:
            historical = reference_map.get(str(ranking["industry_group"]))
            ranking[key] = (
                historical["rank"] - ranking["rank"]
                if historical is not None
                else None
            )


def _group_history_points_from_maps(
    industry_group: str,
    *,
    market_runs: Iterable[FeatureRun],
    ranking_maps: Mapping[int, Mapping[str, Mapping[str, Any]]],
    cutoff_date: date | None = None,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for run in market_runs:
        if cutoff_date is not None and run.as_of_date < cutoff_date:
            continue
        historical = ranking_maps.get(run.id, {}).get(industry_group)
        if historical is None:
            continue
        history.append(
            HistoricalDataPoint(
                date=historical["date"],
                rank=historical["rank"],
                avg_rs_rating=historical["avg_rs_rating"],
                avg_rs_rating_1m=historical.get("avg_rs_rating_1m"),
                avg_rs_rating_3m=historical.get("avg_rs_rating_3m"),
                num_stocks=historical["num_stocks"],
            ).model_dump(mode="json", exclude_none=True)
        )
    return history


def group_history_points(
    industry_group: str,
    *,
    market_runs: Iterable[FeatureRun],
    historical_rankings: Mapping[int, list[dict[str, Any]]],
    cutoff_date: date | None = None,
) -> list[dict[str, Any]]:
    runs = list(market_runs)
    ranking_maps = {
        run.id: group_rank_map(historical_rankings.get(run.id, []))
        for run in runs
    }
    return _group_history_points_from_maps(
        industry_group,
        market_runs=runs,
        ranking_maps=ranking_maps,
        cutoff_date=cutoff_date,
    )


def build_group_detail_payload(
    industry_group: str,
    *,
    ranking: Mapping[str, Any],
    current_rows: Iterable[Mapping[str, Any]],
    market_runs: list[FeatureRun],
    historical_rankings: Mapping[int, list[dict[str, Any]]],
    history_cutoff_date: date | None = None,
    history_limit: int | None = None,
) -> dict[str, Any]:
    history_runs = market_runs if history_limit is None else market_runs[:history_limit]
    return build_group_detail_payload_from_parts(
        industry_group,
        ranking=ranking,
        history=group_history_points(
            industry_group,
            market_runs=history_runs,
            historical_rankings=historical_rankings,
            cutoff_date=history_cutoff_date,
        ),
        stocks=constituent_stock_payloads_from_group_rows(current_rows),
    )


def build_group_detail_payload_from_parts(
    industry_group: str,
    *,
    ranking: Mapping[str, Any],
    history: Iterable[Mapping[str, Any]],
    stocks: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    response = GroupDetailResponse(
        industry_group=industry_group,
        current_rank=ranking["rank"],
        current_avg_rs=ranking["avg_rs_rating"],
        current_avg_rs_1m=ranking.get("avg_rs_rating_1m"),
        current_avg_rs_3m=ranking.get("avg_rs_rating_3m"),
        current_median_rs=ranking.get("median_rs_rating"),
        current_weighted_avg_rs=ranking.get("weighted_avg_rs_rating"),
        current_rs_std_dev=ranking.get("rs_std_dev"),
        num_stocks=ranking["num_stocks"],
        pct_rs_above_80=ranking.get("pct_rs_above_80"),
        top_symbol=ranking.get("top_symbol"),
        top_symbol_name=ranking.get("top_symbol_name"),
        top_rs_rating=ranking.get("top_rs_rating"),
        rs_formula_version=ranking.get(
            "rs_formula_version", LEGACY_RS_FORMULA_VERSION
        ),
        market_rs_run_id=ranking.get("market_rs_run_id"),
        rank_change_1w=ranking.get("rank_change_1w"),
        rank_change_1m=ranking.get("rank_change_1m"),
        rank_change_3m=ranking.get("rank_change_3m"),
        rank_change_6m=ranking.get("rank_change_6m"),
        history=[
            HistoricalDataPoint(**point).model_dump(mode="json", exclude_none=True)
            for point in history
        ],
        stocks=list(stocks),
    )
    payload = response.model_dump(mode="json", exclude_none=True)
    # Rank-change keys are part of the stable detail contract even when the
    # requested lookback is unavailable. Other optional fields retain the
    # historical omit-when-missing behavior.
    for key in GROUP_RANK_CHANGE_OFFSETS:
        field = f"rank_change_{key}"
        payload[field] = getattr(response, field)
    return payload


def build_group_details(
    rankings: Iterable[Mapping[str, Any]],
    *,
    serialized_rows: Iterable[Mapping[str, Any]],
    market_runs: list[FeatureRun],
    historical_rankings: Mapping[int, list[dict[str, Any]]],
    history_limit: int,
) -> dict[str, Any]:
    current_rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in serialized_rows:
        group_name = row.get("ibd_industry_group")
        if group_name:
            current_rows_by_group[str(group_name)].append(row)

    history_runs = market_runs[:history_limit]
    ranking_maps = {
        run.id: group_rank_map(historical_rankings.get(run.id, []))
        for run in history_runs
    }
    details: dict[str, Any] = {}
    for ranking in rankings:
        group_name = str(ranking["industry_group"])
        details[group_name] = build_group_detail_payload_from_parts(
            group_name,
            ranking=ranking,
            history=_group_history_points_from_maps(
                group_name,
                market_runs=history_runs,
                ranking_maps=ranking_maps,
            ),
            stocks=constituent_stock_payloads_from_group_rows(
                current_rows_by_group.get(group_name, [])
            ),
        )
    return details
