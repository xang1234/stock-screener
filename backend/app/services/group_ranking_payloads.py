"""Canonical group ranking calculations for serialized scan rows."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any, Iterable, Mapping

import pandas as pd

from app.models.industry import IBDGroupRank


def rank_record_payload(
    ranking: IBDGroupRank,
    *,
    pct_rs_above_80: float | None,
    top_symbol_name: str | None = None,
) -> dict[str, Any]:
    """Serialize one persisted Group row for every live/static reader."""
    return {
        "industry_group": ranking.industry_group,
        "date": ranking.date.isoformat(),
        "rank": ranking.rank,
        "avg_rs_rating": ranking.avg_rs_rating,
        "avg_rs_rating_1m": ranking.avg_rs_rating_1m,
        "avg_rs_rating_3m": ranking.avg_rs_rating_3m,
        "median_rs_rating": ranking.median_rs_rating,
        "weighted_avg_rs_rating": ranking.weighted_avg_rs_rating,
        "rs_std_dev": ranking.rs_std_dev,
        "num_stocks": ranking.num_stocks,
        "num_stocks_rs_above_80": ranking.num_stocks_rs_above_80,
        "pct_rs_above_80": pct_rs_above_80,
        "top_symbol": ranking.top_symbol,
        "top_symbol_name": top_symbol_name,
        "top_rs_rating": ranking.top_rs_rating,
        "rs_formula_version": ranking.rs_formula_version,
        "market_rs_run_id": ranking.market_rs_run_id,
        "rank_change_1w": None,
        "rank_change_1m": None,
        "rank_change_3m": None,
        "rank_change_6m": None,
    }


def compute_group_rankings_from_serialized_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    ranking_date: date,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group_name = row.get("ibd_industry_group")
        rs_rating = row.get("rs_rating")
        if not group_name or rs_rating is None:
            continue
        grouped[str(group_name)].append(row)

    rankings: list[dict[str, Any]] = []
    ranking_date_str = ranking_date.isoformat()
    for group_name, group_rows in grouped.items():
        rs_values = [
            float(row["rs_rating"])
            for row in group_rows
            if row.get("rs_rating") is not None
        ]
        if not rs_values:
            continue

        avg_rs = round(sum(rs_values) / len(rs_values), 2)
        median_rs = round(float(pd.Series(rs_values).median()), 2)
        std_dev = (
            round(float(pd.Series(rs_values).std(ddof=0)), 2)
            if len(rs_values) > 1
            else 0.0
        )
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
            round(
                sum(weight * value for weight, value in weight_pairs if weight > 0)
                / total_weight,
                2,
            )
            if total_weight > 0
            else None
        )
        top_row = max(
            group_rows,
            key=lambda row: (
                row.get("rs_rating")
                if row.get("rs_rating") is not None
                else float("-inf"),
                row.get("composite_score")
                if row.get("composite_score") is not None
                else float("-inf"),
            ),
        )
        above_80 = sum(1 for value in rs_values if value >= 80)
        rankings.append(
            {
                "industry_group": group_name,
                "date": ranking_date_str,
                "rank": 0,
                "avg_rs_rating": avg_rs,
                "median_rs_rating": median_rs,
                "weighted_avg_rs_rating": weighted_avg,
                "rs_std_dev": std_dev,
                "num_stocks": len(rs_values),
                "num_stocks_rs_above_80": above_80,
                "pct_rs_above_80": (
                    round((above_80 / len(rs_values)) * 100, 2)
                    if rs_values
                    else None
                ),
                "top_symbol": top_row.get("symbol"),
                "top_symbol_name": top_row.get("company_name"),
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
