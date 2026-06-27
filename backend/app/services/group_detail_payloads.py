"""Canonical payload shaping for industry-group detail responses."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from app.domain.scanning.models import ScanResultItemDomain
from app.schemas.groups import ConstituentStock


def scan_result_item_to_group_row(item: ScanResultItemDomain) -> dict[str, Any]:
    extended = item.extended_fields or {}
    return {
        "symbol": item.symbol,
        "company_name": extended.get("company_name"),
        "composite_score": item.composite_score,
        "current_price": item.current_price,
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


def constituent_sort_key(row: Mapping[str, Any]) -> tuple[float, float]:
    rs_rating = row.get("rs_rating")
    composite_score = row.get("composite_score")
    return (
        float(rs_rating) if rs_rating is not None else float("-inf"),
        float(composite_score) if composite_score is not None else float("-inf"),
    )


def constituent_stock_from_group_row(row: Mapping[str, Any]) -> ConstituentStock:
    return ConstituentStock(
        symbol=row["symbol"],
        company_name=row.get("company_name"),
        price=row.get("current_price"),
        rs_rating=row.get("rs_rating"),
        rs_rating_1m=row.get("rs_rating_1m"),
        rs_rating_3m=row.get("rs_rating_3m"),
        rs_rating_12m=row.get("rs_rating_12m"),
        eps_growth_qq=row.get("eps_growth_qq"),
        eps_growth_yy=row.get("eps_growth_yy"),
        sales_growth_qq=row.get("sales_growth_qq"),
        sales_growth_yy=row.get("sales_growth_yy"),
        composite_score=row.get("composite_score"),
        stage=row.get("stage"),
        price_sparkline_data=row.get("price_sparkline_data"),
        price_trend=row.get("price_trend"),
        price_change_1d=row.get("price_change_1d"),
        rs_sparkline_data=row.get("rs_sparkline_data"),
        rs_trend=row.get("rs_trend"),
    )


def constituent_stock_payloads_from_group_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=constituent_sort_key, reverse=True)
    return [
        constituent_stock_from_group_row(row).model_dump(mode="json")
        for row in sorted_rows
    ]


def constituent_stock_payloads_from_scan_items(
    items: Iterable[ScanResultItemDomain],
) -> list[dict[str, Any]]:
    return constituent_stock_payloads_from_group_rows(
        scan_result_item_to_group_row(item)
        for item in items
    )
