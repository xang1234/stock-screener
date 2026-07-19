"""Canonical group ranking calculations for serialized scan rows."""

from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.infra.db.models.relative_strength import MarketRsRun
from app.models.industry import IBDGroupRank
from app.models.stock_universe import StockUniverse


def annotate_top_symbol_names(
    db: Session,
    rows: list[dict[str, Any]],
) -> None:
    """Resolve all top-symbol company names with one universe query."""
    symbols = {
        str(row.get("top_symbol")).strip()
        for row in rows
        if str(row.get("top_symbol") or "").strip()
    }
    name_map = (
        dict(
            db.query(StockUniverse.symbol, StockUniverse.name)
            .filter(StockUniverse.symbol.in_(symbols))
            .all()
        )
        if symbols
        else {}
    )
    for row in rows:
        row["top_symbol_name"] = name_map.get(row.get("top_symbol"))


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

def group_snapshot_metadata(
    db: Session,
    *,
    market: str,
    rankings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and describe the single RS source behind a Group snapshot."""
    if not rankings:
        raise RuntimeError("no Group rankings are available")
    formula_versions = {row.get("rs_formula_version") for row in rankings}
    run_ids = {row.get("market_rs_run_id") for row in rankings}
    dates = {row.get("date") for row in rankings}
    if (
        None in formula_versions
        or None in dates
        or len(formula_versions) != 1
        or len(run_ids) != 1
        or len(dates) != 1
    ):
        raise RuntimeError("group snapshot mixes canonical RS sources")
    formula_version = str(next(iter(formula_versions)))
    run_id = next(iter(run_ids))
    if formula_version == BALANCED_RS_FORMULA_VERSION and run_id is None:
        raise RuntimeError("balanced Group snapshot has no single Market RS run")

    run = db.get(MarketRsRun, int(run_id)) if run_id is not None else None
    normalized_market = market.strip().upper()
    snapshot_date = str(next(iter(dates)))
    if run_id is not None and (
        run is None
        or run.market != normalized_market
        or run.formula_version != formula_version
        or run.as_of_date.isoformat() != snapshot_date
        or run.status != "completed"
    ):
        raise RuntimeError("Group snapshot metadata does not match its Market RS run")
    return {
        "rs_formula_version": formula_version,
        "rs_as_of_date": snapshot_date,
        "rs_universe_size": run.eligible_symbol_count if run is not None else None,
    }
