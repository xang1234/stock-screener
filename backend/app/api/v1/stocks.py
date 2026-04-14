from __future__ import annotations

"""Stock data API endpoints"""
from datetime import UTC, datetime, timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, case, func, or_
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.market_breadth import MarketBreadth
from ...models.stock_universe import StockUniverse
from ...models.theme import ThemeCluster, ThemeConstituent, ThemeMetrics
from ...schemas.scanning import ExplainResponse, ScanResultItem
from ...schemas.stock import (
    StockData,
    StockDecisionDashboardResponse,
    StockInfo,
    StockPriceHistoryPoint,
    StockSearchResult,
    StockTechnicals,
)
from ...services.stock_event_context_service import StockEventContextService
from ...services.strategy_profile_service import DEFAULT_PROFILE, StrategyProfileService
from ...services.symbol_format import require_valid_symbol
from ...schemas.validation import StockValidationResponse
from ...services.validation_service import ValidationService
from ...use_cases.scanning.explain_stock import ExplainStockUseCase
from ...wiring.bootstrap import (
    get_alphavantage_service,
    get_fundamentals_cache,
    get_price_cache,
    get_uow,
    get_yfinance_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_validation_service():
    return ValidationService()


def _get_stock_event_context_service():
    return StockEventContextService()


def _get_strategy_profile_service():
    return StrategyProfileService()


def _get_yfinance_service():
    return get_yfinance_service()


def _build_data_fetcher(db: Session):
    from ...services.data_fetcher import DataFetcher

    return DataFetcher(
        db,
        yfinance_service=get_yfinance_service(),
        alphavantage_service=get_alphavantage_service(),
    )


def _empty_stock_info(symbol: str) -> dict:
    normalized_symbol = symbol.upper()
    return {
        "symbol": normalized_symbol,
        "name": None,
        "sector": None,
        "industry": None,
        "current_price": None,
        "market_cap": None,
    }


def _get_stock_info_payload(symbol: str) -> dict | None:
    info = _get_yfinance_service().get_stock_info(symbol.upper())
    if not info:
        logger.warning(
            "Stock info is unavailable for %s - check yfinance service logs for details",
            symbol,
        )
        return None
    return info


def _get_stock_info_or_404(symbol: str):
    info = _get_stock_info_payload(symbol)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unable to fetch data for {symbol}. This could be due to: invalid symbol, "
                "network issues, or yfinance API problems. Check backend logs for details."
            ),
        )
    return info


def _get_stock_fundamentals_payload(symbol: str, *, force_refresh: bool = False):
    cache = get_fundamentals_cache()
    data = cache.get_fundamentals(symbol.upper(), force_refresh=force_refresh)
    if not data:
        return None
    if "symbol" not in data:
        data["symbol"] = symbol.upper()
    return data


def _get_stock_technicals_payload(
    symbol: str,
    db: Session,
    *,
    force_refresh: bool = False,
):
    fetcher = _build_data_fetcher(db)
    return fetcher.get_stock_technicals(symbol.upper(), force_refresh=force_refresh)


def _load_price_history(symbol: str, period: str = "6mo") -> list[dict]:
    """Get historical price data (OHLCV only) from cache."""

    import pandas as pd

    period_days = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }
    days = period_days.get(period)
    if days is None:
        raise HTTPException(status_code=422, detail=f"Unsupported period: {period}")

    cache_service = get_price_cache()
    cache_period = "5y" if period == "5y" else "2y"
    data = cache_service.get_cached_only(symbol.upper(), period=cache_period)

    if data is None or len(data) == 0:
        logger.warning("No cached data for %s", symbol)
        raise HTTPException(
            status_code=404,
            detail=f"Historical data not available for {symbol}. Run a scan to populate cache.",
        )

    cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=days))
    if data.index.tz is not None:
        cutoff_date = cutoff_date.tz_localize(data.index.tz)

    data = data[data.index >= cutoff_date]
    if len(data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data available for {symbol} in the last {period}",
        )

    df = data.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    result = []
    for _, row in df.iterrows():
        result.append(
            {
                "date": row["Date"],
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
        )

    logger.info("Returning %s price records for %s", len(result), symbol)
    return result


def _build_chart_data_payload(latest_run, item) -> dict:
    ef = item.extended_fields or {}
    return {
        "source": "feature_store",
        "scan_date": latest_run.completed_at.isoformat() if latest_run.completed_at else None,
        "symbol": item.symbol,
        "company_name": ef.get("company_name"),
        "current_price": item.current_price,
        "gics_sector": ef.get("gics_sector"),
        "gics_industry": ef.get("gics_industry"),
        "ibd_industry_group": ef.get("ibd_industry_group"),
        "ibd_group_rank": ef.get("ibd_group_rank"),
        "rs_rating": ef.get("rs_rating"),
        "rs_rating_1m": ef.get("rs_rating_1m"),
        "rs_rating_3m": ef.get("rs_rating_3m"),
        "rs_rating_12m": ef.get("rs_rating_12m"),
        "rs_trend": ef.get("rs_trend"),
        "stage": ef.get("stage"),
        "adr_percent": ef.get("adr_percent"),
        "eps_rating": ef.get("eps_rating"),
        "minervini_score": ef.get("minervini_score"),
        "composite_score": item.composite_score,
        "vcp_detected": ef.get("vcp_detected", False),
        "vcp_score": ef.get("vcp_score"),
        "vcp_pivot": ef.get("vcp_pivot"),
        "vcp_ready_for_breakout": ef.get("vcp_ready_for_breakout", False),
        "ma_alignment": ef.get("ma_alignment"),
        "passes_template": ef.get("passes_template"),
        "eps_growth_qq": ef.get("eps_growth_qq"),
        "sales_growth_qq": ef.get("sales_growth_qq"),
        "eps_growth_yy": ef.get("eps_growth_yy"),
        "sales_growth_yy": ef.get("sales_growth_yy"),
        "canslim_score": ef.get("canslim_score"),
        "ipo_score": ef.get("ipo_score"),
        "custom_score": ef.get("custom_score"),
        "volume_breakthrough_score": ef.get("volume_breakthrough_score"),
        "beta": ef.get("beta"),
        "beta_adj_rs": ef.get("beta_adj_rs"),
        "rating": item.rating,
        "screeners_run": item.screeners_run,
        "se_setup_score": ef.get("se_setup_score"),
        "se_pattern_primary": ef.get("se_pattern_primary"),
        "se_pattern_confidence": ef.get("se_pattern_confidence"),
        "se_quality_score": ef.get("se_quality_score"),
        "se_readiness_score": ef.get("se_readiness_score"),
        "se_setup_ready": ef.get("se_setup_ready"),
    }


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_decision_factor_records(explanation) -> tuple[list[dict], list[dict]]:
    strengths = []
    weaknesses = []

    for screener in explanation.screener_explanations:
        for criterion in screener.criteria:
            max_score = criterion.max_score or 0.0
            ratio = criterion.score / max_score if max_score > 0 else 0.0
            record = {
                "screener_name": screener.screener_name,
                "criterion_name": criterion.name,
                "score": criterion.score,
                "max_score": max_score,
                "passed": criterion.passed,
                "_ratio": ratio,
            }
            if criterion.passed:
                strengths.append(record)
            else:
                weaknesses.append(record)

    strengths.sort(key=lambda row: (-row["_ratio"], -row["score"], row["criterion_name"]))
    weaknesses.sort(key=lambda row: (row["_ratio"], row["score"], row["criterion_name"]))

    def _strip_ratio(rows: list[dict]) -> list[dict]:
        return [{key: value for key, value in row.items() if key != "_ratio"} for row in rows[:3]]

    return _strip_ratio(strengths), _strip_ratio(weaknesses)


def _is_feature_run_stale(latest_run) -> bool:
    if latest_run is None or latest_run.as_of_date is None:
        return True
    age_days = (datetime.now(UTC).date() - latest_run.as_of_date).days
    return age_days > 3


def _build_regime_payload(breadth: MarketBreadth | None, latest_run) -> dict:
    if breadth is None:
        return {
            "label": "unavailable",
            "summary": "Market breadth snapshot is unavailable.",
            "breadth_date": None,
            "up_4pct": None,
            "down_4pct": None,
            "ratio_5day": None,
            "ratio_10day": None,
            "total_stocks_scanned": None,
            "feature_run_stale": _is_feature_run_stale(latest_run),
        }

    score = 0
    if breadth.stocks_up_4pct > breadth.stocks_down_4pct:
        score += 1
    elif breadth.stocks_up_4pct < breadth.stocks_down_4pct:
        score -= 1

    if breadth.ratio_5day is not None:
        score += 1 if breadth.ratio_5day >= 1 else -1
    if breadth.ratio_10day is not None:
        score += 1 if breadth.ratio_10day >= 1 else -1
    if _is_feature_run_stale(latest_run):
        score -= 1

    if score >= 2:
        label = "offense"
    elif score <= -1:
        label = "defense"
    else:
        label = "balanced"

    summary = (
        f"Breadth on {breadth.date.isoformat()} shows {breadth.stocks_up_4pct} stocks up 4%+ "
        f"vs {breadth.stocks_down_4pct} down 4%+, with 5d/10d ratios at "
        f"{breadth.ratio_5day if breadth.ratio_5day is not None else '-'} / "
        f"{breadth.ratio_10day if breadth.ratio_10day is not None else '-'}. "
        f"Current stance: {label}."
    )
    return {
        "label": label,
        "summary": summary,
        "breadth_date": breadth.date.isoformat(),
        "up_4pct": breadth.stocks_up_4pct,
        "down_4pct": breadth.stocks_down_4pct,
        "ratio_5day": breadth.ratio_5day,
        "ratio_10day": breadth.ratio_10day,
        "total_stocks_scanned": breadth.total_stocks_scanned,
        "feature_run_stale": _is_feature_run_stale(latest_run),
    }


def _load_theme_summaries(db: Session, symbol: str) -> list[dict]:
    latest_metrics_subquery = (
        db.query(
            ThemeMetrics.theme_cluster_id.label("theme_cluster_id"),
            func.max(ThemeMetrics.date).label("latest_date"),
        )
        .group_by(ThemeMetrics.theme_cluster_id)
        .subquery()
    )

    rows = (
        db.query(ThemeCluster, ThemeConstituent, ThemeMetrics)
        .join(
            ThemeConstituent,
            ThemeConstituent.theme_cluster_id == ThemeCluster.id,
        )
        .outerjoin(
            latest_metrics_subquery,
            latest_metrics_subquery.c.theme_cluster_id == ThemeCluster.id,
        )
        .outerjoin(
            ThemeMetrics,
            and_(
                ThemeMetrics.theme_cluster_id == ThemeCluster.id,
                ThemeMetrics.date == latest_metrics_subquery.c.latest_date,
            ),
        )
        .filter(
            ThemeConstituent.symbol == symbol,
            ThemeConstituent.is_active.is_(True),
            ThemeCluster.is_active.is_(True),
        )
        .all()
    )

    def _theme_sort_key(row):
        cluster, constituent, metrics = row
        momentum_score = metrics.momentum_score if metrics and metrics.momentum_score is not None else -9999
        confidence = constituent.confidence if constituent.confidence is not None else -9999
        return (
            -momentum_score,
            -confidence,
            cluster.display_name.lower(),
        )

    results = []
    for cluster, constituent, metrics in sorted(rows, key=_theme_sort_key)[:8]:
        results.append(
            {
                "theme_id": cluster.id,
                "display_name": cluster.display_name,
                "pipeline": cluster.pipeline,
                "category": cluster.category,
                "lifecycle_state": cluster.lifecycle_state,
                "is_emerging": bool(cluster.is_emerging),
                "confidence": constituent.confidence,
                "mention_count": constituent.mention_count,
                "correlation_to_theme": constituent.correlation_to_theme,
                "momentum_score": metrics.momentum_score if metrics else None,
                "mention_velocity": metrics.mention_velocity if metrics else None,
                "basket_return_1m": metrics.basket_return_1m if metrics else None,
                "status": metrics.status if metrics else None,
            }
        )
    return results


@router.get("/search", response_model=list[StockSearchResult])
async def search_stocks(
    q: str = Query(..., min_length=1, max_length=50),
    limit: int = Query(8, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Search active universe symbols by symbol or company name."""

    query = q.strip()
    if not query:
        return []

    query_lower = query.lower()
    escaped_query = _escape_like(query_lower)
    symbol_lower = func.lower(func.coalesce(StockUniverse.symbol, ""))
    name_lower = func.lower(func.coalesce(StockUniverse.name, ""))
    rows = (
        db.query(StockUniverse)
        .filter(
            StockUniverse.active_filter(),
            or_(
                symbol_lower.like(f"%{escaped_query}%", escape="\\"),
                name_lower.like(f"%{escaped_query}%", escape="\\"),
            ),
        )
        .order_by(
            case((symbol_lower == query_lower, 0), else_=1),
            case((symbol_lower.like(f"{escaped_query}%", escape="\\"), 0), else_=1),
            case((symbol_lower.like(f"%{escaped_query}%", escape="\\"), 0), else_=1),
            case((name_lower.like(f"{escaped_query}%", escape="\\"), 0), else_=1),
            case((name_lower.like(f"%{escaped_query}%", escape="\\"), 0), else_=1),
            func.length(func.coalesce(StockUniverse.symbol, "")),
            symbol_lower,
        )
        .limit(limit)
        .all()
    )

    return [
        StockSearchResult(
            symbol=row.symbol,
            name=row.name,
            sector=row.sector,
            industry=row.industry,
        )
        for row in rows
    ]


@router.get("/{symbol}/info", response_model=StockInfo)
async def get_stock_info(symbol: str = Depends(require_valid_symbol)):
    """
    Get basic stock information.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)

    Returns:
        Basic stock information
    """
    return _get_stock_info_or_404(symbol)


@router.get("/{symbol}/fundamentals")
async def get_stock_fundamentals(
    symbol: str = Depends(require_valid_symbol),
    force_refresh: bool = False,
    db: Session = Depends(get_db),
):
    """
    Get stock fundamental data.

    Args:
        symbol: Stock ticker symbol
        force_refresh: Force data refresh (ignore cache)

    Returns:
        Fundamental data including earnings, revenue, margins, description
    """
    data = _get_stock_fundamentals_payload(symbol, force_refresh=force_refresh)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Fundamental data not available for {symbol}"
        )
    return data


@router.get("/{symbol}/technicals", response_model=StockTechnicals)
async def get_stock_technicals(
    symbol: str = Depends(require_valid_symbol),
    force_refresh: bool = False,
    db: Session = Depends(get_db),
):
    """
    Get stock technical indicators.

    Args:
        symbol: Stock ticker symbol
        force_refresh: Force data refresh (ignore cache)

    Returns:
        Technical indicators including MAs, RS rating, 52-week range
    """
    data = _get_stock_technicals_payload(symbol, db, force_refresh=force_refresh)

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Technical data not available for {symbol}"
        )

    return data


@router.get("/{symbol}", response_model=StockData)
async def get_stock_data(
    symbol: str = Depends(require_valid_symbol),
    include_fundamentals: bool = True,
    include_technicals: bool = True,
    force_refresh: bool = False,
    db: Session = Depends(get_db),
):
    """
    Get complete stock data (info + fundamentals + technicals).

    Args:
        symbol: Stock ticker symbol
        include_fundamentals: Include fundamental data
        include_technicals: Include technical indicators
        force_refresh: Force data refresh (ignore cache)

    Returns:
        Complete stock data
    """
    # Get basic info
    info = _get_stock_info_or_404(symbol)

    result = {"info": info}

    # Get fundamentals if requested
    if include_fundamentals:
        fundamentals = _get_stock_fundamentals_payload(symbol, force_refresh=force_refresh)
        result["fundamentals"] = fundamentals

    # Get technicals if requested
    if include_technicals:
        technicals = _get_stock_technicals_payload(symbol, db, force_refresh=force_refresh)
        result["technicals"] = technicals

    return result


@router.get("/{symbol}/industry")
async def get_stock_industry(
    symbol: str = Depends(require_valid_symbol),
    db: Session = Depends(get_db),
):
    """
    Get stock industry classification.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Industry classification (sector, industry, ibd_industry_group)
    """
    from ...services.ibd_industry_service import IBDIndustryService

    fetcher = _build_data_fetcher(db)
    classification = fetcher.get_industry_classification(symbol.upper())

    if not classification:
        raise HTTPException(
            status_code=404,
            detail=f"Industry classification not available for {symbol}"
        )

    # Add IBD industry group if available
    try:
        ibd_group = IBDIndustryService.get_industry_group(db, symbol.upper())
        classification['ibd_industry_group'] = ibd_group
    except Exception as e:
        logger.warning(f"Could not fetch IBD industry group for {symbol}: {e}")
        classification['ibd_industry_group'] = None

    return classification


@router.get("/{symbol}/chart-data")
async def get_chart_data(
    symbol: str = Depends(require_valid_symbol),
    uow=Depends(get_uow),
):
    """
    Get all chart modal data in a single call from the feature store.

    Returns all data needed for the chart modal:
    - Basic info (symbol, name, price)
    - Industry classification (GICS sector/industry, IBD group)
    - RS ratings (overall, 1m, 3m, 12m, trend)
    - Technical data (stage, ADR, EPS rating)
    - Minervini/VCP data
    - Growth metrics
    """
    with uow:
        latest_run = uow.feature_runs.get_latest_published()
        if latest_run is None:
            raise HTTPException(
                status_code=404,
                detail=f"No published scan data available for {symbol}",
            )

        item = uow.feature_store.get_by_symbol_for_run(latest_run.id, symbol)
        if item is None:
            raise HTTPException(
                status_code=404,
                detail=f"No scan data found for {symbol}",
            )

    return _build_chart_data_payload(latest_run, item)


@router.get("/{symbol}/decision-dashboard", response_model=StockDecisionDashboardResponse)
async def get_stock_decision_dashboard(
    symbol: str = Depends(require_valid_symbol),
    profile: str | None = Query(None),
    db: Session = Depends(get_db),
    uow=Depends(get_uow),
    event_context_service: StockEventContextService = Depends(_get_stock_event_context_service),
    profile_service: StrategyProfileService = Depends(_get_strategy_profile_service),
):
    """Get a normalized stock decision workspace payload."""

    resolved_profile = profile_service.get_profile(profile or DEFAULT_PROFILE)
    degraded_reasons: list[str] = []

    info = _get_stock_info_payload(symbol)
    if info is None:
        info = _empty_stock_info(symbol)
        degraded_reasons.append("missing_stock_info")
    fundamentals = _get_stock_fundamentals_payload(symbol)
    technicals = _get_stock_technicals_payload(symbol, db)
    if fundamentals is None:
        degraded_reasons.append("missing_fundamentals")
    if technicals is None:
        degraded_reasons.append("missing_technicals")

    try:
        price_history = _load_price_history(symbol, period="6mo")
    except HTTPException:
        price_history = []
        degraded_reasons.append("missing_price_history")

    with uow:
        latest_run = uow.feature_runs.get_latest_published()
        feature_item = None
        feature_row = None
        if latest_run is None:
            degraded_reasons.append("missing_feature_run")
        else:
            feature_item = uow.feature_store.get_by_symbol_for_run(
                latest_run.id,
                symbol,
                include_sparklines=False,
                include_setup_payload=False,
            )
            feature_row = uow.feature_store.get_row_by_symbol(latest_run.id, symbol)
            if feature_item is None or feature_row is None:
                degraded_reasons.append("symbol_missing_from_feature_run")

        breadth = (
            db.query(MarketBreadth)
            .order_by(MarketBreadth.date.desc())
            .first()
        )
        if breadth is None:
            degraded_reasons.append("missing_breadth")

        if feature_item is not None:
            extended_fields = feature_item.extended_fields or {}
            industry_group = extended_fields.get("ibd_industry_group")
            if industry_group:
                peer_items = [
                    peer
                    for peer in uow.feature_store.get_peers_by_industry_for_run(latest_run.id, industry_group)
                    if peer.symbol != symbol
                ][:15]
            else:
                peer_items = []
                degraded_reasons.append("missing_industry_group")
        else:
            peer_items = []

    screener_explanations = []
    decision_summary = {
        "composite_score": None,
        "rating": None,
        "screeners_passed": 0,
        "screeners_total": 0,
        "composite_method": None,
        "top_strengths": [],
        "top_weaknesses": [],
        "freshness": {
            "feature_run_id": latest_run.id if latest_run else None,
            "feature_as_of_date": latest_run.as_of_date.isoformat() if latest_run and latest_run.as_of_date else None,
            "feature_completed_at": latest_run.completed_at.isoformat() if latest_run and latest_run.completed_at else None,
            "breadth_date": breadth.date.isoformat() if breadth else None,
            "has_price_history": bool(price_history),
        },
    }

    if feature_row is not None:
        explanation_item = ExplainStockUseCase._build_item_from_feature_row(feature_row)
        explanation = ExplainStockUseCase.build_explanation_from_item(explanation_item)
        explanation_response = ExplainResponse.from_domain(explanation)
        screener_explanations = explanation_response.screener_explanations
        top_strengths, top_weaknesses = _build_decision_factor_records(explanation)
        decision_summary = {
            "composite_score": explanation.composite_score,
            "rating": explanation.rating,
            "screeners_passed": explanation.screeners_passed,
            "screeners_total": explanation.screeners_total,
            "composite_method": explanation.composite_method,
            "top_strengths": top_strengths,
            "top_weaknesses": top_weaknesses,
            "freshness": decision_summary["freshness"],
        }
    else:
        degraded_reasons.append("missing_explanation")

    themes = _load_theme_summaries(db, symbol)
    if not themes:
        degraded_reasons.append("missing_theme_links")

    chart_data = (
        _build_chart_data_payload(latest_run, feature_item)
        if latest_run is not None and feature_item is not None
        else {
            "source": "unavailable",
            "scan_date": None,
            "symbol": symbol,
            "company_name": info.get("name"),
            "current_price": info.get("current_price"),
        }
    )

    regime = _build_regime_payload(breadth, latest_run)
    event_risk, regime_actions = event_context_service.build(
        db,
        symbol=symbol,
        as_of_date=latest_run.as_of_date if latest_run and latest_run.as_of_date else None,
        regime_label=regime.get("label"),
        profile=resolved_profile.profile,
        fundamentals=fundamentals,
    )

    return {
        "symbol": symbol,
        "as_of_date": latest_run.as_of_date.isoformat() if latest_run and latest_run.as_of_date else None,
        "freshness": decision_summary["freshness"],
        "info": info,
        "fundamentals": fundamentals,
        "technicals": technicals,
        "chart": {
            "price_history": price_history,
            "chart_data": chart_data,
        },
        "decision_summary": decision_summary,
        "screener_explanations": screener_explanations,
        "peers": [
            ScanResultItem.from_domain(peer, include_setup_payload=False)
            for peer in peer_items
        ],
        "themes": themes,
        "regime": regime,
        "event_risk": event_risk,
        "regime_actions": regime_actions,
        "degraded_reasons": sorted(set(degraded_reasons)),
    }


@router.get("/{symbol}/peers", response_model=list[ScanResultItem])
async def get_stock_peers(
    symbol: str = Depends(require_valid_symbol),
    peer_type: str = Query("industry", pattern="^(industry|sector)$"),
    uow=Depends(get_uow),
):
    """Get industry/sector peers from the latest published feature run."""
    from ...domain.scanning.models import PeerType

    pt = PeerType(peer_type)

    with uow:
        latest_run = uow.feature_runs.get_latest_published()
        if latest_run is None:
            raise HTTPException(status_code=404, detail="No published feature run available")

        item = uow.feature_store.get_by_symbol_for_run(
            latest_run.id, symbol, include_sparklines=False, include_setup_payload=False,
        )
        if item is None:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in latest feature run")

        field_key = "ibd_industry_group" if pt == PeerType.INDUSTRY else "gics_sector"
        group_value = (item.extended_fields or {}).get(field_key)
        if not group_value or not str(group_value).strip():
            return []

        if pt == PeerType.INDUSTRY:
            peers = uow.feature_store.get_peers_by_industry_for_run(latest_run.id, group_value)
        else:
            peers = uow.feature_store.get_peers_by_sector_for_run(latest_run.id, group_value)

    return [ScanResultItem.from_domain(p) for p in peers]


@router.get("/{symbol}/history", response_model=list[StockPriceHistoryPoint])
async def get_price_history(
    symbol: str = Depends(require_valid_symbol),
    period: str = "6mo",
):
    return _load_price_history(symbol, period)


@router.get("/{symbol}/validation", response_model=StockValidationResponse)
async def get_stock_validation(
    symbol: str = Depends(require_valid_symbol),
    lookback_days: int = Query(365, ge=30, le=365),
    db: Session = Depends(get_db),
    service: ValidationService = Depends(_get_validation_service),
):
    """Return deterministic historical validation metrics for one symbol."""

    return service.get_stock_validation(
        db,
        symbol=symbol,
        lookback_days=lookback_days,
    )
