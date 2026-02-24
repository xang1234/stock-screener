"""
Database Tools for the chatbot.
Provides read-only access to the application database for financial data queries.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from ....models.scan_result import Scan, ScanResult
from ....models.theme import ThemeCluster, ThemeConstituent, ThemeMetrics, ThemeMention, ContentItem
from ....models.market_breadth import MarketBreadth
from ....models.stock_universe import StockUniverse
from ....models.stock import StockPrice, StockFundamental

logger = logging.getLogger(__name__)


class DatabaseTools:
    """Read-only database access for chatbot queries."""

    def __init__(self, db: Session):
        self.db = db

    def get_stock_scan_results(
        self,
        symbol: str,
        limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Get most recent scan results for a symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Number of results to return

        Returns:
            Dict with scan results or None if not found
        """
        try:
            results = (
                self.db.query(ScanResult)
                .filter(ScanResult.symbol == symbol.upper())
                .order_by(desc(ScanResult.created_at))
                .limit(limit)
                .all()
            )

            if not results:
                return None

            result = results[0]
            return {
                "symbol": result.symbol,
                "composite_score": result.composite_score,
                "minervini_score": result.minervini_score,
                "canslim_score": result.canslim_score,
                "rating": result.rating,
                "price": result.price,
                "stage": result.stage,
                "rs_rating": result.rs_rating,
                "rs_rating_1m": result.rs_rating_1m,
                "rs_rating_3m": result.rs_rating_3m,
                "rs_rating_12m": result.rs_rating_12m,
                "eps_growth_qq": result.eps_growth_qq,
                "sales_growth_qq": result.sales_growth_qq,
                "peg_ratio": result.peg_ratio,
                "ibd_industry_group": result.ibd_industry_group,
                "gics_sector": result.gics_sector,
                "details": result.details,
                "scanned_at": result.created_at.isoformat() if result.created_at else None,
            }
        except Exception as e:
            logger.error(f"Error fetching scan results for {symbol}: {e}")
            return None

    def search_stocks_by_criteria(
        self,
        min_score: Optional[float] = None,
        min_rs_rating: Optional[float] = None,
        stage: Optional[int] = None,
        sector: Optional[str] = None,
        industry_group: Optional[str] = None,
        rating: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search stocks by various criteria.

        Args:
            min_score: Minimum composite score
            min_rs_rating: Minimum RS rating
            stage: Weinstein stage (1-4)
            sector: GICS sector
            industry_group: IBD industry group
            rating: Rating (Strong Buy, Buy, Watch, Pass)
            limit: Maximum results

        Returns:
            List of matching stocks
        """
        try:
            # Get the latest scan
            latest_scan = (
                self.db.query(Scan)
                .filter(Scan.status == "completed")
                .order_by(desc(Scan.completed_at))
                .first()
            )

            if not latest_scan:
                return []

            query = self.db.query(ScanResult).filter(
                ScanResult.scan_id == latest_scan.scan_id
            )

            if min_score is not None:
                query = query.filter(ScanResult.composite_score >= min_score)
            if min_rs_rating is not None:
                query = query.filter(ScanResult.rs_rating >= min_rs_rating)
            if stage is not None:
                query = query.filter(ScanResult.stage == stage)
            if sector is not None:
                query = query.filter(ScanResult.gics_sector.ilike(f"%{sector}%"))
            if industry_group is not None:
                query = query.filter(ScanResult.ibd_industry_group.ilike(f"%{industry_group}%"))
            if rating is not None:
                query = query.filter(ScanResult.rating == rating)

            results = (
                query
                .order_by(desc(ScanResult.composite_score))
                .limit(limit)
                .all()
            )

            return [
                {
                    "symbol": r.symbol,
                    "composite_score": r.composite_score,
                    "rating": r.rating,
                    "price": r.price,
                    "stage": r.stage,
                    "rs_rating": r.rs_rating,
                    "gics_sector": r.gics_sector,
                    "ibd_industry_group": r.ibd_industry_group,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return []

    def get_theme_data(
        self,
        theme_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get theme data with constituents.

        Args:
            theme_name: Specific theme name (optional, returns all if None)
            limit: Maximum themes to return

        Returns:
            List of themes with their data
        """
        try:
            query = self.db.query(ThemeCluster)

            if theme_name:
                query = query.filter(ThemeCluster.display_name.ilike(f"%{theme_name}%"))

            themes = (
                query
                .order_by(desc(ThemeCluster.last_seen_at))
                .limit(limit)
                .all()
            )

            result = []
            for theme in themes:
                # Get constituents
                constituents = (
                    self.db.query(ThemeConstituent)
                    .filter(ThemeConstituent.theme_cluster_id == theme.id)
                    .order_by(desc(ThemeConstituent.confidence))
                    .limit(10)
                    .all()
                )

                # Get latest metrics
                metrics = (
                    self.db.query(ThemeMetrics)
                    .filter(ThemeMetrics.theme_cluster_id == theme.id)
                    .order_by(desc(ThemeMetrics.date))
                    .first()
                )

                result.append({
                    "name": theme.display_name or theme.name,
                    "description": theme.description,
                    "category": theme.category,
                    "is_emerging": theme.is_emerging,
                    "is_validated": theme.is_validated,
                    "first_seen_at": theme.first_seen_at.isoformat() if theme.first_seen_at else None,
                    "last_seen_at": theme.last_seen_at.isoformat() if theme.last_seen_at else None,
                    "constituents": [
                        {
                            "symbol": c.symbol,
                            "confidence": c.confidence,
                            "mention_count": c.mention_count,
                        }
                        for c in constituents
                    ],
                    "metrics": {
                        "mention_count": metrics.mention_count if metrics else 0,
                        "unique_sources": metrics.unique_sources if metrics else 0,
                        "avg_sentiment": metrics.avg_sentiment if metrics else 0,
                        "velocity": metrics.velocity if metrics else 0,
                        "status": metrics.status if metrics else "unknown",
                    } if metrics else None,
                })

            return result
        except Exception as e:
            logger.error(f"Error fetching theme data: {e}")
            return []

    def get_trending_themes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get currently trending themes."""
        try:
            # Get themes with recent metrics showing high velocity
            recent_metrics = (
                self.db.query(ThemeMetrics)
                .filter(ThemeMetrics.status == "trending")
                .order_by(desc(ThemeMetrics.mention_velocity))
                .limit(limit)
                .all()
            )

            result = []
            for metric in recent_metrics:
                theme = self.db.query(ThemeCluster).filter(ThemeCluster.id == metric.theme_cluster_id).first()
                if theme:
                    result.append({
                        "name": theme.display_name or theme.name,
                        "description": theme.description,
                        "velocity": metric.mention_velocity,
                        "mention_count": metric.mentions_7d,
                        "avg_sentiment": metric.sentiment_score,
                        "status": metric.status,
                    })

            return result
        except Exception as e:
            logger.error(f"Error fetching trending themes: {e}")
            return []

    def get_breadth_data(
        self,
        period: str = "1m",
        market: str = "NYSE"
    ) -> Optional[Dict[str, Any]]:
        """
        Get market breadth data.

        Args:
            period: Time period (1d, 1w, 1m, 3m)
            market: Market (NYSE, NASDAQ, SP500)

        Returns:
            Dict with breadth data
        """
        try:
            # Determine date range
            days = {"1d": 1, "1w": 7, "1m": 30, "3m": 90}.get(period, 30)
            start_date = datetime.utcnow() - timedelta(days=days)

            breadth_data = (
                self.db.query(MarketBreadth)
                .filter(MarketBreadth.market == market)
                .filter(MarketBreadth.date >= start_date)
                .order_by(desc(MarketBreadth.date))
                .all()
            )

            if not breadth_data:
                return None

            latest = breadth_data[0]
            return {
                "market": market,
                "date": latest.date.isoformat() if latest.date else None,
                "advance_decline_ratio": latest.advance_decline_ratio,
                "new_highs": latest.new_highs,
                "new_lows": latest.new_lows,
                "percent_above_50ma": latest.percent_above_50ma,
                "percent_above_200ma": latest.percent_above_200ma,
                "mcclellan_oscillator": latest.mcclellan_oscillator,
                "mcclellan_summation": latest.mcclellan_summation,
                "history": [
                    {
                        "date": b.date.isoformat() if b.date else None,
                        "advance_decline_ratio": b.advance_decline_ratio,
                        "percent_above_50ma": b.percent_above_50ma,
                    }
                    for b in breadth_data[:10]  # Last 10 data points
                ],
            }
        except Exception as e:
            logger.error(f"Error fetching breadth data: {e}")
            return None

    def get_stock_universe_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock universe information for a symbol."""
        try:
            stock = (
                self.db.query(StockUniverse)
                .filter(StockUniverse.symbol == symbol.upper())
                .first()
            )

            if not stock:
                return None

            return {
                "symbol": stock.symbol,
                "name": stock.name,
                "exchange": stock.exchange,
                "sector": stock.sector,
                "industry": stock.industry,
                "market_cap": stock.market_cap,
                "is_active": stock.is_active,
            }
        except Exception as e:
            logger.error(f"Error fetching stock universe info for {symbol}: {e}")
            return None

    def get_top_rated_stocks(self, limit: int = 10, rating: str = "Strong Buy") -> List[Dict[str, Any]]:
        """Get top rated stocks from the latest scan."""
        return self.search_stocks_by_criteria(rating=rating, limit=limit)

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Return tool descriptions for the action agent."""
        return [
            {
                "name": "get_scan_results",
                "description": "Get the most recent scan results for a stock symbol including scores, ratings, and technical metrics.",
                "parameters": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, NVDA)"},
                },
                "required": ["symbol"],
            },
            {
                "name": "search_stocks",
                "description": "Search for stocks matching specific criteria like score, RS rating, stage, sector, or rating.",
                "parameters": {
                    "min_score": {"type": "number", "description": "Minimum composite score (0-100)"},
                    "min_rs_rating": {"type": "number", "description": "Minimum RS rating (0-100)"},
                    "stage": {"type": "integer", "description": "Weinstein stage (1-4)"},
                    "sector": {"type": "string", "description": "GICS sector name"},
                    "industry_group": {"type": "string", "description": "IBD industry group"},
                    "rating": {"type": "string", "description": "Rating: Strong Buy, Buy, Watch, Pass"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": [],
            },
            {
                "name": "get_theme_data",
                "description": "Get information about market themes including constituent stocks and metrics.",
                "parameters": {
                    "theme_name": {"type": "string", "description": "Theme name to search for (optional)"},
                    "limit": {"type": "integer", "description": "Max themes to return (default 10)"},
                },
                "required": [],
            },
            {
                "name": "get_trending_themes",
                "description": "Get currently trending market themes sorted by velocity.",
                "parameters": {
                    "limit": {"type": "integer", "description": "Max themes to return (default 10)"},
                },
                "required": [],
            },
            {
                "name": "get_breadth_data",
                "description": "Get market breadth indicators like advance/decline ratio, new highs/lows, and McClellan oscillator.",
                "parameters": {
                    "period": {"type": "string", "description": "Time period: 1d, 1w, 1m, 3m (default 1m)"},
                    "market": {"type": "string", "description": "Market: NYSE, NASDAQ, SP500 (default NYSE)"},
                },
                "required": [],
            },
            {
                "name": "get_top_rated_stocks",
                "description": "Get top rated stocks from the latest scan.",
                "parameters": {
                    "limit": {"type": "integer", "description": "Max stocks to return (default 10)"},
                    "rating": {"type": "string", "description": "Rating filter: Strong Buy, Buy, Watch (default Strong Buy)"},
                },
                "required": [],
            },
        ]

    def get_db_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached fundamentals from stock_fundamentals table.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with comprehensive fundamental data or None if not found
        """
        try:
            fundamental = (
                self.db.query(StockFundamental)
                .filter(StockFundamental.symbol == symbol.upper())
                .first()
            )

            if not fundamental:
                return None

            return {
                "symbol": fundamental.symbol,
                # Market data
                "market_cap": fundamental.market_cap,
                "shares_outstanding": fundamental.shares_outstanding,
                "avg_volume": fundamental.avg_volume,
                "relative_volume": fundamental.relative_volume,
                # Valuation
                "pe_ratio": fundamental.pe_ratio,
                "forward_pe": fundamental.forward_pe,
                "peg_ratio": fundamental.peg_ratio,
                "price_to_book": fundamental.price_to_book,
                "price_to_sales": fundamental.price_to_sales,
                "ev_ebitda": fundamental.ev_ebitda,
                "target_price": fundamental.target_price,
                # EPS Rating (IBD-style)
                "eps_rating": fundamental.eps_rating,
                "eps_current": fundamental.eps_current,
                "eps_growth_quarterly": fundamental.eps_growth_quarterly,
                "eps_growth_annual": fundamental.eps_growth_annual,
                "eps_5yr_cagr": fundamental.eps_5yr_cagr,
                # Revenue
                "revenue_growth": fundamental.revenue_growth,
                "sales_growth_yy": fundamental.sales_growth_yy,
                "sales_growth_qq": fundamental.sales_growth_qq,
                # Profitability
                "profit_margin": fundamental.profit_margin,
                "operating_margin": fundamental.operating_margin,
                "gross_margin": fundamental.gross_margin,
                "roe": fundamental.roe,
                "roa": fundamental.roa,
                "roic": fundamental.roic,
                # Financial health
                "current_ratio": fundamental.current_ratio,
                "debt_to_equity": fundamental.debt_to_equity,
                # Ownership & sentiment
                "insider_ownership": fundamental.insider_ownership,
                "insider_transactions": fundamental.insider_transactions,
                "institutional_ownership": fundamental.institutional_ownership,
                "institutional_transactions": fundamental.institutional_transactions,
                "short_float": fundamental.short_float,
                "short_ratio": fundamental.short_ratio,
                "short_interest": fundamental.short_interest,
                # Performance
                "perf_week": fundamental.perf_week,
                "perf_month": fundamental.perf_month,
                "perf_quarter": fundamental.perf_quarter,
                "perf_half_year": fundamental.perf_half_year,
                "perf_year": fundamental.perf_year,
                "perf_ytd": fundamental.perf_ytd,
                # 52-week range
                "week_52_high": fundamental.week_52_high,
                "week_52_high_distance": fundamental.week_52_high_distance,
                "week_52_low": fundamental.week_52_low,
                # Company info
                "sector": fundamental.sector,
                "industry": fundamental.industry,
                "country": fundamental.country,
                "employees": fundamental.employees,
                "description": fundamental.description_yfinance or fundamental.description_finviz,
                # Analyst
                "recommendation": fundamental.recommendation,
                # Dividend
                "dividend_yield": fundamental.dividend_yield,
                # Technical
                "beta": fundamental.beta,
                "rsi_14": fundamental.rsi_14,
                "sma_50": fundamental.sma_50,
                "sma_200": fundamental.sma_200,
                # Metadata
                "updated_at": fundamental.updated_at.isoformat() if fundamental.updated_at else None,
            }
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    def get_db_price_history(
        self,
        symbol: str,
        days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached price history from stock_prices table.

        Args:
            symbol: Stock ticker symbol
            days: Number of days of history (default 30, max 365)

        Returns:
            Dict with OHLCV data array or None if not found
        """
        try:
            days = min(days, 365)  # Cap at 365 days
            start_date = datetime.utcnow() - timedelta(days=days)

            prices = (
                self.db.query(StockPrice)
                .filter(StockPrice.symbol == symbol.upper())
                .filter(StockPrice.date >= start_date.date())
                .order_by(StockPrice.date)
                .all()
            )

            if not prices:
                return None

            price_data = [
                {
                    "date": p.date.isoformat() if p.date else None,
                    "open": p.open,
                    "high": p.high,
                    "low": p.low,
                    "close": p.close,
                    "volume": p.volume,
                    "adj_close": p.adj_close,
                }
                for p in prices
            ]

            # Calculate summary stats
            closes = [p.close for p in prices if p.close]
            if closes:
                current_price = closes[-1]
                high_price = max(closes)
                low_price = min(closes)
                price_change = ((closes[-1] / closes[0]) - 1) * 100 if closes[0] else 0
            else:
                current_price = high_price = low_price = price_change = None

            return {
                "symbol": symbol.upper(),
                "days_requested": days,
                "data_points": len(price_data),
                "current_price": current_price,
                "period_high": high_price,
                "period_low": low_price,
                "price_change_pct": round(price_change, 2) if price_change else None,
                "prices": price_data,
            }
        except Exception as e:
            logger.error(f"Error fetching price history for {symbol}: {e}")
            return None

    def advanced_stock_search(
        self,
        min_eps_rating: Optional[int] = None,
        max_pe: Optional[float] = None,
        min_profit_margin: Optional[float] = None,
        min_revenue_growth: Optional[float] = None,
        min_roe: Optional[float] = None,
        sector: Optional[str] = None,
        has_description: Optional[bool] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search stocks with advanced fundamental criteria.

        Args:
            min_eps_rating: Minimum EPS rating (0-99)
            max_pe: Maximum P/E ratio
            min_profit_margin: Minimum profit margin (decimal)
            min_revenue_growth: Minimum revenue growth (decimal)
            min_roe: Minimum return on equity (decimal)
            sector: GICS sector filter
            has_description: Only include stocks with descriptions
            limit: Maximum results (default 20)

        Returns:
            List of stocks matching criteria with fundamental data
        """
        try:
            # Get the latest scan to join with
            latest_scan = (
                self.db.query(Scan)
                .filter(Scan.status == "completed")
                .order_by(desc(Scan.completed_at))
                .first()
            )

            if not latest_scan:
                return []

            # Start with scan results and join fundamentals
            query = (
                self.db.query(ScanResult, StockFundamental)
                .join(
                    StockFundamental,
                    ScanResult.symbol == StockFundamental.symbol
                )
                .filter(ScanResult.scan_id == latest_scan.scan_id)
            )

            # Apply fundamental filters
            if min_eps_rating is not None:
                query = query.filter(StockFundamental.eps_rating >= min_eps_rating)
            if max_pe is not None:
                query = query.filter(StockFundamental.pe_ratio <= max_pe)
                query = query.filter(StockFundamental.pe_ratio > 0)  # Exclude negative PE
            if min_profit_margin is not None:
                query = query.filter(StockFundamental.profit_margin >= min_profit_margin)
            if min_revenue_growth is not None:
                query = query.filter(StockFundamental.revenue_growth >= min_revenue_growth)
            if min_roe is not None:
                query = query.filter(StockFundamental.roe >= min_roe)
            if sector is not None:
                query = query.filter(StockFundamental.sector.ilike(f"%{sector}%"))
            if has_description:
                query = query.filter(
                    (StockFundamental.description_yfinance.isnot(None)) |
                    (StockFundamental.description_finviz.isnot(None))
                )

            # Order by composite score and limit
            results = (
                query
                .order_by(desc(ScanResult.composite_score))
                .limit(limit)
                .all()
            )

            return [
                {
                    "symbol": scan_result.symbol,
                    "composite_score": scan_result.composite_score,
                    "rating": scan_result.rating,
                    "price": scan_result.price,
                    "stage": scan_result.stage,
                    "rs_rating": scan_result.rs_rating,
                    # Fundamental data
                    "eps_rating": fundamental.eps_rating,
                    "pe_ratio": fundamental.pe_ratio,
                    "profit_margin": fundamental.profit_margin,
                    "revenue_growth": fundamental.revenue_growth,
                    "roe": fundamental.roe,
                    "sector": fundamental.sector,
                    "industry": fundamental.industry,
                    "short_float": fundamental.short_float,
                    "insider_ownership": fundamental.insider_ownership,
                }
                for scan_result, fundamental in results
            ]
        except Exception as e:
            logger.error(f"Error in advanced stock search: {e}")
            return []

    def research_theme(
        self,
        theme_name: str,
        include_sources: bool = True,
        include_history: bool = False,
        max_sources: int = 10,
        max_constituents: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        All-in-one deep research on a specific theme.
        Returns comprehensive theme data including metrics, constituents, and sources.

        Args:
            theme_name: Theme name to research (e.g., "AI", "Nuclear", "GLP-1")
            include_sources: Include source articles that led to theme discovery
            include_history: Include 30-day historical metrics
            max_sources: Max source articles to return
            max_constituents: Max tickers to return

        Returns:
            Comprehensive theme research object or None if not found
        """
        try:
            # Find theme by name (case-insensitive, check aliases)
            theme = (
                self.db.query(ThemeCluster)
                .filter(
                    (func.lower(ThemeCluster.display_name) == theme_name.lower()) |
                    (ThemeCluster.aliases.contains([theme_name]))
                )
                .first()
            )

            # Fallback to ILIKE search if exact match fails
            if not theme:
                theme = (
                    self.db.query(ThemeCluster)
                    .filter(ThemeCluster.display_name.ilike(f"%{theme_name}%"))
                    .order_by(desc(ThemeCluster.last_seen_at))
                    .first()
                )

            if not theme:
                return None

            # Get latest metrics
            latest_metrics = (
                self.db.query(ThemeMetrics)
                .filter(ThemeMetrics.theme_cluster_id == theme.id)
                .order_by(desc(ThemeMetrics.date))
                .first()
            )

            # Get constituents with details
            constituents = (
                self.db.query(ThemeConstituent)
                .filter(ThemeConstituent.theme_cluster_id == theme.id)
                .filter(ThemeConstituent.is_active == True)
                .order_by(desc(ThemeConstituent.confidence))
                .limit(max_constituents)
                .all()
            )

            result = {
                "theme": {
                    "name": theme.display_name or theme.name,
                    "aliases": theme.aliases or [],
                    "description": theme.description,
                    "category": theme.category,
                    "is_emerging": theme.is_emerging,
                    "first_seen_at": theme.first_seen_at.isoformat() if theme.first_seen_at else None,
                },
                "metrics": None,
                "constituents": [
                    {
                        "symbol": c.symbol,
                        "confidence": c.confidence,
                        "correlation_to_theme": c.correlation_to_theme,
                        "mention_count": c.mention_count,
                        "source": c.source,
                    }
                    for c in constituents
                ],
            }

            # Add metrics if available
            if latest_metrics:
                result["metrics"] = {
                    "momentum_score": latest_metrics.momentum_score,
                    "rank": latest_metrics.rank,
                    "velocity": latest_metrics.mention_velocity,
                    "sentiment_score": latest_metrics.sentiment_score,
                    "basket_rs_vs_spy": latest_metrics.basket_rs_vs_spy,
                    "basket_return_1d": latest_metrics.basket_return_1d,
                    "basket_return_1w": latest_metrics.basket_return_1w,
                    "basket_return_1m": latest_metrics.basket_return_1m,
                    "mentions_1d": latest_metrics.mentions_1d,
                    "mentions_7d": latest_metrics.mentions_7d,
                    "mentions_30d": latest_metrics.mentions_30d,
                    "num_constituents": latest_metrics.num_constituents,
                    "pct_above_50ma": latest_metrics.pct_above_50ma,
                    "pct_above_200ma": latest_metrics.pct_above_200ma,
                    "avg_rs_rating": latest_metrics.avg_rs_rating,
                    "status": latest_metrics.status,
                }

            # Add sources if requested
            if include_sources:
                # Get theme mentions with their content items
                mentions = (
                    self.db.query(ThemeMention, ContentItem)
                    .join(ContentItem, ThemeMention.content_item_id == ContentItem.id)
                    .filter(ThemeMention.theme_cluster_id == theme.id)
                    .order_by(desc(ThemeMention.mentioned_at))
                    .limit(max_sources)
                    .all()
                )

                result["sources"] = [
                    {
                        "source_type": mention.source_type,
                        "source_name": mention.source_name,
                        "title": content_item.title,
                        "url": content_item.url,
                        "author": content_item.author,
                        "published_at": content_item.published_at.isoformat() if content_item.published_at else None,
                        "excerpt": mention.excerpt,
                        "sentiment": mention.sentiment,
                        "confidence": mention.confidence,
                        "tickers": mention.tickers or [],
                        "content_preview": content_item.content[:500] if content_item.content else None,
                    }
                    for mention, content_item in mentions
                ]

            # Add history if requested
            if include_history:
                from datetime import timedelta
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                historical_metrics = (
                    self.db.query(ThemeMetrics)
                    .filter(ThemeMetrics.theme_cluster_id == theme.id)
                    .filter(ThemeMetrics.date >= thirty_days_ago.date())
                    .order_by(ThemeMetrics.date)
                    .all()
                )

                result["history"] = [
                    {
                        "date": m.date.isoformat() if m.date else None,
                        "momentum_score": m.momentum_score,
                        "velocity": m.mention_velocity,
                        "sentiment": m.sentiment_score,
                    }
                    for m in historical_metrics
                ]

            return result

        except Exception as e:
            logger.error(f"Error researching theme '{theme_name}': {e}")
            return None

    def discover_themes(
        self,
        mode: str = "trending",
        theme_names: Optional[List[str]] = None,
        min_velocity: float = 1.0,
        category: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Find and compare themes for discovery and comparison use cases.

        Args:
            mode: One of "emerging", "trending", "compare"
                - emerging: Find newly discovered high-velocity themes
                - trending: Get current trending themes (high momentum)
                - compare: Compare specific themes side by side
            theme_names: Array of theme names (required for "compare" mode)
            min_velocity: For emerging mode - minimum 7d/30d ratio
            category: Filter by technology/healthcare/macro/sector/commodity
            limit: Max themes to return

        Returns:
            Dict with mode, themes list, and as_of_date
        """
        try:
            themes_data = []

            if mode == "compare" and theme_names:
                # Compare specific themes
                for name in theme_names:
                    theme_data = self._get_theme_summary(name)
                    if theme_data:
                        themes_data.append(theme_data)

            elif mode == "emerging":
                # Find emerging/high-velocity themes
                # Get themes where is_emerging=True OR velocity >= threshold
                subquery = (
                    self.db.query(
                        ThemeMetrics.theme_cluster_id,
                        func.max(ThemeMetrics.date).label("max_date")
                    )
                    .group_by(ThemeMetrics.theme_cluster_id)
                    .subquery()
                )

                query = (
                    self.db.query(ThemeCluster, ThemeMetrics)
                    .join(ThemeMetrics, ThemeCluster.id == ThemeMetrics.theme_cluster_id)
                    .join(
                        subquery,
                        (ThemeMetrics.theme_cluster_id == subquery.c.theme_cluster_id) &
                        (ThemeMetrics.date == subquery.c.max_date)
                    )
                    .filter(ThemeCluster.is_active == True)
                    .filter(
                        (ThemeCluster.is_emerging == True) |
                        (ThemeMetrics.mention_velocity >= min_velocity)
                    )
                )

                if category:
                    query = query.filter(ThemeCluster.category.ilike(f"%{category}%"))

                results = (
                    query
                    .order_by(desc(ThemeMetrics.mention_velocity))
                    .limit(limit)
                    .all()
                )

                for theme, metrics in results:
                    themes_data.append(self._format_theme_summary(theme, metrics))

            else:  # trending (default)
                # Get current trending themes by momentum_score
                subquery = (
                    self.db.query(
                        ThemeMetrics.theme_cluster_id,
                        func.max(ThemeMetrics.date).label("max_date")
                    )
                    .group_by(ThemeMetrics.theme_cluster_id)
                    .subquery()
                )

                query = (
                    self.db.query(ThemeCluster, ThemeMetrics)
                    .join(ThemeMetrics, ThemeCluster.id == ThemeMetrics.theme_cluster_id)
                    .join(
                        subquery,
                        (ThemeMetrics.theme_cluster_id == subquery.c.theme_cluster_id) &
                        (ThemeMetrics.date == subquery.c.max_date)
                    )
                    .filter(ThemeCluster.is_active == True)
                )

                if category:
                    query = query.filter(ThemeCluster.category.ilike(f"%{category}%"))

                results = (
                    query
                    .order_by(desc(ThemeMetrics.momentum_score))
                    .limit(limit)
                    .all()
                )

                for theme, metrics in results:
                    themes_data.append(self._format_theme_summary(theme, metrics))

            return {
                "mode": mode,
                "themes": themes_data,
                "as_of_date": datetime.utcnow().strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.error(f"Error discovering themes (mode={mode}): {e}")
            return {
                "mode": mode,
                "themes": [],
                "as_of_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "error": str(e),
            }

    def _get_theme_summary(self, theme_name: str) -> Optional[Dict[str, Any]]:
        """Helper to get theme summary for comparison mode."""
        theme = (
            self.db.query(ThemeCluster)
            .filter(
                (func.lower(ThemeCluster.display_name) == theme_name.lower()) |
                (ThemeCluster.aliases.contains([theme_name]))
            )
            .first()
        )

        if not theme:
            theme = (
                self.db.query(ThemeCluster)
                .filter(ThemeCluster.display_name.ilike(f"%{theme_name}%"))
                .first()
            )

        if not theme:
            return None

        metrics = (
            self.db.query(ThemeMetrics)
            .filter(ThemeMetrics.theme_cluster_id == theme.id)
            .order_by(desc(ThemeMetrics.date))
            .first()
        )

        return self._format_theme_summary(theme, metrics)

    def _format_theme_summary(
        self,
        theme: ThemeCluster,
        metrics: Optional[ThemeMetrics]
    ) -> Dict[str, Any]:
        """Format theme and metrics into summary dict."""
        # Get top tickers
        top_constituents = (
            self.db.query(ThemeConstituent.symbol)
            .filter(ThemeConstituent.theme_cluster_id == theme.id)
            .filter(ThemeConstituent.is_active == True)
            .order_by(desc(ThemeConstituent.confidence))
            .limit(5)
            .all()
        )
        top_tickers = [c.symbol for c in top_constituents]

        summary = {
            "name": theme.display_name or theme.name,
            "category": theme.category,
            "is_emerging": theme.is_emerging,
            "top_tickers": top_tickers,
        }

        if metrics:
            summary.update({
                "momentum_score": metrics.momentum_score,
                "rank": metrics.rank,
                "velocity": metrics.mention_velocity,
                "sentiment_score": metrics.sentiment_score,
                "basket_rs_vs_spy": metrics.basket_rs_vs_spy,
                "basket_return_1w": metrics.basket_return_1w,
                "mentions_7d": metrics.mentions_7d,
                "num_constituents": metrics.num_constituents,
                "status": metrics.status,
            })
        else:
            summary.update({
                "momentum_score": None,
                "rank": None,
                "velocity": None,
                "sentiment_score": None,
                "basket_rs_vs_spy": None,
                "basket_return_1w": None,
                "mentions_7d": None,
                "num_constituents": len(top_tickers),
                "status": "unknown",
            })

        return summary
