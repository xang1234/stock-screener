"""
Theme Correlation Service

Validates themes using price correlations and discovers hidden themes
from stocks that move together but aren't classified in the same group.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.theme import ThemeCluster, ThemeConstituent, ThemeMetrics, ThemeAlert
from ..models.stock import StockPrice
from ..models.industry import IBDIndustryGroup
from ..models.stock_universe import StockUniverse
from .theme_identity_normalization import canonical_theme_key, display_theme_name

logger = logging.getLogger(__name__)


class ThemeCorrelationService:
    """
    Service for price-based theme validation and discovery

    Key capabilities:
    1. Validate existing themes (do constituents correlate?)
    2. Find stocks joining themes (correlation spike detection)
    3. Discover hidden themes via correlation clustering
    4. Identify cross-industry correlations (narrative plays)
    """

    def __init__(self, db: Session):
        self.db = db
        self._price_cache: Optional[pd.DataFrame] = None
        self._returns_cache: Optional[pd.DataFrame] = None
        self._cache_date: Optional[datetime] = None

    def _load_price_data(
        self,
        symbols: list[str],
        days: int = 60,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load price data for symbols into a DataFrame

        Returns DataFrame with dates as index, symbols as columns, close prices as values
        """
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)

        prices = self.db.query(StockPrice).filter(
            StockPrice.symbol.in_(symbols),
            StockPrice.date >= start_date,
            StockPrice.date <= end_date,
        ).all()

        # Pivot to DataFrame
        data = defaultdict(dict)
        for p in prices:
            data[p.symbol][p.date] = p.close

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def _calculate_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from prices"""
        return price_df.pct_change().dropna()

    def calculate_pairwise_correlations(
        self,
        symbols: list[str],
        lookback_days: int = 21
    ) -> pd.DataFrame:
        """
        Calculate pairwise correlation matrix for given symbols

        Returns correlation matrix DataFrame
        """
        price_df = self._load_price_data(symbols, days=lookback_days + 10)
        returns_df = self._calculate_returns(price_df)

        # Use most recent lookback_days
        returns_df = returns_df.tail(lookback_days)

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        return corr_matrix

    def validate_theme(
        self,
        theme_cluster_id: int,
        min_correlation: float = 0.5
    ) -> dict:
        """
        Validate a theme by checking internal correlations

        A valid theme should have high average correlation between constituents.

        Returns:
            is_valid: bool - meets minimum correlation threshold
            avg_correlation: float - average pairwise correlation
            correlation_matrix: dict - full correlation data
            outliers: list - stocks with low correlation to theme
        """
        # Get constituents
        constituents = self.db.query(ThemeConstituent).filter(
            ThemeConstituent.theme_cluster_id == theme_cluster_id,
            ThemeConstituent.is_active == True,
        ).all()

        if len(constituents) < 2:
            return {
                "is_valid": False,
                "avg_correlation": 0,
                "reason": "Need at least 2 constituents to validate",
            }

        symbols = [c.symbol for c in constituents]

        # Calculate correlations
        corr_matrix = self.calculate_pairwise_correlations(symbols)

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()

        avg_correlation = correlations.mean()
        min_corr = correlations.min()
        max_corr = correlations.max()

        # Find outliers (stocks with low avg correlation to others)
        outliers = []
        for symbol in symbols:
            symbol_corrs = corr_matrix[symbol].drop(symbol)
            if symbol_corrs.mean() < min_correlation * 0.7:  # 70% of threshold
                outliers.append({
                    "symbol": symbol,
                    "avg_correlation": round(symbol_corrs.mean(), 3),
                })

        # Update constituent correlation scores
        for constituent in constituents:
            if constituent.symbol in corr_matrix.columns:
                other_corrs = corr_matrix[constituent.symbol].drop(constituent.symbol, errors='ignore')
                constituent.correlation_to_theme = round(other_corrs.mean(), 3)
                constituent.correlation_updated_at = datetime.utcnow()

        self.db.commit()

        is_valid = avg_correlation >= min_correlation

        # Update theme validation status
        cluster = self.db.query(ThemeCluster).filter(ThemeCluster.id == theme_cluster_id).first()
        if cluster:
            cluster.is_validated = is_valid

        self.db.commit()

        return {
            "is_valid": is_valid,
            "avg_correlation": round(avg_correlation, 3),
            "min_correlation": round(min_corr, 3),
            "max_correlation": round(max_corr, 3),
            "num_constituents": len(symbols),
            "outliers": outliers,
        }

    def find_new_theme_entrants(
        self,
        theme_cluster_id: int,
        candidate_symbols: Optional[list[str]] = None,
        correlation_threshold: float = 0.6,
    ) -> list[dict]:
        """
        Find stocks that correlate highly with a theme but aren't yet constituents

        This identifies stocks that may be joining a theme.
        """
        # Get current constituents
        constituents = self.db.query(ThemeConstituent).filter(
            ThemeConstituent.theme_cluster_id == theme_cluster_id,
            ThemeConstituent.is_active == True,
        ).all()

        theme_symbols = [c.symbol for c in constituents]

        if len(theme_symbols) < 2:
            return []

        # Get candidates (if not provided, use broad universe)
        if candidate_symbols is None:
            # Get all symbols from universe, excluding current constituents
            candidates = self.db.query(StockUniverse.symbol).filter(
                ~StockUniverse.symbol.in_(theme_symbols)
            ).limit(500).all()
            candidate_symbols = [c[0] for c in candidates]

        # Filter to only candidates not in theme
        candidate_symbols = [s for s in candidate_symbols if s not in theme_symbols]

        if not candidate_symbols:
            return []

        # Load price data for all symbols
        all_symbols = theme_symbols + candidate_symbols
        price_df = self._load_price_data(all_symbols, days=30)
        returns_df = self._calculate_returns(price_df)

        # Calculate theme basket returns (equal-weight of constituents)
        theme_returns = returns_df[theme_symbols].mean(axis=1)

        # Correlate each candidate with theme basket
        entrants = []
        for candidate in candidate_symbols:
            if candidate not in returns_df.columns:
                continue

            candidate_returns = returns_df[candidate].dropna()
            aligned_theme = theme_returns.loc[candidate_returns.index]

            if len(aligned_theme) < 10:  # Need minimum data
                continue

            correlation = candidate_returns.corr(aligned_theme)

            if correlation >= correlation_threshold:
                # Get industry for context
                industry = self.db.query(IBDIndustryGroup).filter(
                    IBDIndustryGroup.symbol == candidate
                ).first()

                entrants.append({
                    "symbol": candidate,
                    "correlation": round(correlation, 3),
                    "industry": industry.industry_group if industry else "Unknown",
                })

        # Sort by correlation
        entrants.sort(key=lambda x: x["correlation"], reverse=True)

        return entrants[:20]  # Return top 20

    def discover_correlation_clusters(
        self,
        symbols: Optional[list[str]] = None,
        correlation_threshold: float = 0.6,
        min_cluster_size: int = 3,
    ) -> list[dict]:
        """
        Discover hidden themes by clustering stocks with high correlation

        Uses hierarchical clustering on correlation matrix to find
        groups of stocks that move together.
        """
        # Get symbols if not provided
        if symbols is None:
            # Use high-RS stocks from universe
            stocks = self.db.query(StockUniverse).limit(300).all()
            symbols = [s.symbol for s in stocks]

        if len(symbols) < min_cluster_size:
            return []

        # Calculate correlation matrix
        corr_matrix = self.calculate_pairwise_correlations(symbols, lookback_days=21)

        # Remove symbols with all NaN correlations
        valid_symbols = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all').columns.tolist()
        corr_matrix = corr_matrix.loc[valid_symbols, valid_symbols]

        if len(valid_symbols) < min_cluster_size:
            return []

        # Convert correlation to distance (1 - corr)
        # Higher correlation = lower distance
        distance_matrix = 1 - corr_matrix.fillna(0)
        np.fill_diagonal(distance_matrix.values, 0)

        # Perform hierarchical clustering
        try:
            condensed = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed, method='average')

            # Cut tree at threshold (distance = 1 - correlation_threshold)
            distance_threshold = 1 - correlation_threshold
            cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return []

        # Group symbols by cluster
        clusters = defaultdict(list)
        for symbol, label in zip(valid_symbols, cluster_labels):
            clusters[label].append(symbol)

        # Filter and format clusters
        discovered = []
        for label, cluster_symbols in clusters.items():
            if len(cluster_symbols) < min_cluster_size:
                continue

            # Calculate cluster stats
            cluster_corr = corr_matrix.loc[cluster_symbols, cluster_symbols]
            mask = np.triu(np.ones_like(cluster_corr, dtype=bool), k=1)
            avg_corr = cluster_corr.where(mask).stack().mean()

            # Get industries for cluster members
            industries = self.db.query(
                IBDIndustryGroup.industry_group,
                func.count(IBDIndustryGroup.symbol)
            ).filter(
                IBDIndustryGroup.symbol.in_(cluster_symbols)
            ).group_by(IBDIndustryGroup.industry_group).all()

            industry_breakdown = {ind: count for ind, count in industries}

            # Check if this is a cross-industry cluster (potential hidden theme)
            is_cross_industry = len(industry_breakdown) > 1

            discovered.append({
                "cluster_id": int(label),
                "symbols": cluster_symbols,
                "num_stocks": len(cluster_symbols),
                "avg_correlation": round(avg_corr, 3),
                "industries": industry_breakdown,
                "is_cross_industry": is_cross_industry,
            })

        # Sort by size and correlation
        discovered.sort(key=lambda x: (x["num_stocks"], x["avg_correlation"]), reverse=True)

        return discovered

    def find_cross_industry_correlations(
        self,
        min_correlation: float = 0.7,
        lookback_days: int = 21,
    ) -> list[dict]:
        """
        Find stocks from different industries that are highly correlated

        These represent potential hidden themes or narrative plays that
        cross traditional industry boundaries.
        """
        # Get diverse sample of stocks with their industries
        stocks_with_industries = self.db.query(
            StockUniverse.symbol,
            IBDIndustryGroup.industry_group
        ).outerjoin(
            IBDIndustryGroup, StockUniverse.symbol == IBDIndustryGroup.symbol
        ).limit(200).all()

        symbol_industries = {s: ind for s, ind in stocks_with_industries if ind}
        symbols = list(symbol_industries.keys())

        if len(symbols) < 10:
            return []

        # Calculate correlations
        corr_matrix = self.calculate_pairwise_correlations(symbols, lookback_days)

        # Find high-correlation pairs from different industries
        cross_industry_pairs = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 not in corr_matrix.columns or sym2 not in corr_matrix.columns:
                    continue

                ind1 = symbol_industries.get(sym1)
                ind2 = symbol_industries.get(sym2)

                # Skip if same industry or no industry data
                if not ind1 or not ind2 or ind1 == ind2:
                    continue

                correlation = corr_matrix.loc[sym1, sym2]

                if pd.notna(correlation) and correlation >= min_correlation:
                    cross_industry_pairs.append({
                        "symbol1": sym1,
                        "industry1": ind1,
                        "symbol2": sym2,
                        "industry2": ind2,
                        "correlation": round(correlation, 3),
                    })

        # Sort by correlation
        cross_industry_pairs.sort(key=lambda x: x["correlation"], reverse=True)

        # Group into potential themes
        # Stocks that appear together multiple times might form a theme
        symbol_pairs = defaultdict(list)
        for pair in cross_industry_pairs[:50]:  # Top 50 pairs
            for sym in [pair["symbol1"], pair["symbol2"]]:
                other = pair["symbol2"] if sym == pair["symbol1"] else pair["symbol1"]
                symbol_pairs[sym].append({
                    "symbol": other,
                    "correlation": pair["correlation"],
                })

        # Find hub stocks (correlated with many others)
        hub_stocks = [
            {"symbol": sym, "connections": len(pairs), "pairs": pairs[:5]}
            for sym, pairs in symbol_pairs.items()
            if len(pairs) >= 3
        ]
        hub_stocks.sort(key=lambda x: x["connections"], reverse=True)

        return {
            "cross_industry_pairs": cross_industry_pairs[:20],
            "hub_stocks": hub_stocks[:10],
        }

    def create_theme_from_cluster(
        self,
        symbols: list[str],
        name: str,
        description: Optional[str] = None,
    ) -> ThemeCluster:
        """
        Create a new theme cluster from a discovered correlation cluster
        """
        # Check if similar theme exists
        canonical_key = canonical_theme_key(name)
        display_name = display_theme_name(name)
        existing = self.db.query(ThemeCluster).filter(
            ThemeCluster.canonical_key == canonical_key,
            ThemeCluster.pipeline == "technical",
        ).first()

        if existing:
            logger.warning(f"Theme '{name}' already exists")
            return existing

        # Create cluster
        cluster = ThemeCluster(
            canonical_key=canonical_key,
            display_name=display_name,
            name=display_name,
            description=description or f"Theme discovered from price correlation analysis",
            pipeline="technical",
            discovery_source="correlation_clustering",
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
            is_emerging=True,
            is_validated=True,  # Already validated by correlation
        )
        self.db.add(cluster)
        self.db.flush()

        # Add constituents
        for symbol in symbols:
            constituent = ThemeConstituent(
                theme_cluster_id=cluster.id,
                symbol=symbol,
                source="correlation",
                confidence=0.8,
                mention_count=0,
                first_mentioned_at=datetime.utcnow(),
                last_mentioned_at=datetime.utcnow(),
            )
            self.db.add(constituent)

        self.db.commit()

        logger.info(f"Created theme '{display_name}' with {len(symbols)} constituents from correlation cluster")
        return cluster

    def add_entrants_to_theme(
        self,
        theme_cluster_id: int,
        entrants: list[dict],
        min_confidence: float = 0.6,
    ) -> int:
        """
        Add newly discovered entrants to a theme

        Returns count of stocks added
        """
        added = 0
        for entrant in entrants:
            if entrant["correlation"] < min_confidence:
                continue

            # Check if already exists
            existing = self.db.query(ThemeConstituent).filter(
                ThemeConstituent.theme_cluster_id == theme_cluster_id,
                ThemeConstituent.symbol == entrant["symbol"],
            ).first()

            if existing:
                continue

            constituent = ThemeConstituent(
                theme_cluster_id=theme_cluster_id,
                symbol=entrant["symbol"],
                source="correlation",
                confidence=entrant["correlation"],
                correlation_to_theme=entrant["correlation"],
                correlation_updated_at=datetime.utcnow(),
                mention_count=0,
                first_mentioned_at=datetime.utcnow(),
                last_mentioned_at=datetime.utcnow(),
            )
            self.db.add(constituent)
            added += 1

        if added > 0:
            self.db.commit()
            logger.info(f"Added {added} new entrants to theme {theme_cluster_id}")

        return added

    def run_full_validation(self, min_correlation: float = 0.5) -> dict:
        """
        Run validation on all active themes

        Returns summary of validation results
        """
        clusters = self.db.query(ThemeCluster).filter(
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
        ).all()

        results = {
            "themes_validated": 0,
            "themes_valid": 0,
            "themes_invalid": 0,
            "details": [],
        }

        for cluster in clusters:
            try:
                validation = self.validate_theme(cluster.id, min_correlation)
                results["themes_validated"] += 1

                if validation["is_valid"]:
                    results["themes_valid"] += 1
                else:
                    results["themes_invalid"] += 1

                results["details"].append({
                    "theme": cluster.name,
                    "is_valid": validation["is_valid"],
                    "avg_correlation": validation.get("avg_correlation", 0),
                    "num_constituents": validation.get("num_constituents", 0),
                    "outliers": validation.get("outliers", []),
                })
            except Exception as e:
                logger.error(f"Error validating {cluster.name}: {e}")

        return results
