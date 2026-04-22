"""Service for loading and managing IBD Industry Group data"""
import csv
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import delete
from ..config import settings
from ..config.settings import get_project_root
from ..models.industry import IBDIndustryGroup

logger = logging.getLogger(__name__)


class IBDIndustryService:
    """Manage IBD Industry Group data"""

    @staticmethod
    def resolve_tracked_csv_path(csv_path: str | Path | None = None) -> Path:
        """Resolve the tracked IBD CSV, falling back to the repo copy when needed."""
        normalized_csv_path = csv_path
        if isinstance(normalized_csv_path, str) and not normalized_csv_path.strip():
            normalized_csv_path = None

        settings_path = Path(settings.ibd_industry_csv_path)
        configured_path = Path(normalized_csv_path) if normalized_csv_path is not None else settings_path
        if configured_path.exists():
            return configured_path

        fallback_path = get_project_root() / "data" / "IBD_industry_group.csv"
        should_fallback = normalized_csv_path is None or configured_path == settings_path
        if should_fallback and fallback_path.exists() and fallback_path != configured_path:
            logger.warning(
                "Configured IBD industry CSV path %s is missing; falling back to tracked CSV at %s",
                configured_path,
                fallback_path,
            )
            return fallback_path

        if normalized_csv_path is not None:
            raise FileNotFoundError(f"IBD industry CSV file not found at {configured_path}")

        return configured_path

    @staticmethod
    def load_from_csv(db: Session, csv_path: str = None) -> int:
        """
        Load IBD industry groups from CSV file.

        Args:
            db: Database session
            csv_path: Path to CSV file (defaults to data/IBD_industry_group.csv)

        Returns:
            Number of records loaded
        """
        csv_path = IBDIndustryService.resolve_tracked_csv_path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"IBD industry CSV file not found at {csv_path}")

        logger.info(f"Loading IBD industry groups from {csv_path}")

        # Clear existing data
        db.execute(delete(IBDIndustryGroup))
        db.commit()
        logger.info("Cleared existing IBD industry data")

        loaded = 0
        batch = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)

                for row in reader:
                    if len(row) != 2:
                        continue

                    symbol, industry_group = row
                    symbol = symbol.strip().upper()
                    industry_group = industry_group.strip()

                    if not symbol or not industry_group:
                        continue

                    batch.append({
                        'symbol': symbol,
                        'industry_group': industry_group
                    })

                    # Batch insert every 500 records
                    if len(batch) >= 500:
                        db.bulk_insert_mappings(IBDIndustryGroup, batch)
                        db.commit()
                        loaded += len(batch)
                        logger.info(f"Loaded {loaded} IBD industry mappings...")
                        batch = []

                # Insert remaining records
                if batch:
                    db.bulk_insert_mappings(IBDIndustryGroup, batch)
                    db.commit()
                    loaded += len(batch)

            logger.info(f"Successfully loaded {loaded} IBD industry group mappings")
            return loaded

        except Exception as e:
            logger.error(f"Error loading IBD industry data: {e}", exc_info=True)
            db.rollback()
            raise

    @staticmethod
    def get_industry_group(db: Session, symbol: str) -> str:
        """
        Get IBD industry group for a symbol.

        Args:
            db: Database session
            symbol: Stock symbol

        Returns:
            Industry group name or None if not found
        """
        try:
            record = db.query(IBDIndustryGroup).filter(
                IBDIndustryGroup.symbol == symbol.upper()
            ).first()
            return record.industry_group if record else None
        except Exception as e:
            logger.error(f"Error getting industry group for {symbol}: {e}")
            return None

    @staticmethod
    def get_bulk_industry_groups(db: Session, symbols: list) -> dict:
        """
        Bulk fetch IBD industry groups for multiple symbols (performance optimization).

        Args:
            db: Database session
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol -> industry group name
        """
        if not symbols:
            return {}

        try:
            # Normalize symbols to uppercase
            symbols_upper = [s.upper() for s in symbols]

            # Single bulk query instead of N individual queries
            records = db.query(IBDIndustryGroup).filter(
                IBDIndustryGroup.symbol.in_(symbols_upper)
            ).all()

            # Return as dict for O(1) lookup
            return {record.symbol: record.industry_group for record in records}

        except Exception as e:
            logger.error(f"Error bulk fetching industry groups: {e}")
            return {}

    @staticmethod
    def get_group_symbols(db: Session, industry_group: str) -> list:
        """
        Get all symbols in an IBD industry group.

        Args:
            db: Database session
            industry_group: Industry group name

        Returns:
            List of symbols in the group
        """
        try:
            records = db.query(IBDIndustryGroup.symbol).filter(
                IBDIndustryGroup.industry_group == industry_group
            ).all()
            return [r.symbol for r in records]
        except Exception as e:
            logger.error(f"Error getting symbols for group {industry_group}: {e}")
            return []

    @staticmethod
    def get_all_groups(db: Session) -> list:
        """
        Get list of all unique industry groups.

        Args:
            db: Database session

        Returns:
            List of unique industry group names
        """
        try:
            records = db.query(IBDIndustryGroup.industry_group).distinct().all()
            return [r.industry_group for r in records]
        except Exception as e:
            logger.error(f"Error getting all industry groups: {e}", exc_info=True)
            raise
