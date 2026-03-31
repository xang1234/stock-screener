"""Service for loading and managing IBD Industry Group data."""

import csv
import logging
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from ..models.industry import IBDIndustryGroup

logger = logging.getLogger(__name__)


class IBDIndustryService:
    """Manage IBD Industry Group data."""

    @staticmethod
    def default_csv_path() -> Path:
        """Return the canonical bundled IBD industry seed path."""
        return Path(__file__).resolve().parents[2] / "resources" / "IBD_industry_group.csv"

    @classmethod
    def resolve_csv_path(cls, csv_path: str | None = None) -> Path:
        """Resolve the CSV path used to load industry mappings."""
        return Path(csv_path) if csv_path else cls.default_csv_path()

    @staticmethod
    def load_from_csv(db: Session, csv_path: str | None = None) -> int:
        """
        Load IBD industry groups from CSV file.

        Args:
            db: Database session
            csv_path: Path to CSV file (defaults to the configured canonical seed)

        Returns:
            Number of records loaded
        """
        resolved_path = IBDIndustryService.resolve_csv_path(csv_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"IBD industry CSV file not found at {resolved_path}")

        logger.info("Loading IBD industry groups from %s", resolved_path)

        db.execute(delete(IBDIndustryGroup))
        db.commit()
        logger.info("Cleared existing IBD industry data")

        loaded = 0
        batch: list[dict[str, str]] = []

        try:
            with resolved_path.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle)

                for row in reader:
                    if len(row) != 2:
                        continue

                    symbol, industry_group = row
                    symbol = symbol.strip().upper()
                    industry_group = industry_group.strip()
                    if not symbol or not industry_group:
                        continue

                    batch.append(
                        {
                            "symbol": symbol,
                            "industry_group": industry_group,
                        }
                    )

                    if len(batch) >= 500:
                        db.bulk_insert_mappings(IBDIndustryGroup, batch)
                        db.commit()
                        loaded += len(batch)
                        logger.info("Loaded %s IBD industry mappings...", loaded)
                        batch = []

                if batch:
                    db.bulk_insert_mappings(IBDIndustryGroup, batch)
                    db.commit()
                    loaded += len(batch)

            logger.info("Successfully loaded %s IBD industry group mappings", loaded)
            return loaded
        except Exception:
            logger.exception("Error loading IBD industry data")
            db.rollback()
            raise

    @classmethod
    def seed_if_empty(cls, db: Session, csv_path: str | None = None) -> int:
        """
        Seed the industry mapping table if it is currently empty.

        Returns:
            Number of records loaded. Returns 0 when data already exists.
        """
        existing = db.query(IBDIndustryGroup).count()
        if existing > 0:
            logger.info(
                "IBD industry group seed skipped; table already contains %s mappings",
                existing,
            )
            return 0

        resolved_path = cls.resolve_csv_path(csv_path)
        loaded = cls.load_from_csv(db, str(resolved_path))
        logger.info(
            "Seeded IBD industry groups with %s mappings from %s",
            loaded,
            resolved_path,
        )
        return loaded

    @staticmethod
    def get_industry_group(db: Session, symbol: str) -> str | None:
        """Get IBD industry group for a symbol."""
        try:
            record = db.query(IBDIndustryGroup).filter(
                IBDIndustryGroup.symbol == symbol.upper()
            ).first()
            return record.industry_group if record else None
        except Exception as e:
            logger.error("Error getting industry group for %s: %s", symbol, e)
            return None

    @staticmethod
    def get_bulk_industry_groups(db: Session, symbols: list[str]) -> dict[str, str]:
        """Bulk fetch IBD industry groups for multiple symbols."""
        if not symbols:
            return {}

        try:
            symbols_upper = [symbol.upper() for symbol in symbols]
            records = db.query(IBDIndustryGroup).filter(
                IBDIndustryGroup.symbol.in_(symbols_upper)
            ).all()
            return {record.symbol: record.industry_group for record in records}
        except Exception as e:
            logger.error("Error bulk fetching industry groups: %s", e)
            return {}

    @staticmethod
    def get_group_symbols(db: Session, industry_group: str) -> list[str]:
        """Get all symbols in an IBD industry group."""
        try:
            records = db.query(IBDIndustryGroup.symbol).filter(
                IBDIndustryGroup.industry_group == industry_group
            ).all()
            return [record.symbol for record in records]
        except Exception as e:
            logger.error("Error getting symbols for group %s: %s", industry_group, e)
            return []

    @staticmethod
    def get_all_groups(db: Session) -> list[str]:
        """Get list of all unique industry groups."""
        try:
            records = db.query(IBDIndustryGroup.industry_group).distinct().all()
            return [record.industry_group for record in records]
        except Exception as e:
            logger.error("Error getting all industry groups: %s", e)
            return []


ibd_industry_service = IBDIndustryService()
