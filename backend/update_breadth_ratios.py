"""
Update breadth ratios for existing market breadth records.

This script calculates 5-day and 10-day ratios based on existing
breadth data without making any API calls.
"""
import logging
from datetime import timedelta
from app.database import SessionLocal
from app.models.market_breadth import MarketBreadth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_and_update_ratios():
    """Calculate and update ratios for all breadth records."""
    db = SessionLocal()

    try:
        # This script is a one-shot US-scoped utility (predates multi-market
        # breadth). Filter by market='US' so it only mutates US rows.
        all_records = db.query(MarketBreadth).filter(
            MarketBreadth.market == "US",
        ).order_by(
            MarketBreadth.date.asc()
        ).all()

        logger.info(f"Found {len(all_records)} breadth records")

        updated_count = 0

        for i, record in enumerate(all_records):
            lookback_start = record.date - timedelta(days=20)  # Generous window

            prev_records = db.query(MarketBreadth).filter(
                MarketBreadth.date >= lookback_start,
                MarketBreadth.date < record.date,
                MarketBreadth.market == "US",
            ).order_by(MarketBreadth.date.desc()).limit(10).all()

            if len(prev_records) < 5:
                logger.debug(f"Skipping {record.date}: insufficient historical data ({len(prev_records)} days)")
                continue

            # Calculate 5-day ratio
            last_5_days = prev_records[:5]
            up_5day = sum(r.stocks_up_4pct for r in last_5_days)
            down_5day = sum(r.stocks_down_4pct for r in last_5_days)

            ratio_5day = None
            if down_5day > 0:
                ratio_5day = round(up_5day / down_5day, 2)

            # Calculate 10-day ratio
            ratio_10day = None
            if len(prev_records) >= 10:
                last_10_days = prev_records[:10]
                up_10day = sum(r.stocks_up_4pct for r in last_10_days)
                down_10day = sum(r.stocks_down_4pct for r in last_10_days)

                if down_10day > 0:
                    ratio_10day = round(up_10day / down_10day, 2)

            # Update the record
            record.ratio_5day = ratio_5day
            record.ratio_10day = ratio_10day
            updated_count += 1

            logger.info(f"{record.date}: 5-day={ratio_5day}, 10-day={ratio_10day}")

        # Commit all updates
        db.commit()
        logger.info(f"✓ Updated ratios for {updated_count} records")

    except Exception as e:
        logger.error(f"Error updating ratios: {e}", exc_info=True)
        db.rollback()

    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Starting breadth ratio update...")
    calculate_and_update_ratios()
    logger.info("Done!")
