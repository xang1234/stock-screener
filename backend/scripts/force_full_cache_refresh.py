"""
Force a full cache refresh by clearing existing data and re-fetching.

This script will:
1. Delete all existing price data from the database
2. Clear Redis price cache
3. Re-fetch full 2-year history for all active stocks
4. Properly populate both database and Redis cache

Use this to fix the "19 days" issue by ensuring a clean slate.
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import logging
from sqlalchemy import delete
from app.database import SessionLocal
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.services.cache_manager import CacheManager
from app.wiring.bootstrap import get_price_cache, initialize_process_runtime_services

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("FORCE FULL CACHE REFRESH")
print("=" * 80)
print()
print("WARNING: This will delete ALL price data and re-fetch from scratch.")
print("This may take 2-3 hours for 9,650 stocks (respecting yfinance rate limits).")
print()

# Ask for confirmation
confirm = input("Type 'YES' to proceed: ")
if confirm != "YES":
    print("Aborted.")
    exit(0)

db = SessionLocal()
initialize_process_runtime_services(session_factory=SessionLocal)

try:
    # Step 1: Clear price data from database
    print("\n1. Clearing price data from database...")
    count_before = db.query(StockPrice).count()
    logger.info(f"Database has {count_before} price rows before deletion")

    db.execute(delete(StockPrice))
    db.commit()
    logger.info("✓ Deleted all price data from database")

    # Step 2: Clear Redis cache
    print("\n2. Clearing Redis price cache...")
    price_cache = get_price_cache()

    if price_cache._redis_client:
        price_keys = price_cache._redis_client.keys("price:*")
        if price_keys:
            price_cache._redis_client.delete(*price_keys)
            logger.info(f"✓ Deleted {len(price_keys)} keys from Redis")
        else:
            logger.info("✓ No price keys in Redis")
    else:
        logger.warning("Redis not available")

    # Step 3: Get all active symbols
    print("\n3. Getting active symbols from universe...")
    universe_stocks = db.query(StockUniverse).filter(
        StockUniverse.is_active == True
    ).all()

    symbols = [s.symbol for s in universe_stocks]
    logger.info(f"Found {len(symbols)} active symbols to refresh")

    # Step 4: Warm cache with full 2y data
    print(f"\n4. Fetching 5-year data for {len(symbols)} symbols...")
    print("   This will take ~2-3 hours (respecting yfinance rate limits)...")
    print("   Progress will be logged...")

    cache_manager = CacheManager(db)

    # Use smaller batch size to be safe (20 instead of 50)
    result = cache_manager.warm_price_cache(
        symbols,
        period="5y",  # Explicitly set to 2 years
        batch_size=200,  # Smaller batches for reliability
        rate_limit=2.0,  # 2 seconds between batches (conservative)
        force_refresh=True  # Force fetch from yfinance
    )

    print("\n" + "=" * 80)
    print("REFRESH COMPLETE")
    print("=" * 80)
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    print(f"Duration: {result.get('duration', 0):.1f} seconds")
    print()
    print("Next steps:")
    print("1. Run a Minervini scan - it should use cached data (fast)")
    print("2. Verify scan shows ~50-500 results (typical for Minervini)")
    print("3. Check that no \"insufficient data\" errors appear")
    print("=" * 80)

except Exception as e:
    logger.error(f"Error during refresh: {e}", exc_info=True)
    db.rollback()
    raise

finally:
    db.close()
