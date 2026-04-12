"""
Clear Redis price cache so it can be repopulated with full 2-year data.

After changing RECENT_DAYS from 30 to 730, we need to clear the existing
Redis cache (which only has 30 days) so it can be repopulated with 2 years.
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import logging
from app.wiring.bootstrap import get_price_cache, initialize_process_runtime_services

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("CLEAR REDIS PRICE CACHE")
print("=" * 80)
print()
print("This will clear all price data from Redis so it can be repopulated")
print("with full 2-year data (instead of the old 30-day data).")
print()
print("After this:")
print("1. Database still has full 2 years ✓")
print("2. Redis will be empty (will be repopulated on next scan)")
print("3. Next scan will use database fallback (~20-30 min)")
print("4. Redis will be repopulated with 2 years during that scan")
print("5. Subsequent scans will be fast (~17-20 min from Redis)")
print()

confirm = input("Type 'YES' to proceed: ")
if confirm != "YES":
    print("Aborted.")
    exit(0)

try:
    initialize_process_runtime_services()
    price_cache = get_price_cache()

    if not price_cache._redis_client:
        logger.error("Redis not available")
        exit(1)

    # Count keys before deletion
    price_keys = price_cache._redis_client.keys("price:*")
    count = len(price_keys)

    logger.info(f"Found {count} price cache keys in Redis")

    if count > 0:
        # Delete all price keys
        price_cache._redis_client.delete(*price_keys)
        logger.info(f"✓ Deleted {count} keys from Redis")
    else:
        logger.info("No price keys to delete")

    print()
    print("=" * 80)
    print("REDIS CACHE CLEARED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Restart Celery worker (to pick up RECENT_DAYS = 730 change)")
    print("2. Run a Minervini scan")
    print("3. First scan: ~20-30 min (database fallback + Redis population)")
    print("4. Subsequent scans: ~17-20 min (full 2y data from Redis)")
    print()
    print("Redis will now store full 2 years instead of just 30 days!")
    print("=" * 80)

except Exception as e:
    logger.error(f"Error clearing Redis cache: {e}", exc_info=True)
    exit(1)
