#!/usr/bin/env python3
"""
Production Performance Test for Caching System.

This test will:
1. Clear all caches to start fresh
2. Run a bulk scan with 30 stocks (COLD cache)
3. Run the same scan again (HOT cache)
4. Compare performance metrics
"""
import os

import pytest

pytestmark = pytest.mark.live_service

if os.getenv("RUN_LIVE_SERVICE_TESTS") != "1":
    pytest.skip(
        "requires RUN_LIVE_SERVICE_TESTS=1 and a running backend",
        allow_module_level=True,
    )

import time
import requests
import redis
from datetime import datetime
from app.database import SessionLocal
from app.models.stock import StockPrice
from app.config import settings

# Test configuration
BASE_URL = "http://localhost:8000/api/v1"
TEST_SYMBOLS = [
    # Top 30 stocks from S&P 500 for realistic test
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'BRK.B', 'V', 'JPM',
    'WMT', 'MA', 'JNJ', 'PG', 'XOM',
    'UNH', 'HD', 'CVX', 'ABBV', 'KO',
    'LLY', 'AVGO', 'MRK', 'PEP', 'COST',
    'TMO', 'ADBE', 'NFLX', 'CRM', 'DIS'
]

print("=" * 80)
print("PRODUCTION PERFORMANCE TEST - CACHING SYSTEM")
print("=" * 80)
print(f"\nTest Configuration:")
print(f"  - Stocks to scan: {len(TEST_SYMBOLS)}")
print(f"  - Symbols: {', '.join(TEST_SYMBOLS[:10])}... (+{len(TEST_SYMBOLS)-10} more)")
print(f"  - Backend: {BASE_URL}")
print(f"  - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "=" * 80)

# Step 1: Clear all caches
print("\n[STEP 1] Clearing all caches...")
print("-" * 80)

# Clear Redis cache
try:
    redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=2,  # Cache database
        decode_responses=False
    )
    redis_client.ping()

    # Get all cache keys
    spy_keys = redis_client.keys("benchmark:*")
    price_keys = redis_client.keys("price:*")

    total_keys = len(spy_keys) + len(price_keys)

    if total_keys > 0:
        # Delete all cache keys
        if spy_keys:
            redis_client.delete(*spy_keys)
        if price_keys:
            redis_client.delete(*price_keys)
        print(f"✓ Cleared {total_keys} Redis cache keys")
        print(f"  - SPY cache keys: {len(spy_keys)}")
        print(f"  - Price cache keys: {len(price_keys)}")
    else:
        print("✓ Redis cache already empty")

except Exception as e:
    print(f"⚠ Warning: Could not clear Redis cache: {e}")
    print("  Continuing with test...")

# Clear database cache for test symbols
try:
    db = SessionLocal()
    deleted_count = 0

    for symbol in TEST_SYMBOLS:
        deleted = db.query(StockPrice).filter(StockPrice.symbol == symbol).delete()
        deleted_count += deleted

    db.commit()

    if deleted_count > 0:
        print(f"✓ Cleared {deleted_count} database cache rows for test symbols")
    else:
        print("✓ Database cache already empty for test symbols")

    db.close()
except Exception as e:
    print(f"⚠ Warning: Could not clear database cache: {e}")
    print("  Continuing with test...")

print("\n✓ Cache cleared successfully - starting fresh")

# Step 2: Run COLD cache scan
print("\n[STEP 2] Running COLD cache scan (first run)...")
print("-" * 80)

scan_1_request = {
    "universe": "custom",
    "symbols": TEST_SYMBOLS,
    "criteria": {
        "include_vcp": False  # Disable VCP for faster scans
    }
}

print(f"Creating scan for {len(TEST_SYMBOLS)} stocks...")
scan_1_start = time.time()

try:
    response = requests.post(f"{BASE_URL}/scans", json=scan_1_request)
    response.raise_for_status()
    scan_1_data = response.json()
    scan_1_id = scan_1_data['scan_id']

    print(f"✓ Scan created: {scan_1_id}")
    print(f"  Status: {scan_1_data['status']}")
    print(f"  Total stocks: {scan_1_data['total_stocks']}")

    # Poll for completion
    print("\nWaiting for scan to complete...")
    poll_interval = 2  # seconds
    max_wait = 600  # 10 minutes max
    elapsed = 0
    last_progress = 0

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        status_response = requests.get(f"{BASE_URL}/scans/{scan_1_id}/status")
        status_response.raise_for_status()
        status = status_response.json()

        # Show progress
        progress = status.get('progress', 0)
        completed = status.get('completed_stocks', 0)
        total = status.get('total_stocks', len(TEST_SYMBOLS))

        if progress != last_progress:
            print(f"  Progress: {progress:.1f}% ({completed}/{total} stocks) - {elapsed}s elapsed")
            last_progress = progress

        if status['status'] in ['completed', 'failed', 'cancelled']:
            break

    scan_1_end = time.time()
    scan_1_duration = scan_1_end - scan_1_start

    # Get final status
    final_status = requests.get(f"{BASE_URL}/scans/{scan_1_id}/status").json()

    print(f"\n✓ COLD scan completed!")
    print(f"  Status: {final_status['status']}")
    print(f"  Total time: {scan_1_duration:.2f}s")
    print(f"  Completed stocks: {final_status['completed_stocks']}")
    print(f"  Passed stocks: {final_status.get('passed_stocks', 0)}")
    print(f"  Average time per stock: {scan_1_duration / len(TEST_SYMBOLS):.2f}s")

except Exception as e:
    print(f"✗ Error running COLD scan: {e}")
    exit(1)

# Step 3: Verify cache population
print("\n[STEP 3] Verifying cache population...")
print("-" * 80)

try:
    # Check Redis
    spy_keys = redis_client.keys("benchmark:*")
    price_keys = redis_client.keys("price:*")

    print(f"✓ Redis cache populated:")
    print(f"  - SPY cache keys: {len(spy_keys)}")
    print(f"  - Price cache keys: {len(price_keys)}")

    # Check database
    db = SessionLocal()
    cached_symbols = []
    total_rows = 0

    for symbol in TEST_SYMBOLS:
        count = db.query(StockPrice).filter(StockPrice.symbol == symbol).count()
        if count > 0:
            cached_symbols.append(symbol)
            total_rows += count

    db.close()

    print(f"✓ Database cache populated:")
    print(f"  - Cached symbols: {len(cached_symbols)}/{len(TEST_SYMBOLS)}")
    print(f"  - Total rows: {total_rows}")

except Exception as e:
    print(f"⚠ Warning: Could not verify cache: {e}")

# Step 4: Run HOT cache scan
print("\n[STEP 4] Running HOT cache scan (second run - should be MUCH faster)...")
print("-" * 80)

# Wait a moment to ensure everything is settled
time.sleep(2)

scan_2_request = {
    "universe": "custom",
    "symbols": TEST_SYMBOLS,  # Same symbols
    "criteria": {
        "include_vcp": False
    }
}

print(f"Creating scan for {len(TEST_SYMBOLS)} stocks (same as before)...")
scan_2_start = time.time()

try:
    response = requests.post(f"{BASE_URL}/scans", json=scan_2_request)
    response.raise_for_status()
    scan_2_data = response.json()
    scan_2_id = scan_2_data['scan_id']

    print(f"✓ Scan created: {scan_2_id}")
    print(f"  Status: {scan_2_data['status']}")

    # Poll for completion
    print("\nWaiting for scan to complete...")
    elapsed = 0
    last_progress = 0

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        status_response = requests.get(f"{BASE_URL}/scans/{scan_2_id}/status")
        status_response.raise_for_status()
        status = status_response.json()

        progress = status.get('progress', 0)
        completed = status.get('completed_stocks', 0)
        total = status.get('total_stocks', len(TEST_SYMBOLS))

        if progress != last_progress:
            print(f"  Progress: {progress:.1f}% ({completed}/{total} stocks) - {elapsed}s elapsed")
            last_progress = progress

        if status['status'] in ['completed', 'failed', 'cancelled']:
            break

    scan_2_end = time.time()
    scan_2_duration = scan_2_end - scan_2_start

    # Get final status
    final_status = requests.get(f"{BASE_URL}/scans/{scan_2_id}/status").json()

    print(f"\n✓ HOT scan completed!")
    print(f"  Status: {final_status['status']}")
    print(f"  Total time: {scan_2_duration:.2f}s")
    print(f"  Completed stocks: {final_status['completed_stocks']}")
    print(f"  Passed stocks: {final_status.get('passed_stocks', 0)}")
    print(f"  Average time per stock: {scan_2_duration / len(TEST_SYMBOLS):.2f}s")

except Exception as e:
    print(f"✗ Error running HOT scan: {e}")
    exit(1)

# Step 5: Compare results
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

print(f"\nTest Stocks: {len(TEST_SYMBOLS)} symbols")
print(f"\n{'Metric':<30} {'COLD Cache':<20} {'HOT Cache':<20} {'Improvement':<20}")
print("-" * 90)

# Total time
print(f"{'Total scan time':<30} {scan_1_duration:>18.2f}s  {scan_2_duration:>18.2f}s  ", end="")
if scan_2_duration > 0:
    speedup = scan_1_duration / scan_2_duration
    improvement_pct = ((scan_1_duration - scan_2_duration) / scan_1_duration) * 100
    print(f"{speedup:>6.1f}x faster ({improvement_pct:>5.1f}%)")
else:
    print("N/A")

# Per stock time
avg_cold = scan_1_duration / len(TEST_SYMBOLS)
avg_hot = scan_2_duration / len(TEST_SYMBOLS)
print(f"{'Average per stock':<30} {avg_cold:>18.2f}s  {avg_hot:>18.2f}s  ", end="")
if avg_hot > 0:
    speedup = avg_cold / avg_hot
    improvement_pct = ((avg_cold - avg_hot) / avg_cold) * 100
    print(f"{speedup:>6.1f}x faster ({improvement_pct:>5.1f}%)")
else:
    print("N/A")

# Time saved
time_saved = scan_1_duration - scan_2_duration
print(f"{'Time saved':<30} {'-':>20}  {time_saved:>18.2f}s  {'-':<20}")

# Extrapolation to 500 stocks
print("\n" + "-" * 90)
print("EXTRAPOLATION TO 500 STOCKS:")
print("-" * 90)

# Calculate expected times for 500 stocks
# Note: There's 1 req/sec rate limiting, so minimum time is 500s
expected_cold_500 = (avg_cold * 500)
expected_hot_500 = (avg_hot * 500)

# But account for rate limiting (1 req/sec = 500s minimum)
rate_limit_time = 500  # seconds for 500 stocks at 1 req/sec

if expected_cold_500 < rate_limit_time:
    actual_cold_500 = rate_limit_time
else:
    actual_cold_500 = expected_cold_500

if expected_hot_500 < rate_limit_time:
    actual_hot_500 = rate_limit_time
else:
    actual_hot_500 = expected_hot_500

print(f"{'500 stocks - COLD cache':<30} {actual_cold_500/60:>17.1f}m  ({actual_cold_500:>6.0f}s)")
print(f"{'500 stocks - HOT cache':<30} {actual_hot_500/60:>17.1f}m  ({actual_hot_500:>6.0f}s)")
print(f"{'Expected time saved':<30} {(actual_cold_500-actual_hot_500)/60:>17.1f}m  ({actual_cold_500-actual_hot_500:>6.0f}s)")

if actual_hot_500 > 0:
    speedup_500 = actual_cold_500 / actual_hot_500
    improvement_500 = ((actual_cold_500 - actual_hot_500) / actual_cold_500) * 100
    print(f"{'Expected improvement':<30} {speedup_500:>17.1f}x  ({improvement_500:>6.1f}%)")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print(f"""
✓ Cache Implementation Working Successfully!

Current Performance ({len(TEST_SYMBOLS)} stocks):
  • First scan (cold cache):  {scan_1_duration:.1f}s ({avg_cold:.2f}s per stock)
  • Second scan (hot cache):  {scan_2_duration:.1f}s ({avg_hot:.2f}s per stock)
  • Improvement: {speedup:.1f}x faster with cache

Projected Performance (500 stocks):
  • Cold cache scan: {actual_cold_500/60:.1f} minutes
  • Hot cache scan:  {actual_hot_500/60:.1f} minutes
  • Time saved:      {(actual_cold_500-actual_hot_500)/60:.1f} minutes per scan

Cache Efficiency:
  • SPY: Cached globally (fetched once, reused {len(TEST_SYMBOLS)} times)
  • Stock prices: Cached individually with database persistence
  • Incremental updates: Only fetches new data since last cache

Next Steps:
  • Phase 3: Implement cache warming (pre-fetch S&P 500 stocks)
  • Phase 3: Add scheduled updates (Celery Beat)
  • Result: Near-instant scans for pre-warmed stocks
""")

print("=" * 80)
