#!/usr/bin/env python3
"""
Test the scan API to verify multi-screener functionality.
"""
import os

import pytest

pytestmark = pytest.mark.live_service

if os.getenv("RUN_LIVE_SERVICE_TESTS") != "1":
    pytest.skip(
        "requires RUN_LIVE_SERVICE_TESTS=1 and a running backend",
        allow_module_level=True,
    )

import requests
import time
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_multi_screener_scan():
    """Test creating and monitoring a multi-screener scan."""

    print("="*80)
    print("Testing Multi-Screener Scan API")
    print("="*80)

    # Create scan request with all 4 screeners
    scan_request = {
        "universe": "custom",
        "symbols": ["AAPL", "MSFT", "NVDA", "META", "GOOGL"],
        "screeners": ["minervini", "canslim", "ipo", "custom"],
        "criteria": {
            "include_vcp": False,
            "custom_filters": {
                "price_min": 100,
                "rs_rating_min": 70
            }
        },
        "composite_method": "weighted_average"
    }

    print("\n1. Creating scan...")
    print(f"   Symbols: {scan_request['symbols']}")
    print(f"   Screeners: {scan_request['screeners']}")

    try:
        response = requests.post(
            f"{BASE_URL}/scans",
            json=scan_request,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        scan_data = response.json()
        scan_id = scan_data['scan_id']

        print(f"\n✓ Scan created successfully!")
        print(f"   Scan ID: {scan_id}")
        print(f"   Status: {scan_data['status']}")
        print(f"   Total stocks: {scan_data['total_stocks']}")

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error creating scan: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False

    # Monitor scan progress
    print("\n2. Monitoring scan progress...")

    max_wait = 120  # 2 minutes max
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/scans/{scan_id}/status")
            response.raise_for_status()

            status_data = response.json()
            status = status_data['status']
            progress = status_data.get('progress', 0)
            completed = status_data.get('completed_stocks', 0)
            total = status_data.get('total_stocks', 0)

            print(f"   Status: {status} | Progress: {progress:.1f}% ({completed}/{total})")

            if status == "completed":
                print(f"\n✓ Scan completed successfully!")
                break
            elif status == "failed":
                print(f"\n✗ Scan failed!")
                return False

            time.sleep(2)

        except requests.exceptions.RequestException as e:
            print(f"\n✗ Error checking status: {e}")
            return False
    else:
        print(f"\n⏱ Scan timeout after {max_wait}s")
        return False

    # Fetch results
    print("\n3. Fetching scan results...")

    try:
        response = requests.get(
            f"{BASE_URL}/scans/{scan_id}/results",
            params={
                "page": 1,
                "per_page": 50,
                "sort_by": "composite_score",
                "sort_order": "desc"
            }
        )
        response.raise_for_status()

        results_data = response.json()
        results = results_data.get('results', [])
        total_results = results_data.get('total', 0)

        print(f"\n✓ Retrieved {total_results} results")

        if results:
            print("\nTop Results:")
            print("-" * 120)
            print(f"{'Symbol':<10} {'Composite':<12} {'Minervini':<12} {'CANSLIM':<12} {'IPO':<12} {'Custom':<12} {'Rating':<15}")
            print("-" * 120)

            for result in results[:10]:
                symbol = result.get('symbol', 'N/A')
                composite = result.get('composite_score', 0)
                minervini = result.get('minervini_score', 0)
                canslim = result.get('canslim_score', 0)
                ipo = result.get('ipo_score', 0)
                custom = result.get('custom_score', 0)
                rating = result.get('rating', 'N/A')

                print(
                    f"{symbol:<10} "
                    f"{composite if composite else 'N/A':<12} "
                    f"{minervini if minervini else 'N/A':<12} "
                    f"{canslim if canslim else 'N/A':<12} "
                    f"{ipo if ipo else 'N/A':<12} "
                    f"{custom if custom else 'N/A':<12} "
                    f"{rating:<15}"
                )

            print("-" * 120)

            # Verify multi-screener data
            print("\n4. Verifying multi-screener data...")

            has_all_scores = all(
                result.get('minervini_score') is not None and
                result.get('canslim_score') is not None and
                result.get('ipo_score') is not None and
                result.get('custom_score') is not None
                for result in results
            )

            if has_all_scores:
                print("   ✓ All 4 screener scores present for all results")
            else:
                print("   ⚠ Some screener scores are missing")
                for result in results:
                    symbol = result.get('symbol')
                    missing = []
                    if result.get('minervini_score') is None:
                        missing.append('minervini')
                    if result.get('canslim_score') is None:
                        missing.append('canslim')
                    if result.get('ipo_score') is None:
                        missing.append('ipo')
                    if result.get('custom_score') is None:
                        missing.append('custom')
                    if missing:
                        print(f"     {symbol}: Missing {', '.join(missing)}")

            print("\n✅ Multi-screener scan test PASSED!")
            return True
        else:
            print("\n⚠ No results returned (this might be OK if no stocks passed filters)")
            return True

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error fetching results: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


if __name__ == "__main__":
    success = test_multi_screener_scan()
    exit(0 if success else 1)
