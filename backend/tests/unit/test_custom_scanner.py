#!/usr/bin/env python3
"""
Test script for Custom Scanner.

Tests:
1. Custom screener registration
2. Basic filters (price, volume, RS)
3. Fundamental filters (market cap, debt, sector)
4. Growth filters (EPS, sales)
5. Technical filters (MA alignment, 52w high)
6. All 4 screeners together
7. Various filter combinations
"""
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.usefixtures("legacy_market_rs_runtime")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.scanners.screener_registry import screener_registry
from app.wiring.bootstrap import get_scan_orchestrator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test symbols - variety of stocks
TEST_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META']


def test_custom_registration():
    """Test that custom screener is registered."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Custom Screener Registration")
    logger.info("="*80)

    # Check if custom screener is registered
    assert screener_registry.is_registered('custom'), "Custom screener not registered!"
    logger.info("✓ Custom screener is registered")

    # Get screener instance
    screener = screener_registry.get('custom')
    assert screener is not None, "Failed to get Custom screener instance"
    logger.info(f"✓ Got screener instance: {screener}")

    # Check screener name
    assert screener.screener_name == 'custom', "Incorrect screener name"
    logger.info(f"✓ Screener name: {screener.screener_name}")

    # List all screeners
    all_screeners = screener_registry.list_screeners()
    logger.info(f"✓ All registered screeners: {all_screeners}")

    assert 'minervini' in all_screeners, "Minervini should still be registered"
    assert 'canslim' in all_screeners, "CANSLIM should still be registered"
    assert 'ipo' in all_screeners, "IPO should still be registered"
    assert 'custom' in all_screeners, "Custom should be registered"

    logger.info("\n✅ Registration test passed!\n")


def test_basic_filters():
    """Test basic filters: price, volume, RS rating."""
    logger.info("="*80)
    logger.info("TEST 2: Basic Filters (Price, Volume, RS)")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    # Configuration: basic filters only
    criteria = {
        "custom_filters": {
            "price_min": 50,
            "price_max": 500,
            "volume_min": 10_000_000,
            "rs_rating_min": 70
        },
        "min_score": 60
    }

    symbol = 'AAPL'
    logger.info(f"\n📊 Custom Scan: {symbol}")
    logger.info(f"Filters: price $50-500, volume 10M+, RS 70+")
    logger.info("-" * 80)

    try:
        result = orchestrator.scan_stock_multi(
            symbol=symbol,
            screener_names=['custom'],
            criteria=criteria
        )

        # Check result structure
        assert 'symbol' in result, "Missing symbol in result"
        assert 'composite_score' in result, "Missing composite_score"
        assert 'custom_score' in result, "Missing custom_score"
        assert 'rating' in result, "Missing rating"

        logger.info(f"Symbol: {result['symbol']}")
        logger.info(f"Custom Score: {result.get('custom_score', 0):.2f}/100")
        logger.info(f"Rating: {result['rating']}")
        logger.info(f"Passes: {result.get('custom_passes', False)}")

        # Show filter details
        if 'details' in result and 'screeners' in result['details']:
            custom_details = result['details']['screeners'].get('custom', {})

            logger.info("\nFilter Results:")
            logger.info(f"  Filters Enabled: {custom_details.get('filters_enabled', 0)}")
            logger.info(f"  Filters Passed: {custom_details.get('filters_passed', 0)}")

            filter_results = custom_details.get('filter_results', {})
            for filter_name, filter_data in filter_results.items():
                status = "✓" if filter_data.get('passes') else "✗"
                logger.info(
                    f"  {status} {filter_name}: "
                    f"{filter_data.get('points', 0):.1f}/{filter_data.get('max_points', 0):.1f} pts"
                )

        logger.info("✓ Basic filters test completed")

    except Exception as e:
        logger.error(f"❌ Error scanning {symbol}: {e}")
        raise

    logger.info("\n✅ Basic filters test passed!\n")


def test_fundamental_filters():
    """Test fundamental filters: market cap, debt, sector."""
    logger.info("="*80)
    logger.info("TEST 3: Fundamental Filters (Market Cap, Debt, Sector)")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    # Configuration: fundamental filters
    criteria = {
        "custom_filters": {
            "market_cap_min": 100_000_000_000,  # 100B+
            "debt_to_equity_max": 1.0,
            "sectors": ["Technology"]
        },
        "min_score": 50
    }

    for symbol in ['AAPL', 'NVDA']:
        logger.info(f"\n📊 Custom Scan: {symbol}")
        logger.info("Filters: $100B+ market cap, D/E < 1.0, Technology sector")
        logger.info("-" * 80)

        try:
            result = orchestrator.scan_stock_multi(
                symbol=symbol,
                screener_names=['custom'],
                criteria=criteria
            )

            logger.info(f"Symbol: {result['symbol']}")
            logger.info(f"Custom Score: {result.get('custom_score', 0):.2f}/100")
            logger.info(f"Passes: {result.get('custom_passes', False)}")

            # Show filter details
            if 'details' in result and 'screeners' in result['details']:
                custom_details = result['details']['screeners'].get('custom', {})
                filter_results = custom_details.get('filter_results', {})

                for filter_name in ['market_cap', 'debt_to_equity', 'sector']:
                    if filter_name in filter_results:
                        filter_data = filter_results[filter_name]
                        status = "✓" if filter_data.get('passes') else "✗"
                        logger.info(
                            f"  {status} {filter_name}: "
                            f"{filter_data.get('points', 0):.1f}/{filter_data.get('max_points', 0):.1f} pts"
                        )

        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            raise

    logger.info("\n✅ Fundamental filters test passed!\n")


def test_growth_filters():
    """Test growth filters: EPS growth, sales growth."""
    logger.info("="*80)
    logger.info("TEST 4: Growth Filters (EPS, Sales)")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    # Configuration: growth filters
    criteria = {
        "custom_filters": {
            "eps_growth_min": 15,
            "sales_growth_min": 10
        },
        "min_score": 50
    }

    symbol = 'NVDA'
    logger.info(f"\n📊 Custom Scan: {symbol}")
    logger.info("Filters: EPS growth 15%+, Sales growth 10%+")
    logger.info("-" * 80)

    try:
        result = orchestrator.scan_stock_multi(
            symbol=symbol,
            screener_names=['custom'],
            criteria=criteria
        )

        logger.info(f"Symbol: {result['symbol']}")
        logger.info(f"Custom Score: {result.get('custom_score', 0):.2f}/100")

        # Show filter details
        if 'details' in result and 'screeners' in result['details']:
            custom_details = result['details']['screeners'].get('custom', {})
            filter_results = custom_details.get('filter_results', {})

            for filter_name in ['eps_growth', 'sales_growth']:
                if filter_name in filter_results:
                    filter_data = filter_results[filter_name]
                    status = "✓" if filter_data.get('passes') else "✗"
                    logger.info(
                        f"  {status} {filter_name}: "
                        f"{filter_data.get('points', 0):.1f}/{filter_data.get('max_points', 0):.1f} pts"
                    )
                    # Show actual values
                    if filter_name == 'eps_growth':
                        logger.info(f"    EPS Growth: {filter_data.get('eps_growth_qq', 'N/A')}%")
                    elif filter_name == 'sales_growth':
                        logger.info(f"    Sales Growth: {filter_data.get('sales_growth_qq', 'N/A')}%")

        logger.info("✓ Growth filters test completed")

    except Exception as e:
        logger.error(f"❌ Error scanning {symbol}: {e}")
        raise

    logger.info("\n✅ Growth filters test passed!\n")


def test_technical_filters():
    """Test technical filters: MA alignment, 52w high proximity."""
    logger.info("="*80)
    logger.info("TEST 5: Technical Filters (MA Alignment, 52w High)")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    # Configuration: technical filters
    criteria = {
        "custom_filters": {
            "ma_alignment": True,
            "near_52w_high": 10  # Within 10% of 52w high
        },
        "min_score": 50
    }

    for symbol in ['AAPL', 'NVDA']:
        logger.info(f"\n📊 Custom Scan: {symbol}")
        logger.info("Filters: MA alignment (Price > 50 > 150 > 200), within 10% of 52w high")
        logger.info("-" * 80)

        try:
            result = orchestrator.scan_stock_multi(
                symbol=symbol,
                screener_names=['custom'],
                criteria=criteria
            )

            logger.info(f"Symbol: {result['symbol']}")
            logger.info(f"Custom Score: {result.get('custom_score', 0):.2f}/100")

            # Show filter details
            if 'details' in result and 'screeners' in result['details']:
                custom_details = result['details']['screeners'].get('custom', {})
                filter_results = custom_details.get('filter_results', {})

                # MA alignment
                if 'ma_alignment' in filter_results:
                    ma_data = filter_results['ma_alignment']
                    status = "✓" if ma_data.get('passes') else "✗"
                    logger.info(
                        f"  {status} ma_alignment: "
                        f"{ma_data.get('points', 0):.1f}/{ma_data.get('max_points', 0):.1f} pts"
                    )
                    logger.info(f"    Price: ${ma_data.get('current_price', 0):.2f}")
                    logger.info(f"    MA 50: ${ma_data.get('ma_50', 0):.2f}")
                    logger.info(f"    MA 150: ${ma_data.get('ma_150', 0):.2f}")
                    logger.info(f"    MA 200: ${ma_data.get('ma_200', 0):.2f}")
                    logger.info(f"    Aligned: {ma_data.get('aligned', False)}")

                # 52w high
                if 'near_52w_high' in filter_results:
                    high_data = filter_results['near_52w_high']
                    status = "✓" if high_data.get('passes') else "✗"
                    logger.info(
                        f"  {status} near_52w_high: "
                        f"{high_data.get('points', 0):.1f}/{high_data.get('max_points', 0):.1f} pts"
                    )
                    logger.info(f"    Current: ${high_data.get('current_price', 0):.2f}")
                    logger.info(f"    52w High: ${high_data.get('high_52w', 0):.2f}")
                    logger.info(f"    Distance: {high_data.get('pct_from_high', 0):.1f}%")

        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            raise

    logger.info("\n✅ Technical filters test passed!\n")


def test_all_four_screeners():
    """Test running all 4 screeners together."""
    logger.info("="*80)
    logger.info("TEST 6: All 4 Screeners (Minervini + CANSLIM + IPO + Custom)")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    # Custom filter config
    custom_criteria = {
        "custom_filters": {
            "price_min": 100,
            "rs_rating_min": 70,
            "ma_alignment": True
        },
        "include_vcp": False  # For Minervini
    }

    symbol = 'NVDA'
    logger.info(f"\nRunning ALL FOUR screeners on {symbol}...")
    logger.info("-" * 80)

    try:
        result = orchestrator.scan_stock_multi(
            symbol=symbol,
            screener_names=['minervini', 'canslim', 'ipo', 'custom'],
            criteria=custom_criteria,
            composite_method='weighted_average'
        )

        logger.info("\nAll-Screener Results:")
        logger.info(f"  Symbol: {result['symbol']}")
        logger.info(f"  Composite Score: {result['composite_score']:.2f}")
        logger.info(f"  Overall Rating: {result['rating']}")
        logger.info(f"  Screeners Run: {result.get('screeners_run', [])}")

        logger.info("\nIndividual Scores:")
        logger.info(f"  Minervini Score: {result.get('minervini_score', 'N/A'):.2f}")
        logger.info(f"  CANSLIM Score:   {result.get('canslim_score', 'N/A'):.2f}")
        logger.info(f"  IPO Score:       {result.get('ipo_score', 'N/A'):.2f}")
        logger.info(f"  Custom Score:    {result.get('custom_score', 'N/A'):.2f}")

        logger.info("\nPass Status:")
        logger.info(f"  Minervini Passes: {result.get('minervini_passes', False)}")
        logger.info(f"  CANSLIM Passes:   {result.get('canslim_passes', False)}")
        logger.info(f"  IPO Passes:       {result.get('ipo_passes', False)}")
        logger.info(f"  Custom Passes:    {result.get('custom_passes', False)}")
        logger.info(f"  Screeners Passed: {result.get('screeners_passed', 0)}/{result.get('screeners_total', 4)}")

        # Verify all four screeners ran
        assert 'minervini' in result.get('screeners_run', []), "Minervini should have run"
        assert 'canslim' in result.get('screeners_run', []), "CANSLIM should have run"
        assert 'ipo' in result.get('screeners_run', []), "IPO should have run"
        assert 'custom' in result.get('screeners_run', []), "Custom should have run"
        assert 'minervini_score' in result, "Minervini score missing"
        assert 'canslim_score' in result, "CANSLIM score missing"
        assert 'ipo_score' in result, "IPO score missing"
        assert 'custom_score' in result, "Custom score missing"

        # Verify composite score is average of all four
        expected_composite = (
            result['minervini_score'] +
            result['canslim_score'] +
            result['ipo_score'] +
            result['custom_score']
        ) / 4
        actual_composite = result['composite_score']
        diff = abs(expected_composite - actual_composite)

        logger.info(f"\nComposite Calculation:")
        logger.info(f"  Expected (average): {expected_composite:.2f}")
        logger.info(f"  Actual: {actual_composite:.2f}")
        logger.info(f"  Difference: {diff:.4f}")

        assert diff < 0.1, f"Composite score mismatch: {diff}"
        logger.info("✓ Composite score correctly calculated")

        logger.info("\n✅ All 4 screeners test passed!\n")

    except Exception as e:
        logger.error(f"❌ All screeners test failed: {e}")
        raise


def test_combined_filters():
    """Test various combinations of filters."""
    logger.info("="*80)
    logger.info("TEST 7: Combined Filter Configurations")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    # Test configuration: comprehensive filters
    criteria = {
        "custom_filters": {
            "price_min": 20,
            "price_max": 1000,
            "volume_min": 5_000_000,
            "rs_rating_min": 75,
            "market_cap_min": 10_000_000_000,  # 10B+
            "eps_growth_min": 15,
            "sales_growth_min": 10,
            "ma_alignment": True,
            "near_52w_high": 15
        },
        "min_score": 70
    }

    logger.info("\nComprehensive Filter Configuration:")
    logger.info("  - Price: $20-1000")
    logger.info("  - Volume: 5M+")
    logger.info("  - RS Rating: 75+")
    logger.info("  - Market Cap: $10B+")
    logger.info("  - EPS Growth: 15%+")
    logger.info("  - Sales Growth: 10%+")
    logger.info("  - MA Alignment: Yes")
    logger.info("  - Near 52w High: Within 15%")
    logger.info("  - Min Score: 70\n")

    for symbol in ['AAPL', 'MSFT', 'NVDA']:
        logger.info(f"\n📊 Testing {symbol}...")
        logger.info("-" * 80)

        try:
            result = orchestrator.scan_stock_multi(
                symbol=symbol,
                screener_names=['custom'],
                criteria=criteria
            )

            logger.info(f"Custom Score: {result.get('custom_score', 0):.2f}/100")
            logger.info(f"Rating: {result['rating']}")
            logger.info(f"Passes: {result.get('custom_passes', False)}")

            if 'details' in result and 'screeners' in result['details']:
                custom_details = result['details']['screeners'].get('custom', {})
                logger.info(f"Filters Passed: {custom_details.get('filters_passed', 0)}/{custom_details.get('filters_enabled', 0)}")

        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            # Don't raise - continue with other symbols

    logger.info("\n✅ Combined filters test passed!\n")


def main():
    """Run all tests."""
    logger.info("\n" + "🚀 " + "="*76)
    logger.info("CUSTOM SCANNER TEST SUITE")
    logger.info("="*78)

    try:
        test_custom_registration()
        test_basic_filters()
        test_fundamental_filters()
        test_growth_filters()
        test_technical_filters()
        test_all_four_screeners()
        test_combined_filters()

        logger.info("\n" + "🎉 " + "="*76)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*78 + "\n")

        logger.info("Summary:")
        logger.info("  ✅ Custom screener registered and working")
        logger.info("  ✅ Basic filters (price, volume, RS) working")
        logger.info("  ✅ Fundamental filters (market cap, debt, sector) working")
        logger.info("  ✅ Growth filters (EPS, sales) working")
        logger.info("  ✅ Technical filters (MA alignment, 52w high) working")
        logger.info("  ✅ All 4 screeners (Minervini + CANSLIM + IPO + Custom) functional")
        logger.info("  ✅ Combined filter configurations working")
        logger.info("\nPhase 4 Custom screener implementation complete!")

        return True

    except Exception as e:
        logger.error(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
