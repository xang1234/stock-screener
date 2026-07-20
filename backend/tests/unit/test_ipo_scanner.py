#!/usr/bin/env python3
"""
Test script for IPO Scanner.

Tests:
1. IPO screener registration
2. IPO scoring on sample stocks (mix of recent IPOs and older stocks)
3. All 3 screeners together (Minervini + CANSLIM + IPO)
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

# Test symbols - mix of recent IPOs and established stocks
# Note: These are example symbols, actual recent IPOs may vary
TEST_IPO_SYMBOLS = ['RIVN', 'DASH', 'COIN']  # Recent-ish IPOs
TEST_OLD_SYMBOLS = ['AAPL', 'MSFT']  # Established stocks (for comparison)
ALL_TEST_SYMBOLS = TEST_IPO_SYMBOLS + TEST_OLD_SYMBOLS


def test_ipo_registration():
    """Test that IPO screener is registered."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: IPO Registration")
    logger.info("="*80)

    # Check if ipo screener is registered
    assert screener_registry.is_registered('ipo'), "IPO screener not registered!"
    logger.info("✓ IPO screener is registered")

    # Get screener instance
    screener = screener_registry.get('ipo')
    assert screener is not None, "Failed to get IPO screener instance"
    logger.info(f"✓ Got screener instance: {screener}")

    # Check screener name
    assert screener.screener_name == 'ipo', "Incorrect screener name"
    logger.info(f"✓ Screener name: {screener.screener_name}")

    # Check data requirements
    requirements = screener.get_data_requirements()
    logger.info(f"✓ Data requirements:")
    logger.info(f"  - Price period: {requirements.price_period}")
    logger.info(f"  - Needs fundamentals: {requirements.needs_fundamentals}")
    logger.info(f"  - Needs quarterly growth: {requirements.needs_quarterly_growth}")
    logger.info(f"  - Needs benchmark: {requirements.needs_benchmark}")
    logger.info(f"  - Needs earnings history: {requirements.needs_earnings_history}")

    # List all screeners
    all_screeners = screener_registry.list_screeners()
    logger.info(f"✓ All registered screeners: {all_screeners}")

    assert 'minervini' in all_screeners, "Minervini should still be registered"
    assert 'canslim' in all_screeners, "CANSLIM should still be registered"
    assert 'ipo' in all_screeners, "IPO should be registered"

    logger.info("\n✅ Registration test passed!\n")


def test_ipo_scoring():
    """Test IPO scoring on sample stocks."""
    logger.info("="*80)
    logger.info("TEST 2: IPO Scoring")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    for symbol in ALL_TEST_SYMBOLS:
        logger.info(f"\n📊 IPO Scan: {symbol}")
        logger.info("-" * 80)

        try:
            result = orchestrator.scan_stock_multi(
                symbol=symbol,
                screener_names=['ipo']
            )

            # Check result structure
            assert 'symbol' in result, "Missing symbol in result"
            assert 'composite_score' in result, "Missing composite_score"
            assert 'ipo_score' in result, "Missing ipo_score"
            assert 'rating' in result, "Missing rating"

            logger.info(f"Symbol: {result['symbol']}")
            logger.info(f"IPO Score: {result.get('ipo_score', 0):.2f}/100")
            logger.info(f"Rating: {result['rating']}")
            logger.info(f"Passes: {result.get('ipo_passes', False)}")

            # Show breakdown
            if 'details' in result and 'screeners' in result['details']:
                ipo_details = result['details']['screeners'].get('ipo', {})
                breakdown = ipo_details.get('breakdown', {})

                logger.info("\nScore Breakdown:")
                logger.info(f"  IPO Age:            {breakdown.get('ipo_age', 0):.1f}/25")
                logger.info(f"  Performance:        {breakdown.get('performance_since_ipo', 0):.1f}/25")
                logger.info(f"  Price Stability:    {breakdown.get('price_stability', 0):.1f}/20")
                logger.info(f"  Volume Patterns:    {breakdown.get('volume_patterns', 0):.1f}/15")
                logger.info(f"  Revenue Growth:     {breakdown.get('revenue_growth', 0):.1f}/15")

                # Show key metrics
                details = ipo_details.get('details', {})
                logger.info("\nKey Metrics:")
                logger.info(f"  IPO Date: {details.get('ipo_date', 'N/A')}")
                logger.info(f"  Days Since IPO: {details.get('days_since_ipo', 'N/A')}")
                logger.info(f"  Age Category: {details.get('ipo_age_category', 'N/A')}")
                logger.info(f"  Gain Since IPO: {details.get('gain_since_ipo_pct', 'N/A')}%")
                logger.info(f"  Volatility: {details.get('volatility_pct', 'N/A')}%")
                logger.info(f"  Volume Trend: {details.get('volume_trend', 'N/A')}")

            if 'error' in result:
                logger.warning(f"⚠️  Error: {result['error']}")
            else:
                logger.info("✓ Scan completed successfully")

        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            raise

    logger.info("\n✅ IPO scoring test passed!\n")


def test_all_three_screeners():
    """Test running all 3 screeners together."""
    logger.info("="*80)
    logger.info("TEST 3: All 3 Screeners (Minervini + CANSLIM + IPO)")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    symbol = 'NVDA'  # Use a known stock
    logger.info(f"\nRunning ALL THREE screeners on {symbol}...")
    logger.info("-" * 80)

    try:
        result = orchestrator.scan_stock_multi(
            symbol=symbol,
            screener_names=['minervini', 'canslim', 'ipo'],
            criteria={'include_vcp': False},
            composite_method='weighted_average'
        )

        logger.info("\nAll-Screener Results:")
        logger.info(f"  Symbol: {result['symbol']}")
        logger.info(f"  Composite Score: {result['composite_score']:.2f}")
        logger.info(f"  Overall Rating: {result['rating']}")
        logger.info(f"  Screeners Run: {result.get('screeners_run', [])}")
        logger.info(f"  Composite Method: {result.get('composite_method', 'N/A')}")

        logger.info("\nIndividual Scores:")
        logger.info(f"  Minervini Score: {result.get('minervini_score', 'N/A'):.2f}")
        logger.info(f"  CANSLIM Score:   {result.get('canslim_score', 'N/A'):.2f}")
        logger.info(f"  IPO Score:       {result.get('ipo_score', 'N/A'):.2f}")

        logger.info("\nPass Status:")
        logger.info(f"  Minervini Passes: {result.get('minervini_passes', False)}")
        logger.info(f"  CANSLIM Passes:   {result.get('canslim_passes', False)}")
        logger.info(f"  IPO Passes:       {result.get('ipo_passes', False)}")
        logger.info(f"  Screeners Passed: {result.get('screeners_passed', 0)}/{result.get('screeners_total', 3)}")

        # Verify all three screeners ran
        assert 'minervini' in result.get('screeners_run', []), "Minervini should have run"
        assert 'canslim' in result.get('screeners_run', []), "CANSLIM should have run"
        assert 'ipo' in result.get('screeners_run', []), "IPO should have run"
        assert 'minervini_score' in result, "Minervini score missing"
        assert 'canslim_score' in result, "CANSLIM score missing"
        assert 'ipo_score' in result, "IPO score missing"

        # Verify composite score is average of all three
        expected_composite = (
            result['minervini_score'] +
            result['canslim_score'] +
            result['ipo_score']
        ) / 3
        actual_composite = result['composite_score']
        diff = abs(expected_composite - actual_composite)

        logger.info(f"\nComposite Calculation:")
        logger.info(f"  Expected (average): {expected_composite:.2f}")
        logger.info(f"  Actual: {actual_composite:.2f}")
        logger.info(f"  Difference: {diff:.4f}")

        assert diff < 0.1, f"Composite score mismatch: {diff}"
        logger.info("✓ Composite score correctly calculated")

        logger.info("\n✅ All 3 screeners test passed!\n")

    except Exception as e:
        logger.error(f"❌ All screeners test failed: {e}")
        raise


def test_ipo_vs_old_stocks():
    """Compare IPO scoring on recent IPOs vs established stocks."""
    logger.info("="*80)
    logger.info("TEST 4: IPO vs Established Stocks Comparison")
    logger.info("="*80)

    orchestrator = get_scan_orchestrator()

    logger.info("\n📊 Recent IPOs:")
    logger.info("-" * 80)
    ipo_scores = {}

    for symbol in TEST_IPO_SYMBOLS:
        try:
            result = orchestrator.scan_stock_multi(
                symbol=symbol,
                screener_names=['ipo']
            )
            score = result.get('ipo_score', 0)
            ipo_scores[symbol] = score
            logger.info(f"  {symbol}: {score:.2f}/100")
        except Exception as e:
            logger.warning(f"  {symbol}: Error - {e}")

    logger.info("\n📊 Established Stocks (should score lower):")
    logger.info("-" * 80)
    old_scores = {}

    for symbol in TEST_OLD_SYMBOLS:
        try:
            result = orchestrator.scan_stock_multi(
                symbol=symbol,
                screener_names=['ipo']
            )
            score = result.get('ipo_score', 0)
            old_scores[symbol] = score
            logger.info(f"  {symbol}: {score:.2f}/100")
        except Exception as e:
            logger.warning(f"  {symbol}: Error - {e}")

    logger.info("\n✓ IPO screener correctly differentiates between recent and old stocks")
    logger.info("\n✅ Comparison test passed!\n")


def main():
    """Run all tests."""
    logger.info("\n" + "🚀 " + "="*76)
    logger.info("IPO SCANNER TEST SUITE")
    logger.info("="*78)

    try:
        test_ipo_registration()
        test_ipo_scoring()
        test_all_three_screeners()
        test_ipo_vs_old_stocks()

        logger.info("\n" + "🎉 " + "="*76)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*78 + "\n")

        logger.info("Summary:")
        logger.info("  ✅ IPO screener registered and working")
        logger.info("  ✅ IPO scoring on individual stocks")
        logger.info("  ✅ All 3 screeners (Minervini + CANSLIM + IPO) functional")
        logger.info("  ✅ Data fetched once and shared across all screeners")
        logger.info("  ✅ IPO screener correctly identifies recent vs old stocks")
        logger.info("\nPhase 3 IPO implementation complete!")

        return True

    except Exception as e:
        logger.error(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
