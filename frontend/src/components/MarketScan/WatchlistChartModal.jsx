/**
 * Watchlist Chart Modal
 *
 * Full-screen modal for viewing stock charts from watchlists with keyboard navigation.
 * Simplified version of ChartViewerModal that takes a symbols array directly.
 *
 * Performance optimized: Uses consolidated /chart-data endpoint that prioritizes
 * data from recent scans (fast DB lookup) over computing data (slow API calls).
 */
import { useEffect, useState } from 'react';
import {
  Modal,
  Box,
  Typography,
  IconButton,
  CircularProgress,
  Chip,
  Fade,
  Button,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import KeyboardIcon from '@mui/icons-material/Keyboard';
import PeopleIcon from '@mui/icons-material/People';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchPriceHistory, priceHistoryKeys } from '../../api/priceHistory';
import { getChartData, getStockFundamentals } from '../../api/stocks';
import { getGroupDetail } from '../../api/groups';
import { useChartNavigation } from '../../hooks/useChartNavigation';
import CandlestickChart from '../Charts/CandlestickChart';
import StockMetricsSidebar from '../Scan/StockMetricsSidebar';
import PeerComparisonModal from '../Scan/PeerComparisonModal';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';

const inferMarketFromSymbol = (symbol) => {
  const normalized = String(symbol || '').toUpperCase();
  if (normalized.endsWith('.HK')) return 'HK';
  if (normalized.endsWith('.NS') || normalized.endsWith('.BO')) return 'IN';
  if (normalized.endsWith('.T')) return 'JP';
  if (normalized.endsWith('.KS') || normalized.endsWith('.KQ')) return 'KR';
  if (normalized.endsWith('.TW') || normalized.endsWith('.TWO')) return 'TW';
  if (normalized.endsWith('.SS') || normalized.endsWith('.SZ') || normalized.endsWith('.BJ')) return 'CN';
  if (normalized.endsWith('.DE') || normalized.endsWith('.F')) return 'DE';
  return 'US';
};

// Get color for group rank indicator based on thresholds
const getGroupRankColor = (rank) => {
  if (rank == null) return 'grey.500';
  if (rank <= 20) return 'success.main';   // Top ~10%
  if (rank >= 177) return 'error.main';    // Bottom ~10%
  return 'warning.main';                    // Middle
};

/**
 * Full-screen modal for viewing watchlist stock charts with keyboard navigation
 *
 * @param {Object} props
 * @param {boolean} props.open - Whether modal is open
 * @param {Function} props.onClose - Close handler
 * @param {string} props.initialSymbol - Symbol to start with
 * @param {Array<string>} props.symbols - Array of symbols to navigate through
 */
function WatchlistChartModal({ open, onClose, initialSymbol, symbols = [] }) {
  const queryClient = useQueryClient();
  const [visibleRange, setVisibleRange] = useState(null); // Persist zoom across symbol navigation
  const [peerModalOpen, setPeerModalOpen] = useState(false);

  // Use navigation hook
  const { currentIndex, currentSymbol, totalCount, goNext, goPrevious } = useChartNavigation(
    symbols,
    initialSymbol,
    open
  );

  // Fetch consolidated chart data (RS, industry, VCP, growth metrics, etc.)
  // This endpoint prioritizes scan_results lookup (fast) over computation (slow)
  const { data: chartData, isLoading: isChartDataLoading } = useQuery({
    queryKey: ['chartData', currentSymbol],
    queryFn: () => getChartData(currentSymbol),
    enabled: open && !!currentSymbol,
    staleTime: 300000, // 5 minutes
    retry: 2,
  });

  // Fetch fundamentals for sidebar (description, additional metrics)
  const { data: fundamentals } = useQuery({
    queryKey: ['fundamentals', currentSymbol],
    queryFn: () => getStockFundamentals(currentSymbol),
    enabled: open && !!currentSymbol,
    staleTime: 300000,
  });

  // Get industry group from chart data (no waterfall dependency)
  const industryGroup = chartData?.ibd_industry_group;
  const groupMarket = chartData?.market || inferMarketFromSymbol(currentSymbol);
  const marketThemes = chartData?.market_themes || [];

  // Fetch group ranking - only if not already in chartData
  // When source is "scan_results", ibd_group_rank is included
  const needsGroupRankFetch = chartData && !chartData.ibd_group_rank && industryGroup;
  const {
    data: groupRankData,
    isLoading: isGroupRankLoading,
  } = useQuery({
    queryKey: ['groupRank', groupMarket, industryGroup],
    queryFn: () => getGroupDetail(industryGroup, 1, groupMarket),
    enabled: open && !!needsGroupRankFetch,
    staleTime: 300000,
    retry: false,
  });

  // Use chartData directly as stockData (already in correct format)
  const stockData = chartData;

  // Handle keyboard events
  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e) => {
      // Escape to close
      if (e.key === 'Escape') {
        onClose();
        return;
      }

      // Space to navigate
      if (e.key === ' ') {
        e.preventDefault();
        if (e.shiftKey) {
          goPrevious();
        } else {
          goNext();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, goNext, goPrevious, onClose]);

  // Prefetch adjacent stocks for smooth navigation with staggered timing
  useEffect(() => {
    if (!open || !currentSymbol || !symbols || symbols.length === 0) return;

    const prefetchSymbol = (symbol) => {
      // Prefetch price history
      queryClient.prefetchQuery({
        queryKey: priceHistoryKeys.symbol(symbol, '6mo'),
        queryFn: () => fetchPriceHistory(symbol, '6mo'),
      });
      // Prefetch chart data (consolidated endpoint)
      queryClient.prefetchQuery({
        queryKey: ['chartData', symbol],
        queryFn: () => getChartData(symbol),
        staleTime: 300000,
      });
      // Prefetch fundamentals
      queryClient.prefetchQuery({
        queryKey: ['fundamentals', symbol],
        queryFn: () => getStockFundamentals(symbol),
        staleTime: 300000,
      });
    };

    // Clear any pending timeouts from previous renders
    const timeouts = [];

    // Get next 5 and previous 5 stocks
    const nextSymbols = symbols.slice(currentIndex + 1, currentIndex + 6);
    const prevSymbols = symbols.slice(Math.max(0, currentIndex - 5), currentIndex).reverse();

    // High priority: Immediate next/prev (fetch now)
    if (nextSymbols[0]) prefetchSymbol(nextSymbols[0]);
    if (prevSymbols[0]) prefetchSymbol(prevSymbols[0]);

    // Medium priority: Next 2-5 stocks (staggered 100ms apart)
    nextSymbols.slice(1).forEach((symbol, idx) => {
      const timeout = setTimeout(() => prefetchSymbol(symbol), (idx + 1) * 100);
      timeouts.push(timeout);
    });

    // Low priority: Previous 2-5 stocks (staggered after next, starting at 500ms)
    prevSymbols.slice(1).forEach((symbol, idx) => {
      const timeout = setTimeout(() => prefetchSymbol(symbol), 500 + (idx + 1) * 100);
      timeouts.push(timeout);
    });

    // Cleanup timeouts on unmount or dependency change
    return () => {
      timeouts.forEach(clearTimeout);
    };
  }, [open, currentSymbol, currentIndex, symbols, queryClient]);

  const chartHeight = window.innerHeight - 60; // Full height minus header

  return (
  <>
    <Modal
      open={open}
      onClose={onClose}
      aria-labelledby="watchlist-chart-modal"
      closeAfterTransition
    >
      <Fade in={open}>
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            bgcolor: 'background.paper',
            display: 'flex',
            flexDirection: 'column',
            outline: 'none',
          }}
        >
          {/* Header */}
          <Box
            sx={{
              height: 60,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 3,
              borderBottom: 1,
              borderColor: 'divider',
              bgcolor: 'background.default',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h5" fontWeight="bold">
                {currentSymbol || 'Loading...'}
              </Typography>

              {/* Group Rank Box */}
              {industryGroup && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Box sx={{
                    borderRadius: 1,
                    px: 1.5,
                    py: 0.5,
                    textAlign: 'center',
                    minWidth: 36,
                    bgcolor: (isChartDataLoading || isGroupRankLoading) ? 'grey.400' : getGroupRankColor(chartData?.ibd_group_rank ?? groupRankData?.current_rank),
                  }}>
                    {(isChartDataLoading || (needsGroupRankFetch && isGroupRankLoading)) ? (
                      <CircularProgress size={14} sx={{ color: 'white' }} />
                    ) : (
                      <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
                        {chartData?.ibd_group_rank ?? groupRankData?.current_rank ?? '-'}
                      </Typography>
                    )}
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                    Grp Rnk
                  </Typography>
                </Box>
              )}

              {/* ADR Box */}
              {stockData?.adr_percent != null && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Box sx={{
                    borderRadius: 1,
                    px: 1.5,
                    py: 0.5,
                    textAlign: 'center',
                    bgcolor: stockData.adr_percent >= 4
                      ? 'success.main'
                      : stockData.adr_percent >= 2
                        ? 'warning.main'
                        : 'error.main',
                  }}>
                    <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
                      {stockData.adr_percent.toFixed(1)}%
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                    ADR
                  </Typography>
                </Box>
              )}

              {/* EPS Rating Box */}
              {stockData?.eps_rating != null && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Box sx={{
                    borderRadius: 1,
                    px: 1.5,
                    py: 0.5,
                    textAlign: 'center',
                    minWidth: 36,
                    bgcolor: stockData.eps_rating >= 80
                      ? 'success.main'
                      : stockData.eps_rating >= 50
                        ? 'warning.main'
                        : 'error.main',
                  }}>
                    <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
                      {stockData.eps_rating}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                    EPS Rtg
                  </Typography>
                </Box>
              )}

              {/* Industry Info Boxes */}
              {stockData && (
                <Box sx={{ display: 'flex', gap: 1.5, ml: 1 }}>
                  {/* IBD Industry Group */}
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Box sx={{
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      minWidth: 80,
                      maxWidth: 180,
                      textAlign: 'center',
                      bgcolor: 'background.paper'
                    }}>
                      <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem' }}>
                        {stockData.ibd_industry_group || '-'}
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                      Group
                    </Typography>
                  </Box>

                  {/* Sector */}
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Box sx={{
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      minWidth: 80,
                      maxWidth: 180,
                      textAlign: 'center',
                      bgcolor: 'background.paper'
                    }}>
                      <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem' }}>
                        {stockData.gics_sector || '-'}
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                      Sector
                    </Typography>
                  </Box>

                  {/* Industry */}
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Box sx={{
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      minWidth: 80,
                      maxWidth: 180,
                      textAlign: 'center',
                      bgcolor: 'background.paper'
                    }}>
                      <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem' }}>
                        {stockData.gics_industry || '-'}
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                      Industry
                    </Typography>
                  </Box>
                </Box>
              )}

              {marketThemes.length > 0 && (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, maxWidth: 240 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                    Market Themes
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {marketThemes.map((theme) => (
                      <Chip key={theme} label={theme} size="small" variant="outlined" sx={{ height: 20 }} />
                    ))}
                  </Box>
                </Box>
              )}
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {currentSymbol && (
                <AddToWatchlistMenu symbols={currentSymbol} size="medium" />
              )}
              {currentSymbol && stockData?.ibd_industry_group && (
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<PeopleIcon />}
                  onClick={() => setPeerModalOpen(true)}
                  sx={{ textTransform: 'none' }}
                >
                  View Peers
                </Button>
              )}
              {totalCount > 0 && (
                <Typography variant="body2" color="text.secondary">
                  Stock {currentIndex + 1} of {totalCount}
                </Typography>
              )}
              <IconButton onClick={onClose} size="large">
                <CloseIcon />
              </IconButton>
            </Box>
          </Box>

          {/* Main Content */}
          <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
            {/* Sidebar */}
            <StockMetricsSidebar
              stockData={stockData}
              fundamentals={fundamentals}
            />

            {/* Chart Area */}
            <Box sx={{ flex: 1, overflow: 'hidden', bgcolor: 'background.paper' }}>
              {currentSymbol ? (
                <CandlestickChart
                  symbol={currentSymbol}
                  period="6mo"
                  height={chartHeight}
                  visibleRange={visibleRange}
                  onVisibleRangeChange={setVisibleRange}
                />
              ) : (
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: chartHeight,
                  }}
                >
                  <CircularProgress size={60} />
                </Box>
              )}
            </Box>
          </Box>

          {/* Footer - Keyboard Hints */}
          <Box
            sx={{
              position: 'fixed',
              bottom: 0,
              left: 0,
              right: 0,
              p: 1.5,
              bgcolor: 'background.paper',
              borderTop: 1,
              borderColor: 'divider',
              display: 'flex',
              justifyContent: 'center',
              gap: 3,
            }}
          >
            <Chip icon={<KeyboardIcon />} label="Space: Next Stock" size="small" variant="outlined" />
            <Chip icon={<KeyboardIcon />} label="Shift+Space: Previous" size="small" variant="outlined" />
            <Chip icon={<KeyboardIcon />} label="Esc: Close" size="small" variant="outlined" />
          </Box>
        </Box>
      </Fade>
    </Modal>

    {/* Peer Comparison Modal */}
    <PeerComparisonModal
      open={peerModalOpen}
      onClose={() => setPeerModalOpen(false)}
      symbol={currentSymbol}
      onOpenChart={() => setPeerModalOpen(false)}
    />
  </>
  );
}

export default WatchlistChartModal;
