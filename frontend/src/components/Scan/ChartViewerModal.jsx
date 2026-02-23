import { useEffect, useMemo, useRef, useState } from 'react';
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
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getAllFilteredSymbols, getSingleResult } from '../../api/scans';
import { fetchPriceHistory, priceHistoryKeys } from '../../api/priceHistory';
import { getStockFundamentals } from '../../api/stocks';
import { getGroupDetail } from '../../api/groups';
import { useChartNavigation } from '../../hooks/useChartNavigation';
import { buildFilterParams, getStableFilterKey } from '../../utils/filterUtils';
import CandlestickChart from '../Charts/CandlestickChart';
import StockMetricsSidebar from './StockMetricsSidebar';
import PeerComparisonModal from './PeerComparisonModal';
import SetupEngineDrawer from './SetupEngineDrawer';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import { getGroupRankColor } from '../../utils/colorUtils';

/**
 * Full-screen modal for viewing stock charts with keyboard navigation
 *
 * @param {Object} props
 * @param {boolean} props.open - Whether modal is open
 * @param {Function} props.onClose - Close handler
 * @param {string} props.initialSymbol - Symbol to start with
 * @param {string} props.scanId - Scan ID for fetching results
 * @param {Object} props.filters - Current filter state object (from ScanPage)
 * @param {string} props.sortBy - Current sort field
 * @param {string} props.sortOrder - Current sort order
 * @param {Array} props.currentPageResults - Results from current page (for quick lookup)
 */
function ChartViewerModal({
  open,
  onClose,
  initialSymbol,
  scanId,
  filters = {},
  sortBy = 'composite_score',
  sortOrder = 'desc',
  currentPageResults = [],
}) {
  const queryClient = useQueryClient();
  const [peerModalOpen, setPeerModalOpen] = useState(false);
  const [setupDrawerOpen, setSetupDrawerOpen] = useState(false);
  const [visibleRange, setVisibleRange] = useState(null); // Persist zoom across symbol navigation
  const [descriptionExpanded, setDescriptionExpanded] = useState(false);

  // Build API params from filter state
  const filterParams = useMemo(
    () => buildFilterParams(filters, { sortBy, sortOrder }),
    [filters, sortBy, sortOrder]
  );

  // Generate stable cache key for filters
  const filterCacheKey = useMemo(() => getStableFilterKey(filters), [filters]);

  // Invalidate cached symbol list when sort changes so navigation always
  // reflects the current table order — even if the modal is currently closed.
  const prevSortRef = useRef({ sortBy, sortOrder });
  useEffect(() => {
    const prev = prevSortRef.current;
    if (prev.sortBy !== sortBy || prev.sortOrder !== sortOrder) {
      queryClient.removeQueries({ queryKey: ['allFilteredSymbols'] });
      prevSortRef.current = { sortBy, sortOrder };
    }
  }, [sortBy, sortOrder, queryClient]);

  // Fetch all filtered symbols for navigation (sort-aware)
  const { data: allSymbols, isLoading: symbolsLoading } = useQuery({
    queryKey: ['allFilteredSymbols', scanId, filterCacheKey, sortBy, sortOrder],
    queryFn: () => getAllFilteredSymbols(scanId, filterParams),
    enabled: open && !!scanId,
    staleTime: 5000, // 5 seconds — short enough to catch mid-session sort changes
    refetchOnMount: 'always', // Always refetch when modal opens to pick up sort changes
  });

  // Use navigation hook
  const { currentIndex, currentSymbol, totalCount, goNext, goPrevious, goToIndex } = useChartNavigation(
    allSymbols,
    initialSymbol,
    open
  );

  // Navigate to a specific symbol
  const handleNavigateToSymbol = (symbol) => {
    if (!allSymbols) return;
    const index = allSymbols.indexOf(symbol);
    if (index >= 0) {
      goToIndex(index);
    }
  };

  // Get stock data for current symbol
  const stockData = useMemo(() => {
    if (!currentSymbol || !currentPageResults) return null;
    return currentPageResults.find((r) => r.symbol === currentSymbol);
  }, [currentSymbol, currentPageResults]);

  // Fetch stock result if not in current page (using optimized single-stock endpoint)
  const { data: fetchedStockData } = useQuery({
    queryKey: ['stockResult', scanId, currentSymbol],
    queryFn: () => getSingleResult(scanId, currentSymbol),
    enabled: open && !!scanId && !!currentSymbol && !stockData,
    staleTime: 300000, // 5 minutes
  });

  const finalStockData = stockData || fetchedStockData;

  // Fetch fundamentals for current symbol
  const { data: fundamentals } = useQuery({
    queryKey: ['fundamentals', currentSymbol],
    queryFn: () => getStockFundamentals(currentSymbol),
    enabled: open && !!currentSymbol,
    staleTime: 300000, // 5 minutes (data is cached 7 days server-side)
  });

  // Fetch group ranking for current stock's industry group
  const industryGroup = finalStockData?.ibd_industry_group;
  const {
    data: groupRankData,
    isLoading: isGroupRankLoading,
  } = useQuery({
    queryKey: ['groupRank', industryGroup],
    queryFn: () => getGroupDetail(industryGroup, 1),
    enabled: open && !!industryGroup,
    staleTime: 300000, // 5 minutes
    retry: false,
  });

  // Handle keyboard events
  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e) => {
      // Escape: close drawer first, then modal (always active, even in inputs)
      if (e.key === 'Escape') {
        if (setupDrawerOpen) {
          setSetupDrawerOpen(false);
          return;
        }
        onClose();
        return;
      }

      // Skip navigation shortcuts when user is typing in an input/textarea
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable) {
        return;
      }

      // D to toggle setup drawer (only when explain data exists)
      if (e.key === 'd' || e.key === 'D') {
        if (finalStockData?.se_explain) {
          setSetupDrawerOpen((prev) => !prev);
        }
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
  }, [open, goNext, goPrevious, onClose, setupDrawerOpen, finalStockData]);

  // Reset description expansion and setup drawer when symbol changes
  useEffect(() => {
    setDescriptionExpanded(false);
    setSetupDrawerOpen(false);
  }, [currentSymbol]);

  // Prefetch adjacent stocks for smooth navigation with staggered timing
  useEffect(() => {
    if (!open || !currentSymbol || !allSymbols) return;

    const prefetchSymbol = (symbol) => {
      // Prefetch price history
      queryClient.prefetchQuery({
        queryKey: priceHistoryKeys.symbol(symbol, '6mo'),
        queryFn: () => fetchPriceHistory(symbol, '6mo'),
      });
      // Prefetch fundamentals
      queryClient.prefetchQuery({
        queryKey: ['fundamentals', symbol],
        queryFn: () => getStockFundamentals(symbol),
        staleTime: 300000,
      });
      // Prefetch stock result
      if (scanId) {
        queryClient.prefetchQuery({
          queryKey: ['stockResult', scanId, symbol],
          queryFn: () => getSingleResult(scanId, symbol),
          staleTime: 300000,
        });
      }
    };

    // Clear any pending timeouts from previous renders
    const timeouts = [];

    // Get next 5 and previous 5 stocks
    const nextSymbols = allSymbols.slice(currentIndex + 1, currentIndex + 6);
    const prevSymbols = allSymbols.slice(Math.max(0, currentIndex - 5), currentIndex).reverse();

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
  }, [open, currentSymbol, currentIndex, allSymbols, queryClient, scanId]);

  const chartHeight = window.innerHeight - 60; // Full height minus header

  return (
    <>
      <Modal
        open={open}
        onClose={onClose}
        aria-labelledby="chart-viewer-modal"
        closeAfterTransition
        disableEscapeKeyDown
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
                {symbolsLoading && <CircularProgress size={20} />}

                {/* Group Rank Box */}
                {industryGroup && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Box sx={{
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      textAlign: 'center',
                      minWidth: 36,
                      bgcolor: isGroupRankLoading ? 'grey.400' : getGroupRankColor(groupRankData?.current_rank),
                    }}>
                      {isGroupRankLoading ? (
                        <CircularProgress size={14} sx={{ color: 'white' }} />
                      ) : (
                        <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
                          {groupRankData?.current_rank ?? '-'}
                        </Typography>
                      )}
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                      Grp Rnk
                    </Typography>
                  </Box>
                )}

                {/* ADR Box */}
                {(finalStockData?.adr_percent != null || fundamentals?.adr_percent != null) && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Box sx={{
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      textAlign: 'center',
                      bgcolor: (finalStockData?.adr_percent ?? fundamentals?.adr_percent) >= 4
                        ? 'success.main'
                        : (finalStockData?.adr_percent ?? fundamentals?.adr_percent) >= 2
                          ? 'warning.main'
                          : 'error.main',
                    }}>
                      <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
                        {(finalStockData?.adr_percent ?? fundamentals?.adr_percent).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                      ADR
                    </Typography>
                  </Box>
                )}

                {/* EPS Rating Box */}
                {(finalStockData?.eps_rating != null || fundamentals?.eps_rating != null) && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Box sx={{
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      textAlign: 'center',
                      minWidth: 36,
                      bgcolor: (finalStockData?.eps_rating ?? fundamentals?.eps_rating) >= 80
                        ? 'success.main'
                        : (finalStockData?.eps_rating ?? fundamentals?.eps_rating) >= 50
                          ? 'warning.main'
                          : 'error.main',
                    }}>
                      <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
                        {finalStockData?.eps_rating ?? fundamentals?.eps_rating}
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                      EPS Rtg
                    </Typography>
                  </Box>
                )}

                {/* Industry Info Boxes */}
                {finalStockData && (
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
                          {finalStockData.ibd_industry_group || '-'}
                        </Typography>
                      </Box>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                        IBD
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
                          {finalStockData.gics_sector || '-'}
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
                          {finalStockData.gics_industry || '-'}
                        </Typography>
                      </Box>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                        Industry
                      </Typography>
                    </Box>
                  </Box>
                )}

              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {currentSymbol && (
                  <AddToWatchlistMenu symbols={currentSymbol} size="medium" />
                )}
                {currentSymbol && finalStockData?.ibd_industry_group && (
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
                stockData={finalStockData}
                fundamentals={fundamentals}
                onViewPeers={() => setPeerModalOpen(true)}
                onViewSetupDetails={() => setSetupDrawerOpen(true)}
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
              {finalStockData?.se_explain && (
                <Chip icon={<KeyboardIcon />} label="D: Setup Details" size="small" variant="outlined" />
              )}
              <Chip icon={<KeyboardIcon />} label="Esc: Close" size="small" variant="outlined" />
            </Box>
          </Box>
        </Fade>
      </Modal>

      {/* Peer Comparison Modal */}
      <PeerComparisonModal
        open={peerModalOpen}
        onClose={() => setPeerModalOpen(false)}
        scanId={scanId}
        symbol={currentSymbol}
        onOpenChart={(sym) => {
          setPeerModalOpen(false);
          handleNavigateToSymbol(sym);
        }}
      />

      {/* Setup Engine Drawer */}
      <SetupEngineDrawer
        open={setupDrawerOpen}
        onClose={() => setSetupDrawerOpen(false)}
        stockData={finalStockData}
      />
    </>
  );
}

export default ChartViewerModal;
