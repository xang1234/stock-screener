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
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getAllFilteredSymbols, getSetupDetails, getSingleResult } from '../../api/scans';
import { prefetchPriceHistoryBatch } from '../../api/priceHistory';
import { getStockFundamentals } from '../../api/stocks';
import { getGroupDetail } from '../../api/groups';
import { useChartNavigation } from '../../hooks/useChartNavigation';
import { buildFilterParams, getStableFilterKey } from '../../utils/filterUtils';
import CandlestickChart from '../Charts/CandlestickChart';
import StockMetricsSidebar from './StockMetricsSidebar';
import PeerComparisonModal from './PeerComparisonModal';
import SetupEngineDrawer from './SetupEngineDrawer';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import MarketThemesList from '../Stock/MarketThemesList';
import { getGroupRankColor } from '../../utils/colorUtils';

const inferMarketFromSymbol = (symbol) => {
  const normalized = String(symbol || '').toUpperCase();
  if (normalized.endsWith('.HK')) return 'HK';
  if (normalized.endsWith('.T')) return 'JP';
  if (normalized.endsWith('.TW') || normalized.endsWith('.TWO')) return 'TW';
  return 'US';
};

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
 * @param {Array<string>} props.navigationSymbolsOverride - Explicit navigation set
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
  navigationSymbolsOverride = null,
  currentPageResults = [],
}) {
  const queryClient = useQueryClient();
  const [peerModalOpen, setPeerModalOpen] = useState(false);
  const [setupDrawerOpen, setSetupDrawerOpen] = useState(false);
  const [visibleRange, setVisibleRange] = useState(null); // Persist zoom across symbol navigation

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
    enabled: open && !!scanId && !Array.isArray(navigationSymbolsOverride),
    staleTime: 5000, // 5 seconds — short enough to catch mid-session sort changes
    refetchOnMount: 'always', // Always refetch when modal opens to pick up sort changes
  });

  const currentPageSymbols = useMemo(() => {
    if (!Array.isArray(currentPageResults) || currentPageResults.length === 0) return [];
    const seen = new Set();
    const symbols = [];
    currentPageResults.forEach((row) => {
      if (row?.symbol && !seen.has(row.symbol)) {
        seen.add(row.symbol);
        symbols.push(row.symbol);
      }
    });
    return symbols;
  }, [currentPageResults]);

  // Make navigation immediately usable from the current page, then
  // upgrade to the full filtered symbol list when hydration finishes.
  const navigationSymbols = useMemo(() => {
    if (Array.isArray(navigationSymbolsOverride)) {
      return navigationSymbolsOverride;
    }
    if (Array.isArray(allSymbols) && allSymbols.length > 0) {
      return allSymbols;
    }
    return currentPageSymbols;
  }, [allSymbols, currentPageSymbols, navigationSymbolsOverride]);

  // Use navigation hook
  const { currentIndex, currentSymbol, totalCount, goNext, goPrevious, goToIndex } = useChartNavigation(
    navigationSymbols,
    initialSymbol,
    open
  );

  // Navigate to a specific symbol
  const handleNavigateToSymbol = (symbol) => {
    if (!navigationSymbols) return;
    const index = navigationSymbols.indexOf(symbol);
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
  const hasInlineSetupPayload =
    finalStockData?.se_explain != null ||
    finalStockData?.se_candidates != null;

  // Lazy-load heavy setup explain payload only when the drawer opens.
  const { data: setupDetails, isLoading: setupDetailsLoading } = useQuery({
    queryKey: ['setupDetails', scanId, currentSymbol],
    queryFn: async () => {
      try {
        return await getSetupDetails(scanId, currentSymbol);
      } catch (error) {
        // Non-setup scans may legitimately return 404 from setup endpoint.
        if (error?.response?.status === 404) return null;
        throw error;
      }
    },
    enabled: open && setupDrawerOpen && !!scanId && !!currentSymbol && !hasInlineSetupPayload,
    staleTime: 300000,
    retry: false,
  });

  const drawerStockData = useMemo(() => {
    if (!finalStockData) return finalStockData;
    if (!setupDetails) return finalStockData;
    return {
      ...finalStockData,
      se_explain: setupDetails.se_explain ?? null,
      se_candidates: setupDetails.se_candidates ?? null,
    };
  }, [finalStockData, setupDetails]);

  // Fetch fundamentals for current symbol
  const { data: fundamentals } = useQuery({
    queryKey: ['fundamentals', currentSymbol],
    queryFn: () => getStockFundamentals(currentSymbol),
    enabled: open && !!currentSymbol,
    staleTime: 300000, // 5 minutes (data is cached 7 days server-side)
  });

  // Fetch group ranking for current stock's industry group
  const industryGroup = finalStockData?.ibd_industry_group;
  const groupMarket = finalStockData?.market || inferMarketFromSymbol(currentSymbol);
  const marketThemes = finalStockData?.market_themes || [];
  const adrValue = finalStockData?.adr_percent ?? fundamentals?.adr_percent ?? null;
  const {
    data: groupRankData,
    isLoading: isGroupRankLoading,
  } = useQuery({
    queryKey: ['groupRank', groupMarket, industryGroup],
    queryFn: () => getGroupDetail(industryGroup, 1, groupMarket),
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

      // D to toggle setup drawer
      if (e.key === 'd' || e.key === 'D') {
        setSetupDrawerOpen((prev) => !prev);
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
  }, [open, goNext, goPrevious, onClose, setupDrawerOpen]);

  // Reset setup drawer when symbol changes
  useEffect(() => {
    setSetupDrawerOpen(false);
  }, [currentSymbol]);

  // Prefetch adjacent stocks (next 5 / prev 5) for smooth navigation.
  // Price history fetches in a single batch; fundamentals + scan result still
  // go individually (no batch endpoints for those yet).
  useEffect(() => {
    if (!open || !currentSymbol || !navigationSymbols) return;

    const nextSymbols = navigationSymbols.slice(currentIndex + 1, currentIndex + 6);
    const prevSymbols = navigationSymbols.slice(Math.max(0, currentIndex - 5), currentIndex);
    const adjacent = [...nextSymbols, ...prevSymbols].filter(Boolean);
    if (adjacent.length === 0) return;

    let cancelled = false;
    prefetchPriceHistoryBatch(queryClient, adjacent, '6mo');

    adjacent.forEach((symbol) => {
      if (cancelled) return;
      queryClient.prefetchQuery({
        queryKey: ['fundamentals', symbol],
        queryFn: () => getStockFundamentals(symbol),
        staleTime: 300000,
      });
      if (scanId) {
        queryClient.prefetchQuery({
          queryKey: ['stockResult', scanId, symbol],
          queryFn: () => getSingleResult(scanId, symbol),
          staleTime: 300000,
        });
      }
    });

    return () => {
      cancelled = true;
    };
  }, [open, currentSymbol, currentIndex, navigationSymbols, queryClient, scanId]);

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
                {adrValue != null && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Box sx={{
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      textAlign: 'center',
                      bgcolor: Number(adrValue) >= 4
                        ? 'success.main'
                        : Number(adrValue) >= 2
                          ? 'warning.main'
                          : 'error.main',
                    }}>
                      <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
                        {Number(adrValue).toFixed(1)}%
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

                {marketThemes.length > 0 && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, maxWidth: 280 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                      Market Themes
                    </Typography>
                    <MarketThemesList themes={marketThemes} variant="wrap" />
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
              {currentSymbol && (
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
        stockData={drawerStockData}
        isLoading={setupDetailsLoading}
      />
    </>
  );
}

export default ChartViewerModal;
