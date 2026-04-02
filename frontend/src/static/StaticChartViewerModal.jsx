import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  Fade,
  IconButton,
  Modal,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import KeyboardIcon from '@mui/icons-material/Keyboard';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import CandlestickChart from '../components/Charts/CandlestickChart';
import StockMetricsSidebar from '../components/Scan/StockMetricsSidebar';
import { getGroupRankColor } from '../utils/colorUtils';
import { useChartNavigation } from '../hooks/useChartNavigation';
import { fetchStaticChartPayload, staticChartKeys } from './chartClient';

function StaticChartViewerModal({
  open,
  onClose,
  initialSymbol,
  chartIndex,
  navigationSymbols = null,
}) {
  const queryClient = useQueryClient();
  const [visibleRange, setVisibleRange] = useState(null);

  const entries = useMemo(() => chartIndex?.symbols || [], [chartIndex]);
  const entryBySymbol = useMemo(
    () => new Map(entries.map((entry) => [entry.symbol, entry])),
    [entries]
  );
  const symbols = useMemo(() => {
    if (Array.isArray(navigationSymbols) && navigationSymbols.length > 0) {
      return navigationSymbols.filter((symbol) => entryBySymbol.has(symbol));
    }
    return entries.map((entry) => entry.symbol);
  }, [entries, entryBySymbol, navigationSymbols]);

  const { currentIndex, currentSymbol, totalCount, goNext, goPrevious } = useChartNavigation(
    symbols,
    initialSymbol,
    open
  );

  const currentEntry = currentSymbol ? entryBySymbol.get(currentSymbol) : null;
  const {
    data: chartPayload,
    isLoading,
    isError,
  } = useQuery({
    queryKey: staticChartKeys.payload(currentSymbol, currentEntry?.path),
    queryFn: () => fetchStaticChartPayload(currentEntry.path),
    enabled: open && Boolean(currentEntry?.path),
    staleTime: Infinity,
    gcTime: Infinity,
  });

  useEffect(() => {
    if (!open || !symbols.length) {
      return undefined;
    }

    const prefetch = (entry) => {
      if (!entry?.path) {
        return;
      }
      queryClient.prefetchQuery({
        queryKey: staticChartKeys.payload(entry.symbol, entry.path),
        queryFn: () => fetchStaticChartPayload(entry.path),
        staleTime: Infinity,
        gcTime: Infinity,
      });
    };

    const timeouts = [];
    const nextEntries = symbols
      .slice(currentIndex + 1, currentIndex + 3)
      .map((symbol) => entryBySymbol.get(symbol))
      .filter(Boolean);
    const previousEntries = symbols
      .slice(Math.max(0, currentIndex - 2), currentIndex)
      .reverse()
      .map((symbol) => entryBySymbol.get(symbol))
      .filter(Boolean);

    if (nextEntries[0]) {
      prefetch(nextEntries[0]);
    }
    if (previousEntries[0]) {
      prefetch(previousEntries[0]);
    }

    nextEntries.slice(1).forEach((entry, index) => {
      timeouts.push(setTimeout(() => prefetch(entry), (index + 1) * 120));
    });
    previousEntries.slice(1).forEach((entry, index) => {
      timeouts.push(setTimeout(() => prefetch(entry), 320 + (index + 1) * 120));
    });

    return () => {
      timeouts.forEach(clearTimeout);
    };
  }, [currentIndex, entryBySymbol, open, queryClient, symbols]);

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose();
        return;
      }
      if (event.key === ' ') {
        event.preventDefault();
        if (event.shiftKey) {
          goPrevious();
        } else {
          goNext();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [goNext, goPrevious, onClose, open]);

  const stockData = chartPayload?.stock_data || null;
  const fundamentals = chartPayload?.fundamentals || null;
  const adrValue = stockData?.adr_percent ?? fundamentals?.adr_percent ?? null;
  const epsRating = stockData?.eps_rating ?? fundamentals?.eps_rating ?? null;
  const groupRank = stockData?.ibd_group_rank ?? null;
  const viewportHeight = typeof window !== 'undefined' ? window.innerHeight : 900;
  const chartHeight = Math.max(viewportHeight - 60, 500);
  const dataUpdatedAtOverride = chartPayload?.generated_at ? Date.parse(chartPayload.generated_at) : null;

  return (
    <Modal
      open={open}
      onClose={onClose}
      aria-labelledby="static-chart-viewer-modal"
      closeAfterTransition
    >
      <Fade in={open}>
        <Box
          sx={{
            position: 'fixed',
            inset: 0,
            bgcolor: 'background.paper',
            display: 'flex',
            flexDirection: 'column',
            outline: 'none',
          }}
        >
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
              {isLoading ? <CircularProgress size={18} /> : null}

              {stockData?.ibd_industry_group ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Box
                    sx={{
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      textAlign: 'center',
                      minWidth: 36,
                      bgcolor: getGroupRankColor(groupRank),
                    }}
                  >
                    <Typography
                      variant="body2"
                      noWrap
                      sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}
                    >
                      {groupRank ?? '-'}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                    Grp Rnk
                  </Typography>
                </Box>
              ) : null}

              {adrValue != null ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Box
                    sx={{
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      textAlign: 'center',
                      bgcolor: Number(adrValue) >= 4
                        ? 'success.main'
                        : Number(adrValue) >= 2
                          ? 'warning.main'
                          : 'error.main',
                    }}
                  >
                    <Typography
                      variant="body2"
                      noWrap
                      sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}
                    >
                      {Number(adrValue).toFixed(1)}%
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                    ADR
                  </Typography>
                </Box>
              ) : null}

              {epsRating != null ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Box
                    sx={{
                      borderRadius: 1,
                      px: 1.5,
                      py: 0.5,
                      textAlign: 'center',
                      minWidth: 36,
                      bgcolor: epsRating >= 80
                        ? 'success.main'
                        : epsRating >= 50
                          ? 'warning.main'
                          : 'error.main',
                    }}
                  >
                    <Typography
                      variant="body2"
                      noWrap
                      sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}
                    >
                      {epsRating}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                    EPS Rtg
                  </Typography>
                </Box>
              ) : null}

              {stockData ? (
                <Box sx={{ display: 'flex', gap: 1.5, ml: 1 }}>
                  {[
                    ['IBD', stockData.ibd_industry_group],
                    ['Sector', stockData.gics_sector],
                    ['Industry', stockData.gics_industry],
                  ].map(([label, value]) => (
                    <Box key={label} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <Box
                        sx={{
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 1,
                          px: 1.5,
                          py: 0.5,
                          minWidth: 80,
                          maxWidth: 180,
                          textAlign: 'center',
                          bgcolor: 'background.paper',
                        }}
                      >
                        <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem' }}>
                          {value || '-'}
                        </Typography>
                      </Box>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
                        {label}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              ) : null}
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {totalCount > 0 ? (
                <Typography variant="body2" color="text.secondary">
                  Stock {currentIndex + 1} of {totalCount}
                </Typography>
              ) : null}
              <Chip label="Static" size="small" color="info" />
              <IconButton onClick={onClose} size="large">
                <CloseIcon />
              </IconButton>
            </Box>
          </Box>

          <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
            <StockMetricsSidebar stockData={stockData} fundamentals={fundamentals} />

            <Box sx={{ flex: 1, overflow: 'hidden', bgcolor: 'background.paper' }}>
              {isError ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: chartHeight, p: 3 }}>
                  <Alert severity="error">Failed to load the static chart payload.</Alert>
                </Box>
              ) : isLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: chartHeight }}>
                  <CircularProgress size={56} />
                </Box>
              ) : currentSymbol ? (
                <CandlestickChart
                  symbol={currentSymbol}
                  period="6mo"
                  height={chartHeight}
                  visibleRange={visibleRange}
                  onVisibleRangeChange={setVisibleRange}
                  priceData={chartPayload?.bars || []}
                  dataUpdatedAtOverride={dataUpdatedAtOverride}
                />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: chartHeight }}>
                  <CircularProgress size={56} />
                </Box>
              )}
            </Box>
          </Box>

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
  );
}

export default StaticChartViewerModal;
