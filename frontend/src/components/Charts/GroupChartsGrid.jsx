import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Card,
  CircularProgress,
  Typography,
  useTheme,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';

import CandlestickChart from './CandlestickChart';
import GroupChartsLayout, { GroupChartCell } from './GroupChartsLayout';
import {
  fetchPriceHistoryBatch,
  priceHistoryKeys,
  PRICE_HISTORY_STALE_TIME,
} from '../../api/priceHistory';

const MAX_SYMBOLS = 40;
// Card header (~33px) plus grid gap (8px), rounded up: one row is chart height + this.
const CARD_CHROME_PX = 48;

const SCROLLABLE = /(auto|scroll|overlay)/;

/** Nearest scrolling ancestor (the group dialog's content), else the viewport. */
function scrollParent(element) {
  for (let node = element?.parentElement; node; node = node.parentElement) {
    if (SCROLLABLE.test(getComputedStyle(node).overflowY)) return node;
  }
  return null;
}

/**
 * Track which cells have come within `rootMargin` of the scroll area, using one
 * shared IntersectionObserver. Revealed cells stay revealed, so a mounted chart
 * is never rebuilt by scrolling away and back. Without IntersectionObserver
 * every cell counts as revealed (the old eager behaviour).
 */
function useRevealOnScroll(rootMargin) {
  const supported = typeof IntersectionObserver !== 'undefined';
  const [revealed, setRevealed] = useState(() => new Set());
  const observerRef = useRef(null);
  const keyByElement = useRef(new Map());
  const refCallbacks = useRef(new Map());

  useEffect(() => {
    if (!supported) return undefined;
    const elements = keyByElement.current;
    let active = true;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!active) return;
        const shown = entries.filter((entry) => entry.isIntersecting);
        if (shown.length === 0) return;
        shown.forEach((entry) => observer.unobserve(entry.target));
        setRevealed((previous) => {
          const next = new Set(previous);
          shown.forEach((entry) => next.add(elements.get(entry.target)));
          return next;
        });
      },
      { root: scrollParent(elements.keys().next().value), rootMargin },
    );
    observerRef.current = observer;
    elements.forEach((_, element) => observer.observe(element));
    return () => {
      active = false;
      observer.disconnect();
      observerRef.current = null;
    };
  }, [supported, rootMargin]);

  // One stable ref callback per key, so re-renders do not re-observe cells.
  const cellRef = useCallback((key) => {
    if (!refCallbacks.current.has(key)) {
      let current = null;
      refCallbacks.current.set(key, (element) => {
        if (current) {
          keyByElement.current.delete(current);
          observerRef.current?.unobserve(current);
        }
        current = element;
        if (element) {
          keyByElement.current.set(element, key);
          observerRef.current?.observe(element);
        }
      });
    }
    return refCallbacks.current.get(key);
  }, []);

  const isRevealed = useCallback((key) => !supported || revealed.has(key), [supported, revealed]);
  return { cellRef, isRevealed };
}

/**
 * Grid of mini candlestick charts for a group of constituent symbols.
 *
 * One batch network call fetches all OHLCV payloads, and each cell hands its
 * slice to a `compact` CandlestickChart so no per-symbol request fires.
 * Charts are built only once their cell nears the visible area (about one row
 * beyond it); until then the cell keeps its label and reserved height.
 *
 * @param {Object} props
 * @param {string[]} props.symbols - Constituent ticker symbols
 * @param {string} props.period - Time period (default '6mo')
 * @param {number} props.height - Height per chart cell (default 200)
 */
function GroupChartsGrid({ symbols = [], period = '6mo', height = 200 }) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

  const normalizedSymbols = useMemo(
    () =>
      Array.from(
        new Set(
          (symbols || [])
            .filter((s) => typeof s === 'string' && s.trim().length > 0)
            .map((s) => s.trim().toUpperCase()),
        ),
      ),
    [symbols],
  );

  const truncatedSymbols = useMemo(
    () => normalizedSymbols.slice(0, MAX_SYMBOLS),
    [normalizedSymbols],
  );

  const {
    data: batch,
    isLoading,
    isError,
    error,
    refetch,
    dataUpdatedAt,
  } = useQuery({
    queryKey: priceHistoryKeys.batch(truncatedSymbols, period),
    queryFn: () => fetchPriceHistoryBatch(truncatedSymbols, period),
    enabled: truncatedSymbols.length > 0,
    staleTime: PRICE_HISTORY_STALE_TIME,
  });

  if (normalizedSymbols.length === 0) {
    return (
      <Alert severity="info" sx={{ mt: 1 }}>
        No constituent stocks to chart.
      </Alert>
    );
  }

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" py={6}>
        <CircularProgress />
        <Typography variant="body2" sx={{ ml: 2 }} color="text.secondary">
          Loading {truncatedSymbols.length} charts…
        </Typography>
      </Box>
    );
  }

  if (isError) {
    return (
      <Alert severity="error" sx={{ mt: 1 }}>
        <AlertTitle>Failed to load group charts</AlertTitle>
        {error?.message || 'Unknown error fetching price history'}
        <Box mt={1}>
          <Button size="small" variant="outlined" onClick={() => refetch()}>
            Retry
          </Button>
        </Box>
      </Alert>
    );
  }

  const truncated = normalizedSymbols.length > truncatedSymbols.length;

  return (
    <Box>
      {truncated && (
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Showing first {truncatedSymbols.length} of {normalizedSymbols.length} stocks.
        </Typography>
      )}
      {/* Keyed by identity: a new group or period gets fresh observations. */}
      <GroupChartCells
        key={`${period}|${truncatedSymbols.join(',')}`}
        symbols={truncatedSymbols}
        batch={batch}
        period={period}
        height={height}
        dataUpdatedAt={dataUpdatedAt}
        isDarkMode={isDarkMode}
      />
    </Box>
  );
}

function GroupChartCells({ symbols, batch, period, height, dataUpdatedAt, isDarkMode }) {
  const { cellRef, isRevealed } = useRevealOnScroll(`${height + CARD_CHROME_PX}px 0px`);
  const dataMap = batch?.data || {};
  const missingSet = new Set(batch?.missing || []);

  return (
      <GroupChartsLayout data-testid="group-charts-grid" gap={1}>
        {symbols.map((sym) => {
          const priceData = dataMap[sym];
          const isMissing = missingSet.has(sym) || !priceData || priceData.length === 0;
          const lastClose = priceData && priceData.length > 0 ? priceData[priceData.length - 1].close : null;

          return (
            <GroupChartCell key={sym} ref={cellRef(sym)} data-testid="group-chart-cell">
              <Card
                variant="outlined"
                sx={{
                  overflow: 'hidden',
                  bgcolor: isDarkMode ? '#1e1e1e' : 'background.paper',
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    px: 1,
                    py: 0.5,
                    borderBottom: `1px solid ${isDarkMode ? '#363a45' : '#e0e0e0'}`,
                  }}
                >
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, fontFamily: 'monospace' }}>
                    {sym}
                  </Typography>
                  {lastClose !== null && (
                    <Typography
                      variant="caption"
                      sx={{ fontFamily: 'monospace', color: 'text.secondary' }}
                    >
                      {lastClose.toFixed(2)}
                    </Typography>
                  )}
                </Box>
                {isMissing ? (
                  <Box
                    sx={{
                      height,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Typography variant="caption" color="text.secondary">
                      No price data
                    </Typography>
                  </Box>
                ) : isRevealed(sym) ? (
                  <CandlestickChart
                    symbol={sym}
                    period={period}
                    height={height}
                    priceData={priceData}
                    dataUpdatedAtOverride={dataUpdatedAt || null}
                    compact
                  />
                ) : (
                  <Box
                    sx={{
                      height,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Typography variant="caption" color="text.secondary">
                      Chart loads when scrolled into view
                    </Typography>
                  </Box>
                )}
              </Card>
            </GroupChartCell>
          );
        })}
      </GroupChartsLayout>
  );
}

export default GroupChartsGrid;
