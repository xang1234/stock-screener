import { useMemo } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Card,
  CircularProgress,
  Grid,
  Typography,
  useTheme,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';

import CandlestickChart from './CandlestickChart';
import {
  fetchPriceHistoryBatch,
  priceHistoryKeys,
  PRICE_HISTORY_STALE_TIME,
} from '../../api/priceHistory';

const MAX_SYMBOLS = 40;

/**
 * Grid of mini candlestick charts for a group of constituent symbols.
 *
 * One batch network call fetches all OHLCV payloads, and each cell hands its
 * slice to a `compact` CandlestickChart so no per-symbol request fires.
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

  const dataMap = batch?.data || {};
  const missingSet = new Set(batch?.missing || []);
  const truncated = normalizedSymbols.length > truncatedSymbols.length;

  return (
    <Box>
      {truncated && (
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Showing first {truncatedSymbols.length} of {normalizedSymbols.length} stocks.
        </Typography>
      )}
      <Grid container spacing={1}>
        {truncatedSymbols.map((sym) => {
          const priceData = dataMap[sym];
          const isMissing = missingSet.has(sym) || !priceData || priceData.length === 0;
          const lastClose = priceData && priceData.length > 0 ? priceData[priceData.length - 1].close : null;

          return (
            <Grid item xs={12} sm={6} md={4} lg={3} key={sym}>
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
                ) : (
                  <CandlestickChart
                    symbol={sym}
                    period={period}
                    height={height}
                    priceData={priceData}
                    dataUpdatedAtOverride={dataUpdatedAt || null}
                    compact
                  />
                )}
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
}

export default GroupChartsGrid;
