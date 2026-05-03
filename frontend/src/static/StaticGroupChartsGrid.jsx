import { useMemo } from 'react';
import {
  Alert,
  Box,
  Card,
  CircularProgress,
  Grid,
  Typography,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';

import CandlestickChart from '../components/Charts/CandlestickChart';
import { getGroupRankColor } from '../utils/colorUtils';
import { fetchStaticChartPayload, staticChartKeys } from './chartClient';

const MAX_SYMBOLS = 50;
const CHART_HEIGHT = 360;

function StatBadge({ value, label, bgcolor }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box
        sx={{
          borderRadius: 1,
          px: 1.5,
          py: 0.5,
          minWidth: 36,
          textAlign: 'center',
          bgcolor,
        }}
      >
        <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
          {value}
        </Typography>
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
        {label}
      </Typography>
    </Box>
  );
}

function StaticGroupChartCard({ symbol, entry }) {
  const { data: payload, isLoading, isError } = useQuery({
    queryKey: staticChartKeys.payload(symbol, entry?.path),
    queryFn: () => fetchStaticChartPayload(entry.path),
    enabled: Boolean(entry?.path),
    staleTime: Infinity,
    gcTime: Infinity,
  });

  const stockData = payload?.stock_data || null;
  const fundamentals = payload?.fundamentals || null;
  const bars = payload?.bars || null;
  const hasBarsPayload = Array.isArray(bars);
  const generatedAt = payload?.generated_at ? Date.parse(payload.generated_at) : null;
  const lastClose = bars && bars.length > 0 ? bars[bars.length - 1].close : null;
  const groupRank = stockData?.ibd_group_rank ?? null;
  const adrValue = stockData?.adr_percent ?? fundamentals?.adr_percent ?? null;
  const epsRating = stockData?.eps_rating ?? fundamentals?.eps_rating ?? null;
  const companyName = stockData?.company_name || fundamentals?.company_name || null;

  const adrBg = adrValue == null
    ? null
    : Number(adrValue) >= 4
      ? 'success.main'
      : Number(adrValue) >= 2
        ? 'warning.main'
        : 'error.main';

  const epsBg = epsRating == null
    ? null
    : epsRating >= 80
      ? 'success.main'
      : epsRating >= 50
        ? 'warning.main'
        : 'error.main';

  return (
    <Card variant="outlined" sx={{ overflow: 'hidden' }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 2,
          px: 2,
          py: 1,
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.default',
          flexWrap: 'wrap',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          <Box>
            <Typography variant="h6" sx={{ fontFamily: 'monospace', fontWeight: 700, lineHeight: 1.1 }}>
              {symbol}
            </Typography>
            {companyName ? (
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                {companyName}
              </Typography>
            ) : null}
          </Box>
          {groupRank != null ? (
            <StatBadge value={groupRank} label="Grp Rnk" bgcolor={getGroupRankColor(groupRank)} />
          ) : null}
          {adrValue != null ? (
            <StatBadge value={`${Number(adrValue).toFixed(1)}%`} label="ADR" bgcolor={adrBg} />
          ) : null}
          {epsRating != null ? (
            <StatBadge value={epsRating} label="EPS Rtg" bgcolor={epsBg} />
          ) : null}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {lastClose != null ? (
            <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
              {lastClose.toFixed(2)}
            </Typography>
          ) : null}
        </Box>
      </Box>

      {isError ? (
        <Box sx={{ height: CHART_HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>
          <Alert severity="error" sx={{ width: '100%' }}>
            Failed to load chart payload for {symbol}.
          </Alert>
        </Box>
      ) : isLoading ? (
        <Box sx={{ height: CHART_HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <CircularProgress size={32} />
        </Box>
      ) : !hasBarsPayload ? (
        <Box sx={{ height: CHART_HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>
          <Alert severity="warning" sx={{ width: '100%' }}>
            No price data available for {symbol}.
          </Alert>
        </Box>
      ) : (
        <CandlestickChart
          symbol={symbol}
          period="6mo"
          height={CHART_HEIGHT}
          priceData={bars}
          dataUpdatedAtOverride={generatedAt}
        />
      )}
    </Card>
  );
}

/**
 * Vertical list of full-featured candlestick charts for a group's constituent
 * stocks, rendered from pre-generated static chart payloads (no live API).
 *
 * Each chart matches the appearance of `StaticChartViewerModal` (EMAs, OHLC
 * legend, daily/weekly toggle) with a header carrying symbol, last close, and
 * the same metric badges as the scan-result popup.
 *
 * @param {Object} props
 * @param {string[]} props.symbols - Constituent ticker symbols
 * @param {Object|null} props.chartIndex - Static chart index `{ symbols: [{ symbol, path }, ...] }`
 */
function StaticGroupChartsGrid({ symbols = [], chartIndex = null }) {
  const entryBySymbol = useMemo(() => {
    const entries = chartIndex?.symbols || [];
    return new Map(entries.map((entry) => [entry.symbol, entry]));
  }, [chartIndex]);

  const normalizedSymbols = useMemo(
    () => Array.from(
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

  if (normalizedSymbols.length === 0) {
    return (
      <Alert severity="info" sx={{ mt: 1 }}>
        No constituent stocks to chart.
      </Alert>
    );
  }

  if (!chartIndex) {
    return (
      <Alert severity="warning" sx={{ mt: 1 }}>
        Static chart payloads are unavailable for this market.
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
      <Grid container spacing={2}>
        {truncatedSymbols.map((sym) => {
          const entry = entryBySymbol.get(sym);
          if (!entry) {
            return (
              <Grid item xs={12} md={6} key={sym}>
                <Card variant="outlined">
                  <Box
                    sx={{
                      px: 2,
                      py: 1,
                      borderBottom: 1,
                      borderColor: 'divider',
                      bgcolor: 'background.default',
                    }}
                  >
                    <Typography variant="h6" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                      {sym}
                    </Typography>
                  </Box>
                  <Box sx={{ height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      No price data
                    </Typography>
                  </Box>
                </Card>
              </Grid>
            );
          }
          return (
            <Grid item xs={12} md={6} key={sym}>
              <StaticGroupChartCard symbol={sym} entry={entry} />
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
}

export default StaticGroupChartsGrid;
