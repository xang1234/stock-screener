import { useMemo, useState } from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  CircularProgress,
  Grid,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

import { getWatchlist } from '../../api/marketScan';
import { getScanBootstrap, getScanResults } from '../../api/scans';
import { getCurrentRankings } from '../../api/groups';
import { fetchPriceHistory, priceHistoryKeys, PRICE_HISTORY_STALE_TIME } from '../../api/priceHistory';
import PriceSparkline from '../Scan/PriceSparkline';
import RSSparkline from '../Scan/RSSparkline';
import ChartViewerModal from '../Scan/ChartViewerModal';
import RankChangeCell from '../shared/RankChangeCell';
import { MARKET_CAP_OPTIONS } from '../../features/scan/components/filterPanel/constants';
import { getGroupRankColor } from '../../utils/colorUtils';
import { formatLocalCurrency } from '../../utils/formatUtils';
import { resolveMarketCapDisplay } from '../../utils/marketCapUtils';

const EMPTY_ROWS = [];
const DEFAULT_MIN_VOLUME = 100_000_000;
const DEFAULT_TOP_RESULTS = 20;

function formatNumber(value, digits = 0) {
  if (value == null) return '-';
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function DailyMarketSnapshotTab() {
  const watchlistQuery = useQuery({
    queryKey: ['marketScan', 'key_markets'],
    queryFn: () => getWatchlist('key_markets'),
  });

  const scanBootstrapQuery = useQuery({
    queryKey: ['scanBootstrap', 'latest'],
    queryFn: () => getScanBootstrap(),
    retry: false,
    staleTime: 60_000,
  });

  const groupsQuery = useQuery({
    queryKey: ['groupRankings', 'dailySnapshot', 10],
    queryFn: () => getCurrentRankings(10),
    staleTime: 60_000,
  });

  const watchlistSymbols = useMemo(
    () => watchlistQuery.data?.symbols ?? EMPTY_ROWS,
    [watchlistQuery.data],
  );
  const keyMarketHistories = useQueries({
    queries: watchlistSymbols.map((entry) => ({
      queryKey: priceHistoryKeys.symbol(entry.symbol, '3mo'),
      queryFn: () => fetchPriceHistory(entry.symbol, '3mo'),
      staleTime: PRICE_HISTORY_STALE_TIME,
      enabled: Boolean(entry.symbol),
    })),
  });

  const keyMarkets = useMemo(() => (
    watchlistSymbols.map((entry, idx) => {
      const history = keyMarketHistories[idx]?.data ?? [];
      const closes = history.map((h) => h.close).filter((c) => c != null);
      const latestClose = closes.length ? closes[closes.length - 1] : null;
      const priorClose = closes.length >= 2 ? closes[closes.length - 2] : null;
      const change1d = latestClose != null && priorClose
        ? ((latestClose - priorClose) / priorClose) * 100
        : null;
      return {
        symbol: entry.symbol,
        display_name: entry.display_name || entry.symbol,
        closes,
        latestClose,
        change1d,
      };
    })
  ), [watchlistSymbols, keyMarketHistories]);

  const scanPayload = scanBootstrapQuery.data?.payload;
  const scanId = scanPayload?.selected_scan?.scan_id
    ?? scanPayload?.results_page?.scan_id
    ?? null;
  const scanAsOfDate = scanPayload?.selected_scan?.as_of_date
    ?? scanPayload?.results_page?.as_of_date
    ?? null;
  const [marketCapMin, setMarketCapMin] = useState('');
  const topResultsParams = useMemo(() => ({
    page: 1,
    per_page: DEFAULT_TOP_RESULTS,
    sort_by: 'composite_score',
    sort_order: 'desc',
    min_volume: DEFAULT_MIN_VOLUME,
    ...(marketCapMin !== '' ? { min_market_cap: Number(marketCapMin) } : {}),
  }), [marketCapMin]);
  const topResultsQuery = useQuery({
    queryKey: ['marketScan', 'dailySnapshot', scanId, topResultsParams],
    queryFn: () => getScanResults(scanId, topResultsParams),
    enabled: Boolean(scanId),
    staleTime: 60_000,
    placeholderData: (previous) => previous,
  });
  const topResults = topResultsQuery.data?.results ?? EMPTY_ROWS;
  const topResultSymbols = useMemo(() => {
    const seen = new Set();
    return topResults
      .map((row) => row?.symbol)
      .filter((symbol) => symbol && !seen.has(symbol) && seen.add(symbol));
  }, [topResults]);

  const topGroups = (groupsQuery.data?.rankings ?? EMPTY_ROWS).slice(0, 10);
  const groupsDate = groupsQuery.data?.date ?? null;

  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState(null);

  const isInitialLoading =
    watchlistQuery.isLoading || scanBootstrapQuery.isLoading || groupsQuery.isLoading;

  if (isInitialLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (scanBootstrapQuery.isError && groupsQuery.isError) {
    return <Alert severity="error">Failed to load the daily snapshot.</Alert>;
  }

  const handleRowClick = (symbol) => {
    if (!scanId) return;
    setSelectedSymbol(symbol);
    setChartModalOpen(true);
  };

  return (
    <Box sx={{ height: '100%', overflow: 'auto', pr: 1 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          columnGap: 2,
          rowGap: 0.5,
          mb: 2,
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: '-0.5px' }}>
          Daily Snapshot
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontFamily: 'monospace', fontSize: '11px' }}
        >
          {`Scan ${scanAsOfDate || '-'} · Groups ${groupsDate || '-'}`}
        </Typography>
      </Box>

      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        {keyMarkets.map((item) => {
          const trend = item.closes.length >= 2
            ? (item.closes[item.closes.length - 1] > item.closes[0] ? 1
              : item.closes[item.closes.length - 1] < item.closes[0] ? -1 : 0)
            : 0;
          return (
            <Grid item xs={6} sm={4} md={2.4} key={item.symbol}>
              <Paper elevation={0} sx={{ p: 1.5, height: '100%', border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '13px' }}>
                  {item.symbol}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '10px' }}>
                  {item.display_name}
                </Typography>
                <Typography variant="body1" sx={{ mt: 0.5, fontFamily: 'monospace', fontWeight: 600 }}>
                  {formatLocalCurrency(item.latestClose, 'USD')}
                </Typography>
                <Box display="flex" alignItems="center" sx={{ mt: 0.5 }}>
                  {item.change1d > 0 && <TrendingUpIcon sx={{ fontSize: 14, mr: 0.25, color: 'success.main' }} />}
                  {item.change1d < 0 && <TrendingDownIcon sx={{ fontSize: 14, mr: 0.25, color: 'error.main' }} />}
                  <Typography
                    variant="body2"
                    sx={{
                      color: item.change1d > 0 ? 'success.main' : item.change1d < 0 ? 'error.main' : 'text.secondary',
                      fontFamily: 'monospace',
                      fontWeight: 600,
                      fontSize: '12px',
                    }}
                  >
                    {item.change1d != null
                      ? `${item.change1d > 0 ? '+' : ''}${formatNumber(item.change1d, 2)}%`
                      : '-'}
                  </Typography>
                </Box>
                {item.closes.length > 1 ? (
                  <Box sx={{ mt: 0.75 }}>
                    <PriceSparkline
                      data={item.closes}
                      trend={trend}
                      change1d={null}
                      width="100%"
                      height={36}
                      showChange={false}
                    />
                  </Box>
                ) : null}
              </Paper>
            </Grid>
          );
        })}
      </Grid>

      <Paper elevation={0} sx={{ p: 1.5, mb: 2, border: '1px solid', borderColor: 'divider' }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 1,
            flexWrap: 'wrap',
            mb: 1,
          }}
        >
          <Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
              Top Scan Candidates
            </Typography>
            <Typography variant="caption" color="text.disabled" sx={{ display: 'block', fontSize: '10px' }}>
              Dollar volume &gt; $100M. Click a row for chart details.
            </Typography>
          </Box>
          <TextField
            select
            size="small"
            label="Mkt Cap"
            value={marketCapMin}
            onChange={(event) => {
              const nextValue = event.target.value;
              setMarketCapMin(nextValue === '' ? '' : Number(nextValue));
            }}
            sx={{ minWidth: 140 }}
          >
            <MenuItem value="">All</MenuItem>
            {MARKET_CAP_OPTIONS.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </TextField>
        </Box>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell align="center">Symbol</TableCell>
                <TableCell align="center">Score</TableCell>
                <TableCell align="center">Price</TableCell>
                <TableCell align="center">MCap ($)</TableCell>
                <TableCell align="center">Rating</TableCell>
                <TableCell align="center">Price Trend</TableCell>
                <TableCell align="center">RS Trend</TableCell>
                <TableCell align="center">IBD Group</TableCell>
                <TableCell align="center">Grp Rank</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {topResultsQuery.isLoading && topResults.length === 0 ? (
                <TableRow>
                  <TableCell align="center" colSpan={9}>
                    <CircularProgress size={18} />
                  </TableCell>
                </TableRow>
              ) : null}
              {topResults.map((row) => (
                <TableRow
                  key={row.symbol}
                  hover
                  tabIndex={0}
                  onClick={() => handleRowClick(row.symbol)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      handleRowClick(row.symbol);
                    }
                  }}
                  sx={{ cursor: scanId ? 'pointer' : 'default' }}
                >
                  <TableCell align="center" sx={{ fontWeight: 600 }}>{row.symbol}</TableCell>
                  <TableCell align="center">{formatNumber(row.composite_score, 1)}</TableCell>
                  <TableCell align="center">{formatLocalCurrency(row.current_price, row.currency)}</TableCell>
                  <TableCell align="center">
                    {resolveMarketCapDisplay(row, null, { preferUsd: true }).formattedValue}
                  </TableCell>
                  <TableCell align="center">{row.rating}</TableCell>
                  <TableCell align="center">
                    {row.price_sparkline_data ? (
                      <Box display="flex" justifyContent="center">
                        <PriceSparkline
                          data={row.price_sparkline_data}
                          trend={row.price_trend}
                          change1d={row.price_change_1d}
                          industry={row.ibd_industry_group}
                          width={130}
                          height={28}
                        />
                      </Box>
                    ) : '-'}
                  </TableCell>
                  <TableCell align="center">
                    {row.rs_sparkline_data ? (
                      <Box display="flex" justifyContent="center">
                        <RSSparkline
                          data={row.rs_sparkline_data}
                          trend={row.rs_trend}
                          width={78}
                          height={20}
                        />
                      </Box>
                    ) : '-'}
                  </TableCell>
                  <TableCell align="center" sx={{
                    color: 'text.secondary', fontSize: '12px',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 140,
                  }}>
                    {row.ibd_industry_group || '-'}
                  </TableCell>
                  <TableCell align="center" sx={{
                    fontFamily: 'monospace',
                    fontWeight: row.ibd_group_rank && row.ibd_group_rank <= 20 ? 600 : 400,
                    color: getGroupRankColor(row.ibd_group_rank),
                  }}>
                    {row.ibd_group_rank ?? '-'}
                  </TableCell>
                </TableRow>
              ))}
              {!topResultsQuery.isLoading && topResults.length === 0 && (
                <TableRow>
                  <TableCell colSpan={9} align="center" sx={{ color: 'text.disabled', py: 2 }}>
                    No scan candidates match the current filters.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <Paper elevation={0} sx={{ p: 1.5, border: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
          Top 10 Groups
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell align="center">Rank</TableCell>
                <TableCell>Group</TableCell>
                <TableCell align="right">1W</TableCell>
                <TableCell>Top Stock</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {topGroups.map((group) => (
                <TableRow key={group.industry_group}>
                  <TableCell align="center" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>{group.rank}</TableCell>
                  <TableCell>{group.industry_group}</TableCell>
                  <TableCell align="right"><RankChangeCell value={group.rank_change_1w} /></TableCell>
                  <TableCell sx={{ fontWeight: 500, fontFamily: 'monospace', fontSize: '11px' }}>
                    {group.top_symbol || '-'}
                  </TableCell>
                </TableRow>
              ))}
              {topGroups.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} align="center" sx={{ color: 'text.disabled', py: 2 }}>
                    No group rankings available.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {scanId && (
        <ChartViewerModal
          open={chartModalOpen}
          onClose={() => setChartModalOpen(false)}
          initialSymbol={selectedSymbol}
          scanId={scanId}
          filters={{}}
          sortBy="composite_score"
          sortOrder="desc"
          navigationSymbolsOverride={topResultSymbols}
          currentPageResults={topResults}
        />
      )}
    </Box>
  );
}

export default DailyMarketSnapshotTab;
