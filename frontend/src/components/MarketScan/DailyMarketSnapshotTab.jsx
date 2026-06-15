import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
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

import { getDailySnapshot } from '../../api/marketScan';
import { getScanResults } from '../../api/scans';
import PriceSparkline from '../Scan/PriceSparkline';
import ChartViewerModal from '../Scan/ChartViewerModalLazy';
import DailyScanRowsTable from '../shared/DailyScanRowsTable';
import RankChangeCell from '../shared/RankChangeCell';
import TickerCell from '../common/TickerCell';
import { MARKET_CAP_OPTIONS } from '../../features/scan/components/filterPanel/constants';
import { useMarket } from '../../contexts/MarketContext';
import { marketFlag } from '../../utils/marketFlags';
import { formatLocalCurrency } from '../../utils/formatUtils';

const EMPTY_ROWS = [];
const DEFAULT_TOP_RESULTS = 20;

function formatNumber(value, digits = 0) {
  if (value == null) return '-';
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function DailyMarketSnapshotTab() {
  const { selectedMarket } = useMarket();
  const snapshotQuery = useQuery({
    queryKey: ['dailySnapshot', selectedMarket],
    queryFn: () => getDailySnapshot(selectedMarket),
    staleTime: 60_000,
  });
  const snapshot = snapshotQuery.data;
  const scanId = snapshot?.scan_id ?? null;
  const freshness = snapshot?.freshness ?? {};
  const minDollarVolume = snapshot?.top_candidates?.min_dollar_volume ?? null;

  const [marketCapMin, setMarketCapMin] = useState('');
  // The default view comes entirely from the aggregated snapshot payload;
  // a market-cap override re-queries the scan results for this scan only.
  const filteredResultsQuery = useQuery({
    queryKey: ['dailySnapshot', 'mcapFilter', scanId, marketCapMin, minDollarVolume],
    queryFn: () => getScanResults(scanId, {
      page: 1,
      per_page: DEFAULT_TOP_RESULTS,
      sort_by: 'composite_score',
      sort_order: 'desc',
      ...(minDollarVolume != null ? { min_volume: minDollarVolume } : {}),
      min_market_cap_usd: Number(marketCapMin),
    }),
    enabled: Boolean(scanId) && marketCapMin !== '',
    staleTime: 60_000,
    // Keep rows visible across market-cap bucket changes, but only within
    // the same scan — a market switch swaps the scan and must not show the
    // previous market's rows while loading.
    placeholderData: (previous, previousQuery) => (
      previousQuery?.queryKey?.[2] === scanId ? previous : undefined
    ),
  });

  const topResults = marketCapMin === ''
    ? (snapshot?.top_candidates?.rows ?? EMPTY_ROWS)
    : (filteredResultsQuery.data?.results ?? EMPTY_ROWS);
  const leaders = snapshot?.leaders?.rows ?? EMPTY_ROWS;
  const topGroups = (snapshot?.top_groups ?? EMPTY_ROWS).slice(0, 10);

  const keyMarkets = useMemo(() => (
    (snapshot?.key_markets ?? EMPTY_ROWS)
      .map((item) => ({
        ...item,
        closes: (item.history || []).map((h) => h.close).filter((c) => c != null),
      }))
      .filter((item) => item.latest_close != null && item.closes.length > 1)
  ), [snapshot]);

  const topResultSymbols = useMemo(() => {
    const seen = new Set();
    return topResults
      .map((row) => row?.symbol)
      .filter((symbol) => symbol && !seen.has(symbol) && seen.add(symbol));
  }, [topResults]);

  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState(null);

  if (snapshotQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (snapshotQuery.isError) {
    return <Alert severity="error">Failed to load the daily snapshot.</Alert>;
  }

  const handleRowClick = (symbol) => {
    if (!scanId) return;
    setSelectedSymbol(symbol);
    setChartModalOpen(true);
  };

  const flag = marketFlag(snapshot?.market);
  const marketDisplay = snapshot?.market_display_name || snapshot?.market || '';

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
          {flag ? `${flag}  ` : ''}{marketDisplay} Snapshot
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontFamily: 'monospace', fontSize: '11px' }}
        >
          {`Snapshot ${freshness.scan_as_of_date || '-'} · Breadth ${freshness.breadth_latest_date || '-'} · Groups ${freshness.groups_latest_date || '-'}`}
        </Typography>
      </Box>

      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        {keyMarkets.map((item) => {
          const trend = item.closes[item.closes.length - 1] > item.closes[0]
            ? 1
            : item.closes[item.closes.length - 1] < item.closes[0] ? -1 : 0;
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
                  {formatLocalCurrency(item.latest_close, item.currency || 'USD')}
                </Typography>
                <Box display="flex" alignItems="center" sx={{ mt: 0.5 }}>
                  {item.change_1d > 0 && <TrendingUpIcon sx={{ fontSize: 14, mr: 0.25, color: 'success.main' }} />}
                  {item.change_1d < 0 && <TrendingDownIcon sx={{ fontSize: 14, mr: 0.25, color: 'error.main' }} />}
                  <Typography
                    variant="body2"
                    sx={{
                      color: item.change_1d > 0 ? 'success.main' : item.change_1d < 0 ? 'error.main' : 'text.secondary',
                      fontFamily: 'monospace',
                      fontWeight: 600,
                      fontSize: '12px',
                    }}
                  >
                    {item.change_1d != null
                      ? `${item.change_1d > 0 ? '+' : ''}${formatNumber(item.change_1d, 2)}%`
                      : '-'}
                  </Typography>
                </Box>
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
              </Paper>
            </Grid>
          );
        })}
      </Grid>

      <DailyScanRowsTable
        title="Top Scan Candidates"
        subtitle={
          minDollarVolume == null
            ? 'No default liquidity floor. Click a row for chart details.'
            : `Dollar volume >= ${formatNumber(minDollarVolume)}. Click a row for chart details.`
        }
        rows={topResults}
        isLoading={marketCapMin !== '' && filteredResultsQuery.isLoading}
        isError={marketCapMin !== '' && filteredResultsQuery.isError}
        errorMessage="Failed to load scan candidates."
        emptyMessage="No scan candidates match the current filters."
        showRating
        onOpenChart={scanId ? handleRowClick : null}
        action={(
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
        )}
      />

      <DailyScanRowsTable
        title="Leaders in Leading Groups"
        subtitle={`Top 20 by report card: group rank <= ${snapshot?.leaders?.criteria?.max_group_rank ?? 40}, RS >= ${snapshot?.leaders?.criteria?.min_rs_rating ?? 80}${minDollarVolume != null ? `, dollar volume >= ${formatNumber(minDollarVolume)}` : ''}.`}
        rows={leaders}
        emptyMessage="No leaders in leading groups match the current snapshot."
        showRs
        onOpenChart={scanId ? handleRowClick : null}
      />

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
                <TableCell align="right">1M</TableCell>
                <TableCell>Top Stock</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {topGroups.map((group) => (
                <TableRow key={group.industry_group}>
                  <TableCell align="center" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>{group.rank}</TableCell>
                  <TableCell>{group.industry_group}</TableCell>
                  <TableCell align="right"><RankChangeCell value={group.rank_change_1w} /></TableCell>
                  <TableCell align="right"><RankChangeCell value={group.rank_change_1m} /></TableCell>
                  <TableCell>
                    <TickerCell symbol={group.top_symbol} companyName={group.top_symbol_name} />
                  </TableCell>
                </TableRow>
              ))}
              {topGroups.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} align="center" sx={{ color: 'text.disabled', py: 2 }}>
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
