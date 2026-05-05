import { useState, useMemo } from 'react';
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
import { useStaticManifest, fetchStaticJson, resolveStaticMarketEntry } from '../dataClient';
import { useStaticChartIndex } from '../chartClient';
import PriceSparkline from '../../components/Scan/PriceSparkline';
import RSSparkline from '../../components/Scan/RSSparkline';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import StaticChartViewerModal from '../StaticChartViewerModal';
import RankChangeCell from '../../components/shared/RankChangeCell';
import TickerCell from '../../components/common/TickerCell';
import { getGroupRankColor } from '../../utils/colorUtils';
import { formatLocalCurrency } from '../../utils/formatUtils';
import { useStaticMarket } from '../StaticMarketContext';
import { marketFlag } from '../marketFlags';
import { MARKET_CAP_OPTIONS } from '../../features/scan/components/filterPanel/constants';
import { applyScanFilterDefaults } from '../../features/scan/defaultFilters';
import { filterStaticScanRows, sortStaticScanRows } from '../scanClient';
import { resolveMarketCapDisplay } from '../../utils/marketCapUtils';

const EMPTY_RESULTS = [];
const DEFAULT_MIN_VOLUME = 100_000_000;
const DEFAULT_TOP_RESULTS = 20;

const formatNumber = (value, digits = 0) => {
  if (value == null) return '-';
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
};

function StaticHomePage() {
  const manifestQuery = useStaticManifest();
  const { selectedMarket } = useStaticMarket();
  const marketEntry = useMemo(
    () => resolveStaticMarketEntry(manifestQuery.data, selectedMarket),
    [manifestQuery.data, selectedMarket],
  );
  const homeQuery = useQuery({
    queryKey: ['staticHome', marketEntry.pages?.home?.path],
    queryFn: () => fetchStaticJson(marketEntry.pages.home.path),
    enabled: Boolean(marketEntry.pages?.home?.path),
    staleTime: Infinity,
  });
  const scanRowsQuery = useQuery({
    queryKey: ['staticHomeScanRows', marketEntry.pages?.scan?.path],
    queryFn: async () => {
      const scanManifest = await fetchStaticJson(marketEntry.pages.scan.path);
      const rowsBySymbol = new Map(
        (scanManifest.initial_rows || []).map((row) => [row.symbol, row])
      );
      const chunkPayloads = await Promise.all(
        (scanManifest.chunks || []).map((chunk) => fetchStaticJson(chunk.path))
      );
      chunkPayloads.forEach((payload) => {
        (payload.rows || []).forEach((row) => {
          rowsBySymbol.set(row.symbol, row);
        });
      });
      return Array.from(rowsBySymbol.values());
    },
    enabled: Boolean(marketEntry.pages?.scan?.path),
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const chartIndexQuery = useStaticChartIndex(marketEntry.assets?.charts?.path);

  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedChartSymbol, setSelectedChartSymbol] = useState(null);
  const [marketCapMin, setMarketCapMin] = useState('');
  const topGroups = homeQuery.data?.top_groups ?? EMPTY_RESULTS;
  const topCandidateFilters = useMemo(
    () => applyScanFilterDefaults({
      minVolume: DEFAULT_MIN_VOLUME,
      ...(marketCapMin !== '' ? { marketCapUsd: { min: Number(marketCapMin), max: null } } : {}),
    }),
    [marketCapMin]
  );
  const topResults = useMemo(() => {
    const allRows = scanRowsQuery.data ?? EMPTY_RESULTS;
    return sortStaticScanRows(
      filterStaticScanRows(allRows, topCandidateFilters),
      'composite_score',
      'desc'
    ).slice(0, DEFAULT_TOP_RESULTS);
  }, [scanRowsQuery.data, topCandidateFilters]);

  const chartEntries = useMemo(() => chartIndexQuery.data?.symbols || [], [chartIndexQuery.data]);
  const chartEnabledSymbols = useMemo(() => new Set(chartEntries.map((e) => e.symbol)), [chartEntries]);
  const navigationSymbols = useMemo(
    () => topResults.map((r) => r.symbol).filter((s) => chartEnabledSymbols.has(s)),
    [topResults, chartEnabledSymbols],
  );

  if (manifestQuery.isLoading || homeQuery.isLoading || scanRowsQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError || homeQuery.isError || scanRowsQuery.isError) {
    return (
      <Alert severity="error">
        Failed to load the daily snapshot.
      </Alert>
    );
  }

  const home = homeQuery.data;
  const freshness = home?.freshness || {};
  const marketDisplay = home?.market_display_name || marketEntry.display_name;
  const flag = marketFlag(marketEntry.market);

  const handleRowClick = (symbol) => {
    if (chartEnabledSymbols.has(symbol)) {
      setSelectedChartSymbol(symbol);
      setChartModalOpen(true);
    }
  };

  return (
    <Box>
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
        {(home.key_markets || [])
          .filter((item) => item.latest_close != null && (item.history || []).length > 0)
          .map((item) => {
          const closes = (item.history || []).map((h) => h.close).filter((c) => c != null);
          const trend = closes.length >= 2
            ? (closes[closes.length - 1] > closes[0] ? 1 : closes[closes.length - 1] < closes[0] ? -1 : 0)
            : 0;
          return (
            <Grid item xs={12} sm={6} md={4} lg={2.4} key={item.symbol}>
              <Paper
                elevation={0}
                sx={{
                  p: 1.5,
                  height: '100%',
                  border: '1px solid',
                  borderColor: 'divider',
                  display: 'flex',
                  alignItems: 'stretch',
                  gap: 1.5,
                }}
              >
                <Box sx={{ flex: '0 0 auto', minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '13px' }}>
                    {item.symbol}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '10px' }}>
                    {item.display_name}
                  </Typography>
                  <Typography variant="body1" sx={{ mt: 0.5, fontFamily: 'monospace', fontWeight: 600 }}>
                    {formatLocalCurrency(item.latest_close, item.currency)}
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
                </Box>
                {closes.length > 1 ? (
                  <Box sx={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'stretch' }}>
                    <PriceSparkline
                      data={closes}
                      trend={trend}
                      change1d={null}
                      width="100%"
                      height="100%"
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
                <TableCell align="center">MCap</TableCell>
                <TableCell align="center">Rating</TableCell>
                <TableCell align="center">Price Trend (30d)</TableCell>
                <TableCell align="center">RS Trend (30d)</TableCell>
                <TableCell align="center">IBD Group</TableCell>
                <TableCell align="center">Grp Rank</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {topResults.map((row) => (
                <TableRow
                  key={row.symbol}
                  hover={chartEnabledSymbols.has(row.symbol)}
                  tabIndex={chartEnabledSymbols.has(row.symbol) ? 0 : -1}
                  onClick={() => handleRowClick(row.symbol)}
                  onKeyDown={(event) => {
                    if (!chartEnabledSymbols.has(row.symbol)) return;
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      handleRowClick(row.symbol);
                    }
                  }}
                  sx={{ cursor: chartEnabledSymbols.has(row.symbol) ? 'pointer' : 'default' }}
                >
                  <TableCell align="center">
                    <TickerCell symbol={row.symbol} companyName={row.company_name} align="center" />
                  </TableCell>
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
                          width={137}
                          height={28}
                          sparklineWidth={105}
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
                          width={117}
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
                    fontFamily: 'monospace', fontWeight: row.ibd_group_rank && row.ibd_group_rank <= 20 ? 600 : 400,
                    color: getGroupRankColor(row.ibd_group_rank),
                  }}>
                    {row.ibd_group_rank ?? '-'}
                  </TableCell>
                </TableRow>
              ))}
              {topResults.length === 0 ? (
                <TableRow>
                  <TableCell align="center" colSpan={9}>
                    No scan candidates match the current filters.
                  </TableCell>
                </TableRow>
              ) : null}
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
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <StaticChartViewerModal
        open={chartModalOpen}
        onClose={() => setChartModalOpen(false)}
        initialSymbol={selectedChartSymbol}
        chartIndex={chartIndexQuery.data}
        navigationSymbols={navigationSymbols}
      />
    </Box>
  );
}

export default StaticHomePage;
