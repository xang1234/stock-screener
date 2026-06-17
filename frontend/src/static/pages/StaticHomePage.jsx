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
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import StaticChartViewerModal from '../StaticChartViewerModal';
import RankChangeCell from '../../components/shared/RankChangeCell';
import TickerCell from '../../components/common/TickerCell';
import { formatLocalCurrency } from '../../utils/formatUtils';
import { useStaticMarket } from '../StaticMarketContext';
import { marketFlag } from '../../utils/marketFlags';
import { MARKET_CAP_OPTIONS } from '../../features/scan/components/filterPanel/constants';
import { applyScanFilterDefaults } from '../../features/scan/defaultFilters';
import { filterStaticScanRows, sortStaticScanRows } from '../scanClient';
import DailyScanRowsTable from '../../components/shared/DailyScanRowsTable';
import { buildFiltersFromPreset } from '../hooks/usePresetScreens';

const EMPTY_RESULTS = [];
const DEFAULT_TOP_RESULTS = 20;
const LEADERS_SCREEN_ID = 'leaders_in_leading_groups';

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
  const scanBundleQuery = useQuery({
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
      return {
        rows: Array.from(rowsBySymbol.values()),
        defaultFilters: scanManifest.default_filters || {},
        presetScreens: scanManifest.preset_screens || [],
      };
    },
    enabled: Boolean(marketEntry.pages?.scan?.path),
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const chartIndexQuery = useStaticChartIndex(marketEntry.assets?.charts?.path);

  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedChartSymbol, setSelectedChartSymbol] = useState(null);
  const [modalNavigationSymbols, setModalNavigationSymbols] = useState([]);
  const [marketCapMin, setMarketCapMin] = useState('');
  const topGroups = homeQuery.data?.top_groups ?? EMPTY_RESULTS;
  const scanDefaultFilters = useMemo(
    () => scanBundleQuery.data?.defaultFilters ?? {},
    [scanBundleQuery.data?.defaultFilters]
  );
  const topCandidateFilters = useMemo(
    () => applyScanFilterDefaults({
      ...scanDefaultFilters,
      ...(marketCapMin !== '' ? { marketCapUsd: { min: Number(marketCapMin), max: null } } : {}),
    }),
    [marketCapMin, scanDefaultFilters]
  );
  const scanRows = scanBundleQuery.data?.rows ?? EMPTY_RESULTS;
  const topResults = useMemo(() => {
    return sortStaticScanRows(
      filterStaticScanRows(scanRows, topCandidateFilters),
      'composite_score',
      'desc'
    ).slice(0, DEFAULT_TOP_RESULTS);
  }, [scanRows, topCandidateFilters]);
  const leadingGroupScreen = useMemo(
    () => scanBundleQuery.data?.presetScreens?.find((screen) => screen.id === LEADERS_SCREEN_ID) ?? null,
    [scanBundleQuery.data?.presetScreens]
  );
  const leadingGroupRows = useMemo(() => {
    if (!leadingGroupScreen) {
      return EMPTY_RESULTS;
    }
    return sortStaticScanRows(
      filterStaticScanRows(scanRows, buildFiltersFromPreset(leadingGroupScreen)),
      leadingGroupScreen.sort_by,
      leadingGroupScreen.sort_order,
      { prioritizeCompositeScanMode: false }
    ).slice(0, DEFAULT_TOP_RESULTS);
  }, [leadingGroupScreen, scanRows]);

  const chartEntries = useMemo(() => chartIndexQuery.data?.symbols || [], [chartIndexQuery.data]);
  const chartEnabledSymbols = useMemo(() => new Set(chartEntries.map((e) => e.symbol)), [chartEntries]);
  const topNavigationSymbols = useMemo(
    () => topResults.map((r) => r.symbol).filter((s) => chartEnabledSymbols.has(s)),
    [topResults, chartEnabledSymbols],
  );
  const leadingGroupNavigationSymbols = useMemo(
    () => leadingGroupRows.map((r) => r.symbol).filter((s) => chartEnabledSymbols.has(s)),
    [leadingGroupRows, chartEnabledSymbols],
  );
  const leadingGroupMinVolume = leadingGroupScreen?.filters?.minVolume;
  const leadingGroupSubtitle = leadingGroupMinVolume == null
    ? 'Top 20 by report card: group rank <=40, RS >=80.'
    : `Top 20 by report card: group rank <=40, RS >=80, dollar volume >= ${formatNumber(leadingGroupMinVolume)}.`;

  if (manifestQuery.isLoading || homeQuery.isLoading || scanBundleQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError || homeQuery.isError || scanBundleQuery.isError) {
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

  const handleRowClick = (symbol, navigationSymbols) => {
    if (chartEnabledSymbols.has(symbol)) {
      setSelectedChartSymbol(symbol);
      setModalNavigationSymbols(navigationSymbols);
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
          .map((item) => ({
            ...item,
            _closes: (item.history || []).map((h) => h.close).filter((c) => c != null),
          }))
          .filter((item) => item.latest_close != null && item._closes.length > 1)
          .map((item) => {
          const closes = item._closes;
          const trend = closes[closes.length - 1] > closes[0]
            ? 1
            : closes[closes.length - 1] < closes[0]
              ? -1
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
              </Paper>
            </Grid>
          );
        })}
      </Grid>

      <DailyScanRowsTable
        testId="top-scan-candidates-section"
        title="Top Scan Candidates"
        subtitle={
          topCandidateFilters.minVolume == null
            ? 'No default liquidity floor. Click a row for chart details.'
            : `Dollar volume >= ${formatNumber(topCandidateFilters.minVolume)}. Click a row for chart details.`
        }
        rows={topResults}
        chartEnabledSymbols={chartEnabledSymbols}
        navigationSymbols={topNavigationSymbols}
        onOpenChart={handleRowClick}
        emptyMessage="No scan candidates match the current filters."
        showRating
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
        testId="leaders-in-leading-groups-section"
        title="Leaders in Leading Groups"
        subtitle={leadingGroupSubtitle}
        rows={leadingGroupRows}
        chartEnabledSymbols={chartEnabledSymbols}
        navigationSymbols={leadingGroupNavigationSymbols}
        onOpenChart={handleRowClick}
        emptyMessage="No leaders in leading groups match the current snapshot."
        showRs
        priceSparklineWidth={195}
        priceSparklineInnerWidth={150}
      />

      <Paper elevation={0} sx={{ p: 1.5, border: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
          Top 10 Groups
        </Typography>
        <TableContainer>
          <Table size="small" sx={{ minWidth: 520 }}>
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
        navigationSymbols={modalNavigationSymbols}
      />
    </Box>
  );
}

export default StaticHomePage;
