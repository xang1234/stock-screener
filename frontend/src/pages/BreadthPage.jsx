import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Container,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Paper,
  Grid,
  Chip,
  FormControl,
  Tab,
  Tabs,
  InputLabel,
  MenuItem,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import {
  getBreadthBootstrap,
  getCurrentBreadth,
  getHistoricalBreadth,
  getBreadthSummary,
} from '../api/breadth';
import { getPriceHistory } from '../api/stocks';
import BreadthChart from '../components/Charts/BreadthChart';
import { format } from 'date-fns';
import { useRuntime } from '../contexts/RuntimeContext';

// Helper function to calculate date range based on time selection
const getDateRange = (range) => {
  const endDate = new Date();
  const startDate = new Date();

  switch (range) {
    case '1M':
      startDate.setMonth(startDate.getMonth() - 1);
      break;
    case '3M':
      startDate.setMonth(startDate.getMonth() - 3);
      break;
    case '6M':
      startDate.setMonth(startDate.getMonth() - 6);
      break;
    case '1Y':
    default:
      startDate.setFullYear(startDate.getFullYear() - 1);
  }

  return {
    startDate: startDate.toISOString().split('T')[0],
    endDate: endDate.toISOString().split('T')[0],
  };
};

// Determine market sentiment for a breadth row (up vs down 4%+ movers)
const getRowSentiment = (row) => {
  if (row.stocks_up_4pct == null || row.stocks_down_4pct == null) return 'neutral';
  if (row.stocks_up_4pct > row.stocks_down_4pct) return 'bullish';
  if (row.stocks_down_4pct > row.stocks_up_4pct) return 'bearish';
  return 'neutral';
};

// Determine sentiment for a ratio value (above/below 1.0)
const getRatioSentiment = (ratio) => {
  if (ratio == null) return 'neutral';
  if (ratio > 1.0) return 'bullish';
  if (ratio < 1.0) return 'bearish';
  return 'neutral';
};

// StockBee-style background colors for row/cell coloring
const sentimentBg = {
  bullish: { row: 'rgba(76, 175, 80, 0.12)', cell: 'rgba(76, 175, 80, 0.18)' },
  bearish: { row: 'rgba(244, 67, 54, 0.12)', cell: 'rgba(244, 67, 54, 0.18)' },
  neutral: { row: 'transparent', cell: 'transparent' },
};

const MARKET_LABELS = {
  US: 'United States',
  HK: 'Hong Kong',
  IN: 'India',
  JP: 'Japan',
  KR: 'South Korea',
  TW: 'Taiwan',
  CN: 'China A-shares',
  SG: 'Singapore',
  CA: 'Canada',
};

const MARKET_LIVE_BENCHMARK_SYMBOLS = {
  US: 'SPY',
  HK: '2800.HK',
  IN: 'NIFTYBEES.NS',
  JP: '1306.T',
  KR: '069500.KS',
  TW: '0050.TW',
  CN: '000300.SS',
  SG: 'ES3.SI',
  CA: 'XIU.TO',
};

function normalizeMarket(market) {
  const normalized = String(market || 'US').trim().toUpperCase();
  return MARKET_LABELS[normalized] ? normalized : 'US';
}

function BreadthPage() {
  const {
    runtimeReady,
    uiSnapshots,
    primaryMarket = 'US',
    enabledMarkets = ['US'],
    supportedMarkets = ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'SG', 'CA', 'DE'],
  } = useRuntime();
  const queryClient = useQueryClient();
  const [selectedTab, setSelectedTab] = useState(0);
  const [chartTimeRange, setChartTimeRange] = useState('1M');
  const [bootstrapSettled, setBootstrapSettled] = useState(false);
  const [selectedMarket, setSelectedMarket] = useState(() => normalizeMarket(primaryMarket));
  const userSelectedMarketRef = useRef(false);
  const previousPrimaryMarketRef = useRef(normalizeMarket(primaryMarket));
  const marketOptions = useMemo(() => {
    const enabled = (enabledMarkets || [])
      .map(normalizeMarket)
      .filter((market, index, markets) => markets.indexOf(market) === index);
    if (enabled.length > 0) {
      return enabled;
    }
    return (supportedMarkets || ['US'])
      .map(normalizeMarket)
      .filter((market, index, markets) => markets.indexOf(market) === index);
  }, [enabledMarkets, supportedMarkets]);
  useEffect(() => {
    if (marketOptions.length === 0) {
      return;
    }
    const normalizedPrimary = normalizeMarket(primaryMarket);
    const fallbackMarket = marketOptions.includes(normalizedPrimary)
      ? normalizedPrimary
      : marketOptions[0];
    const primaryChanged = previousPrimaryMarketRef.current !== normalizedPrimary;

    setSelectedMarket((currentMarket) => {
      if (!marketOptions.includes(currentMarket)) {
        userSelectedMarketRef.current = false;
        return fallbackMarket;
      }
      if (
        !userSelectedMarketRef.current
        && marketOptions.includes(normalizedPrimary)
        && currentMarket !== normalizedPrimary
      ) {
        return normalizedPrimary;
      }
      if (
        primaryChanged
        && !userSelectedMarketRef.current
        && currentMarket !== fallbackMarket
      ) {
        return fallbackMarket;
      }
      return currentMarket;
    });
    previousPrimaryMarketRef.current = normalizedPrimary;
  }, [marketOptions, primaryMarket]);
  useEffect(() => {
    setBootstrapSettled(false);
  }, [selectedMarket]);
  const snapshotEnabled = runtimeReady && Boolean(uiSnapshots?.breadth);
  const liveQueriesEnabled = runtimeReady && (!snapshotEnabled || bootstrapSettled);

  // Calculate date range for chart based on selected time range
  const chartDateRange = getDateRange(chartTimeRange);
  const defaultChartDateRange = getDateRange('1M');
  const endDate = new Date().toISOString().split('T')[0];
  const startDate = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  const spyPeriodMap = { '1M': '1mo', '3M': '3mo', '6M': '6mo', '1Y': '1y' };
  const spyPeriod = spyPeriodMap[chartTimeRange] || '1y';
  const benchmarkSymbol = MARKET_LIVE_BENCHMARK_SYMBOLS[selectedMarket] || null;

  const breadthBootstrapQuery = useQuery({
    queryKey: ['breadthBootstrap', selectedMarket],
    queryFn: () => getBreadthBootstrap(selectedMarket),
    enabled: snapshotEnabled && !bootstrapSettled,
    retry: false,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (!snapshotEnabled) {
      return;
    }
    if (breadthBootstrapQuery.isError) {
      setBootstrapSettled(true);
      return;
    }
    if (!breadthBootstrapQuery.isSuccess) {
      return;
    }
    if (breadthBootstrapQuery.data?.is_stale) {
      setBootstrapSettled(true);
      return;
    }

    const payload = breadthBootstrapQuery.data?.payload ?? {};
    queryClient.setQueryData(['breadth', 'current', selectedMarket], payload.current ?? null);
    queryClient.setQueryData(['breadth', 'historical', selectedMarket, startDate, endDate], payload.history_90d ?? []);
    queryClient.setQueryData(['breadth', 'summary', selectedMarket], payload.summary ?? {});
    if (payload.chart_range === '1M') {
      queryClient.setQueryData(
        ['breadth', 'chart', selectedMarket, defaultChartDateRange.startDate, defaultChartDateRange.endDate],
        payload.chart_data ?? []
      );
      if (benchmarkSymbol) {
        queryClient.setQueryData(
          ['benchmark', 'history', selectedMarket, benchmarkSymbol, '1mo'],
          payload.benchmark_overlay ?? payload.spy_overlay ?? []
        );
      }
    }
    setBootstrapSettled(true);
  }, [
    benchmarkSymbol,
    breadthBootstrapQuery.data,
    breadthBootstrapQuery.isError,
    breadthBootstrapQuery.isSuccess,
    defaultChartDateRange.endDate,
    defaultChartDateRange.startDate,
    endDate,
    queryClient,
    selectedMarket,
    snapshotEnabled,
    startDate,
  ]);

  // Fetch current breadth data
  const {
    data: currentBreadth,
    isLoading: isLoadingCurrent,
    error: errorCurrent,
  } = useQuery({
    queryKey: ['breadth', 'current', selectedMarket],
    queryFn: () => getCurrentBreadth(selectedMarket),
    enabled: liveQueriesEnabled,
    refetchInterval: 60000, // Refetch every minute
    staleTime: 60_000,
  });

  // Fetch historical data (last 90 days)
  const {
    data: historicalBreadth,
  } = useQuery({
    queryKey: ['breadth', 'historical', selectedMarket, startDate, endDate],
    queryFn: () => getHistoricalBreadth(startDate, endDate, 365, selectedMarket),
    enabled: liveQueriesEnabled,
    staleTime: 60_000,
  });

  // Fetch summary statistics
  useQuery({
    queryKey: ['breadth', 'summary', selectedMarket],
    queryFn: () => getBreadthSummary(selectedMarket),
    enabled: liveQueriesEnabled,
    staleTime: 60_000,
  });

  // Fetch extended breadth data for chart (up to 2 years)
  const {
    data: chartBreadthData,
    isLoading: isLoadingChartBreadth,
    error: errorChartBreadth,
  } = useQuery({
    queryKey: ['breadth', 'chart', selectedMarket, chartDateRange.startDate, chartDateRange.endDate],
    queryFn: () => getHistoricalBreadth(chartDateRange.startDate, chartDateRange.endDate, 730, selectedMarket),
    enabled: liveQueriesEnabled,
    staleTime: 60_000,
  });

  // Fetch optional benchmark history for the overlay
  const {
    data: benchmarkData,
  } = useQuery({
    queryKey: ['benchmark', 'history', selectedMarket, benchmarkSymbol, spyPeriod],
    queryFn: () => getPriceHistory(benchmarkSymbol, spyPeriod),
    enabled: liveQueriesEnabled && Boolean(benchmarkSymbol),
    staleTime: 60_000,
  });

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const marketSelector = (
    <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
      <FormControl size="small" sx={{ minWidth: 180 }}>
        <InputLabel id="breadth-market-label">Market</InputLabel>
        <Select
          labelId="breadth-market-label"
          value={selectedMarket}
          label="Market"
          onChange={(event) => {
            userSelectedMarketRef.current = true;
            setSelectedMarket(event.target.value);
          }}
        >
          {marketOptions.map((market) => (
            <MenuItem key={market} value={market}>
              {MARKET_LABELS[market] || market}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </Box>
  );

  if (!runtimeReady) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (errorCurrent) {
    return (
      <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
        {marketSelector}
        <Alert severity="error">
          Error loading {selectedMarket} breadth data: {errorCurrent.message}
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
      {isLoadingCurrent ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      ) : (
        <>
          {marketSelector}

          {/* Top Section - Side by Side Layout */}
          <Grid container spacing={2} sx={{ mb: 2 }}>
            {/* Left: Chart (60%) */}
            <Grid item xs={12} lg={7} sx={{ height: 450 }}>
              <BreadthChart
                breadthData={chartBreadthData}
                spyData={benchmarkData || []}
                benchmarkLabel={benchmarkSymbol || 'Benchmark'}
                isLoading={isLoadingChartBreadth}
                error={errorChartBreadth}
                timeRange={chartTimeRange}
                onTimeRangeChange={setChartTimeRange}
                fillContainer
              />
            </Grid>

            {/* Right: Data Panel (40%) */}
            <Grid item xs={12} lg={5} sx={{ height: 450 }}>
              <Paper elevation={1} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                {/* Header */}
                <Box sx={{ p: 1, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ fontSize: '13px', fontWeight: 600 }}>
                    Latest Breadth Data
                  </Box>
                  {currentBreadth && (
                    <Chip
                      label={format(new Date(currentBreadth.date), 'MMM dd, yyyy')}
                      size="small"
                    />
                  )}
                </Box>

                {currentBreadth && (
                  <>
                    {/* Daily Movers and Ratios - 40% of panel height */}
                    <Box sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider', flex: '0 0 40%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      <Grid container spacing={1}>
                        {/* Daily Movers */}
                        <Grid item xs={6}>
                          <Box sx={{ fontSize: '13px', fontWeight: 600, color: 'text.secondary', mb: 1, textAlign: 'center' }}>
                            Daily Movers (4%+)
                          </Box>
                          <Box display="flex" justifyContent="center" gap={2}>
                            <Box textAlign="center">
                              <TrendingUpIcon color="success" sx={{ fontSize: 16 }} />
                              <Box sx={{ fontSize: '18px', fontWeight: 700, color: 'success.main', fontFamily: 'monospace' }}>
                                {currentBreadth.stocks_up_4pct}
                              </Box>
                            </Box>
                            <Box textAlign="center">
                              <TrendingDownIcon color="error" sx={{ fontSize: 16 }} />
                              <Box sx={{ fontSize: '18px', fontWeight: 700, color: 'error.main', fontFamily: 'monospace' }}>
                                {currentBreadth.stocks_down_4pct}
                              </Box>
                            </Box>
                          </Box>
                        </Grid>

                        {/* Ratios */}
                        <Grid item xs={6}>
                          <Box sx={{ fontSize: '13px', fontWeight: 600, color: 'text.secondary', mb: 1, textAlign: 'center' }}>
                            Multi-Day Ratios
                          </Box>
                          <Box display="flex" justifyContent="center" gap={2}>
                            <Box textAlign="center">
                              <Box sx={{
                                fontSize: '18px',
                                fontWeight: 700,
                                fontFamily: 'monospace',
                                color: getRatioSentiment(currentBreadth.ratio_5day) === 'bullish' ? 'success.main'
                                  : getRatioSentiment(currentBreadth.ratio_5day) === 'bearish' ? 'error.main'
                                  : 'text.primary',
                              }}>
                                {currentBreadth.ratio_5day?.toFixed(2) || 'N/A'}
                              </Box>
                              <Box sx={{ fontSize: '9px', color: 'text.secondary' }}>5D</Box>
                            </Box>
                            <Box textAlign="center">
                              <Box sx={{
                                fontSize: '18px',
                                fontWeight: 700,
                                fontFamily: 'monospace',
                                color: getRatioSentiment(currentBreadth.ratio_10day) === 'bullish' ? 'success.main'
                                  : getRatioSentiment(currentBreadth.ratio_10day) === 'bearish' ? 'error.main'
                                  : 'text.primary',
                              }}>
                                {currentBreadth.ratio_10day?.toFixed(2) || 'N/A'}
                              </Box>
                              <Box sx={{ fontSize: '9px', color: 'text.secondary' }}>10D</Box>
                            </Box>
                          </Box>
                        </Grid>
                      </Grid>
                    </Box>

                    {/* Tabs for different period views - 60% of panel height */}
                    <Box sx={{ flex: '0 0 60%', display: 'flex', flexDirection: 'column' }}>
                      <Tabs
                        value={selectedTab}
                        onChange={handleTabChange}
                        variant="scrollable"
                        scrollButtons="auto"
                        sx={{
                          minHeight: 36,
                          '& .MuiTab-root': {
                            minHeight: 36,
                            py: 0.5,
                            px: 1,
                            fontSize: '11px',
                            minWidth: 'auto'
                          }
                        }}
                      >
                        <Tab label="Quarterly" />
                        <Tab label="Monthly" />
                        <Tab label="Explosive" />
                        <Tab label="34-Day" />
                      </Tabs>

                      <Box sx={{ p: 1.5, flex: 1, display: 'flex', alignItems: 'center' }}>
                        {selectedTab === 0 && (
                          <Grid container spacing={1}>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Box sx={{ fontSize: '28px', fontWeight: 700, color: 'success.main', fontFamily: 'monospace' }}>
                                  {currentBreadth.stocks_up_25pct_quarter}
                                </Box>
                                <Box sx={{ fontSize: '12px', fontWeight: 600, color: 'text.secondary' }}>Up 25%+ (63d)</Box>
                              </Box>
                            </Grid>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Box sx={{ fontSize: '28px', fontWeight: 700, color: 'error.main', fontFamily: 'monospace' }}>
                                  {currentBreadth.stocks_down_25pct_quarter}
                                </Box>
                                <Box sx={{ fontSize: '12px', fontWeight: 600, color: 'text.secondary' }}>Down 25%+ (63d)</Box>
                              </Box>
                            </Grid>
                          </Grid>
                        )}

                        {selectedTab === 1 && (
                          <Grid container spacing={1}>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Box sx={{ fontSize: '28px', fontWeight: 700, color: 'success.main', fontFamily: 'monospace' }}>
                                  {currentBreadth.stocks_up_25pct_month}
                                </Box>
                                <Box sx={{ fontSize: '12px', fontWeight: 600, color: 'text.secondary' }}>Up 25%+ (21d)</Box>
                              </Box>
                            </Grid>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Box sx={{ fontSize: '28px', fontWeight: 700, color: 'error.main', fontFamily: 'monospace' }}>
                                  {currentBreadth.stocks_down_25pct_month}
                                </Box>
                                <Box sx={{ fontSize: '12px', fontWeight: 600, color: 'text.secondary' }}>Down 25%+ (21d)</Box>
                              </Box>
                            </Grid>
                          </Grid>
                        )}

                        {selectedTab === 2 && (
                          <Grid container spacing={1}>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Box sx={{ fontSize: '28px', fontWeight: 700, color: 'success.main', fontFamily: 'monospace' }}>
                                  {currentBreadth.stocks_up_50pct_month}
                                </Box>
                                <Box sx={{ fontSize: '12px', fontWeight: 600, color: 'text.secondary' }}>Up 50%+ (21d)</Box>
                              </Box>
                            </Grid>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Box sx={{ fontSize: '28px', fontWeight: 700, color: 'error.main', fontFamily: 'monospace' }}>
                                  {currentBreadth.stocks_down_50pct_month}
                                </Box>
                                <Box sx={{ fontSize: '12px', fontWeight: 600, color: 'text.secondary' }}>Down 50%+ (21d)</Box>
                              </Box>
                            </Grid>
                          </Grid>
                        )}

                        {selectedTab === 3 && (
                          <Grid container spacing={1}>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Box sx={{ fontSize: '28px', fontWeight: 700, color: 'success.main', fontFamily: 'monospace' }}>
                                  {currentBreadth.stocks_up_13pct_34days}
                                </Box>
                                <Box sx={{ fontSize: '12px', fontWeight: 600, color: 'text.secondary' }}>Up 13%+ (34d)</Box>
                              </Box>
                            </Grid>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Box sx={{ fontSize: '28px', fontWeight: 700, color: 'error.main', fontFamily: 'monospace' }}>
                                  {currentBreadth.stocks_down_13pct_34days}
                                </Box>
                                <Box sx={{ fontSize: '12px', fontWeight: 600, color: 'text.secondary' }}>Down 13%+ (34d)</Box>
                              </Box>
                            </Grid>
                          </Grid>
                        )}
                      </Box>
                    </Box>
                  </>
                )}
              </Paper>
            </Grid>
          </Grid>

          {/* Historical Data Table */}
          {historicalBreadth && historicalBreadth.length > 0 && (
            <Paper sx={{ mt: 2 }} elevation={1}>
              <Box sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider' }}>
                <Box sx={{ fontSize: '14px', fontWeight: 600 }}>Recent History (Last 90 Days)</Box>
              </Box>
              <TableContainer sx={{ maxHeight: 'calc(100vh - 500px)' }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Date</TableCell>
                      <TableCell align="right">Up 4%+</TableCell>
                      <TableCell align="right">Down 4%+</TableCell>
                      <TableCell align="right">5D Ratio</TableCell>
                      <TableCell align="right">10D Ratio</TableCell>
                      <TableCell align="right">Scanned</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {historicalBreadth.slice(0, 30).map((row) => {
                      const rowSentiment = getRowSentiment(row);
                      const ratio5Sentiment = getRatioSentiment(row.ratio_5day);
                      const ratio10Sentiment = getRatioSentiment(row.ratio_10day);

                      return (
                        <TableRow
                          key={row.date}
                          hover
                          sx={{
                            '&&': { backgroundColor: sentimentBg[rowSentiment].row },
                          }}
                        >
                          <TableCell sx={{ fontFamily: 'monospace' }}>{format(new Date(row.date), 'MM/dd')}</TableCell>
                          <TableCell align="right" sx={{ color: 'success.main', fontWeight: 600, fontFamily: 'monospace' }}>
                            {row.stocks_up_4pct}
                          </TableCell>
                          <TableCell align="right" sx={{ color: 'error.main', fontWeight: 600, fontFamily: 'monospace' }}>
                            {row.stocks_down_4pct}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              fontFamily: 'monospace',
                              fontWeight: ratio5Sentiment !== 'neutral' ? 600 : 400,
                              backgroundColor: sentimentBg[ratio5Sentiment].cell,
                              color: ratio5Sentiment === 'bullish' ? 'success.main'
                                : ratio5Sentiment === 'bearish' ? 'error.main'
                                : 'text.primary',
                            }}
                          >
                            {row.ratio_5day?.toFixed(2) || '-'}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              fontFamily: 'monospace',
                              fontWeight: ratio10Sentiment !== 'neutral' ? 600 : 400,
                              backgroundColor: sentimentBg[ratio10Sentiment].cell,
                              color: ratio10Sentiment === 'bullish' ? 'success.main'
                                : ratio10Sentiment === 'bearish' ? 'error.main'
                                : 'text.primary',
                            }}
                          >
                            {row.ratio_10day?.toFixed(2) || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.total_stocks_scanned}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          )}

          {/* Metadata */}
          {currentBreadth && (
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Total stocks scanned: {currentBreadth.total_stocks_scanned} |
                Calculation time: {currentBreadth.calculation_duration_seconds?.toFixed(2) || 'N/A'}s
              </Typography>
            </Box>
          )}
        </>
      )}
    </Container>
  );
}

export default BreadthPage;
