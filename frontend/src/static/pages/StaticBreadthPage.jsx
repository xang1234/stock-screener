import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  CircularProgress,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import BreadthChart from '../../components/Charts/BreadthChart';
import { useStaticManifest, fetchStaticJson, resolveStaticMarketEntry } from '../dataClient';
import { useStaticMarket } from '../StaticMarketContext';

const RANGE_DAYS = { '1M': 31, '3M': 90 };

function MetricCard({ label, value }) {
  return (
    <Paper elevation={0} sx={{ p: 1.5, height: '100%', border: '1px solid', borderColor: 'divider' }}>
      <Typography variant="caption" sx={{ fontSize: '10px', letterSpacing: '0.5px', textTransform: 'uppercase', color: 'text.disabled' }}>
        {label}
      </Typography>
      <Typography variant="body1" sx={{ mt: 0.25, fontFamily: 'monospace', fontWeight: 600 }}>
        {value ?? '-'}
      </Typography>
    </Paper>
  );
}

function StaticBreadthPage() {
  const manifestQuery = useStaticManifest();
  const { selectedMarket } = useStaticMarket();
  const marketEntry = useMemo(
    () => resolveStaticMarketEntry(manifestQuery.data, selectedMarket),
    [manifestQuery.data, selectedMarket],
  );
  const breadthQuery = useQuery({
    queryKey: ['staticBreadth', marketEntry.pages?.breadth?.path],
    queryFn: () => fetchStaticJson(marketEntry.pages.breadth.path),
    enabled: Boolean(marketEntry.pages?.breadth?.path),
    staleTime: Infinity,
  });
  const [timeRange, setTimeRange] = useState('1M');

  const payload = breadthQuery.data?.payload || {};
  const displayName = marketEntry.display_name;
  const filteredChartData = useMemo(() => {
    const allData = payload.chart_data || payload.history_90d || [];
    return allData.slice(-(RANGE_DAYS[timeRange] || 31));
  }, [payload.chart_data, payload.history_90d, timeRange]);
  const filteredSpyData = useMemo(() => {
    const allSpy = payload.benchmark_overlay ?? payload.spy_overlay ?? [];
    return allSpy.slice(-(RANGE_DAYS[timeRange] || 31));
  }, [payload.benchmark_overlay, payload.spy_overlay, timeRange]);
  const benchmarkLabel = payload.benchmark_symbol || (marketEntry.market === 'US' ? 'SPY' : 'Benchmark');

  if (manifestQuery.isLoading || breadthQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError || breadthQuery.isError) {
    return <Alert severity="error">Failed to load breadth data.</Alert>;
  }

  if (breadthQuery.data?.available === false) {
    return <Alert severity="info">{breadthQuery.data?.message || 'No breadth snapshot is available.'}</Alert>;
  }

  const current = payload.current || {};
  const history = payload.history_90d || [];

  return (
    <Box>
      <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: '-0.5px', mb: 0.5 }}>
        {displayName} Breadth
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: '12px' }}>
        Breadth snapshot published {breadthQuery.data.published_at || breadthQuery.data.generated_at}.
      </Typography>

      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard label="Date" value={current.date} />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard label="Stocks Up 4%+" value={current.stocks_up_4pct} />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard label="Stocks Down 4%+" value={current.stocks_down_4pct} />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard label="10-day Ratio" value={current.ratio_10day?.toFixed?.(2) ?? '-'} />
        </Grid>
      </Grid>

      <BreadthChart
        breadthData={filteredChartData}
        spyData={filteredSpyData}
        benchmarkLabel={benchmarkLabel}
        isLoading={false}
        error={null}
        timeRange={timeRange}
        onTimeRangeChange={setTimeRange}
        availableRanges={['1M', '3M']}
      />

      <Paper elevation={0} sx={{ p: 1.5, border: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
          Recent Sessions
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Date</TableCell>
                <TableCell align="right">Up 4%+</TableCell>
                <TableCell align="right">Down 4%+</TableCell>
                <TableCell align="right">5-day Ratio</TableCell>
                <TableCell align="right">10-day Ratio</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {history.slice(0, 20).map((row) => (
                <TableRow key={row.date}>
                  <TableCell>{row.date}</TableCell>
                  <TableCell align="right">{row.stocks_up_4pct}</TableCell>
                  <TableCell align="right">{row.stocks_down_4pct}</TableCell>
                  <TableCell align="right">{row.ratio_5day?.toFixed?.(2) ?? '-'}</TableCell>
                  <TableCell align="right">{row.ratio_10day?.toFixed?.(2) ?? '-'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  );
}

export default StaticBreadthPage;
