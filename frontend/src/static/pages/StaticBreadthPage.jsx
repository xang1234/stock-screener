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
import { useStaticManifest, fetchStaticJson } from '../dataClient';

function MetricCard({ label, value }) {
  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="h6" sx={{ mt: 0.5 }}>
        {value ?? '-'}
      </Typography>
    </Paper>
  );
}

function StaticBreadthPage() {
  const manifestQuery = useStaticManifest();
  const breadthQuery = useQuery({
    queryKey: ['staticBreadth', manifestQuery.data?.pages?.breadth?.path],
    queryFn: () => fetchStaticJson(manifestQuery.data.pages.breadth.path),
    enabled: Boolean(manifestQuery.data?.pages?.breadth?.path),
    staleTime: Infinity,
  });

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

  const payload = breadthQuery.data?.payload || {};
  const current = payload.current || {};
  const history = payload.history_90d || [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Market Breadth
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Read-only breadth snapshot published {breadthQuery.data.published_at || breadthQuery.data.generated_at}.
      </Typography>

      <Grid container spacing={2} sx={{ mb: 3 }}>
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
        breadthData={payload.chart_data || []}
        spyData={payload.spy_overlay || []}
        isLoading={false}
        error={null}
        timeRange="1M"
        onTimeRangeChange={() => {}}
        availableRanges={['1M']}
      />

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
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
