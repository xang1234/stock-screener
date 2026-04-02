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
import { useStaticManifest, fetchStaticJson } from '../dataClient';

const formatNumber = (value, digits = 0) => {
  if (value == null) return '-';
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
};

function SummaryCard({ label, value, helper }) {
  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="h6" sx={{ mt: 0.5, fontWeight: 600 }}>
        {value}
      </Typography>
      {helper ? (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75 }}>
          {helper}
        </Typography>
      ) : null}
    </Paper>
  );
}

function StaticHomePage() {
  const manifestQuery = useStaticManifest();
  const homeQuery = useQuery({
    queryKey: ['staticHome', manifestQuery.data?.pages?.home?.path],
    queryFn: () => fetchStaticJson(manifestQuery.data.pages.home.path),
    enabled: Boolean(manifestQuery.data?.pages?.home?.path),
    staleTime: Infinity,
  });

  if (manifestQuery.isLoading || homeQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError || homeQuery.isError) {
    return (
      <Alert severity="error">
        Failed to load the daily snapshot.
      </Alert>
    );
  }

  const home = homeQuery.data;
  const freshness = home?.freshness || {};
  const topResults = home?.scan_summary?.top_results || [];
  const topGroups = home?.top_groups || [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Daily Market Snapshot
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Generated {home.generated_at}. Data reflects the latest published daily snapshot as of {home.as_of_date}.
      </Typography>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <SummaryCard
            label="Feature Snapshot"
            value={freshness.scan_as_of_date || '-'}
            helper={`Run ${freshness.scan_run_id ?? '-'}`}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <SummaryCard
            label="Breadth Date"
            value={freshness.breadth_latest_date || '-'}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <SummaryCard
            label="Group Rankings"
            value={freshness.groups_latest_date || '-'}
          />
        </Grid>
      </Grid>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        {(home.key_markets || []).map((item) => (
          <Grid item xs={12} sm={6} md={2.4} key={item.symbol}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                {item.symbol}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {item.display_name}
              </Typography>
              <Typography variant="h6" sx={{ mt: 1 }}>
                {item.latest_close != null ? `$${formatNumber(item.latest_close, 2)}` : '-'}
              </Typography>
              <Typography
                variant="body2"
                color={item.change_1d > 0 ? 'success.main' : item.change_1d < 0 ? 'error.main' : 'text.secondary'}
              >
                {item.change_1d != null ? `${formatNumber(item.change_1d, 2)}%` : 'No daily change'}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={2}>
        <Grid item xs={12} lg={7}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Top Scan Candidates
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell align="right">Score</TableCell>
                    <TableCell align="right">Price</TableCell>
                    <TableCell>Rating</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {topResults.map((row) => (
                    <TableRow key={row.symbol}>
                      <TableCell sx={{ fontWeight: 600 }}>{row.symbol}</TableCell>
                      <TableCell align="right">{formatNumber(row.composite_score, 1)}</TableCell>
                      <TableCell align="right">{row.current_price != null ? `$${formatNumber(row.current_price, 2)}` : '-'}</TableCell>
                      <TableCell>{row.rating}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} lg={5}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Leading Groups
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Group</TableCell>
                    <TableCell align="right">Rank</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {topGroups.map((group) => (
                    <TableRow key={group.industry_group}>
                      <TableCell>{group.industry_group}</TableCell>
                      <TableCell align="right">{group.rank}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default StaticHomePage;
