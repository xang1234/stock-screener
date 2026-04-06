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
import { useStaticManifest, fetchStaticJson } from '../dataClient';
import { useStaticChartIndex } from '../chartClient';
import PriceSparkline from '../../components/Scan/PriceSparkline';
import RSSparkline from '../../components/Scan/RSSparkline';
import StaticChartViewerModal from '../StaticChartViewerModal';
import { getGroupRankColor } from '../../utils/colorUtils';

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
  const chartIndexQuery = useStaticChartIndex(manifestQuery.data?.assets?.charts?.path);

  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedChartSymbol, setSelectedChartSymbol] = useState(null);

  const topResults = homeQuery.data?.scan_summary?.top_results || [];
  const topGroups = homeQuery.data?.top_groups || [];

  const chartEntries = useMemo(() => chartIndexQuery.data?.symbols || [], [chartIndexQuery.data]);
  const chartEnabledSymbols = useMemo(() => new Set(chartEntries.map((e) => e.symbol)), [chartEntries]);
  const navigationSymbols = useMemo(
    () => topResults.map((r) => r.symbol).filter((s) => chartEnabledSymbols.has(s)),
    [topResults, chartEnabledSymbols],
  );

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

  const handleRowClick = (symbol) => {
    if (chartEnabledSymbols.has(symbol)) {
      setSelectedChartSymbol(symbol);
      setChartModalOpen(true);
    }
  };

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

      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Top Scan Candidates
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Default view uses dollar volume &gt; $100M. Click a row for chart details.
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell align="center">Symbol</TableCell>
                <TableCell align="center">Score</TableCell>
                <TableCell align="center">Price</TableCell>
                <TableCell align="center">Rating</TableCell>
                <TableCell align="center">Price Trend</TableCell>
                <TableCell align="center">RS Trend</TableCell>
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
                  <TableCell align="center" sx={{ fontWeight: 600 }}>{row.symbol}</TableCell>
                  <TableCell align="center">{formatNumber(row.composite_score, 1)}</TableCell>
                  <TableCell align="center">{row.current_price != null ? `$${formatNumber(row.current_price, 2)}` : '-'}</TableCell>
                  <TableCell align="center">{row.rating}</TableCell>
                  <TableCell align="center">
                    {row.price_sparkline_data ? (
                      <Box display="flex" justifyContent="center">
                        <PriceSparkline
                          data={row.price_sparkline_data}
                          trend={row.price_trend}
                          change1d={row.price_change_1d}
                          industry={row.ibd_industry_group}
                          width={100}
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
                          width={60}
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
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <Paper sx={{ p: 2 }}>
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
