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
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import { useStaticManifest, fetchStaticJson } from '../dataClient';
import { useStaticChartIndex } from '../chartClient';
import PriceSparkline from '../../components/Scan/PriceSparkline';
import RSSparkline from '../../components/Scan/RSSparkline';
import StaticChartViewerModal from '../StaticChartViewerModal';
import RankChangeCell from '../../components/shared/RankChangeCell';
import { getGroupRankColor } from '../../utils/colorUtils';

const EMPTY_RESULTS = [];

const formatNumber = (value, digits = 0) => {
  if (value == null) return '-';
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
};

function FreshnessRow({ freshness }) {
  const items = [
    { label: 'Scan', value: freshness.scan_as_of_date },
    { label: 'Breadth', value: freshness.breadth_latest_date },
    { label: 'Groups', value: freshness.groups_latest_date },
  ];
  return (
    <Box display="flex" alignItems="center" gap={2} sx={{ mb: 2 }}>
      {items.map((item) => (
        <Box key={item.label} display="flex" alignItems="center" gap={0.5}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '10px' }}>
            {item.label}:
          </Typography>
          <Typography variant="caption" sx={{ fontFamily: 'monospace', fontSize: '10px', color: 'text.secondary' }}>
            {item.value || '-'}
          </Typography>
        </Box>
      ))}
    </Box>
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

  const topResults = homeQuery.data?.scan_summary?.top_results ?? EMPTY_RESULTS;
  const topGroups = homeQuery.data?.top_groups ?? EMPTY_RESULTS;

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
      <Box display="flex" alignItems="baseline" gap={1.5} sx={{ mb: 2 }}>
        <Typography variant="h4">
          Daily Market Snapshot
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {home.as_of_date}
        </Typography>
      </Box>

      <FreshnessRow freshness={freshness} />

      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        {(home.key_markets || []).map((item) => (
          <Grid item xs={12} sm={6} md={2.4} key={item.symbol}>
            <Paper sx={{ p: 1.5, height: '100%' }}>
              <Box display="flex" alignItems="baseline" justifyContent="space-between">
                <Typography variant="caption" sx={{ fontWeight: 600, letterSpacing: '0.05em' }}>
                  {item.symbol}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ fontSize: '9px' }}>
                  {item.display_name}
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ mt: 0.5, fontFamily: 'monospace', fontWeight: 600 }}>
                {item.latest_close != null ? `$${formatNumber(item.latest_close, 2)}` : '-'}
              </Typography>
              <Box display="flex" alignItems="center" gap={0.25} sx={{ mt: 0.25 }}>
                {item.change_1d > 0 && (
                  <TrendingUpIcon sx={{ fontSize: 14, color: 'success.main' }} />
                )}
                {item.change_1d < 0 && (
                  <TrendingDownIcon sx={{ fontSize: 14, color: 'error.main' }} />
                )}
                <Typography
                  variant="body2"
                  component="span"
                  sx={{
                    fontFamily: 'monospace',
                    fontWeight: 600,
                    fontSize: '11px',
                    color: item.change_1d > 0 ? 'success.main' : item.change_1d < 0 ? 'error.main' : 'text.secondary',
                  }}
                >
                  {item.change_1d != null ? `${item.change_1d > 0 ? '+' : ''}${formatNumber(item.change_1d, 2)}%` : '-'}
                </Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Paper sx={{ p: 1.5, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Top Scan Candidates
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontSize: '10px' }}>
          Dollar volume &gt; $100M. Click a row for chart details.
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
                    color: 'text.secondary', fontSize: '11px',
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

      <Paper sx={{ p: 1.5 }}>
        <Typography variant="h6" gutterBottom>
          Leading Groups
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell align="center">Rank</TableCell>
                <TableCell>Group</TableCell>
                <TableCell>Top Stock</TableCell>
                <TableCell align="right">1W Chg</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {topGroups.map((group) => (
                <TableRow key={group.industry_group}>
                  <TableCell align="center" sx={{
                    fontWeight: 600,
                    fontFamily: 'monospace',
                    color: getGroupRankColor(group.rank),
                  }}>
                    {group.rank}
                  </TableCell>
                  <TableCell>{group.industry_group}</TableCell>
                  <TableCell sx={{ fontWeight: 500, color: 'text.secondary', fontSize: '11px' }}>
                    {group.top_symbol || '-'}
                  </TableCell>
                  <TableCell align="right">
                    <RankChangeCell value={group.rank_change_1w} />
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
