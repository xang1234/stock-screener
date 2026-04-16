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
import { useStaticManifest, fetchStaticJson, resolveStaticMarketEntry } from '../dataClient';
import { useStaticChartIndex } from '../chartClient';
import PriceSparkline from '../../components/Scan/PriceSparkline';
import RSSparkline from '../../components/Scan/RSSparkline';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import StaticChartViewerModal from '../StaticChartViewerModal';
import RankChangeCell from '../../components/shared/RankChangeCell';
import { getGroupRankColor } from '../../utils/colorUtils';
import { formatLocalCurrency } from '../../utils/formatUtils';
import { useStaticMarket } from '../StaticMarketContext';

const EMPTY_RESULTS = [];

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
  const chartIndexQuery = useStaticChartIndex(marketEntry.assets?.charts?.path);

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
  const marketDisplay = home?.market_display_name || marketEntry.display_name;

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
          {marketDisplay} Snapshot
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
        {(home.key_markets || []).map((item) => {
          const closes = (item.history || []).map((h) => h.close).filter((c) => c != null);
          const trend = closes.length >= 2
            ? (closes[closes.length - 1] > closes[0] ? 1 : closes[closes.length - 1] < closes[0] ? -1 : 0)
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
                {closes.length > 1 ? (
                  <Box sx={{ mt: 0.75 }}>
                    <PriceSparkline
                      data={closes}
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
        <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
          Top Scan Candidates
        </Typography>
        <Typography variant="caption" color="text.disabled" sx={{ mb: 1, display: 'block', fontSize: '10px' }}>
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
