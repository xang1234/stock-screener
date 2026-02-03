import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Container,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Paper,
  Grid,
  Card,
  CardContent,
  Chip,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import CategoryIcon from '@mui/icons-material/Category';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';
import {
  getCurrentRankings,
  getRankMovers,
  getGroupDetail,
  triggerCalculation,
  getCalculationStatus,
} from '../api/groups';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

// Helper to format rank change with color
const RankChangeCell = ({ value }) => {
  if (value === null || value === undefined) {
    return <Box sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>-</Box>;
  }

  const color = value > 0 ? 'success.main' : value < 0 ? 'error.main' : 'text.secondary';
  const prefix = value > 0 ? '+' : '';

  return (
    <Box display="flex" alignItems="center" justifyContent="flex-end" sx={{ fontFamily: 'monospace' }}>
      {value > 0 && <TrendingUpIcon sx={{ fontSize: 12, mr: 0.25, color }} />}
      {value < 0 && <TrendingDownIcon sx={{ fontSize: 12, mr: 0.25, color }} />}
      <Box component="span" sx={{ color, fontWeight: value !== 0 ? 600 : 400, fontSize: '11px' }}>
        {prefix}{value}
      </Box>
    </Box>
  );
};

// Helper to show historical rank (current rank + rank change = historical rank)
const HistoricalRankCell = ({ currentRank, rankChange }) => {
  if (rankChange === null || rankChange === undefined || currentRank === null || currentRank === undefined) {
    return <Box sx={{ color: 'text.secondary', fontFamily: 'monospace', fontSize: '11px' }}>-</Box>;
  }

  // Historical rank = current rank + rank change
  // (rank_change = historical_rank - current_rank, so historical_rank = current_rank + rank_change)
  const historicalRank = currentRank + rankChange;

  return (
    <Box sx={{ fontFamily: 'monospace', fontSize: '11px', textAlign: 'right' }}>
      {historicalRank}
    </Box>
  );
};

// Movers card component
const MoversCard = ({ title, groups, isGainers, period }) => {
  const color = isGainers ? 'success' : 'error';
  const Icon = isGainers ? TrendingUpIcon : TrendingDownIcon;

  return (
    <Card variant="outlined">
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Box display="flex" alignItems="center" mb={1}>
          <Icon sx={{ color: `${color}.main`, mr: 0.5, fontSize: 18 }} />
          <Box sx={{ fontWeight: 600, fontSize: '12px' }}>
            {title}
          </Box>
        </Box>
        {groups && groups.length > 0 ? (
          <Box>
            {groups.slice(0, 5).map((group, index) => {
              const changeKey = period ? `rank_change_${period}` : null;
              const change = changeKey ? group[changeKey] : null;

              return (
                <Box
                  key={group.industry_group}
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                  py={0.25}
                  borderBottom={index < 4 ? 1 : 0}
                  borderColor="divider"
                >
                  <Box sx={{ minWidth: 0, flex: 1 }}>
                    <Box sx={{ fontSize: '11px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 160 }}>
                      #{group.rank} {group.industry_group}
                    </Box>
                    <Box sx={{ fontSize: '10px', color: 'text.secondary' }}>
                      RS: {group.avg_rs_rating?.toFixed(1)}
                    </Box>
                  </Box>
                      <RankChangeCell value={change} />
                    </Box>
                  );
                })}
          </Box>
        ) : (
          <Box sx={{ fontSize: '11px', color: 'text.secondary' }}>
            No data available
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Group Detail Modal
const GroupDetailModal = ({ group, open, onClose }) => {
  const { data: detail, isLoading } = useQuery({
    queryKey: ['groupDetail', group],
    queryFn: () => getGroupDetail(group, 365),
    enabled: !!group && open,
  });

  // Prepare chart data (reverse to show oldest first)
  const chartData = detail?.history
    ? [...detail.history].reverse().map(item => {
        const d = new Date(item.date);
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        return {
          date: item.date,
          rank: item.rank,
          displayDate: `${months[d.getMonth()]} '${String(d.getFullYear()).slice(2)}`, // "Jan '25" format
        };
      })
    : [];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">{group}</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : detail ? (
          <Box>
            {/* Current Stats */}
            <Grid container spacing={2} mb={3}>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4">{detail.current_rank}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Current Rank
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4">{detail.current_avg_rs?.toFixed(1)}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Avg RS Rating
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4">{detail.num_stocks}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Stocks
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="body1">{detail.top_symbol || '-'}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Top Stock (RS: {detail.top_rs_rating?.toFixed(1) || '-'})
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            {/* Rank History Chart */}
            {chartData.length > 0 && (
              <Box mb={3}>
                <Typography variant="subtitle2" gutterBottom>
                  Rank History (1 Year)
                </Typography>
                <Box sx={{ width: '100%', height: 220, bgcolor: 'background.paper', borderRadius: 1, p: 1 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis
                        dataKey="displayDate"
                        tick={{ fontSize: 11 }}
                        interval={Math.floor(chartData.length / 6)}
                        angle={-45}
                        textAnchor="end"
                        height={50}
                      />
                      <YAxis
                        scale="log"
                        domain={[1, 200]}
                        reversed
                        tick={{ fontSize: 10 }}
                        tickFormatter={(value) => value}
                        ticks={[1, 5, 10, 20, 50, 100, 197]}
                      />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: 'rgba(0, 0, 0, 0.8)',
                          border: 'none',
                          borderRadius: 4,
                          fontSize: 12,
                        }}
                        labelStyle={{ color: '#fff' }}
                        itemStyle={{ color: '#fff' }}
                        formatter={(value) => [`Rank: ${value}`, '']}
                        labelFormatter={(label, payload) => payload?.[0]?.payload?.date || label}
                      />
                      <ReferenceLine y={20} stroke="#4caf50" strokeDasharray="3 3" opacity={0.5} />
                      <ReferenceLine y={177} stroke="#f44336" strokeDasharray="3 3" opacity={0.5} />
                      <Line
                        type="monotone"
                        dataKey="rank"
                        stroke="#2196f3"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
            )}

            {/* Rank Changes */}
            <Typography variant="subtitle2" gutterBottom>
              Rank Changes
            </Typography>
            <Grid container spacing={2} mb={3}>
              {[
                { label: '1 Week', key: 'rank_change_1w' },
                { label: '1 Month', key: 'rank_change_1m' },
                { label: '3 Months', key: 'rank_change_3m' },
                { label: '6 Months', key: 'rank_change_6m' },
              ].map(({ label, key }) => (
                <Grid item xs={3} key={key}>
                  <Box textAlign="center" p={1} bgcolor="action.hover" borderRadius={1}>
                    <RankChangeCell value={detail[key]} />
                    <Typography variant="caption" color="text.secondary">
                      {label}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>

            {/* Constituent Stocks Table */}
            {detail.stocks && detail.stocks.length > 0 && (
              <Box mb={2}>
                <Box sx={{ fontSize: '12px', fontWeight: 600, mb: 0.5 }}>
                  Constituent Stocks ({detail.stocks.length})
                </Box>
                <TableContainer sx={{ maxHeight: 300 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Sym</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell align="right">RS</TableCell>
                        <TableCell align="right">1M</TableCell>
                        <TableCell align="right">3M</TableCell>
                        <TableCell align="right">EPS Q</TableCell>
                        <TableCell align="right">EPS Y</TableCell>
                        <TableCell align="right">Sls Q</TableCell>
                        <TableCell align="right">Sls Y</TableCell>
                        <TableCell align="right">Stg</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {detail.stocks.map((stock) => (
                        <TableRow key={stock.symbol} hover>
                          <TableCell sx={{ fontWeight: 600 }}>
                            {stock.symbol}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {stock.price?.toFixed(2) || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{
                            fontFamily: 'monospace',
                            fontWeight: 600,
                            color: stock.rs_rating >= 80 ? 'success.main' : stock.rs_rating <= 30 ? 'error.main' : 'text.primary'
                          }}>
                            {stock.rs_rating?.toFixed(0) || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {stock.rs_rating_1m?.toFixed(0) || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {stock.rs_rating_3m?.toFixed(0) || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{
                            fontFamily: 'monospace',
                            color: stock.eps_growth_qq > 0 ? 'success.main' : stock.eps_growth_qq < 0 ? 'error.main' : 'text.secondary'
                          }}>
                            {stock.eps_growth_qq != null ? `${stock.eps_growth_qq > 0 ? '+' : ''}${stock.eps_growth_qq.toFixed(0)}%` : '-'}
                          </TableCell>
                          <TableCell align="right" sx={{
                            fontFamily: 'monospace',
                            color: stock.eps_growth_yy > 0 ? 'success.main' : stock.eps_growth_yy < 0 ? 'error.main' : 'text.secondary'
                          }}>
                            {stock.eps_growth_yy != null ? `${stock.eps_growth_yy > 0 ? '+' : ''}${stock.eps_growth_yy.toFixed(0)}%` : '-'}
                          </TableCell>
                          <TableCell align="right" sx={{
                            fontFamily: 'monospace',
                            color: stock.sales_growth_qq > 0 ? 'success.main' : stock.sales_growth_qq < 0 ? 'error.main' : 'text.secondary'
                          }}>
                            {stock.sales_growth_qq != null ? `${stock.sales_growth_qq > 0 ? '+' : ''}${stock.sales_growth_qq.toFixed(0)}%` : '-'}
                          </TableCell>
                          <TableCell align="right" sx={{
                            fontFamily: 'monospace',
                            color: stock.sales_growth_yy > 0 ? 'success.main' : stock.sales_growth_yy < 0 ? 'error.main' : 'text.secondary'
                          }}>
                            {stock.sales_growth_yy != null ? `${stock.sales_growth_yy > 0 ? '+' : ''}${stock.sales_growth_yy.toFixed(0)}%` : '-'}
                          </TableCell>
                          <TableCell align="center">
                            <Box
                              component="span"
                              sx={{
                                backgroundColor: stock.stage === 2 ? 'success.main' : 'grey.400',
                                color: 'white',
                                padding: '1px 4px',
                                borderRadius: '2px',
                                fontSize: '10px',
                                fontWeight: 500,
                              }}
                            >
                              S{stock.stage || '-'}
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}

            {/* History Table */}
            {detail.history && detail.history.length > 0 && (
              <>
                <Box sx={{ fontSize: '12px', fontWeight: 600, mb: 0.5 }}>
                  Rank History
                </Box>
                <TableContainer sx={{ maxHeight: 180 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Date</TableCell>
                        <TableCell align="right">Rank</TableCell>
                        <TableCell align="right">Avg RS</TableCell>
                        <TableCell align="right">Stocks</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {detail.history.slice(0, 20).map((row) => (
                        <TableRow key={row.date} hover>
                          <TableCell sx={{ fontFamily: 'monospace' }}>{row.date}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.rank}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.avg_rs_rating?.toFixed(1)}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.num_stocks || '-'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </>
            )}
          </Box>
        ) : (
          <Typography color="text.secondary">No data available</Typography>
        )}
      </DialogContent>
    </Dialog>
  );
};

function GroupRankingsPage() {
  const [selectedPeriod, setSelectedPeriod] = useState('1w');
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [orderBy, setOrderBy] = useState('rank');
  const [order, setOrder] = useState('asc');
  const [isCalculating, setIsCalculating] = useState(false);
  const [calculationTaskId, setCalculationTaskId] = useState(null);
  const [showHistoricalRanks, setShowHistoricalRanks] = useState(false); // Toggle between change vs actual rank

  // Fetch current rankings
  const {
    data: rankings,
    isLoading: isLoadingRankings,
    error: errorRankings,
    refetch: refetchRankings,
  } = useQuery({
    queryKey: ['groupRankings'],
    queryFn: () => getCurrentRankings(197),
    refetchInterval: 60000,
  });

  // Fetch movers for selected period
  const {
    data: movers,
    isLoading: isLoadingMovers,
  } = useQuery({
    queryKey: ['groupMovers', selectedPeriod],
    queryFn: () => getRankMovers(selectedPeriod, 10),
  });

  // Poll calculation status while task is running
  const { data: calcStatus } = useQuery({
    queryKey: ['calculationStatus', calculationTaskId],
    queryFn: () => getCalculationStatus(calculationTaskId),
    enabled: !!calculationTaskId,
    refetchInterval: 2000,
  });

  // Handle calculation completion/failure
  useEffect(() => {
    if (calcStatus?.status === 'completed' || calcStatus?.status === 'failed') {
      setCalculationTaskId(null);
      setIsCalculating(false);
      if (calcStatus.status === 'completed') {
        refetchRankings();
      } else {
        console.error('Calculation failed:', calcStatus.error);
      }
    }
  }, [calcStatus, refetchRankings]);

  const handlePeriodChange = (event, newValue) => {
    setSelectedPeriod(newValue);
  };

  const handleSort = (property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const handleCalculate = async () => {
    setIsCalculating(true);
    try {
      const response = await triggerCalculation(null);
      setCalculationTaskId(response.task_id);
    } catch (error) {
      console.error('Calculation error:', error);
      setIsCalculating(false);
    }
  };

  // Helper to get sort value - when showing historical ranks, calculate actual rank instead of change
  const getSortValue = (row, column) => {
    const rankChangeColumns = ['rank_change_1w', 'rank_change_1m', 'rank_change_3m', 'rank_change_6m'];

    if (showHistoricalRanks && rankChangeColumns.includes(column)) {
      // Historical rank = current rank + rank change
      const change = row[column];
      if (change === null || change === undefined) return null;
      return row.rank + change;
    }

    return row[column];
  };

  // Sort rankings
  const sortedRankings = rankings?.rankings
    ? [...rankings.rankings].sort((a, b) => {
        let aVal = getSortValue(a, orderBy);
        let bVal = getSortValue(b, orderBy);

        // Handle null values
        if (aVal === null || aVal === undefined) aVal = order === 'asc' ? Infinity : -Infinity;
        if (bVal === null || bVal === undefined) bVal = order === 'asc' ? Infinity : -Infinity;

        if (order === 'asc') {
          return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        }
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      })
    : [];

  if (errorRankings) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">
          Error loading rankings: {errorRankings.message}
        </Alert>
        <Box mt={2}>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={handleCalculate}
            disabled={isCalculating}
          >
            {isCalculating ? 'Calculating...' : 'Calculate Rankings'}
          </Button>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
      {/* Refresh Button */}
      <Box sx={{ mb: 1.5, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="outlined"
          size="small"
          startIcon={isCalculating ? <CircularProgress size={14} /> : <RefreshIcon sx={{ fontSize: 16 }} />}
          onClick={handleCalculate}
          disabled={isCalculating}
        >
          {isCalculating ? 'Calculating...' : 'Refresh'}
        </Button>
      </Box>

      {isLoadingRankings ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      ) : (
        <>
          <Grid container spacing={2}>
            {/* Left Column: Movers */}
            <Grid item xs={12} md={5}>
              <Paper sx={{ height: '100%' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={selectedPeriod} onChange={handlePeriodChange}>
                <Tab label="1 Week" value="1w" />
                <Tab label="1 Month" value="1m" />
                <Tab label="3 Months" value="3m" />
                <Tab label="6 Months" value="6m" />
              </Tabs>
            </Box>

            {/* Movers Cards */}
            <Box sx={{ p: 2 }}>
              {isLoadingMovers ? (
                <Box display="flex" justifyContent="center" p={2}>
                  <CircularProgress size={24} />
                </Box>
              ) : (
                <Grid container spacing={1} direction="column">
                  <Grid item>
                    <MoversCard
                      title="Top Rank Gainers"
                      groups={movers?.gainers}
                      isGainers={true}
                      period={selectedPeriod}
                    />
                  </Grid>
                  <Grid item>
                    <MoversCard
                      title="Top Rank Losers"
                      groups={movers?.losers}
                      isGainers={false}
                      period={selectedPeriod}
                    />
                  </Grid>
                </Grid>
              )}
            </Box>
          </Paper>
            </Grid>

            {/* Right Column: Full Rankings Table */}
            <Grid item xs={12} md={7}>
              <Paper elevation={1} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ fontSize: '14px', fontWeight: 600 }}>
                All Industry Groups
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <Tooltip title={showHistoricalRanks ? "Showing actual historical ranks" : "Showing rank changes"}>
                  <ToggleButtonGroup
                    value={showHistoricalRanks ? 'ranks' : 'changes'}
                    exclusive
                    onChange={(e, newValue) => {
                      if (newValue !== null) {
                        setShowHistoricalRanks(newValue === 'ranks');
                      }
                    }}
                    size="small"
                    sx={{ height: 24 }}
                  >
                    <ToggleButton value="changes" sx={{ px: 1, py: 0.25, fontSize: '10px' }}>
                      Change
                    </ToggleButton>
                    <ToggleButton value="ranks" sx={{ px: 1, py: 0.25, fontSize: '10px' }}>
                      Prior Rank
                    </ToggleButton>
                  </ToggleButtonGroup>
                </Tooltip>
                {rankings && (
                  <Chip
                    label={`${rankings.total_groups} groups | ${rankings.date}`}
                    size="small"
                  />
                )}
              </Box>
            </Box>
            <TableContainer sx={{ maxHeight: 'calc(100vh - 200px)', flexGrow: 1 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>
                      <TableSortLabel
                        active={orderBy === 'rank'}
                        direction={orderBy === 'rank' ? order : 'asc'}
                        onClick={() => handleSort('rank')}
                      >
                        Rank
                      </TableSortLabel>
                    </TableCell>
                    <TableCell>Industry Group</TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'avg_rs_rating'}
                        direction={orderBy === 'avg_rs_rating' ? order : 'asc'}
                        onClick={() => handleSort('avg_rs_rating')}
                      >
                        RS
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'num_stocks'}
                        direction={orderBy === 'num_stocks' ? order : 'asc'}
                        onClick={() => handleSort('num_stocks')}
                      >
                        #
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">Top</TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rank_change_1w'}
                        direction={orderBy === 'rank_change_1w' ? order : 'asc'}
                        onClick={() => handleSort('rank_change_1w')}
                      >
                        1W
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rank_change_1m'}
                        direction={orderBy === 'rank_change_1m' ? order : 'asc'}
                        onClick={() => handleSort('rank_change_1m')}
                      >
                        1M
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rank_change_3m'}
                        direction={orderBy === 'rank_change_3m' ? order : 'asc'}
                        onClick={() => handleSort('rank_change_3m')}
                      >
                        3M
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rank_change_6m'}
                        direction={orderBy === 'rank_change_6m' ? order : 'asc'}
                        onClick={() => handleSort('rank_change_6m')}
                      >
                        6M
                      </TableSortLabel>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sortedRankings.map((row) => (
                    <TableRow
                      key={row.industry_group}
                      hover
                      onClick={() => setSelectedGroup(row.industry_group)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>
                        <Box
                          component="span"
                          sx={{
                            backgroundColor: row.rank <= 20 ? 'success.main' : row.rank >= 177 ? 'error.main' : 'warning.main',
                            color: row.rank <= 20 || row.rank >= 177 ? 'white' : 'warning.contrastText',
                            padding: '1px 5px',
                            borderRadius: '2px',
                            fontSize: '10px',
                            fontWeight: 600,
                            fontFamily: 'monospace',
                          }}
                        >
                          {row.rank}
                        </Box>
                      </TableCell>
                      <TableCell sx={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {row.industry_group}
                      </TableCell>
                      <TableCell align="right" sx={{
                        fontFamily: 'monospace',
                        fontWeight: 600,
                        color: row.avg_rs_rating >= 70 ? 'success.main' : row.avg_rs_rating <= 30 ? 'error.main' : 'text.primary'
                      }}>
                        {row.avg_rs_rating?.toFixed(1)}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.num_stocks}</TableCell>
                      <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                        {row.top_symbol || '-'}
                      </TableCell>
                      <TableCell align="right">
                        {showHistoricalRanks ? (
                          <HistoricalRankCell currentRank={row.rank} rankChange={row.rank_change_1w} />
                        ) : (
                          <RankChangeCell value={row.rank_change_1w} />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        {showHistoricalRanks ? (
                          <HistoricalRankCell currentRank={row.rank} rankChange={row.rank_change_1m} />
                        ) : (
                          <RankChangeCell value={row.rank_change_1m} />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        {showHistoricalRanks ? (
                          <HistoricalRankCell currentRank={row.rank} rankChange={row.rank_change_3m} />
                        ) : (
                          <RankChangeCell value={row.rank_change_3m} />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        {showHistoricalRanks ? (
                          <HistoricalRankCell currentRank={row.rank} rankChange={row.rank_change_6m} />
                        ) : (
                          <RankChangeCell value={row.rank_change_6m} />
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
              </Paper>
            </Grid>
          </Grid>

          {/* Metadata */}
          {rankings && (
            <Box sx={{ mt: 2, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Data as of {rankings.date} | {rankings.total_groups} groups ranked
              </Typography>
            </Box>
          )}
        </>
      )}

      {/* Group Detail Modal */}
      <GroupDetailModal
        group={selectedGroup}
        open={!!selectedGroup}
        onClose={() => setSelectedGroup(null)}
      />
    </Container>
  );
}

export default GroupRankingsPage;
