import { useMemo } from 'react';
import {
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  Grid,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
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
import RankChangeCell from '../components/shared/RankChangeCell';
import PriceSparkline from '../components/Scan/PriceSparkline';
import RSSparkline from '../components/Scan/RSSparkline';

function StaticGroupDetailModal({ group, detail, open, onClose }) {
  const chartData = useMemo(() => {
    if (!detail?.history) return [];
    return [...detail.history].reverse().map((item) => {
      const [year, month] = item.date.split('-');
      const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      return {
        date: item.date,
        rank: item.rank,
        displayDate: `${months[parseInt(month, 10) - 1]} '${year.slice(2)}`,
      };
    });
  }, [detail?.history]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">{group}</Typography>
          <IconButton onClick={onClose} size="small" aria-label="Close group detail">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {!detail ? (
          <Typography color="text.secondary">No data available</Typography>
        ) : (
          <Box>
            {/* Current Stats */}
            <Grid container spacing={2} mb={3}>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4">{detail.current_rank}</Typography>
                  <Typography variant="caption" color="text.secondary">Current Rank</Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4">{detail.current_avg_rs?.toFixed(1)}</Typography>
                  <Typography variant="caption" color="text.secondary">Avg RS Rating</Typography>
                </Box>
              </Grid>
              <Grid item xs={3}>
                <Box textAlign="center">
                  <Typography variant="h4">{detail.num_stocks}</Typography>
                  <Typography variant="caption" color="text.secondary">Stocks</Typography>
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
                <Typography variant="subtitle2" gutterBottom>Rank History</Typography>
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
                      <Line type="monotone" dataKey="rank" stroke="#2196f3" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
            )}

            {/* Rank Changes */}
            <Typography variant="subtitle2" gutterBottom>Rank Changes</Typography>
            <Grid container spacing={2} mb={3}>
              {[
                { label: '1 Week', key: 'rank_change_1w' },
                { label: '1 Month', key: 'rank_change_1m' },
                { label: '3 Months', key: 'rank_change_3m' },
                { label: '6 Months', key: 'rank_change_6m' },
              ].map(({ label, key }) => (
                <Grid item xs={3} key={key}>
                  <Box textAlign="center" p={1} bgcolor="action.hover" borderRadius={1}>
                    <RankChangeCell value={detail[key]} justifyContent="center" />
                    <Typography variant="caption" color="text.secondary">{label}</Typography>
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
                        <TableCell align="center" sx={{ p: '2px' }}>Price 30d</TableCell>
                        <TableCell align="center" sx={{ p: '2px' }}>RS 30d</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell align="right">RS</TableCell>
                        <TableCell align="right">1M</TableCell>
                        <TableCell align="right">3M</TableCell>
                        <TableCell align="right">EPS Q</TableCell>
                        <TableCell align="right">EPS Y</TableCell>
                        <TableCell align="right">Sls Q</TableCell>
                        <TableCell align="right">Sls Y</TableCell>
                        <TableCell align="center">Stg</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {detail.stocks.map((stock) => (
                        <TableRow key={stock.symbol} hover>
                          <TableCell sx={{ fontWeight: 600 }}>{stock.symbol}</TableCell>
                          <TableCell align="center" sx={{ p: '2px' }}>
                            <PriceSparkline
                              data={stock.price_sparkline_data}
                              trend={stock.price_trend}
                              change1d={stock.price_change_1d}
                              width={80}
                              height={22}
                              showChange={false}
                            />
                          </TableCell>
                          <TableCell align="center" sx={{ p: '2px' }}>
                            <RSSparkline
                              data={stock.rs_sparkline_data}
                              trend={stock.rs_trend}
                              width={60}
                              height={20}
                            />
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {stock.price?.toFixed(2) || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{
                            fontFamily: 'monospace', fontWeight: 600,
                            color: stock.rs_rating == null ? 'text.primary'
                              : stock.rs_rating >= 80 ? 'success.main'
                              : stock.rs_rating <= 30 ? 'error.main' : 'text.primary',
                          }}>
                            {stock.rs_rating?.toFixed(0) || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {stock.rs_rating_1m?.toFixed(0) || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {stock.rs_rating_3m?.toFixed(0) || '-'}
                          </TableCell>
                          {['eps_growth_qq', 'eps_growth_yy', 'sales_growth_qq', 'sales_growth_yy'].map((field) => (
                            <TableCell key={field} align="right" sx={{
                              fontFamily: 'monospace',
                              color: stock[field] > 0 ? 'success.main' : stock[field] < 0 ? 'error.main' : 'text.secondary',
                            }}>
                              {stock[field] != null ? `${stock[field] > 0 ? '+' : ''}${stock[field].toFixed(0)}%` : '-'}
                            </TableCell>
                          ))}
                          <TableCell align="center">
                            <Box component="span" sx={{
                              backgroundColor: stock.stage === 2 ? 'success.main' : 'grey.400',
                              color: 'white', padding: '1px 4px', borderRadius: '2px', fontSize: '10px', fontWeight: 500,
                            }}>
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
                <Box sx={{ fontSize: '12px', fontWeight: 600, mb: 0.5 }}>Rank History</Box>
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
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.avg_rs_rating?.toFixed(1) ?? '-'}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.num_stocks || '-'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </>
            )}
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default StaticGroupDetailModal;
