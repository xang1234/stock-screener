import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box, Typography, Paper, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Chip, CircularProgress, Alert, Select, MenuItem,
  FormControl, InputLabel, Grid, Card, CardContent, TablePagination,
} from '@mui/material';
import apiClient from '../api/client';

const fetchSignals = async ({ outcome, screener, limit, offset }) => {
  const params = { limit, offset };
  if (outcome) params.outcome = outcome;
  if (screener) params.screener = screener;
  const res = await apiClient.get('/v1/signals', { params });
  return res.data;
};

const fetchStats = async () => {
  const res = await apiClient.get('/v1/signals/stats');
  return res.data;
};

const outcomeChip = (outcome) => {
  if (!outcome || outcome === 'open') return <Chip label="Open" color="primary" size="small" />;
  if (outcome === 'target_hit') return <Chip label="Target Hit" color="success" size="small" />;
  if (outcome === 'stop_hit') return <Chip label="Stop Hit" color="error" size="small" />;
  return <Chip label={outcome} size="small" />;
};

export default function SignalsPage() {
  const [outcome, setOutcome] = useState('');
  const [screener, setScreener] = useState('');
  const [page, setPage] = useState(0);
  const rowsPerPage = 50;

  const { data, isLoading, error } = useQuery({
    queryKey: ['signals', outcome, screener, page],
    queryFn: () => fetchSignals({ outcome, screener, limit: rowsPerPage, offset: page * rowsPerPage }),
  });

  const { data: stats } = useQuery({
    queryKey: ['signal-stats'],
    queryFn: fetchStats,
  });

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" fontWeight={700} mb={2}>Signal Archive</Typography>

      {/* Stats cards */}
      {stats && (
        <Grid container spacing={2} mb={3}>
          {[
            { label: 'Total Signals', value: stats.total },
            { label: 'Open', value: stats.open },
            { label: 'Win Rate', value: stats.win_rate != null ? `${stats.win_rate}%` : '—' },
            { label: 'Avg Return', value: stats.avg_return_pct != null ? `${stats.avg_return_pct > 0 ? '+' : ''}${stats.avg_return_pct}%` : '—' },
          ].map(({ label, value }) => (
            <Grid item xs={6} sm={3} key={label}>
              <Card variant="outlined">
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Typography variant="caption" color="text.secondary">{label}</Typography>
                  <Typography variant="h6" fontWeight={700}>{value ?? '—'}</Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Filters */}
      <Box display="flex" gap={2} mb={2}>
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Outcome</InputLabel>
          <Select value={outcome} label="Outcome" onChange={e => { setOutcome(e.target.value); setPage(0); }}>
            <MenuItem value="">All</MenuItem>
            <MenuItem value="open">Open</MenuItem>
            <MenuItem value="target_hit">Target Hit</MenuItem>
            <MenuItem value="stop_hit">Stop Hit</MenuItem>
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 160 }}>
          <InputLabel>Screener</InputLabel>
          <Select value={screener} label="Screener" onChange={e => { setScreener(e.target.value); setPage(0); }}>
            <MenuItem value="">All</MenuItem>
            <MenuItem value="minervini">Minervini</MenuItem>
            <MenuItem value="canslim">CANSLIM</MenuItem>
            <MenuItem value="setup_engine">Setup Engine</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }}>Failed to load signals</Alert>}
      {isLoading && <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />}

      {data && (
        <>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow sx={{ '& th': { fontWeight: 700, bgcolor: 'background.default' } }}>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Date</TableCell>
                  <TableCell>Screener</TableCell>
                  <TableCell>Stage</TableCell>
                  <TableCell align="right">Entry</TableCell>
                  <TableCell align="right">Stop Loss</TableCell>
                  <TableCell align="right">Target</TableCell>
                  <TableCell align="right">Score</TableCell>
                  <TableCell>Outcome</TableCell>
                  <TableCell align="right">Return</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.signals.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={10} align="center" sx={{ py: 4, color: 'text.secondary' }}>
                      No signals yet. Run a Minervini scan to generate signals.
                    </TableCell>
                  </TableRow>
                )}
                {data.signals.map(row => (
                  <TableRow key={row.id} hover>
                    <TableCell><Typography fontWeight={700}>{row.symbol}</Typography></TableCell>
                    <TableCell>{row.signal_date}</TableCell>
                    <TableCell>{row.screener ?? '—'}</TableCell>
                    <TableCell>{row.stage ?? '—'}</TableCell>
                    <TableCell align="right">{row.entry_price != null ? `$${row.entry_price.toFixed(2)}` : '—'}</TableCell>
                    <TableCell align="right" sx={{ color: 'error.main' }}>{row.stop_loss != null ? `$${row.stop_loss.toFixed(2)}` : '—'}</TableCell>
                    <TableCell align="right" sx={{ color: 'success.main' }}>{row.target_price != null ? `$${row.target_price.toFixed(2)}` : '—'}</TableCell>
                    <TableCell align="right">{row.composite_score != null ? row.composite_score.toFixed(1) : '—'}</TableCell>
                    <TableCell>{outcomeChip(row.outcome)}</TableCell>
                    <TableCell align="right" sx={{ color: row.pct_return > 0 ? 'success.main' : row.pct_return < 0 ? 'error.main' : 'inherit', fontWeight: 600 }}>
                      {row.pct_return != null ? `${row.pct_return > 0 ? '+' : ''}${row.pct_return.toFixed(1)}%` : '—'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            component="div"
            count={data.total}
            page={page}
            onPageChange={(_, p) => setPage(p)}
            rowsPerPage={rowsPerPage}
            rowsPerPageOptions={[rowsPerPage]}
          />
        </>
      )}
    </Box>
  );
}
