import { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Chip,
  Collapse,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

const formatPct = (value) => {
  if (value == null || Number.isNaN(value)) return '-';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
};

function StockList({ title, stocks, color }) {
  if (!stocks || stocks.length === 0) {
    return null;
  }
  return (
    <Box sx={{ flex: 1, minWidth: 220 }}>
      <Typography
        variant="caption"
        sx={{
          fontSize: '10px',
          fontWeight: 600,
          color,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}
      >
        {title} ({stocks.length})
      </Typography>
      <Table size="small" sx={{ mt: 0.5 }}>
        <TableBody>
          {stocks.map((stock) => (
            <TableRow key={stock.symbol}>
              <TableCell sx={{ py: 0.25, fontFamily: 'monospace', fontWeight: 600, width: 70 }}>
                {stock.symbol}
              </TableCell>
              <TableCell sx={{ py: 0.25, fontSize: '11px', color: 'text.secondary' }}>
                {stock.name || ''}
              </TableCell>
              <TableCell
                align="right"
                sx={{ py: 0.25, fontFamily: 'monospace', color, fontWeight: 600, width: 80 }}
              >
                {formatPct(stock.pct_change)}
              </TableCell>
              <TableCell
                align="right"
                sx={{ py: 0.25, fontFamily: 'monospace', color: 'text.secondary', width: 70 }}
              >
                {stock.close != null ? `$${stock.close.toFixed(2)}` : '-'}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

function GroupRow({ row }) {
  const [open, setOpen] = useState(false);
  const totalActivity = row.up_count + row.down_count;
  const toggle = () => setOpen((prev) => !prev);

  return (
    <>
      <TableRow
        hover
        onClick={toggle}
        sx={{ cursor: 'pointer', '& > *': { borderBottom: 'unset' } }}
      >
        <TableCell sx={{ width: 32, py: 0.5 }}>
          <IconButton
            size="small"
            sx={{ p: 0.25 }}
            aria-label={open ? `Collapse ${row.group}` : `Expand ${row.group}`}
            aria-expanded={open}
            onClick={(event) => {
              event.stopPropagation();
              toggle();
            }}
          >
            {open ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
          </IconButton>
        </TableCell>
        <TableCell sx={{ py: 0.5, fontWeight: 600 }}>{row.group}</TableCell>
        <TableCell
          align="right"
          sx={{ py: 0.5, fontFamily: 'monospace', color: 'success.main', fontWeight: 600 }}
        >
          {row.up_count}
        </TableCell>
        <TableCell
          align="right"
          sx={{ py: 0.5, fontFamily: 'monospace', color: 'error.main', fontWeight: 600 }}
        >
          {row.down_count}
        </TableCell>
        <TableCell
          align="right"
          sx={{
            py: 0.5,
            fontFamily: 'monospace',
            fontWeight: 700,
            color: row.net > 0 ? 'success.main' : row.net < 0 ? 'error.main' : 'text.primary',
          }}
        >
          {row.net > 0 ? `+${row.net}` : row.net}
        </TableCell>
        <TableCell align="right" sx={{ py: 0.5, fontFamily: 'monospace' }}>
          {totalActivity}
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell sx={{ py: 0, borderBottom: open ? undefined : 'none' }} colSpan={6}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', py: 1, px: 4 }}>
              <StockList title="Up 4%+" stocks={row.up_stocks} color="success.main" />
              <StockList title="Down 4%+" stocks={row.down_stocks} color="error.main" />
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

function BreadthGroupAttribution({ attribution }) {
  const history = useMemo(() => attribution?.history || [], [attribution]);
  const [selectedDate, setSelectedDate] = useState(history[0]?.date || null);

  const selectedDay = useMemo(
    () => history.find((day) => day.date === selectedDate) || history[0] || null,
    [history, selectedDate]
  );

  if (!attribution || attribution.available === false) {
    return (
      <Alert severity="info" sx={{ fontSize: '12px' }}>
        {attribution?.reason || 'Group attribution is not available for this market yet.'}
      </Alert>
    );
  }

  if (!selectedDay || (selectedDay.groups || []).length === 0) {
    return (
      <Alert severity="info" sx={{ fontSize: '12px' }}>
        No 4%+ movers were attributed for the selected session.
      </Alert>
    );
  }

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: 1.5,
          mb: 1.5,
          justifyContent: 'space-between',
        }}
      >
        <FormControl size="small" sx={{ minWidth: 160 }}>
          <InputLabel id="breadth-attribution-date-label">Session</InputLabel>
          <Select
            labelId="breadth-attribution-date-label"
            label="Session"
            value={selectedDay.date}
            onChange={(event) => setSelectedDate(event.target.value)}
          >
            {history.map((day) => (
              <MenuItem key={day.date} value={day.date}>
                {day.date}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            size="small"
            label={`Up 4%+: ${selectedDay.stocks_up_4pct}`}
            sx={{ bgcolor: 'rgba(76,175,80,0.16)', color: 'success.main', fontWeight: 600 }}
          />
          <Chip
            size="small"
            label={`Down 4%+: ${selectedDay.stocks_down_4pct}`}
            sx={{ bgcolor: 'rgba(244,67,54,0.16)', color: 'error.main', fontWeight: 600 }}
          />
          <Chip
            size="small"
            label={`Groups: ${selectedDay.groups.length}`}
            variant="outlined"
          />
        </Box>
      </Box>

      <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider' }}>
        <TableContainer sx={{ maxHeight: 'calc(100vh - 320px)' }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ width: 32 }} />
                <TableCell>IBD Industry Group</TableCell>
                <TableCell align="right">Up 4%+</TableCell>
                <TableCell align="right">Down 4%+</TableCell>
                <TableCell align="right">Net</TableCell>
                <TableCell align="right">Total</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {selectedDay.groups.map((row) => (
                <GroupRow key={row.group} row={row} />
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        Click a group to expand its 4%+ movers. Stocks without an IBD industry group are bucketed
        under &quot;No Group&quot;.
      </Typography>
    </Box>
  );
}

export default BreadthGroupAttribution;
