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
  TableSortLabel,
  Tooltip,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';

const UP_COLOR = '#4caf50';
const DOWN_COLOR = '#f44336';
const FLAT_COLOR = '#9e9e9e';
const NO_GROUP_LABEL = 'No Group';

const formatPct = (value) => {
  if (value == null || Number.isNaN(value)) return '-';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
};

const totalActivity = (row) => (row?.up_count ?? 0) + (row?.down_count ?? 0);

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

function SplitBar({ upCount, downCount }) {
  const total = upCount + downCount;
  const tooltip = `${upCount} up, ${downCount} down`;
  return (
    <Tooltip title={tooltip} arrow placement="top">
      <Box
        role="img"
        aria-label={tooltip}
        sx={{
          display: 'flex',
          width: 80,
          height: 8,
          borderRadius: 0.5,
          overflow: 'hidden',
          border: '1px solid',
          borderColor: 'divider',
          bgcolor: total === 0 ? 'action.disabledBackground' : 'transparent',
        }}
      >
        {upCount > 0 && (
          <Box sx={{ flex: upCount, bgcolor: UP_COLOR }} />
        )}
        {downCount > 0 && (
          <Box sx={{ flex: downCount, bgcolor: DOWN_COLOR }} />
        )}
      </Box>
    </Tooltip>
  );
}

function NetTrendSparkline({ data }) {
  const latestNet = data?.length ? data[data.length - 1].net : 0;
  const stroke = latestNet > 0 ? UP_COLOR : latestNet < 0 ? DOWN_COLOR : FLAT_COLOR;

  if (!data || data.length === 0) {
    return (
      <Box
        sx={{
          width: 80,
          height: 24,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'text.disabled',
          fontSize: 10,
        }}
      >
        -
      </Box>
    );
  }

  return (
    <Box
      data-testid="net-trend-sparkline"
      sx={{ width: 80, height: 24 }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 0, left: 0, bottom: 2 }}>
          <ReferenceLine y={0} stroke="#bdbdbd" strokeWidth={0.5} />
          <Line
            type="monotone"
            dataKey="net"
            stroke={stroke}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}

function HeroBarChartTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const entry = payload[0].payload;
  return (
    <Paper
      elevation={3}
      sx={{
        px: 1.5,
        py: 1,
        fontSize: 11,
        fontFamily: 'monospace',
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Typography sx={{ fontSize: 11, fontWeight: 700, mb: 0.5 }}>{entry.group}</Typography>
      <Box sx={{ color: UP_COLOR }}>Up 4%+: {entry.up_count}</Box>
      <Box sx={{ color: DOWN_COLOR }}>Down 4%+: {entry.down_count}</Box>
      <Box sx={{ color: entry.net >= 0 ? UP_COLOR : DOWN_COLOR }}>
        Net: {entry.net > 0 ? `+${entry.net}` : entry.net}
      </Box>
    </Paper>
  );
}

function GroupActivityBarChart({ topGroups, date }) {
  if (!topGroups.length) {
    return null;
  }

  // Tall enough for ~26px per bar plus padding.
  const chartHeight = Math.max(180, topGroups.length * 26 + 40);

  return (
    <Paper
      elevation={0}
      data-testid="group-activity-hero-chart"
      sx={{ border: '1px solid', borderColor: 'divider', p: 1.5, mb: 1.5 }}
    >
      <Typography
        variant="caption"
        sx={{
          fontSize: 11,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          color: 'text.secondary',
          display: 'block',
          mb: 0.5,
        }}
      >
        Top {topGroups.length} groups driving breadth · {date}
      </Typography>
      <Box sx={{ width: '100%', height: chartHeight }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={topGroups}
            layout="vertical"
            margin={{ top: 4, right: 16, bottom: 4, left: 0 }}
            barCategoryGap={4}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#eeeeee" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 10 }} allowDecimals={false} />
            <YAxis
              type="category"
              dataKey="group"
              tick={{ fontSize: 10 }}
              width={170}
              interval={0}
            />
            <RechartsTooltip content={<HeroBarChartTooltip />} cursor={{ fill: 'rgba(0,0,0,0.04)' }} />
            <Bar dataKey="up_count" stackId="activity" fill={UP_COLOR} />
            <Bar dataKey="down_count" stackId="activity" fill={DOWN_COLOR} />
          </BarChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
}

function GroupRow({ row, maxAbsNet, trendData }) {
  const [open, setOpen] = useState(false);
  const total = totalActivity(row);
  const toggle = () => setOpen((prev) => !prev);

  const netIntensity = maxAbsNet > 0 ? Math.min(Math.abs(row.net) / maxAbsNet, 1) : 0;
  const netBgColor =
    row.net > 0
      ? `rgba(76, 175, 80, ${(netIntensity * 0.32).toFixed(3)})`
      : row.net < 0
        ? `rgba(244, 67, 54, ${(netIntensity * 0.32).toFixed(3)})`
        : 'transparent';

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
        <TableCell sx={{ py: 0.5, width: 96 }}>
          <SplitBar upCount={row.up_count} downCount={row.down_count} />
        </TableCell>
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
            backgroundColor: netBgColor,
          }}
        >
          {row.net > 0 ? `+${row.net}` : row.net}
        </TableCell>
        <TableCell align="right" sx={{ py: 0.5, fontFamily: 'monospace' }}>
          {total}
        </TableCell>
        <TableCell sx={{ py: 0.5, width: 96 }}>
          <NetTrendSparkline data={trendData} />
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell sx={{ py: 0, borderBottom: open ? undefined : 'none' }} colSpan={8}>
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

const getSortValue = (row, column) => {
  if (column === 'group') return row.group ?? '';
  if (column === 'up') return row.up_count;
  if (column === 'down') return row.down_count;
  if (column === 'net') return row.net;
  if (column === 'total') return totalActivity(row);
  return null;
};

const sortGroups = (groups, orderBy, order) => {
  const dir = order === 'asc' ? 1 : -1;
  return [...groups].sort((a, b) => {
    // Pin "No Group" to the bottom regardless of direction — it's a synthetic bucket.
    if (a.group === NO_GROUP_LABEL && b.group !== NO_GROUP_LABEL) return 1;
    if (b.group === NO_GROUP_LABEL && a.group !== NO_GROUP_LABEL) return -1;

    let aVal = getSortValue(a, orderBy);
    let bVal = getSortValue(b, orderBy);

    if (orderBy === 'group') {
      return dir * String(aVal).localeCompare(String(bVal));
    }

    if (aVal == null) aVal = order === 'asc' ? Infinity : -Infinity;
    if (bVal == null) bVal = order === 'asc' ? Infinity : -Infinity;

    if (aVal < bVal) return -1 * dir;
    if (aVal > bVal) return 1 * dir;
    return 0;
  });
};

function BreadthGroupAttribution({ attribution }) {
  const history = useMemo(() => attribution?.history || [], [attribution]);
  const [selectedDate, setSelectedDate] = useState(history[0]?.date || null);
  const [orderBy, setOrderBy] = useState('total');
  const [order, setOrder] = useState('desc');

  const selectedDay = useMemo(
    () => history.find((day) => day.date === selectedDate) || history[0] || null,
    [history, selectedDate]
  );

  // Derive a per-group 10-day net trend from the existing history payload so
  // each row can render a sparkline without any backend changes.
  const groupTrendMap = useMemo(() => {
    const trends = new Map();
    if (!history.length) return trends;
    const allGroups = new Set();
    for (const day of history) {
      for (const g of day.groups || []) {
        allGroups.add(g.group);
      }
    }
    // history is newest-first; reverse so the line plots oldest→newest left-to-right.
    const ordered = [...history].reverse();
    for (const groupName of allGroups) {
      trends.set(
        groupName,
        ordered.map((day) => {
          const entry = (day.groups || []).find((g) => g.group === groupName);
          return { date: day.date, net: entry?.net ?? 0 };
        })
      );
    }
    return trends;
  }, [history]);

  const sortedGroups = useMemo(() => {
    if (!selectedDay?.groups?.length) return [];
    return sortGroups(selectedDay.groups, orderBy, order);
  }, [selectedDay, orderBy, order]);

  const maxAbsNet = useMemo(
    () => sortedGroups.reduce((acc, g) => Math.max(acc, Math.abs(g.net ?? 0)), 0),
    [sortedGroups]
  );

  const topGroups = useMemo(() => {
    if (!selectedDay?.groups?.length) return [];
    return [...selectedDay.groups]
      .map((g) => ({
        group: g.group,
        up_count: g.up_count,
        down_count: g.down_count,
        net: g.net,
        total: totalActivity(g),
      }))
      .sort((a, b) => b.total - a.total)
      .slice(0, 10);
  }, [selectedDay]);

  const handleSort = (column) => {
    if (orderBy === column) {
      setOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
      return;
    }
    setOrderBy(column);
    setOrder(column === 'group' ? 'asc' : 'desc');
  };

  if (!attribution || attribution.available === false) {
    return (
      <Alert severity="info" sx={{ fontSize: '12px' }}>
        {attribution?.reason || 'Group attribution is not available for this market yet.'}
      </Alert>
    );
  }

  if (!selectedDay) {
    return (
      <Alert severity="info" sx={{ fontSize: '12px' }}>
        No 4%+ movers were attributed for the lookback window.
      </Alert>
    );
  }

  const hasGroups = sortedGroups.length > 0;

  const sortableHeader = (column, label, align = 'left') => (
    <TableCell align={align}>
      <TableSortLabel
        active={orderBy === column}
        direction={orderBy === column ? order : 'asc'}
        onClick={() => handleSort(column)}
      >
        {label}
      </TableSortLabel>
    </TableCell>
  );

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
            label={`Groups: ${(selectedDay.groups || []).length}`}
            variant="outlined"
          />
        </Box>
      </Box>

      {hasGroups ? (
        <>
          <GroupActivityBarChart topGroups={topGroups} date={selectedDay.date} />

          <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider' }}>
            <TableContainer sx={{ maxHeight: 'calc(100vh - 320px)' }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ width: 32 }} />
                    {sortableHeader('group', 'IBD Industry Group')}
                    <TableCell sx={{ width: 96 }}>Split</TableCell>
                    {sortableHeader('up', 'Up 4%+', 'right')}
                    {sortableHeader('down', 'Down 4%+', 'right')}
                    {sortableHeader('net', 'Net', 'right')}
                    {sortableHeader('total', 'Total', 'right')}
                    <TableCell sx={{ width: 96 }}>10-day Net</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sortedGroups.map((row) => (
                    <GroupRow
                      key={row.group}
                      row={row}
                      maxAbsNet={maxAbsNet}
                      trendData={groupTrendMap.get(row.group)}
                    />
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Click a group to expand its 4%+ movers. Stocks without an IBD industry group are
            bucketed under &quot;No Group&quot;.
          </Typography>
        </>
      ) : (
        <Alert severity="info" sx={{ fontSize: '12px' }}>
          No 4%+ movers were attributed for {selectedDay.date}. Pick another session above to see
          the groups that drove its breadth.
        </Alert>
      )}
    </Box>
  );
}

export default BreadthGroupAttribution;
