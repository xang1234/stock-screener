import { Box, Button, Stack, Typography } from '@mui/material';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import {
  buildCotChartRows,
  cotRangeLabel,
  formatCotNumber,
  formatCotPercent,
  priceStateLabel,
} from './cotPresentation';

const COLORS = ['#4f7cac', '#d96c75', '#d9a441', '#6da77f', '#8e7cc3'];

const CotTooltip = ({ active, payload, history, mode, participant, visibleParticipants }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  const positions = Object.values(row.positions).filter((position) => (
    mode === 'net'
      ? visibleParticipants.includes(position.participant)
      : position.participant === participant
  ));
  return (
    <Box sx={{ bgcolor: 'background.paper', border: 1, borderColor: 'divider', p: 1, maxWidth: 300 }}>
      <Typography variant="subtitle2">{row.reportDate}</Typography>
      {positions.map((position) => (
        <Typography variant="caption" display="block" key={position.participant}>
          {position.label}: Long {formatCotNumber(position.long)} · Short {formatCotNumber(position.short)} · Net {formatCotNumber(position.net)} · 3Y {position.percentile_status === 'available' ? formatCotPercent(position.percentile_3y, 0) : 'building'}
        </Typography>
      ))}
      <Typography variant="caption" display="block">Open interest: {formatCotNumber(row.openInterest)}</Typography>
      <Typography variant="caption" display="block">
        Price: {row.price === null ? '—' : `${formatCotNumber(row.price)} (${row.priceDate})`} · {priceStateLabel(history.price_mapping_kind, history.price_coverage_state)}
      </Typography>
    </Box>
  );
};

const CotPositioningChart = ({ history, mode, participant, visibleParticipants, onToggleParticipant }) => {
  const rows = buildCotChartRows(history, { mode, participant });
  const participants = history.weeks[0]?.positions || [];
  const visible = visibleParticipants || participants.map((item) => item.participant);
  const description = `${history.display_name} ${mode === 'net' ? 'net positioning' : 'long and short positioning'} for ${mode === 'net' ? `${participants.length} CFTC participant groups` : participants.find((item) => item.participant === participant)?.label || 'the selected CFTC participant'} over ${cotRangeLabel(history.range)} report weeks; dashed line is ${priceStateLabel(history.price_mapping_kind, history.price_coverage_state).toLowerCase()} price context.`;
  const descriptionId = `cot-chart-description-${history.slug}`;

  return (
    <Box>
      {mode === 'net' && (
        <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ mb: 1 }} aria-label="COT participant legend">
          {participants.map((item, index) => (
            <Button
              key={item.participant}
              size="small"
              variant={visible.includes(item.participant) ? 'contained' : 'outlined'}
              aria-pressed={visible.includes(item.participant)}
              onClick={() => onToggleParticipant?.(item.participant)}
              sx={{ color: visible.includes(item.participant) ? undefined : COLORS[index % COLORS.length] }}
            >
              {item.label}
            </Button>
          ))}
        </Stack>
      )}
      <Box role="img" aria-label={description} aria-describedby={descriptionId} sx={{ width: '100%', height: 370 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={rows}
            margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
            stackOffset="sign"
            barGap={0}
            barCategoryGap="12%"
          >
            <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
            <XAxis dataKey="reportDate" minTickGap={32} />
            <YAxis yAxisId="position" tickFormatter={formatCotNumber} />
            <YAxis yAxisId="price" orientation="right" tickFormatter={formatCotNumber} />
            <ReferenceLine yAxisId="position" y={0} stroke="currentColor" opacity={0.45} />
            <Tooltip content={(
              <CotTooltip
                history={history}
                mode={mode}
                participant={participant}
                visibleParticipants={visible}
              />
            )} />
            <Legend />
            {mode === 'net' ? participants.map((item, index) => (
              visible.includes(item.participant) && (
                <Bar
                  key={item.participant}
                  yAxisId="position"
                  dataKey={item.participant}
                  name={item.label}
                  fill={COLORS[index % COLORS.length]}
                  stackId="net"
                />
              )
            )) : (
              <>
                <Bar yAxisId="position" dataKey="long" name="Long" fill="#4f7cac" maxBarSize={24} />
                <Bar yAxisId="position" dataKey="short" name="Short" fill="#d96c75" maxBarSize={24} />
              </>
            )}
            <Line yAxisId="price" dataKey="price" name="Price" stroke="#8796a5" strokeDasharray="5 4" dot={false} connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </Box>
      <Typography id={descriptionId} variant="caption" color="text.secondary">{description}</Typography>
      <Typography variant="overline" display="block" sx={{ mt: 1 }}>Open Interest</Typography>
      <Box sx={{ width: '100%', height: 120 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 4, right: 20, left: 10, bottom: 4 }} barCategoryGap="12%">
            <XAxis dataKey="reportDate" hide />
            <YAxis tickFormatter={formatCotNumber} />
            <Tooltip formatter={(value) => formatCotNumber(value)} />
            <Bar dataKey="openInterest" name="Open interest" fill="#6da77f" />
          </BarChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
};

export default CotPositioningChart;
