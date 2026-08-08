/**
 * MetricHistoryChart
 *
 * Time-series view of Max Pain or GEX for one symbol, over a selectable
 * timeframe (Daily/Weekly/Monthly/Quarterly/Yearly).
 *
 * Every plotted point is a real, dated end-of-day snapshot -- never an
 * average. Daily/Weekly/Monthly show every trading day in the window;
 * Quarterly/Yearly downsample to the last snapshot of each week/month (see
 * backend/app/services/options_history_service.py) to keep point counts
 * chart-friendly, still never a computed mean. The caption under the title
 * and the tooltip's exact date are there so that's never ambiguous.
 */
import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ComposedChart,
  Line,
  Bar,
  Cell,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import { format, parseISO } from 'date-fns';
import apiClient from '../../api/client';
import LastUpdated from './LastUpdated';

const TIMEFRAMES = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly'];
const TIMEFRAME_LABELS = { daily: '1D', weekly: '1W', monthly: '1M', quarterly: '1Q', yearly: '1Y' };

const SAMPLING_CAPTION = {
  point_in_time_eod: 'Every trading day shown — EOD snapshot',
  last_of_period_week: 'Last trading day of each week — EOD snapshot',
  last_of_period_month: 'Last trading day of each month — EOD snapshot',
};

function samplingCaption(sampling, period) {
  if (sampling === 'last_of_period' && period) {
    return SAMPLING_CAPTION[`last_of_period_${period}`] || `Last snapshot of each ${period} — EOD snapshot`;
  }
  return SAMPLING_CAPTION.point_in_time_eod;
}

function fmtCompact(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  const num = Number(value);
  const abs = Math.abs(num);
  if (abs >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
  return num.toFixed(2);
}

const CustomTooltip = ({ active, payload, label, metric }) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <Box
      sx={{
        backgroundColor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        p: 1.5,
        boxShadow: 2,
      }}
    >
      <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
        {format(parseISO(label), 'MMM dd, yyyy')}
      </Typography>
      {payload.map((entry) => (
        <Typography key={entry.dataKey} variant="body2" sx={{ color: entry.color }}>
          {entry.name}:{' '}
          {metric === 'maxPain' ? `$${Number(entry.value).toFixed(2)}` : fmtCompact(entry.value)}
        </Typography>
      ))}
    </Box>
  );
};

/**
 * @param {Object} props
 * @param {string} props.symbol - Ticker symbol
 * @param {'maxPain'|'gex'} props.metric - Which metric's history to show
 * @param {number} [props.height=320]
 */
export default function MetricHistoryChart({ symbol, metric, expiration = null, height = 320 }) {
  const [timeframe, setTimeframe] = useState('monthly');

  const endpoint = metric === 'gex' ? '/v1/gex/history' : '/v1/max-pain/history';
  const title = metric === 'gex' ? 'GEX History' : 'Max Pain History';

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['metric-history', metric, symbol, timeframe, expiration],
    queryFn: async () => {
      const params = { symbol, timeframe };
      if (expiration) params.expiration = expiration;
      const resp = await apiClient.get(endpoint, { params });
      return resp.data;
    },
    enabled: Boolean(symbol),
    staleTime: 5 * 60 * 1000,
  });

  const chartData = useMemo(() => data?.points ?? [], [data]);

  if (!symbol) return null;

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            {title}
            {expiration && (
              <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                (exp {expiration})
              </Typography>
            )}
          </Typography>
          <ToggleButtonGroup
            value={timeframe}
            exclusive
            size="small"
            onChange={(e, next) => {
              if (next !== null) setTimeframe(next);
            }}
          >
            {TIMEFRAMES.map((tf) => (
              <ToggleButton key={tf} value={tf}>
                {TIMEFRAME_LABELS[tf]}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', mb: 1, flexWrap: 'wrap' }}>
        <Typography variant="caption" color="text.secondary">
          {data ? samplingCaption(data.sampling, data.period) : ' '}
          {data?.points?.length ? ` · ${data.points.length} points` : ''}
        </Typography>
          <LastUpdated timestamp={data?.as_of} />
        </Box>

        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
            <CircularProgress size={28} />
          </Box>
        )}

        {isError && (
          <Alert severity="error">
            Failed to load {title.toLowerCase()}: {error?.message || 'unknown error'}
          </Alert>
        )}

        {!isLoading && !isError && chartData.length === 0 && (
          <Alert severity="info">No history available for {symbol} in this window yet.</Alert>
        )}

        {!isLoading && !isError && chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={height}>
            <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11 }}
                tickFormatter={(d) => format(parseISO(d), 'MMM dd')}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 11 }}
                tickFormatter={metric === 'gex' ? fmtCompact : (v) => `$${v}`}
                domain={metric === 'gex' ? ['auto', 'auto'] : ['auto', 'auto']}
              />
              <Tooltip content={<CustomTooltip metric={metric} />} />
              <Legend />

              {metric === 'maxPain' ? (
                <>
                  <Line
                    type="monotone"
                    dataKey="max_pain_strike"
                    name="Max Pain Strike"
                    stroke="#ff9800"
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="last_price"
                    name="Last Price"
                    stroke="#2196f3"
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                </>
              ) : (
                <>
                  <ReferenceLine y={0} stroke="#9e9e9e" strokeWidth={1} />
                  <Bar dataKey="total_gex" name="Total GEX">
                    {chartData.map((point) => (
                      <Cell
                        key={point.date}
                        fill={Number(point.total_gex) >= 0 ? '#4caf50' : '#f44336'}
                      />
                    ))}
                  </Bar>
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
