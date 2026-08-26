/**
 * IVTermStructureChart
 *
 * ATM implied volatility across the nearest N expirations for one ticker --
 * a live cross-section (contango/backwardation), not a history. Curving up
 * toward later expirations (contango) is the normal shape; a hump or
 * downward slope into the front month (backwardation) usually means the
 * market is pricing in near-term event risk (earnings, a binary event).
 *
 * This is its own fetch (GET /v1/options/iv-term-structure/{symbol}), not
 * part of the default options-metrics payload -- it costs up to N live
 * yfinance option-chain fetches (one per expiration sampled), meaningfully
 * more than the single-fetch endpoints everything else on this dashboard
 * uses. The backend Redis-caches it for 30 minutes so repeat views of the
 * same ticker don't re-pay that cost.
 */
import { useQuery } from '@tanstack/react-query';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Box, Card, CardContent, Typography, CircularProgress, Alert } from '@mui/material';
import apiClient from '../../api/client';
import LastUpdated from './LastUpdated';

const CURVE_COLOR = '#7e57c2';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;
  const point = payload[0].payload;
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
      <Typography variant="body2" sx={{ fontWeight: 600 }}>
        {point.expiration}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        {point.days_to_expiry} days to expiry
      </Typography>
      <Typography variant="body2" sx={{ color: CURVE_COLOR }}>
        ATM IV: {(Number(point.atm_iv) * 100).toFixed(1)}%
      </Typography>
    </Box>
  );
};

/**
 * @param {Object} props
 * @param {string} props.symbol
 * @param {number} [props.maxExpirations=8]
 * @param {number} [props.height=280]
 */
export default function IVTermStructureChart({ symbol, maxExpirations = 8, height = 280 }) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['iv-term-structure', symbol, maxExpirations],
    queryFn: async () => {
      const resp = await apiClient.get(`/v1/options/iv-term-structure/${symbol}`, {
        params: { max_expirations: maxExpirations },
      });
      return resp.data;
    },
    enabled: Boolean(symbol),
    staleTime: 15 * 60 * 1000,
  });

  if (!symbol) return null;

  const chartData = data?.curve ?? [];

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', mb: 0.5, flexWrap: 'wrap' }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            IV Term Structure
          </Typography>
          <LastUpdated timestamp={data?.as_of} />
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          ATM IV across the nearest {maxExpirations} expirations, computed live -- a snapshot, not a history. Upward sloping (contango) is typical; a hump or downward slope near-term (backwardation) usually flags priced-in event risk.
        </Typography>

        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
            <CircularProgress size={28} />
          </Box>
        )}

        {isError && (
          <Alert severity="error">
            Failed to load IV term structure: {error?.message || 'unknown error'}
          </Alert>
        )}

        {!isLoading && !isError && chartData.length === 0 && (
          <Alert severity="info">No usable IV term structure available for {symbol}.</Alert>
        )}

        {!isLoading && !isError && chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={height}>
            <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="days_to_expiry"
                type="number"
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `${v}d`}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="atm_iv"
                name="ATM IV"
                stroke={CURVE_COLOR}
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
