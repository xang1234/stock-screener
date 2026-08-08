/**
 * IVSmileChart
 *
 * Implied volatility across strikes for one expiration -- calls and puts as
 * two separate lines, so the "smile"/skew shape between them is visible
 * (that gap is the whole point; a blended average would erase it).
 *
 * Data comes straight from `iv_smile` in the same options-metrics payload
 * SummaryCards/StrikeExposureChart already use (compute_options_metrics'
 * extract_iv_smile()) -- no separate fetch.
 */
import { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Box, Card, CardContent, Typography, Alert } from '@mui/material';

// Same blue/orange pair MetricHistoryChart already uses for its two-line
// max-pain chart (Last Price / Max Pain Strike) -- reused here rather than
// introducing a third color pairing for the same "two named series" job.
const CALL_COLOR = '#2196f3';
const PUT_COLOR = '#ff9800';

const CustomTooltip = ({ active, payload, label }) => {
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
        Strike ${Number(label).toFixed(2)}
      </Typography>
      {payload.map((entry) => (
        entry.value != null && (
          <Typography key={entry.dataKey} variant="body2" sx={{ color: entry.color }}>
            {entry.name}: {(Number(entry.value) * 100).toFixed(1)}%
          </Typography>
        )
      ))}
    </Box>
  );
};

/**
 * @param {Object} props
 * @param {{calls: Array<{strike:number, iv:number}>, puts: Array<{strike:number, iv:number}>}} [props.ivSmile]
 * @param {number} [props.spot] - underlying price, drawn as a reference line
 * @param {number} [props.windowPct=0.2] - only plot strikes within +/- this fraction of spot
 * @param {number} [props.height=320]
 */
export default function IVSmileChart({ ivSmile, spot, windowPct = 0.2, height = 320 }) {
  const chartData = useMemo(() => {
    const calls = ivSmile?.calls ?? [];
    const puts = ivSmile?.puts ?? [];
    if (calls.length === 0 && puts.length === 0) return [];

    const lo = spot != null ? spot * (1 - windowPct) : -Infinity;
    const hi = spot != null ? spot * (1 + windowPct) : Infinity;

    const byStrike = new Map();
    for (const { strike, iv } of calls) {
      if (strike < lo || strike > hi) continue;
      byStrike.set(strike, { ...(byStrike.get(strike) || { strike }), call_iv: iv });
    }
    for (const { strike, iv } of puts) {
      if (strike < lo || strike > hi) continue;
      byStrike.set(strike, { ...(byStrike.get(strike) || { strike }), put_iv: iv });
    }

    return Array.from(byStrike.values()).sort((a, b) => a.strike - b.strike);
  }, [ivSmile, spot, windowPct]);

  if (!ivSmile || (ivSmile.calls?.length ?? 0) === 0 && (ivSmile.puts?.length ?? 0) === 0) return null;

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
          IV Smile / Skew
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Implied volatility by strike for this expiration. A curve that tilts up toward lower strikes (put skew) shows demand for downside protection; a flat curve is unusual and worth a second look. Showing strikes within ±{Math.round(windowPct * 100)}% of spot.
        </Typography>

        {chartData.length === 0 && (
          <Alert severity="info">No strikes with a usable IV quote in this range.</Alert>
        )}

        {chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={height}>
            <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="strike"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {spot != null && (
                <ReferenceLine
                  x={spot}
                  stroke="#9e9e9e"
                  strokeDasharray="4 4"
                  label={{ value: 'Spot', position: 'insideTopRight', fontSize: 10, fill: '#9e9e9e' }}
                />
              )}
              <Line
                type="monotone"
                dataKey="call_iv"
                name="Call IV"
                stroke={CALL_COLOR}
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="put_iv"
                name="Put IV"
                stroke={PUT_COLOR}
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
