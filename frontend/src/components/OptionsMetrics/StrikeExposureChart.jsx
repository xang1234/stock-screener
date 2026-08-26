/**
 * StrikeExposureChart
 *
 * Strike-by-strike breakdown of dealer exposure for one expiration -- a
 * visual map of exactly which strikes are acting as gamma "magnets"
 * (positive GEX, dealers hedge into the move, dampens it) or "repellents"
 * (negative GEX, dealers hedge with the move, amplifies it), plus how that
 * would shift if IV moves (VEX) or as expiration approaches (CEX).
 *
 * Data comes straight from the same `strikes[]` array already returned by
 * compute_options_metrics() -- no separate fetch, this just visualizes what
 * SummaryCards' Net GEX/VEX/CEX totals are already summed from.
 *
 * GEX shows call_gex and put_gex as two stacked series (call_gex is always
 * >= 0, put_gex always <= 0 per aggregate_by_strike's convention), so a
 * strike's split between call-side and put-side pressure is visible, not
 * just the net. VEX/CEX have no call/put split in the API response (net
 * only), so those render as a single bar per strike.
 */
import { useMemo, useState } from 'react';
import {
  ComposedChart,
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
  Alert,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';

// Matches the green/red already used for signed GEX bars in
// MetricHistoryChart.jsx and the --color-positive/--color-negative CSS
// variables -- kept identical rather than introducing a second palette for
// the same "positive vs negative dollar exposure" concept. Note: this pair
// alone is CVD-thin (deutan ΔE ~3.6); the zero baseline + above/below bar
// position carries sign as a second, color-independent channel here.
const POSITIVE_COLOR = '#4caf50';
const NEGATIVE_COLOR = '#f44336';
const REFERENCE_LINE_COLOR = '#9e9e9e';

const METRICS = {
  gex: { label: 'GEX', unit: '$' },
  vex: { label: 'VEX', unit: '$' },
  cex: { label: 'CEX', unit: '$' },
};

function fmtCompact(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  const num = Number(value);
  const abs = Math.abs(num);
  const sign = num < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(1)}K`;
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
        Strike ${Number(label).toFixed(2)}
      </Typography>
      {payload.map((entry) => (
        entry.value != null && (
          <Typography key={entry.dataKey} variant="body2" sx={{ color: entry.color }}>
            {entry.name}: {fmtCompact(entry.value)}
          </Typography>
        )
      ))}
      {metric === 'gex' && payload[0]?.payload?.oi != null && (
        <Typography variant="body2" color="text.secondary">
          OI: {Number(payload[0].payload.oi).toLocaleString()}
        </Typography>
      )}
    </Box>
  );
};

/**
 * @param {Object} props
 * @param {Array} props.strikes - strikes[] from compute_options_metrics (strike, call_gex, put_gex, total_gex, dex, vex, cex, oi, iv_avg)
 * @param {number} [props.spot] - underlying price, drawn as a reference line
 * @param {number} [props.callWall]
 * @param {number} [props.putWall]
 * @param {number} [props.zeroGamma] - flip level
 * @param {number} [props.expectedMove] - +/- 1sd expected move (from ATM straddle pricing); drawn as spot +/- this value
 * @param {number} [props.windowPct=0.2] - only plot strikes within +/- this fraction of spot (matches the ATM window used elsewhere in this codebase for max pain)
 * @param {number} [props.height=360]
 */
export default function StrikeExposureChart({
  strikes,
  spot,
  callWall,
  putWall,
  zeroGamma,
  expectedMove,
  windowPct = 0.2,
  height = 360,
}) {
  const [metric, setMetric] = useState('gex');

  const chartData = useMemo(() => {
    if (!Array.isArray(strikes) || strikes.length === 0) return [];
    const lo = spot != null ? spot * (1 - windowPct) : -Infinity;
    const hi = spot != null ? spot * (1 + windowPct) : Infinity;
    return strikes
      .filter((s) => s.strike >= lo && s.strike <= hi)
      .sort((a, b) => a.strike - b.strike);
  }, [strikes, spot, windowPct]);

  if (!strikes || strikes.length === 0) return null;

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5, flexWrap: 'wrap', gap: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Exposure by Strike
          </Typography>
          <ToggleButtonGroup
            value={metric}
            exclusive
            size="small"
            onChange={(e, next) => {
              if (next !== null) setMetric(next);
            }}
          >
            {Object.entries(METRICS).map(([key, { label }]) => (
              <ToggleButton key={key} value={key}>
                {label}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>

        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          {metric === 'gex'
            ? 'Call-side (green, above zero) vs put-side (red, below zero) gamma exposure at each strike -- where dealer hedging is concentrated.'
            : metric === 'vex'
              ? 'Net vanna exposure per strike -- how much dealer hedging would shift at that strike if implied volatility itself changes.'
              : 'Net charm exposure per strike -- how much dealer hedging drifts at that strike purely from time passing.'}
          {' '}Showing strikes within ±{Math.round(windowPct * 100)}% of spot.
        </Typography>

        {chartData.length === 0 && (
          <Alert severity="info">No strikes with open interest in this range.</Alert>
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
              <YAxis tick={{ fontSize: 11 }} tickFormatter={fmtCompact} />
              <Tooltip content={<CustomTooltip metric={metric} />} />

              <ReferenceLine y={0} stroke={REFERENCE_LINE_COLOR} strokeWidth={1} />
              {spot != null && (
                <ReferenceLine
                  x={spot}
                  stroke="#2196f3"
                  strokeDasharray="4 4"
                  label={{ value: 'Spot', position: 'insideTopRight', fontSize: 10, fill: '#2196f3' }}
                />
              )}
              {callWall != null && (
                <ReferenceLine
                  x={callWall}
                  stroke="#ff9800"
                  strokeDasharray="4 4"
                  label={{ value: 'Call Wall', position: 'insideTopLeft', fontSize: 10, fill: '#ff9800' }}
                />
              )}
              {putWall != null && (
                <ReferenceLine
                  x={putWall}
                  stroke="#ab47bc"
                  strokeDasharray="4 4"
                  label={{ value: 'Put Wall', position: 'insideBottomLeft', fontSize: 10, fill: '#ab47bc' }}
                />
              )}
              {zeroGamma != null && (
                <ReferenceLine
                  x={zeroGamma}
                  stroke="#9e9e9e"
                  strokeDasharray="2 2"
                  label={{ value: 'Flip', position: 'insideBottomRight', fontSize: 10, fill: '#9e9e9e' }}
                />
              )}
              {spot != null && expectedMove != null && (
                <ReferenceLine
                  x={spot - expectedMove}
                  stroke="#9e9e9e"
                  strokeDasharray="3 3"
                  strokeOpacity={0.6}
                  label={{ value: 'Expected Move (-)', position: 'insideTopLeft', fontSize: 9, fill: '#9e9e9e' }}
                />
              )}
              {spot != null && expectedMove != null && (
                <ReferenceLine
                  x={spot + expectedMove}
                  stroke="#9e9e9e"
                  strokeDasharray="3 3"
                  strokeOpacity={0.6}
                  label={{ value: 'Expected Move (+)', position: 'insideTopRight', fontSize: 9, fill: '#9e9e9e' }}
                />
              )}

              {metric === 'gex' ? (
                <>
                  <Bar dataKey="call_gex" name="Call GEX" stackId="gex" fill={POSITIVE_COLOR} />
                  <Bar dataKey="put_gex" name="Put GEX" stackId="gex" fill={NEGATIVE_COLOR} />
                  <Legend />
                </>
              ) : (
                <Bar dataKey={metric} name={METRICS[metric].label}>
                  {chartData.map((point) => (
                    <Cell
                      key={point.strike}
                      fill={Number(point[metric]) >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR}
                    />
                  ))}
                </Bar>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
