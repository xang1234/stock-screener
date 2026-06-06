/**
 * Relative Rotation Graph (RRG)
 *
 * Plots each IBD group (or sector roll-up) by RS-Ratio (x) vs RS-Momentum (y),
 * with a weekly "tail" tracing its path through the four quadrants:
 *
 *     Leading   (x>=100, y>=100)  green
 *     Weakening (x>=100, y<100)   orange
 *     Lagging   (x<100,  y<100)   red
 *     Improving (x<100,  y>=100)  blue
 *
 * Coordinates are pre-computed server-side (see backend rrg_service.py), so this
 * component is purely presentational. It is shared by the live Group Rankings
 * page and the static-site Groups page — both pass the same `{ groups: [...] }`.
 */
import { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  Cell,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
} from 'recharts';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import { QUADRANT_COLORS, QUADRANT_FILLS, quadrantColor } from './rrgColors';

/** Symmetric axis bounds around the 100/100 cross, padded to the data extent. */
const computeBound = (groups) => {
  let maxAbs = 8;
  for (const g of groups) {
    const pts = [g.current, ...(g.tail || [])];
    for (const p of pts) {
      maxAbs = Math.max(maxAbs, Math.abs(p.x - 100), Math.abs(p.y - 100));
    }
  }
  return Math.min(20, Math.ceil(maxAbs) + 1);
};

const RRGTooltip = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;
  const g = payload[0]?.payload;
  if (!g) return null;
  return (
    <Box
      sx={{
        backgroundColor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        p: 1.5,
        boxShadow: 2,
        minWidth: 180,
      }}
    >
      <Typography variant="body2" sx={{ fontWeight: 700, mb: 0.5 }}>
        {g.industry_group}
      </Typography>
      <Typography variant="caption" sx={{ color: quadrantColor(g.quadrant), fontWeight: 600 }}>
        {g.quadrant}
        {g.is_provisional ? ' · provisional' : ''}
      </Typography>
      <Typography variant="body2" sx={{ mt: 0.5 }}>
        Rank: {g.rank ?? '—'} · RS: {g.avg_rs_rating != null ? g.avg_rs_rating.toFixed(1) : '—'}
      </Typography>
      <Typography variant="body2" sx={{ color: 'text.secondary' }}>
        Ratio {g.x?.toFixed(1)} · Momentum {g.y?.toFixed(1)}
      </Typography>
    </Box>
  );
};

/**
 * @param {Object[]} props.data.groups - RRGGroupResponse[] from the API
 */
export default function RRGChart({ data, isLoading, error, onSelectGroup, height = 560 }) {
  const groups = useMemo(() => (data?.groups ?? []), [data]);

  const bound = useMemo(() => computeBound(groups), [groups]);
  const lo = 100 - bound;
  const hi = 100 + bound;

  // Flatten current points (each carries the full group for tooltip/click).
  const currentPoints = useMemo(
    () => groups.map((g) => ({ ...g, ...g.current })),
    [groups],
  );

  if (isLoading) {
    return (
      <Card>
        <CardContent sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height }}>
          <CircularProgress />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">
            Failed to load RRG data{error?.message ? `: ${error.message}` : '.'}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!groups.length) {
    return (
      <Card>
        <CardContent>
          <Alert severity="info">
            No RRG data available yet. Group-ranking history is required to plot rotation.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Relative Rotation Graph — {data?.scope === 'sectors' ? 'Sectors' : 'Groups'}
          {data?.date ? ` · ${data.date}` : ''}
        </Typography>
        <ResponsiveContainer width="100%" height={height}>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />

            {/* Quadrant backdrops */}
            <ReferenceArea x1={100} x2={hi} y1={100} y2={hi} fill={QUADRANT_FILLS.Leading} fillOpacity={1}
              label={{ value: 'Leading', position: 'insideTopRight', fill: QUADRANT_COLORS.Leading, fontSize: 12 }} />
            <ReferenceArea x1={100} x2={hi} y1={lo} y2={100} fill={QUADRANT_FILLS.Weakening} fillOpacity={1}
              label={{ value: 'Weakening', position: 'insideBottomRight', fill: QUADRANT_COLORS.Weakening, fontSize: 12 }} />
            <ReferenceArea x1={lo} x2={100} y1={lo} y2={100} fill={QUADRANT_FILLS.Lagging} fillOpacity={1}
              label={{ value: 'Lagging', position: 'insideBottomLeft', fill: QUADRANT_COLORS.Lagging, fontSize: 12 }} />
            <ReferenceArea x1={lo} x2={100} y1={100} y2={hi} fill={QUADRANT_FILLS.Improving} fillOpacity={1}
              label={{ value: 'Improving', position: 'insideTopLeft', fill: QUADRANT_COLORS.Improving, fontSize: 12 }} />

            <ReferenceLine x={100} stroke="#9e9e9e" strokeDasharray="4 4" />
            <ReferenceLine y={100} stroke="#9e9e9e" strokeDasharray="4 4" />

            <XAxis
              type="number"
              dataKey="x"
              name="RS-Ratio"
              domain={[lo, hi]}
              tickCount={5}
              label={{ value: 'RS-Ratio', position: 'insideBottom', offset: -10, fontSize: 12 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="RS-Momentum"
              domain={[lo, hi]}
              tickCount={5}
              label={{ value: 'RS-Momentum', angle: -90, position: 'insideLeft', fontSize: 12 }}
            />
            <ZAxis type="number" dataKey="num_stocks" range={[60, 500]} name="Constituents" />
            <Tooltip content={<RRGTooltip />} cursor={{ strokeDasharray: '3 3' }} />

            {/* Tails: one connected polyline per group, dots hidden. */}
            {groups.map((g) => (
              <Scatter
                key={`tail-${g.industry_group}`}
                data={g.tail}
                line={{ stroke: quadrantColor(g.quadrant), strokeWidth: 1.25, strokeOpacity: 0.5 }}
                lineType="joint"
                shape={() => null}
                isAnimationActive={false}
                legendType="none"
              />
            ))}

            {/* Current dots — colored by quadrant, clickable. */}
            <Scatter
              data={currentPoints}
              isAnimationActive={false}
              onClick={(pt) => onSelectGroup?.(pt?.industry_group)}
              cursor="pointer"
            >
              {currentPoints.map((p) => (
                <Cell
                  key={`dot-${p.industry_group}`}
                  fill={quadrantColor(p.quadrant)}
                  fillOpacity={p.is_provisional ? 0.35 : 0.9}
                  stroke={quadrantColor(p.quadrant)}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
