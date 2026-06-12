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
 * Each tail vertex is a hoverable dot (date + weeks-ago), the trace is a
 * Catmull-Rom spline graduated oldest->newest, and direction arrows point the
 * way each series is travelling. A filter narrows the plot to the groups/
 * sectors of interest, and dragging a rectangle zooms into it (Reset zoom to
 * restore the full view).
 *
 * Coordinates are pre-computed server-side (see backend rrg_service.py), so this
 * component is purely presentational. It is shared by the live Group Rankings
 * page and the static-site Groups page — both pass the same `{ groups: [...] }`.
 */
import { useId, useMemo, useState } from 'react';
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
  Customized,
} from 'recharts';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  FormControlLabel,
  Switch,
} from '@mui/material';
import { QUADRANT_COLORS, QUADRANT_FILLS, QUADRANT_LAYOUT, quadrantColor } from './rrgColors';
import { buildTailPoints, catmullRomPath, splineSegmentMidpoint, layoutLabels } from './rrgTrace';
import { useDragZoom } from './useDragZoom';
import { useRRGFilters } from './useRRGFilters';
import RRGFilters from './RRGFilters';

// Below this many series shown, the plot renders the full per-week detail:
// graduated, hoverable tail dots + per-segment direction arrows. Above it (e.g.
// the unfiltered ~197-series view) tails are line-only with a single most-recent
// arrow per series, to stay light and readable. Filter down to get the detail.
const DETAIL_LIMIT = 20;

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

/** Graduated tail vertex: faint/small when old, brighter toward the head. The
 *  head itself is drawn by the larger "current" dot, so it's skipped here. */
const TailDot = ({ cx, cy, payload }) => {
  if (cx == null || cy == null || payload?.isHead) return null;
  const t = payload.t ?? 0.5;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={2 + 2.5 * t}
      fill={quadrantColor(payload.quadrant)}
      fillOpacity={0.25 + 0.5 * t}
    />
  );
};

const ArrowHead = ({ x, y, angle, color }) => {
  const size = 5;
  return (
    <polygon
      points={`${-size},${-size * 0.6} ${size},0 ${-size},${size * 0.6}`}
      transform={`translate(${x},${y}) rotate(${(angle * 180) / Math.PI})`}
      fill={color}
      fillOpacity={0.9}
    />
  );
};

/** Recharts <Customized> child: draws everything that lives in pixel space —
 *  the smoothed Catmull-Rom trail per series, direction arrows riding the
 *  curve, and (when toggled) collision-avoiding head labels. All of it is
 *  clipped to the plot area so nothing spills over the axes when zoomed.
 *  Degrades to nothing if scales aren't ready (e.g. SSR/jsdom). */
const RRGOverlay = ({ shown, perSegment, showLabels, xAxisMap, yAxisMap }) => {
  const clipId = useId();
  const xAxis = xAxisMap && xAxisMap[Object.keys(xAxisMap)[0]];
  const yAxis = yAxisMap && yAxisMap[Object.keys(yAxisMap)[0]];
  const xScale = xAxis?.scale;
  const yScale = yAxis?.scale;
  if (typeof xScale !== 'function' || typeof yScale !== 'function') return null;

  const [rx0, rx1] = xScale.range();
  const [ry0, ry1] = yScale.range();
  const clip = {
    x: Math.min(rx0, rx1),
    y: Math.min(ry0, ry1),
    width: Math.abs(rx1 - rx0),
    height: Math.abs(ry1 - ry0),
  };

  const splines = [];
  const arrows = [];
  shown.forEach((g) => {
    const tail = g.tail || [];
    if (tail.length < 2) return;
    const pts = tail.map((p) => ({ x: xScale(p.x), y: yScale(p.y) }));
    if (pts.some((p) => p.x == null || p.y == null || Number.isNaN(p.x) || Number.isNaN(p.y))) return;
    const color = quadrantColor(g.quadrant);
    splines.push(
      <path
        key={`spline-${g.industry_group}`}
        d={catmullRomPath(pts)}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeOpacity={0.5}
      />,
    );
    const from = perSegment ? 1 : pts.length - 1;
    for (let i = from; i < pts.length; i += 1) {
      const mid = splineSegmentMidpoint(pts[i - 2], pts[i - 1], pts[i], pts[i + 1]);
      arrows.push(
        <ArrowHead
          key={`${g.industry_group}-${i}`}
          x={mid.x}
          y={mid.y}
          angle={mid.angle}
          color={color}
        />,
      );
    }
  });

  let labels = null;
  if (showLabels) {
    const anchors = [];
    shown.forEach((g) => {
      const cur = g.current;
      if (!cur) return;
      const cx = xScale(cur.x);
      const cy = yScale(cur.y);
      if ([cx, cy].some((v) => v == null || Number.isNaN(v))) return;
      // Heads zoomed out of view get no label (it would be clipped anyway).
      if (cx < clip.x || cx > clip.x + clip.width || cy < clip.y || cy > clip.y + clip.height) return;
      anchors.push({ cx, cy, text: g.industry_group, color: quadrantColor(g.quadrant) });
    });
    labels = layoutLabels(anchors).map((l) => (
      <text
        key={`label-${l.text}`}
        x={l.x}
        y={l.y}
        fontSize={10}
        fontWeight={600}
        fill={l.color}
        pointerEvents="none"
      >
        {l.text}
      </text>
    ));
  }

  return (
    <g>
      <defs>
        <clipPath id={clipId}>
          <rect {...clip} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        {splines}
        {arrows}
        {labels}
      </g>
    </g>
  );
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
      {g.isCurrent ? (
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          Rank: {g.rank ?? '—'} · RS: {g.avg_rs_rating != null ? g.avg_rs_rating.toFixed(1) : '—'}
        </Typography>
      ) : (
        <Typography variant="body2" sx={{ mt: 0.5, color: 'text.secondary' }}>
          {g.weeksAgo === 0 ? 'Current' : `${g.weeksAgo}w ago`}
          {g.date ? ` · ${g.date}` : ''}
        </Typography>
      )}
      <Typography variant="body2" sx={{ color: 'text.secondary' }}>
        Ratio {g.x?.toFixed(1)} · Momentum {g.y?.toFixed(1)}
      </Typography>
    </Box>
  );
};

/**
 * @param {Object[]} props.data.groups - RRGGroupResponse[] from the API
 */
export default function RRGChart({ data, isLoading, error, onSelectGroup, height = 720 }) {
  const groups = useMemo(() => (data?.groups ?? []), [data]);
  const asOf = data?.date ?? null;
  const scopeLabel = data?.scope === 'sectors' ? 'Sectors' : 'Groups';

  const { shown, filter } = useRRGFilters(groups, { scope: data?.scope, market: data?.market });

  // Display preference (not a filter) — kept local so it persists across
  // scope/market switches, unlike the filters in useRRGFilters.
  const [showLabels, setShowLabels] = useState(false);

  const bound = useMemo(() => computeBound(shown), [shown]);
  const lo = 100 - bound;
  const hi = 100 + bound;

  const zoom = useDragZoom({ x: [lo, hi], y: [lo, hi] }, `${data?.scope}|${data?.market}`);
  const [xLo, xHi] = zoom.xDomain;
  const [yLo, yHi] = zoom.yDomain;
  // The 100/100 cross, clamped into the visible window so the quadrant
  // backdrops always tile exactly the visible plot area when zoomed.
  const xMid = Math.min(Math.max(100, xLo), xHi);
  const yMid = Math.min(Math.max(100, yLo), yHi);

  // Zoomed domains land on arbitrary floats, so ticks need explicit rounding —
  // with decimals adapted to the window size so narrow zooms stay distinct.
  const span = Math.max(xHi - xLo, yHi - yLo);
  const tickDecimals = span > 8 ? 0 : span > 2 ? 1 : 2;
  const formatTick = (v) => v.toFixed(tickDecimals);

  // Single "detail level" driving both tail-dot richness and arrow density, so
  // the default (all-series) view stays light and the filtered view gets the
  // full per-week detail.
  const detailed = shown.length <= DETAIL_LIMIT;

  const currentPoints = useMemo(
    () => shown.map((g) => ({ ...g, ...g.current, isCurrent: true })),
    [shown],
  );
  // Hoverable per-week tail dots exist only in the detailed view (the trail
  // itself is RRGOverlay's spline), so the point enrichment is skipped — and
  // no tail Scatters are mounted — when many series are shown.
  const tails = useMemo(
    () => (detailed ? shown.map((g) => ({ name: g.industry_group, points: buildTailPoints(g, asOf) })) : []),
    [detailed, shown, asOf],
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
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
          <Typography variant="subtitle2">
            Relative Rotation Graph — {scopeLabel}
            {asOf ? ` · ${asOf}` : ''}
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            drag to zoom
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          {zoom.isZoomed && (
            <Button size="small" variant="outlined" onClick={zoom.reset}>
              Reset zoom
            </Button>
          )}
          <FormControlLabel
            control={
              <Switch
                size="small"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
              />
            }
            label="Labels"
            sx={{ mr: 0 }}
          />
          <RRGFilters
            scopeLabel={scopeLabel}
            names={filter.names}
            selected={filter.selected}
            onSelected={filter.setSelected}
            quadrants={filter.quadrants}
            onQuadrants={filter.setQuadrants}
            maxRank={filter.maxRank}
            rankValue={filter.rankValue}
            onRankChange={filter.setRankRange}
          />
        </Box>

        {shown.length === 0 ? (
          <Alert severity="info">No {scopeLabel.toLowerCase()} match the current filter.</Alert>
        ) : (
          <ResponsiveContainer width="100%" height={height}>
            <ScatterChart
              margin={{ top: 20, right: 30, bottom: 20, left: 10 }}
              {...zoom.mouseHandlers}
              style={{ cursor: 'crosshair', userSelect: 'none' }}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />

              {/* Quadrant backdrops, clamped to the (possibly zoomed) window.
                  A quadrant fully outside the window collapses and is skipped. */}
              {QUADRANT_LAYOUT.map(({ name, x, y, labelPosition }) => {
                const [x1, x2] = x === 'hi' ? [xMid, xHi] : [xLo, xMid];
                const [y1, y2] = y === 'hi' ? [yMid, yHi] : [yLo, yMid];
                if (!(x1 < x2 && y1 < y2)) return null;
                return (
                  <ReferenceArea
                    key={name}
                    x1={x1}
                    x2={x2}
                    y1={y1}
                    y2={y2}
                    fill={QUADRANT_FILLS[name]}
                    fillOpacity={1}
                    label={{ value: name, position: labelPosition, fill: QUADRANT_COLORS[name], fontSize: 12 }}
                  />
                );
              })}

              {xLo < 100 && 100 < xHi && <ReferenceLine x={100} stroke="#9e9e9e" strokeDasharray="4 4" />}
              {yLo < 100 && 100 < yHi && <ReferenceLine y={100} stroke="#9e9e9e" strokeDasharray="4 4" />}

              <XAxis
                type="number"
                dataKey="x"
                name="RS-Ratio"
                domain={[xLo, xHi]}
                allowDataOverflow
                tickCount={5}
                tickFormatter={formatTick}
                label={{ value: 'RS-Ratio', position: 'insideBottom', offset: -10, fontSize: 12 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="RS-Momentum"
                domain={[yLo, yHi]}
                allowDataOverflow
                tickCount={5}
                tickFormatter={formatTick}
                label={{ value: 'RS-Momentum', angle: -90, position: 'insideLeft', fontSize: 12 }}
              />
              <ZAxis type="number" dataKey="num_stocks" range={[60, 500]} name="Constituents" />
              <Tooltip content={<RRGTooltip />} cursor={{ strokeDasharray: '3 3' }} />

              {/* Graduated hoverable per-week tail dots (empty unless detailed).
                  The connecting trail itself is RRGOverlay's spline below. */}
              {tails.map((t) => (
                <Scatter
                  key={`tail-${t.name}`}
                  data={t.points}
                  shape={<TailDot />}
                  isAnimationActive={false}
                  legendType="none"
                />
              ))}

              {/* Spline trails, direction arrows (per-segment when detailed),
                  and collision-avoiding head labels (toggle). */}
              <Customized
                component={(props) => (
                  <RRGOverlay shown={shown} perSegment={detailed} showLabels={showLabels} {...props} />
                )}
              />

              {/* In-progress drag-to-zoom selection rectangle. */}
              {zoom.drag && (
                <ReferenceArea
                  x1={zoom.drag.x1}
                  x2={zoom.drag.x2}
                  y1={zoom.drag.y1}
                  y2={zoom.drag.y2}
                  fill="#90caf9"
                  fillOpacity={0.2}
                  stroke="#90caf9"
                  strokeOpacity={0.7}
                />
              )}

              {/* Current head dots — sized by constituents, colored by quadrant, clickable. */}
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
        )}
      </CardContent>
    </Card>
  );
}
