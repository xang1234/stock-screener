/**
 * Pure helpers for RRG tail rendering (kept out of the component file so they
 * are unit-testable and don't trip react-refresh).
 */

const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

/** Whole weeks between the as-of date and a tail point's date (>= 0, or null). */
export const weeksAgo = (asOfISO, pointISO) => {
  const asOf = Date.parse(asOfISO);
  const point = Date.parse(pointISO);
  if (Number.isNaN(asOf) || Number.isNaN(point)) return null;
  return Math.max(0, Math.round((asOf - point) / WEEK_MS));
};

/**
 * Enrich a group's weekly tail (oldest -> newest) with per-point metadata used
 * for hover tooltips and graduated styling:
 *   - weeksAgo: how far back the point is from the as-of date
 *   - isHead:   the most-recent point (the current position)
 *   - t:        0 (oldest) -> 1 (newest), for size/opacity gradients
 */
/**
 * Filter RRG series by selected names, quadrants, and/or an inclusive
 * current-rank range. All filters compose with AND; an empty `names`/`quadrants`
 * array disables that filter, and `rankRange = null` disables rank filtering
 * (when set, series with no rank are excluded).
 */
export const filterGroups = (groups, { names = [], quadrants = [], rankRange = null } = {}) => {
  const nameSet = names.length ? new Set(names) : null;
  const quadSet = quadrants.length ? new Set(quadrants) : null;
  return (groups ?? []).filter((g) => {
    if (nameSet && !nameSet.has(g.industry_group)) return false;
    if (quadSet && !quadSet.has(g.quadrant)) return false;
    if (rankRange) {
      const [lo, hi] = rankRange;
      if (g.rank == null || g.rank < lo || g.rank > hi) return false;
    }
    return true;
  });
};

export const buildTailPoints = (group, asOfISO) => {
  const tail = group?.tail ?? [];
  const last = tail.length - 1;
  return tail.map((p, i) => ({
    ...p,
    industry_group: group.industry_group,
    quadrant: group.quadrant,
    weeksAgo: weeksAgo(asOfISO, p.date),
    isHead: i === last,
    t: last > 0 ? i / last : 1,
  }));
};

/**
 * Catmull-Rom control points for the spline segment p1 -> p2 (p0/p3 are the
 * neighbouring vertices, or the segment endpoints at the ends of the tail).
 * The uniform variant interpolates *through* every vertex, so the smoothed
 * trail still passes exactly where the series was each week.
 */
const splineControls = (p0, p1, p2, p3) => ({
  c1: { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 },
  c2: { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 },
});

/** SVG path string for a Catmull-Rom spline through `pts` ({x,y} pixel coords). */
export const catmullRomPath = (pts) => {
  if (!pts || pts.length < 2) return '';
  let d = `M${pts[0].x},${pts[0].y}`;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const p0 = pts[i - 1] ?? pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] ?? p2;
    const { c1, c2 } = splineControls(p0, p1, p2, p3);
    d += ` C${c1.x},${c1.y} ${c2.x},${c2.y} ${p2.x},${p2.y}`;
  }
  return d;
};

/**
 * Point and tangent angle at the middle (t=0.5) of the spline segment
 * p1 -> p2, so direction arrows sit on the curve rather than the chord.
 */
export const splineSegmentMidpoint = (p0, p1, p2, p3) => {
  const { c1, c2 } = splineControls(p0 ?? p1, p1, p2, p3 ?? p2);
  const x = (p1.x + 3 * c1.x + 3 * c2.x + p2.x) / 8;
  const y = (p1.y + 3 * c1.y + 3 * c2.y + p2.y) / 8;
  const dx = 0.75 * (c1.x - p1.x) + 1.5 * (c2.x - c1.x) + 0.75 * (p2.x - c2.x);
  const dy = 0.75 * (c1.y - p1.y) + 1.5 * (c2.y - c1.y) + 0.75 * (p2.y - c2.y);
  return { x, y, angle: Math.atan2(dy, dx) };
};

// Label collision layout: estimated glyph metrics for fontSize=10, weight 600.
const LABEL_HEIGHT = 12;
const LABEL_CHAR_WIDTH = 6;

const boxesOverlap = (a, b) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/**
 * Greedy collision-avoiding placement for head-dot labels. Each anchor is
 * `{cx, cy, text, ...}` in pixel space; the default spot is above-right of the
 * dot, falling back to below-right / above-left / below-left / further out
 * until a spot free of already-placed labels is found. Returns the anchors
 * (extra fields preserved) with `{x, y}` text positions (SVG baseline coords)
 * added, in input order.
 */
export const layoutLabels = (anchors) => {
  const placed = [];
  return (anchors ?? []).map((anchor) => {
    const { cx, cy, text } = anchor;
    const w = String(text ?? '').length * LABEL_CHAR_WIDTH;
    const candidates = [
      { x: cx + 7, y: cy - 7 },
      { x: cx + 7, y: cy + 14 },
      { x: cx - w - 7, y: cy - 7 },
      { x: cx - w - 7, y: cy + 14 },
      { x: cx + 7, y: cy - 20 },
      { x: cx + 7, y: cy + 27 },
    ];
    let pick = candidates[0];
    for (const c of candidates) {
      const box = { x: c.x, y: c.y - LABEL_HEIGHT, w, h: LABEL_HEIGHT + 2 };
      if (!placed.some((p) => boxesOverlap(p, box))) {
        pick = c;
        break;
      }
    }
    placed.push({ x: pick.x, y: pick.y - LABEL_HEIGHT, w, h: LABEL_HEIGHT + 2 });
    return { ...anchor, x: pick.x, y: pick.y };
  });
};
