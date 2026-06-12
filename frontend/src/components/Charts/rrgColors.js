/**
 * Shared RRG quadrant colors — single source of truth for the live and static
 * charts (and any consumer that colors by quadrant, e.g. a detail modal).
 */
export const QUADRANT_COLORS = {
  Leading: '#4caf50',
  Weakening: '#ff9800',
  Lagging: '#f44336',
  Improving: '#2196f3',
};

export const QUADRANT_FILLS = {
  Leading: 'rgba(76, 175, 80, 0.08)',
  Weakening: 'rgba(255, 152, 0, 0.08)',
  Lagging: 'rgba(244, 67, 54, 0.08)',
  Improving: 'rgba(33, 150, 243, 0.08)',
};

export const quadrantColor = (q) => QUADRANT_COLORS[q] || '#9e9e9e';

/**
 * Canonical quadrant order (clockwise rotation cycle) — single source of truth
 * for the quadrant filter buttons so they stay consistent with the colors above.
 */
export const QUADRANTS = ['Improving', 'Leading', 'Weakening', 'Lagging'];

/**
 * Per-quadrant plot geometry: which half of each axis the quadrant occupies
 * relative to the 100/100 cross ('lo' below, 'hi' above) and where its
 * backdrop label sits. Drives the RRG quadrant backdrops.
 */
export const QUADRANT_LAYOUT = [
  { name: 'Leading', x: 'hi', y: 'hi', labelPosition: 'insideTopRight' },
  { name: 'Weakening', x: 'hi', y: 'lo', labelPosition: 'insideBottomRight' },
  { name: 'Lagging', x: 'lo', y: 'lo', labelPosition: 'insideBottomLeft' },
  { name: 'Improving', x: 'lo', y: 'hi', labelPosition: 'insideTopLeft' },
];
