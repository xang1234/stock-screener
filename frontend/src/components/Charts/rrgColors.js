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
