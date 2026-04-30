import { Chip, Tooltip } from '@mui/material';
import { UNIVERSE_MARKETS } from '../../features/scan/constants';

// Per-market colors pulled from MUI palette keys rather than hard-coded
// hex values — keeps the badge automatically theme-aware (light/dark).
const MARKET_COLOR = Object.freeze({
  US: 'primary',
  HK: 'success',
  IN: 'secondary',
  JP: 'warning',
  KR: 'error',
  TW: 'info',
  CN: 'secondary',
});

// Derive full-name labels from the canonical universe constants so renames
// (e.g. "Hong Kong" → "Hong Kong SAR") stay in one place.
const MARKET_LABEL = Object.freeze(
  Object.fromEntries(UNIVERSE_MARKETS.map(({ value, label }) => [value, label])),
);

/**
 * Per-row market-origin badge shown next to the symbol. Returns null when
 * the market is missing so US-only scans with stale `market=null` rows
 * (legacy universe data) stay visually quiet instead of rendering a
 * placeholder chip.
 */
function MarketBadge({ market, exchange }) {
  if (!market) return null;
  const tooltip = exchange
    ? `${MARKET_LABEL[market] ?? market} (${exchange})`
    : MARKET_LABEL[market] ?? market;
  return (
    <Tooltip title={tooltip} arrow disableInteractive enterDelay={300}>
      <Chip
        size="small"
        label={market}
        color={MARKET_COLOR[market] ?? 'default'}
        variant="outlined"
        data-testid={`market-badge-${market}`}
        sx={{
          height: 16,
          fontSize: 10,
          fontWeight: 600,
          ml: 0.5,
          '& .MuiChip-label': { px: 0.5 },
        }}
      />
    </Tooltip>
  );
}

export default MarketBadge;
