import { Chip, Tooltip } from '@mui/material';

// Per-market colors pulled from MUI palette keys rather than hard-coded
// hex values — keeps the badge automatically theme-aware (light/dark).
const MARKET_COLOR = Object.freeze({
  US: 'primary',
  HK: 'success',
  JP: 'warning',
  TW: 'info',
});

const MARKET_LABEL = Object.freeze({
  US: 'United States',
  HK: 'Hong Kong',
  JP: 'Japan',
  TW: 'Taiwan',
});

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
    <Tooltip title={tooltip} arrow>
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
