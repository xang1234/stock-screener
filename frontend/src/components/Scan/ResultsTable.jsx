import { useLayoutEffect, useMemo, useRef, useState, useCallback, memo } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Paper,
  Chip,
  Typography,
  Box,
  IconButton,
  ListSubheader,
  MenuItem,
  TextField,
  styled,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import RSSparkline from './RSSparkline';
import PriceSparkline from './PriceSparkline';
import FieldAvailabilityChip from './FieldAvailabilityChip';
import MarketBadge from './MarketBadge';
import MarketThemesList from '../Stock/MarketThemesList';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import OpportunityEvidenceDrawer from '../shared/OpportunityEvidenceDrawer';
import ActionStateBadge from '../shared/ActionStateBadge';
import StockRowCard from '../shared/StockRowCard';
import { useIsMobileShell } from '../Layout/shellNav';
import { TableSkeleton } from '../common/LoadingSkeletons';
import OpportunityResultCells from './OpportunityResultCells';
import { useOpportunityEvidenceSelection } from './useOpportunityEvidenceSelection';
import {
  getStageColor,
  getRatingColor,
  getGrowthColor,
  getEpsRatingColor,
  getGroupRankColor,
} from '../../utils/colorUtils';
import {
  formatLargeNumber,
  formatIpoAge,
  getIpoAgeColor,
  getCurrencyPrefix,
  formatLocalCurrency,
  formatPatternName,
} from '../../utils/formatUtils';
import { heatClass, heatClassStyles } from '../../utils/scoreHeat';

// Row height constant for virtualization
const ROW_HEIGHT = 48;
// Phone layout: one StockRowCard per result (fixed height keeps the virtualizer
// exact). Content needs ~75px; the rest is headroom for larger system text.
const MOBILE_CARD_HEIGHT = 84;
// Phone cards show the value of the active sort (pill label + desktop-style
// formatting), so the visible number matches the order. Covers every sortable
// column the card doesn't already show, plus preset-only sort fields (units from
// the preset filters in backend/app/services/preset_screens.py). Sorts not
// listed are text (sector, pattern, action state, RS blue dot) or already on
// the card (composite, symbol, price, 1D change, group rank, industry), and
// keep the composite.
const oneDp = (value) => Number(value).toFixed(1);
const pct = (value) => `${Number(value).toFixed(1)}%`;
const ratio = (value) => `${Number(value).toFixed(1)}×`;
const whole = (value) => Number(value).toFixed(0);
const CARD_METRICS = {
  minervini_score: { label: 'Min', name: 'Minervini', format: oneDp },
  canslim_score: { label: 'CAN', name: 'CANSLIM', format: oneDp },
  ipo_score: { label: 'IPO', name: 'IPO score', format: oneDp },
  custom_score: { label: 'Cust', name: 'Custom score', format: oneDp },
  volume_breakthrough_score: { label: 'VolB', name: 'Volume breakthrough', format: oneDp },
  se_setup_score: { label: 'SE', name: 'Setup score', format: oneDp },
  vcp_score: { label: 'VScr', name: 'VCP score', format: oneDp },
  resilience_score: { label: 'Res', name: 'Resilience', format: oneDp },
  rs_rating: { label: 'RS', name: 'RS rating', format: whole },
  rs_rating_1m: { label: 'RS1M', name: 'RS rating 1 month', format: whole },
  rs_rating_3m: { label: 'RS3M', name: 'RS rating 3 months', format: whole },
  rs_rating_12m: { label: 'RS12M', name: 'RS rating 12 months', format: whole },
  rs_trend: { label: 'RS', name: 'RS trend', format: (v) => (v > 0 ? '▲' : v < 0 ? '▼' : '–') },
  eps_rating: { label: 'EPS', name: 'EPS rating', format: whole },
  beta: { label: 'β', name: 'Beta', format: (v) => Number(v).toFixed(2) },
  beta_adj_rs: { label: 'βRS', name: 'Beta-adjusted RS', format: whole },
  stage: { label: 'Stg', name: 'Stage', format: (v) => `S${whole(v)}` },
  eps_growth_qq: { label: 'EPS', name: 'EPS growth Q/Q', format: pct },
  sales_growth_qq: { label: 'Sales', name: 'Sales growth Q/Q', format: pct },
  adr_percent: { label: 'ADR', name: 'ADR', format: pct },
  se_distance_to_pivot_pct: { label: 'Pvt', name: 'Distance to pivot', format: pct },
  se_bb_width_pctile_252: { label: 'Sqz', name: 'Squeeze percentile', format: whole },
  se_volume_vs_50d: { label: 'V50', name: 'Volume vs 50-day', format: ratio },
  se_pivot_price: { label: 'Pvt', name: 'Pivot price', format: (v, row) => formatLocalCurrency(v, row.currency) },
  vcp_pivot: { label: 'Pvt', name: 'VCP pivot', format: (v, row) => formatLocalCurrency(v, row.currency) },
  volume: { label: '$Vol', name: 'Dollar volume', format: (v, row) => formatLargeNumber(v, getCurrencyPrefix(row.currency)) },
  // market_cap sorts by the native value (backend mixed_market_policy), so show
  // that, in local currency; market_cap_usd is a separate sort.
  market_cap: { label: 'MCap', name: 'Market cap', format: (v, row) => formatLargeNumber(v, getCurrencyPrefix(row.currency)) },
  market_cap_usd: { label: 'MCap$', name: 'Market cap USD', format: (v) => formatLargeNumber(v, '$') },
  adv_usd: { label: 'ADV', name: 'Average dollar volume', format: (v) => formatLargeNumber(v, '$') },
  // The list sorts by the raw date, so show the date (an age would read in
  // the opposite direction to the sort arrow).
  ipo_date: { label: 'IPO', name: 'IPO date', format: (v) => String(v).slice(0, 10), text: true },
  gap_percent: { label: 'Gap', name: 'Gap', format: pct },
  // Preset-only fields: present on static-site rows (static_scan_bundle_exporter
  // extended fields); the live scan API doesn't return them yet, so live cards
  // fall back to the composite for these sorts.
  perf_week: { label: '1W', name: 'Performance 1 week', format: pct },
  perf_month: { label: '1M', name: 'Performance 1 month', format: pct },
  perf_3m: { label: '3M', name: 'Performance 3 months', format: pct },
  perf_6m: { label: '6M', name: 'Performance 6 months', format: pct },
  week_52_high_distance: { label: '52W', name: 'From 52-week high', format: pct },
  week_52_low_distance: { label: '52WL', name: 'From 52-week low', format: pct },
  volume_surge: { label: 'Vol', name: 'Volume surge', format: ratio },
  se_up_down_volume_ratio_10d: { label: 'U/D', name: 'Up/down volume 10 days', format: ratio },
};
// Same text as the desktop symbol cell's matched-setups chip.
const matchedGroupsLabel = (row) => {
  const groups = row.matched_groups || [];
  if (!groups.length) return null;
  return groups.length === 1 ? groups[0].name : `${groups[0].name} +${groups.length - 1}`;
};
// Text sorts the pill can't fit: their value leads the card's group line.
const CARD_SORT_NOTES = {
  gics_sector: (row) => row.gics_sector || null,
  se_pattern_primary: (row) => (row.se_pattern_primary ? formatPatternName(row.se_pattern_primary) : null),
  rs_line_blue_dot_recent: (row) => (row.rs_line_blue_dot_recent ? 'RS blue dot' : null),
  correction_survivor: (row) => (row.correction_survivor ? 'Correction survivor' : null),
};
// The card's labelled metric for this row, or null to keep the composite.
const cardMetricFor = (row, sortBy) => {
  const spec = CARD_METRICS[sortBy];
  const raw = row[sortBy];
  if (!spec || raw == null || raw === '') return null;
  if (!spec.text && !Number.isFinite(Number(raw))) return null;
  const value = spec.format(raw, row);
  return value && value !== '-' ? { label: spec.label, value, name: spec.name } : null;
};
// Friendly picker names for sortable columns the metric table doesn't name.
const SORT_NAMES = {
  symbol: 'Symbol',
  price_change_1d: '1D change',
  current_price: 'Price',
  price: 'Price', // sortable alias of current_price (scanFilterFields.json)
  correction_survivor: 'Correction survivor',
  gics_sector: 'Sector',
  ibd_industry_group: 'Industry',
  ibd_group_rank: 'Group rank',
  composite_score: 'Composite score',
  se_pattern_primary: 'Pattern',
  rs_line_blue_dot_recent: 'RS blue dot',
  action_state: 'Action state',
};
// Sortable aliases of a table column (scanFilterFields.json): `price` sorts
// the same column as current_price.
const SORT_ALIASES = { price: 'current_price' };
const sortName = (id) => SORT_NAMES[id] || CARD_METRICS[id]?.name || columns.find((column) => column.id === id)?.label || 'Current sort';
// Common sorts lead the phone picker; every other sortable column follows.
const MOBILE_SORT_FIELDS = [
  ['composite_score', 'Composite score'],
  ['minervini_score', 'Minervini'],
  ['canslim_score', 'CANSLIM'],
  ['se_setup_score', 'Setup score'],
  ['rs_rating', 'RS rating'],
  ['price_change_1d', '1D change'],
  ['ibd_group_rank', 'Group rank'],
  ['symbol', 'Symbol'],
];
const SYMBOL_COLUMN_WIDTH = 210;
const OPPORTUNITY_COLUMNS_WIDTH = 153;
const SOCIAL_COLUMN_WIDTH = 110;
const TABLE_MIN_WIDTH = Object.freeze({
  WITH_ACTIONS: 2673,
  WITHOUT_ACTIONS: 2613,
});

// MCap column display modes. USD is the default per 3axp: cross-market
// parity is the common case; local is one click away. Kept as constants
// (rather than bare strings) so callers grep-reliably and typos fail fast.
const MCAP_DISPLAY = Object.freeze({
  USD: 'usd',
  LOCAL: 'local',
});

const formatGroupRankDate = (value) => {
  if (!value) return null;
  const [year, month, day] = String(value).split('-').map((part) => Number(part));
  if (year && month && day) {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      timeZone: 'UTC',
    }).format(new Date(Date.UTC(year, month - 1, day)));
  }
  return String(value);
};

const getGroupRankTooltip = (row) => {
  if (row.ibd_group_rank == null || !row.ibd_group_rank_date) return undefined;
  return `Group rank ${row.ibd_group_rank} as of ${formatGroupRankDate(row.ibd_group_rank_date)}`;
};

// Column definitions with explicit widths
const columns = [
  { id: 'chart', label: '', sortable: false, width: 60 },
  // Width fits "0700.HK" + MarketBadge + FieldAvailabilityChip on a single
  // line without overflow (nowrap guards the rest).
  { id: 'symbol', label: 'Sym', sortable: true, width: SYMBOL_COLUMN_WIDTH },
  { id: 'rs_trend', label: 'RS Trend', sortable: true, width: 110 },
  { id: 'price_change_1d', label: 'Price', sortable: true, width: 110 },
  { id: 'gics_sector', label: 'Sector', sortable: true, width: 80 },
  { id: 'ibd_industry_group', label: 'IBD Industry', sortable: true, width: 140 },
  { id: 'market_themes', label: 'Themes', sortable: false, width: 180 },
  { id: 'ibd_group_rank', label: 'Grp', sortable: true, width: 45 },
  { id: 'composite_score', label: 'Comp', sortable: true, width: 50 },
  { id: 'minervini_score', label: 'Min', sortable: true, width: 45 },
  { id: 'canslim_score', label: 'CAN', sortable: true, width: 45 },
  { id: 'ipo_score', label: 'IPO', sortable: true, width: 45 },
  { id: 'custom_score', label: 'Cust', sortable: true, width: 45 },
  { id: 'volume_breakthrough_score', label: 'VolB', sortable: true, width: 50 },
  { id: 'se_setup_score', label: 'SE', sortable: true, width: 45 },
  { id: 'se_pattern_primary', label: 'Pat', sortable: true, width: 55 },
  { id: 'se_distance_to_pivot_pct', label: 'Pvt%', sortable: true, width: 50 },
  { id: 'se_bb_width_pctile_252', label: 'Sqz', sortable: true, width: 45 },
  { id: 'se_volume_vs_50d', label: 'V50', sortable: true, width: 45 },
  { id: 'se_rs_line_new_high', label: 'RSH', sortable: false, width: 35 },
  { id: 'se_rs_line_blue_dot', label: 'SEBD', sortable: false, width: 42 },
  { id: 'rs_line_blue_dot_recent', label: 'RSBD', sortable: true, width: 42 },
  { id: 'se_pivot_price', label: 'Pvt$', sortable: true, width: 55 },
  { id: 'resilience_score', label: 'Res', sortable: true, width: 48 },
  { id: 'action_state', label: 'Action', sortable: true, width: 105 },
  { id: 'social', label: 'Social 7D', sortable: false, width: SOCIAL_COLUMN_WIDTH },
  { id: 'rs_rating', label: 'RS', sortable: true, width: 40 },
  { id: 'rs_rating_1m', label: '1M', sortable: true, width: 40 },
  { id: 'rs_rating_3m', label: '3M', sortable: true, width: 40 },
  { id: 'rs_rating_12m', label: '12M', sortable: true, width: 45 },
  { id: 'beta', label: 'β', sortable: true, width: 45 },
  { id: 'beta_adj_rs', label: 'βRS', sortable: true, width: 45 },
  { id: 'eps_rating', label: 'EPS Rtg', sortable: true, width: 55 },
  { id: 'stage', label: 'Stg', sortable: true, width: 40 },
  { id: 'current_price', label: 'Price', sortable: true, width: 65 },
  { id: 'volume', label: '$Vol', sortable: true, width: 60 },
  // MCap column header label is overridden per-render based on the USD/Local
  // toggle; keep the underlying sort key stable at 'market_cap' so the
  // sort-by dropdown / URL state doesn't shift when the user flips modes.
  { id: 'market_cap', label: 'MCap', sortable: true, width: 75 },
  { id: 'adv_usd', label: 'ADV ($)', sortable: true, width: 70 },
  { id: 'ipo_date', label: 'IPO', sortable: true, width: 50 },
  { id: 'eps_growth_qq', label: 'EPS', sortable: true, width: 50 },
  { id: 'sales_growth_qq', label: 'Sales', sortable: true, width: 50 },
  { id: 'adr_percent', label: 'ADR', sortable: true, width: 50 },
  { id: 'ma_alignment', label: 'MA', sortable: false, width: 35 },
  { id: 'vcp_detected', label: 'VCP', sortable: false, width: 40 },
  { id: 'vcp_score', label: 'VScr', sortable: true, width: 50 },
  { id: 'vcp_pivot', label: 'Pvt', sortable: true, width: 55 },
  { id: 'vcp_ready_for_breakout', label: 'Rdy', sortable: false, width: 35 },
  { id: 'passes_template', label: 'Pass', sortable: false, width: 40 },
  { id: 'rating', label: 'Rate', sortable: false, width: 80 },
];

const getStatusChipProps = (row) => {
  const isInsufficientHistoryRow =
    row.data_status === 'insufficient_history' || row.rating === 'Insufficient Data';

  if (row.scan_mode === 'listing_only' && isInsufficientHistoryRow) {
    return {
      label: 'New IPO',
      color: 'warning',
      title: 'Visible in the scan table, but not yet scannable because price history is still limited.',
    };
  }
  if (row.scan_mode === 'ipo_weighted' && isInsufficientHistoryRow) {
    return {
      label: 'IPO Weighted',
      color: 'info',
      title: row.composite_reason === 'ipo_uplift'
        ? 'Composite uses applicable screeners plus an IPO uplift while the stock is still young.'
        : 'Composite uses only the screeners that have enough history to run.',
    };
  }
  return null;
};

// Static body-cell styles live on the row as class selectors so each render
// only resolves one style object per row, instead of running MUI's `sx`
// resolution for ~48 cells per row on every page, sort, or scroll update.
const CELL_WIDTHS = [35, 40, 42, 45, 50, 55, 60, 65, 70, 75, 80, 110, 140, 180];
const COLOR_CLASS_PATHS = [
  'text.primary',
  'text.secondary',
  'text.disabled',
  'primary.main',
  'success.main',
  'warning.main',
  'error.main',
];
const colorClass = (path) => (
  COLOR_CLASS_PATHS.includes(path) ? `rt-c-${path.replace('.', '-')}` : ''
);
const resolvePaletteColor = (palette, path) => (
  path.split('.').reduce((value, key) => value?.[key], palette)
);
const stopPropagation = (event) => event.stopPropagation();
// Screener scores: 0 means "could not score" (insufficient data, error, no
// breakthrough; see VolumeBreakthroughScanner._insufficient_data_result), not a
// weak score, so it stays untinted.
const scoreHeatClass = (score) => heatClass(Number(score?.toFixed(1)) === 0 ? null : score);
const stageFill = (theme, stage) => {
  const fill = resolvePaletteColor(theme.palette, getStageColor(stage));
  return { backgroundColor: fill, color: theme.palette.getContrastText(fill) };
};

const ResultRow = styled(TableRow)(({ theme }) => ({
  height: ROW_HEIGHT,
  cursor: 'default',
  '&.rt-clickable': { cursor: 'pointer' },
  ...Object.fromEntries(CELL_WIDTHS.map((width) => [
    `& .rt-w${width}`,
    { width, minWidth: width },
  ])),
  ...Object.fromEntries(COLOR_CLASS_PATHS.map((path) => [
    `& .${colorClass(path)}`,
    { color: resolvePaletteColor(theme.palette, path) },
  ])),
  ...heatClassStyles(theme.palette),
  '& .rt-mono': { fontFamily: 'monospace' },
  '& .rt-strong': { fontWeight: 600 },
  '& .rt-ellipsis': { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  '& .rt-p2': { padding: '2px' },
  '& .rt-p4': { padding: '4px' },
  '& .rt-py4': { paddingTop: theme.spacing(0.5), paddingBottom: theme.spacing(0.5) },
  '& .rt-symbol': {
    width: SYMBOL_COLUMN_WIDTH,
    minWidth: SYMBOL_COLUMN_WIDTH,
    maxWidth: SYMBOL_COLUMN_WIDTH,
    paddingTop: '4px',
    paddingBottom: '4px',
    overflow: 'hidden',
  },
  '& .rt-sym-stack': {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(0.25),
    minWidth: 0,
  },
  '& .rt-sym-line': {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(0.5),
    minWidth: 0,
    whiteSpace: 'nowrap',
  },
  '& .rt-sym-sub': {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(0.5),
    minWidth: 0,
  },
  '& .rt-sym-text': { fontWeight: 600, lineHeight: 1.2, flexShrink: 0 },
  '& .rt-sym-company': { display: 'block', lineHeight: 1.2, minWidth: 0, flex: 1 },
  '& .rt-chart-button': { color: theme.palette.primary.main, padding: 0 },
  '& .rt-icon': { fontSize: 14 },
  '& .rt-dot': {
    width: 10,
    height: 10,
    borderRadius: '50%',
    backgroundColor: theme.palette.primary.main,
    display: 'inline-block',
  },
  '& .rt-stage': {
    ...stageFill(theme, null),
    padding: '1px 4px',
    borderRadius: '2px',
    fontSize: '10px',
    fontWeight: 500,
  },
  // After .rt-stage: same specificity, so the per-stage fill must come later.
  ...Object.fromEntries([1, 2, 3, 4].map((stage) => [`& .rt-stage-${stage}`, stageFill(theme, stage)])),
}));

/**
 * Memoized table row component to prevent unnecessary re-renders
 */
const VirtualTableRow = memo(function VirtualTableRow({
  row,
  onRowClick,
  onRowHover,
  onOpenChart,
  showActions,
  showWatchlistMenu,
  chartEnabled,
  mcapDisplay,
  showOpportunityState,
  onOpenOpportunity,
  renderSocialCell,
}) {
  const statusChip = getStatusChipProps(row);
  const handleRowClick = useCallback(() => {
    if (!chartEnabled) {
      return;
    }
    onRowClick?.(row.symbol);
  }, [chartEnabled, onRowClick, row.symbol]);

  const handleRowHover = useCallback(() => {
    onRowHover?.(row.symbol);
  }, [onRowHover, row.symbol]);

  const handleChartClick = useCallback((e) => {
    e.stopPropagation();
    if (!chartEnabled) {
      return;
    }
    onOpenChart?.(row.symbol);
  }, [chartEnabled, onOpenChart, row.symbol]);

  const check = (value, falseClass = 'rt-c-text-disabled') => (
    value
      ? <CheckIcon className="rt-icon rt-c-success-main" />
      : <CloseIcon className={`rt-icon ${falseClass}`} />
  );

  return (
    <ResultRow
      hover
      onClick={handleRowClick}
      onMouseEnter={handleRowHover}
      className={onRowClick && chartEnabled ? 'rt-clickable' : undefined}
    >
      {showActions && (
        <TableCell align="center" onClick={stopPropagation} className="rt-p2 rt-w60">
          {chartEnabled ? (
            <IconButton
              size="small"
              onClick={handleChartClick}
              className="rt-chart-button"
            >
              <ShowChartIcon className="rt-icon" />
            </IconButton>
          ) : null}
          {showWatchlistMenu ? <AddToWatchlistMenu symbols={row.symbol} size="small" /> : null}
        </TableCell>
      )}

      <TableCell className="rt-symbol">
        <div className="rt-sym-stack">
          <div className="rt-sym-line">
            <Typography component="span" variant="body2" className="rt-sym-text">
              {row.symbol}
            </Typography>
            <MarketBadge market={row.market} exchange={row.exchange} />
            <FieldAvailabilityChip
              fieldAvailability={row.field_availability}
              growthMetricBasis={row.growth_metric_basis}
            />
          </div>
          {row.company_name || statusChip || row.matched_groups?.length ? (
            <div className="rt-sym-sub">
              {row.company_name ? (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  noWrap
                  title={row.company_name}
                  className="rt-sym-company"
                >
                  {row.company_name}
                </Typography>
              ) : null}
              {statusChip ? (
                <Chip
                  label={statusChip.label}
                  color={statusChip.color}
                  size="small"
                  title={statusChip.title}
                  sx={{ height: 18, fontSize: 10, flexShrink: 0 }}
                />
              ) : null}
              {row.matched_groups?.length ? (
                <Chip
                  label={row.matched_groups.length === 1
                    ? row.matched_groups[0].name
                    : `${row.matched_groups[0].name} +${row.matched_groups.length - 1}`}
                  size="small"
                  color="primary"
                  variant="outlined"
                  title={`Matched setups: ${row.matched_groups.map((group) => group.name).join(', ')}`}
                  sx={{
                    height: 18,
                    maxWidth: 105,
                    fontSize: 9,
                    flexShrink: 0,
                    '& .MuiChip-label': { px: 0.65, overflow: 'hidden', textOverflow: 'ellipsis' },
                  }}
                />
              ) : null}
            </div>
          ) : null}
        </div>
      </TableCell>

      <TableCell align="center" className="rt-p4 rt-w110">
        <RSSparkline
          data={row.rs_sparkline_data}
          trend={row.rs_trend}
          width={100}
          height={28}
        />
      </TableCell>

      <TableCell align="center" className="rt-p4 rt-w110">
        <PriceSparkline
          data={row.price_sparkline_data}
          trend={row.price_trend}
          change1d={row.price_change_1d}
          industry={row.ibd_industry_group}
          width={100}
          height={28}
        />
      </TableCell>

      <TableCell align="center" className="rt-c-text-secondary rt-w80 rt-ellipsis">
        {row.gics_sector || '-'}
      </TableCell>

      <TableCell align="left" className="rt-c-text-secondary rt-w140 rt-ellipsis">
        {row.ibd_industry_group || '-'}
      </TableCell>

      <TableCell align="left" className="rt-c-text-secondary rt-w180 rt-py4">
        <MarketThemesList themes={row.market_themes} variant="compact" />
      </TableCell>

      <TableCell
        align="center"
        title={getGroupRankTooltip(row)}
        className={`rt-mono rt-w45 ${colorClass(getGroupRankColor(row.ibd_group_rank))}${
          row.ibd_group_rank && row.ibd_group_rank <= 20 ? ' rt-strong' : ''
        }`}
      >
        {row.ibd_group_rank ?? '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w50 rt-strong ${scoreHeatClass(row.composite_score)}`}>
        {row.composite_score?.toFixed(1) || '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w45 ${scoreHeatClass(row.minervini_score)}`}>
        {row.minervini_score != null ? row.minervini_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w45 ${scoreHeatClass(row.canslim_score)}`}>
        {row.canslim_score != null ? row.canslim_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w45 ${scoreHeatClass(row.ipo_score)}`}>
        {row.ipo_score != null ? row.ipo_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w45 ${scoreHeatClass(row.custom_score)}`}>
        {row.custom_score != null ? row.custom_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w50 ${scoreHeatClass(row.volume_breakthrough_score)}`}>
        {row.volume_breakthrough_score != null ? row.volume_breakthrough_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w45 ${scoreHeatClass(row.se_setup_score)}`}>
        {row.se_setup_score != null ? row.se_setup_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-c-text-secondary rt-w55 rt-ellipsis">
        {row.se_pattern_primary || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w50">
        {row.se_distance_to_pivot_pct != null ? `${row.se_distance_to_pivot_pct.toFixed(1)}%` : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.se_bb_width_pctile_252 != null ? row.se_bb_width_pctile_252.toFixed(0) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.se_volume_vs_50d != null ? `${row.se_volume_vs_50d.toFixed(1)}x` : '-'}
      </TableCell>

      <TableCell align="center" className="rt-w35">
        {row.se_rs_line_new_high == null ? '-' : check(row.se_rs_line_new_high)}
      </TableCell>

      <TableCell align="center" className="rt-w42">
        {row.se_rs_line_blue_dot ? (
          <span className="rt-dot" title="Setup Engine RS line at new high before price" />
        ) : (
          <span className="rt-c-text-disabled">-</span>
        )}
      </TableCell>

      <TableCell align="center" className="rt-w42">
        {row.rs_line_blue_dot_recent ? (
          <span
            className="rt-dot"
            title={row.rs_line_new_high_date ? `RS blue dot: latest RS high ${row.rs_line_new_high_date}` : 'RS blue dot within 5 trading days'}
          />
        ) : (
          <span className="rt-c-text-disabled">-</span>
        )}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w55">
        {row.se_pivot_price != null ? `$${row.se_pivot_price.toFixed(2)}` : '-'}
      </TableCell>

      <OpportunityResultCells
        row={row}
        visible={showOpportunityState}
        onOpenEvidence={onOpenOpportunity}
      />

      {renderSocialCell ? <TableCell align="center" className="rt-w110">{renderSocialCell(row)}</TableCell> : null}

      <TableCell align="center" className="rt-mono rt-w40">
        {row.rs_rating?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w40">
        {row.rs_rating_1m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w40">
        {row.rs_rating_3m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.rs_rating_12m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.beta != null ? row.beta.toFixed(2) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w45">
        {row.beta_adj_rs != null ? row.beta_adj_rs.toFixed(0) : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w55 ${colorClass(getEpsRatingColor(row.eps_rating))}`}>
        {row.eps_rating != null ? row.eps_rating : '-'}
      </TableCell>

      <TableCell align="center" className="rt-w40">
        {row.stage != null ? (
          <span className={`rt-stage rt-stage-${row.stage}`}>
            S{row.stage}
          </span>
        ) : (
          '-'
        )}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w65">
        {formatLocalCurrency(row.current_price, row.currency)}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w60">
        {formatLargeNumber(row.volume, getCurrencyPrefix(row.currency))}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w75">
        {mcapDisplay === MCAP_DISPLAY.USD
          ? formatLargeNumber(row.market_cap_usd, '$')
          : formatLargeNumber(row.market_cap, getCurrencyPrefix(row.currency))}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w70">
        {formatLargeNumber(row.adv_usd, '$')}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w50 ${colorClass(getIpoAgeColor(row.ipo_date))}`}>
        {formatIpoAge(row.ipo_date)}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w50 ${colorClass(getGrowthColor(row.eps_growth_qq))}`}>
        {row.eps_growth_qq != null ? `${row.eps_growth_qq.toFixed(0)}%` : '-'}
      </TableCell>

      <TableCell align="center" className={`rt-mono rt-w50 ${colorClass(getGrowthColor(row.sales_growth_qq))}`}>
        {row.sales_growth_qq != null ? `${row.sales_growth_qq.toFixed(0)}%` : '-'}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w50">
        {row.adr_percent != null ? `${row.adr_percent.toFixed(1)}%` : '-'}
      </TableCell>

      <TableCell align="center" className="rt-w35">
        {check(row.ma_alignment, 'rt-c-error-main')}
      </TableCell>

      <TableCell align="center" className="rt-w40">
        {check(row.vcp_detected)}
      </TableCell>

      <TableCell align="center" className="rt-mono rt-w50">
        {row.vcp_score != null ? row.vcp_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="right" className="rt-mono rt-w55">
        {row.vcp_pivot != null ? row.vcp_pivot.toFixed(2) : '-'}
      </TableCell>

      <TableCell align="center" className="rt-w35">
        {check(row.vcp_ready_for_breakout)}
      </TableCell>

      <TableCell align="center" className="rt-w40">
        {check(row.passes_template)}
      </TableCell>

      <TableCell align="center" className="rt-w80">
        <Chip
          label={row.rating}
          color={getRatingColor(row.rating)}
          size="small"
        />
      </TableCell>
    </ResultRow>
  );
});

/**
 * Display scan results in a sortable, paginated table with row virtualization
 * @param {Function} onRowHover - Optional callback when hovering over a row (for prefetching)
 */
function ResultsTable({
  results,
  total,
  page,
  perPage,
  sortBy,
  sortOrder,
  onPageChange,
  onPerPageChange,
  onSortChange,
  onOpenChart,
  loading,
  onRowHover,
  showActions = true,
  showWatchlistMenu = true,
  sortingEnabled = true,
  isChartEnabled,
  showOpportunityState = false,
  opportunityTelemetrySurface,
  renderSocialCell = null,
}) {
  const parentRef = useRef(null);
  // MCap column display mode — kept as local state; scan-level persistence
  // can lift this up later if users want it to survive navigation.
  const [mcapDisplay, setMcapDisplay] = useState(MCAP_DISPLAY.USD);
  const isMobile = useIsMobileShell();
  const {
    selectedRow: opportunityRow,
    openEvidence: handleOpenOpportunity,
    closeEvidence: handleCloseOpportunity,
  } = useOpportunityEvidenceSelection({
    rows: results,
    enabled: showOpportunityState,
  });
  const visibleColumns = useMemo(() => {
    const base = columns.filter((column) => (
      (showActions || column.id !== 'chart') &&
      (showOpportunityState || !['resilience_score', 'action_state'].includes(column.id)) &&
      (renderSocialCell || column.id !== 'social')
    ));
    return base.map((column) =>
      column.id === 'market_cap'
        ? { ...column, label: mcapDisplay === MCAP_DISPLAY.USD ? 'MCap ($)' : 'MCap (local)' }
        : column,
    );
  }, [showActions, showOpportunityState, renderSocialCell, mcapDisplay]);

  const toggleMcapDisplay = useCallback(() => {
    setMcapDisplay((mode) =>
      mode === MCAP_DISPLAY.USD ? MCAP_DISPLAY.LOCAL : MCAP_DISPLAY.USD,
    );
  }, []);

  const handleChangePage = useCallback((event, newPage) => {
    onPageChange(newPage + 1); // Material-UI uses 0-based pages, API uses 1-based
  }, [onPageChange]);

  const handleRequestSort = useCallback((property) => {
    const isAsc = sortBy === property && sortOrder === 'asc';
    onSortChange(property, isAsc ? 'desc' : 'asc');
  }, [sortBy, sortOrder, onSortChange]);

  const handleRowClick = useCallback((symbol) => {
    onOpenChart?.(symbol);
  }, [onOpenChart]);

  // Virtualize rows - only render visible rows plus overscan
  const rowVirtualizer = useVirtualizer({
    count: results?.length || 0,
    getScrollElement: () => parentRef.current,
    estimateSize: () => (isMobile ? MOBILE_CARD_HEIGHT : ROW_HEIGHT),
    overscan: 10, // Render 10 extra rows above/below viewport
  });
  // Row height changes with the layout (e.g. rotating a tablet); drop cached
  // sizes before paint so cards never draw at the table's row spacing.
  useLayoutEffect(() => {
    rowVirtualizer.measure();
  }, [isMobile, rowVirtualizer]);

  // Memoize virtual items to prevent recalculation
  const virtualRows = rowVirtualizer.getVirtualItems();

  if (loading) {
    return (
      <TableSkeleton label="Loading scan results" rows={10} columns={8} title={false} />
    );
  }

  if (!results || results.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No results found
        </Typography>
      </Paper>
    );
  }

  const pagination = (
    <TablePagination
      rowsPerPageOptions={[10, 25, 50, 100]}
      component="div"
      count={total}
      rowsPerPage={perPage}
      page={page - 1} // Material-UI uses 0-based pages
      onPageChange={handleChangePage}
      onRowsPerPageChange={(e) => {
        const nextPerPage = Number(e.target.value);
        onPerPageChange?.(nextPerPage);
        onPageChange(1); // Reset to first page when changing per-page
      }}
    />
  );
  const opportunityDrawer = (
    <OpportunityEvidenceDrawer
      open={showOpportunityState && Boolean(opportunityRow)}
      row={opportunityRow}
      onClose={handleCloseOpportunity}
      opportunityTelemetrySurface={opportunityTelemetrySurface}
    />
  );

  // One badge per card, most explanatory first: the scan status says why a
  // card can't open (e.g. "New IPO"), then the action state, then the rating
  // the desktop table shows in its last column.
  const mobileBadge = (row) => {
    const status = getStatusChipProps(row);
    if (status) {
      return <Chip label={status.label} color={status.color} title={status.title} size="small" sx={{ height: 18, fontSize: 10 }} />;
    }
    if (showOpportunityState) {
      return (
        <ActionStateBadge
          state={row.action_state}
          onClick={row.opportunity_state
            ? (event) => {
              event.stopPropagation();
              handleOpenOpportunity(row);
            }
            : undefined}
        />
      );
    }
    return row.rating
      ? <Chip label={row.rating} color={getRatingColor(row.rating)} variant="outlined" size="small" sx={{ height: 18, fontSize: 10 }} />
      : null;
  };

  if (isMobile) {
    // Same sortable columns as the desktop headers (visibleColumns already
    // drops opportunity columns when that capability is off), common ones
    // first; a preset-only active sort is appended so the picker can show it.
    const commonIds = new Set(MOBILE_SORT_FIELDS.map(([id]) => id));
    const moreSorts = visibleColumns
      .filter((column) => column.sortable && !commonIds.has(column.id))
      // An active alias takes over its column's option, so the picker lists it
      // once and keeps the preset's own sort key.
      .map((column) => [SORT_ALIASES[sortBy] === column.id ? sortBy : column.id, sortName(column.id)]);
    const listedIds = new Set([...commonIds, ...moreSorts.map(([id]) => id)]);
    if (sortBy && !listedIds.has(sortBy)) moreSorts.push([sortBy, sortName(sortBy)]);
    return (
      <Paper elevation={1}>
        {sortingEnabled ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1.5, py: 1, borderBottom: 1, borderColor: 'divider' }}>
            <TextField
              select
              size="small"
              label="Sort by"
              value={sortBy ?? ''}
              onChange={(event) => onSortChange(event.target.value, sortOrder)}
              sx={{ minWidth: 170 }}
            >
              {MOBILE_SORT_FIELDS.map(([id, label]) => <MenuItem key={id} value={id}>{label}</MenuItem>)}
              {moreSorts.length ? <ListSubheader>More sorts</ListSubheader> : null}
              {moreSorts.map(([id, label]) => <MenuItem key={id} value={id}>{label}</MenuItem>)}
            </TextField>
            <IconButton
              aria-label={sortOrder === 'asc' ? 'Sorted ascending; switch to descending' : 'Sorted descending; switch to ascending'}
              onClick={() => onSortChange(sortBy, sortOrder === 'asc' ? 'desc' : 'asc')}
            >
              {sortOrder === 'asc' ? <ArrowUpwardIcon fontSize="small" /> : <ArrowDownwardIcon fontSize="small" />}
            </IconButton>
          </Box>
        ) : null}
        <Box ref={parentRef} role="list" aria-label="Scan results" sx={{ maxHeight: 'calc(100dvh - 300px)', minHeight: 240, overflow: 'auto' }}>
          <Box sx={{ height: rowVirtualizer.getTotalSize(), position: 'relative' }}>
            {virtualRows.map((virtualRow) => {
              const row = results[virtualRow.index];
              const chartEnabled = row.is_scannable !== false &&
                (isChartEnabled ? isChartEnabled(row.symbol) : Boolean(onOpenChart));
              return (
                <Box
                  key={row.symbol}
                  role="listitem"
                  aria-setsize={results.length}
                  aria-posinset={virtualRow.index + 1}
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: MOBILE_CARD_HEIGHT,
                    transform: `translateY(${virtualRow.start}px)`,
                    borderBottom: 1,
                    borderColor: 'divider',
                    overflow: 'hidden',
                  }}
                >
                  <StockRowCard
                    row={row}
                    metric={cardMetricFor(row, sortBy)}
                    sortNote={[CARD_SORT_NOTES[sortBy]?.(row), matchedGroupsLabel(row)].filter(Boolean).join(' · ') || null}
                    badge={(
                      <>
                        <MarketBadge market={row.market} exchange={row.exchange} />
                        {mobileBadge(row)}
                      </>
                    )}
                    actions={(
                      <>
                        <FieldAvailabilityChip
                          fieldAvailability={row.field_availability}
                          growthMetricBasis={row.growth_metric_basis}
                        />
                        {showActions && showWatchlistMenu ? <AddToWatchlistMenu symbols={row.symbol} size="small" /> : null}
                      </>
                    )}
                    onOpen={chartEnabled ? () => handleRowClick(row.symbol) : null}
                  />
                </Box>
              );
            })}
          </Box>
        </Box>
        {pagination}
        {opportunityDrawer}
      </Paper>
    );
  }

  return (
    <Paper elevation={1}>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', px: 2, py: 0.5, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
          Market Cap display:
        </Typography>
        <Chip
          label={mcapDisplay === MCAP_DISPLAY.USD ? 'USD' : 'Local'}
          size="small"
          variant="outlined"
          onClick={toggleMcapDisplay}
          data-testid="mcap-display-toggle"
          sx={{ cursor: 'pointer', fontSize: 11, height: 20 }}
        />
      </Box>
      <TableContainer
        ref={parentRef}
        sx={{
          maxHeight: 'calc(100vh - 280px)',
          overflow: 'auto',
        }}
      >
        <Table
          stickyHeader
          size="small"
          sx={{
            minWidth: (showActions
              ? TABLE_MIN_WIDTH.WITH_ACTIONS
              : TABLE_MIN_WIDTH.WITHOUT_ACTIONS) +
              (showOpportunityState ? OPPORTUNITY_COLUMNS_WIDTH : 0) +
              (renderSocialCell ? SOCIAL_COLUMN_WIDTH : 0),
          }}
        >
          <TableHead>
            <TableRow>
              {visibleColumns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.id === 'symbol' ? 'left' : 'center'}
                  sx={{
                    width: column.width,
                    minWidth: column.width,
                    maxWidth: column.width,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {column.sortable && sortingEnabled ? (
                    <TableSortLabel
                      active={sortBy === column.id}
                      direction={sortBy === column.id ? sortOrder : 'asc'}
                      onClick={() => handleRequestSort(column.id)}
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {/* Spacer for virtualization - pushes content down to correct position */}
            {virtualRows.length > 0 && virtualRows[0].start > 0 && (
              <tr style={{ height: virtualRows[0].start }} />
            )}
            {virtualRows.map((virtualRow) => {
              const row = results[virtualRow.index];
              return (
                <VirtualTableRow
                  key={row.symbol}
                  row={row}
                  onRowClick={onOpenChart ? handleRowClick : null}
                  onRowHover={onRowHover}
                  onOpenChart={onOpenChart}
                  showActions={showActions}
                  showWatchlistMenu={showWatchlistMenu}
                  chartEnabled={
                    row.is_scannable !== false &&
                    (isChartEnabled ? isChartEnabled(row.symbol) : Boolean(onOpenChart))
                  }
                  mcapDisplay={mcapDisplay}
                  showOpportunityState={showOpportunityState}
                  onOpenOpportunity={handleOpenOpportunity}
                  renderSocialCell={renderSocialCell}
                />
              );
            })}
            {/* Bottom spacer for virtualization */}
            {virtualRows.length > 0 && (
              <tr style={{ height: rowVirtualizer.getTotalSize() - (virtualRows[virtualRows.length - 1]?.end || 0) }} />
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {pagination}

      {opportunityDrawer}
    </Paper>
  );
}

// Wrap with React.memo for component-level memoization
export default memo(ResultsTable, (prevProps, nextProps) => {
  // Only re-render if these key props change
  return (
    prevProps.results === nextProps.results &&
    prevProps.total === nextProps.total &&
    prevProps.page === nextProps.page &&
    prevProps.perPage === nextProps.perPage &&
    prevProps.sortBy === nextProps.sortBy &&
    prevProps.sortOrder === nextProps.sortOrder &&
    prevProps.loading === nextProps.loading &&
    prevProps.showActions === nextProps.showActions &&
    prevProps.showWatchlistMenu === nextProps.showWatchlistMenu &&
    prevProps.isChartEnabled === nextProps.isChartEnabled &&
    prevProps.sortingEnabled === nextProps.sortingEnabled &&
    prevProps.showOpportunityState === nextProps.showOpportunityState &&
    prevProps.opportunityTelemetrySurface === nextProps.opportunityTelemetrySurface &&
    prevProps.renderSocialCell === nextProps.renderSocialCell
  );
});
