import { MARKET_CAP_OPTIONS, VOLUME_OPTIONS } from './constants';

const NULL_RESET_KEYS = new Set(['stage', 'minVolume', 'minMarketCap', 'ipoAfter']);
const ARRAY_RESET_KEYS = new Set(['ratings']);
const MODED_MULTI_RESET_KEYS = new Set(['ibdIndustries', 'gicsSectors']);
const BOOLEAN_RESET_KEYS = new Set([
  'vcpDetected',
  'vcpReady',
  'maAlignment',
  'passesTemplate',
  'seSetupReady',
  'seRsLineNewHigh',
]);

const SCORE_FILTERS = [
  { key: 'compositeScore', label: 'Composite' },
  { key: 'minerviniScore', label: 'Minervini' },
  { key: 'canslimScore', label: 'CANSLIM' },
  { key: 'ipoScore', label: 'IPO' },
  { key: 'customScore', label: 'Custom' },
  { key: 'volBreakthroughScore', label: 'Vol BT' },
  { key: 'seSetupScore', label: 'SE Score' },
];

const RS_FILTERS = [
  { key: 'rsRating', label: 'RS' },
  { key: 'rs1m', label: 'RS 1M' },
  { key: 'rs3m', label: 'RS 3M' },
  { key: 'rs12m', label: 'RS 12M' },
  { key: 'epsRating', label: 'EPS Rtg' },
];

const TECH_FILTERS = [
  { key: 'perfDay', label: '1D Chg' },
  { key: 'perfWeek', label: '1W Chg' },
  { key: 'perfMonth', label: '1M Chg' },
  { key: 'perf3m', label: '3M Chg' },
  { key: 'perf6m', label: '6M Chg' },
  { key: 'gapPercent', label: 'Gap' },
  { key: 'volumeSurge', label: 'Vol Surge' },
  { key: 'ema10Distance', label: 'vs EMA10' },
  { key: 'ema20Distance', label: 'vs EMA20' },
  { key: 'ema50Distance', label: 'vs EMA50' },
  { key: 'week52HighDistance', label: '52W Hi' },
  { key: 'week52LowDistance', label: '52W Lo' },
  { key: 'beta', label: 'Beta' },
  { key: 'betaAdjRs', label: 'β-adj RS' },
  { key: 'seDistanceToPivot', label: 'Pvt Dist' },
  { key: 'seBbSqueeze', label: 'Squeeze' },
  { key: 'seVolumeVs50d', label: 'Vol/50d' },
];

function hasRangeValue(range) {
  return Boolean(range && (range.min != null || range.max != null));
}

function formatRangeLabel(range, { minPrefix = '≥', maxPrefix = '≤', suffix = '' } = {}) {
  const minStr = range?.min != null ? `${minPrefix}${range.min}${suffix}` : '';
  const maxStr = range?.max != null ? `${maxPrefix}${range.max}${suffix}` : '';
  return `${minStr}${minStr && maxStr ? ', ' : ''}${maxStr}`;
}

export function isFilterActive(filters, key) {
  const value = filters[key];
  if (value === null || value === undefined) return false;
  if (typeof value === 'string') return value.length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'boolean') return true;
  if (typeof value === 'object') {
    if ('min' in value || 'max' in value) {
      return value.min != null || value.max != null;
    }
    if ('values' in value) {
      return value.values?.length > 0;
    }
  }
  return true;
}

export function countActiveInCategory(filters, keys) {
  return keys.filter((key) => isFilterActive(filters, key)).length;
}

export function buildActiveFilters(filters) {
  const active = [];

  if (filters.symbolSearch) {
    active.push({ key: 'symbolSearch', label: `Symbol: ${filters.symbolSearch}` });
  }
  if (filters.stage != null) {
    active.push({ key: 'stage', label: `Stage: ${filters.stage}` });
  }
  if (filters.ratings?.length) {
    active.push({ key: 'ratings', label: `Rating: ${filters.ratings.join(', ')}` });
  }
  if (filters.ibdIndustries?.values?.length) {
    const modeLabel = filters.ibdIndustries.mode === 'exclude' ? ' (Exclude)' : '';
    active.push({ key: 'ibdIndustries', label: `Industry${modeLabel}: ${filters.ibdIndustries.values.length} selected` });
  }
  if (filters.gicsSectors?.values?.length) {
    const modeLabel = filters.gicsSectors.mode === 'exclude' ? ' (Exclude)' : '';
    active.push({ key: 'gicsSectors', label: `Sector${modeLabel}: ${filters.gicsSectors.values.length} selected` });
  }
  if (filters.minVolume != null) {
    const volLabel = VOLUME_OPTIONS.find((option) => option.value === filters.minVolume)?.label || `>${filters.minVolume}`;
    active.push({ key: 'minVolume', label: `Dollar Vol: ${volLabel}` });
  }
  if (filters.minMarketCap != null) {
    const capLabel = MARKET_CAP_OPTIONS.find((option) => option.value === filters.minMarketCap)?.label || `>${filters.minMarketCap}`;
    active.push({ key: 'minMarketCap', label: `Mkt Cap: ${capLabel}` });
  }
  if (filters.ipoAfter) {
    active.push({ key: 'ipoAfter', label: `IPO: >${filters.ipoAfter.toUpperCase()}` });
  }

  for (const { key, label } of SCORE_FILTERS) {
    const range = filters[key];
    if (hasRangeValue(range)) {
      active.push({ key, label: `${label}: ${formatRangeLabel(range)}` });
    }
  }

  for (const { key, label } of RS_FILTERS) {
    const range = filters[key];
    if (hasRangeValue(range)) {
      active.push({ key, label: `${label}: ${formatRangeLabel(range)}` });
    }
  }

  if (hasRangeValue(filters.price)) {
    const { min, max } = filters.price;
    active.push({
      key: 'price',
      label: `Price: ${min != null ? `≥$${min}` : ''}${max != null ? ` ≤$${max}` : ''}`,
    });
  }
  if (hasRangeValue(filters.adrPercent)) {
    const { min, max } = filters.adrPercent;
    active.push({
      key: 'adrPercent',
      label: `ADR: ${min != null ? `≥${min}%` : ''}${max != null ? ` ≤${max}%` : ''}`,
    });
  }
  if (hasRangeValue(filters.epsGrowth)) {
    const { min, max } = filters.epsGrowth;
    active.push({
      key: 'epsGrowth',
      label: `EPS: ${min != null ? `≥${min}%` : ''}${max != null ? ` ≤${max}%` : ''}`,
    });
  }
  if (hasRangeValue(filters.salesGrowth)) {
    const { min, max } = filters.salesGrowth;
    active.push({
      key: 'salesGrowth',
      label: `Sales: ${min != null ? `≥${min}%` : ''}${max != null ? ` ≤${max}%` : ''}`,
    });
  }

  if (hasRangeValue(filters.vcpScore)) {
    const { min, max } = filters.vcpScore;
    active.push({
      key: 'vcpScore',
      label: `VCP Score: ${min != null ? `≥${min}` : ''}${max != null ? ` ≤${max}` : ''}`,
    });
  }
  if (hasRangeValue(filters.vcpPivot)) {
    const { min, max } = filters.vcpPivot;
    active.push({
      key: 'vcpPivot',
      label: `VCP Pivot: ${min != null ? `≥$${min}` : ''}${max != null ? ` ≤$${max}` : ''}`,
    });
  }

  if (filters.vcpDetected != null) {
    active.push({ key: 'vcpDetected', label: `VCP: ${filters.vcpDetected ? 'Yes' : 'No'}` });
  }
  if (filters.vcpReady != null) {
    active.push({ key: 'vcpReady', label: `VCP Ready: ${filters.vcpReady ? 'Yes' : 'No'}` });
  }
  if (filters.maAlignment != null) {
    active.push({ key: 'maAlignment', label: `MA Align: ${filters.maAlignment ? 'Yes' : 'No'}` });
  }
  if (filters.passesTemplate != null) {
    active.push({ key: 'passesTemplate', label: `Passes: ${filters.passesTemplate ? 'Yes' : 'No'}` });
  }
  if (filters.seSetupReady != null) {
    active.push({ key: 'seSetupReady', label: `SE Ready: ${filters.seSetupReady ? 'Yes' : 'No'}` });
  }
  if (filters.seRsLineNewHigh != null) {
    active.push({ key: 'seRsLineNewHigh', label: `RS New Hi: ${filters.seRsLineNewHigh ? 'Yes' : 'No'}` });
  }

  for (const { key, label } of TECH_FILTERS) {
    const range = filters[key];
    if (hasRangeValue(range)) {
      active.push({
        key,
        label: `${label}: ${formatRangeLabel(range, { suffix: '%' })}`,
      });
    }
  }

  return active;
}

export function resetFilterValue(key) {
  if (key === 'symbolSearch') {
    return '';
  }
  if (NULL_RESET_KEYS.has(key)) {
    return null;
  }
  if (ARRAY_RESET_KEYS.has(key)) {
    return [];
  }
  if (MODED_MULTI_RESET_KEYS.has(key)) {
    return { values: [], mode: 'include' };
  }
  if (BOOLEAN_RESET_KEYS.has(key)) {
    return null;
  }
  return { min: null, max: null };
}
