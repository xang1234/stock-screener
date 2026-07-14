import { createEmptyExpression } from './filterExpressionModel';
import {
  BOOLEAN_FILTER_TO_FIELD,
  FIELD_TO_BOOLEAN_FILTER,
  FIELD_TO_RANGE_FILTER,
  RANGE_FILTER_TO_FIELD,
} from './scanFilterFields';

const IPO_PRESET_MONTHS = { '6m': 6, '1y': 12, '2y': 24, '3y': 36, '5y': 60 };

export function resolveIpoCutoff(preset, now = new Date()) {
  if (!preset) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(preset)) return preset;
  const months = IPO_PRESET_MONTHS[preset];
  if (months == null) return null;
  const cutoff = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  cutoff.setUTCMonth(cutoff.getUTCMonth() - months);
  return cutoff.toISOString().slice(0, 10);
}

function rangeCondition(field, range) {
  if (!range || (range.min == null && range.max == null)) return null;
  return { kind: 'range', field, min: range.min ?? null, max: range.max ?? null };
}

function categoricalCondition(field, values, mode = 'include') {
  if (!Array.isArray(values) || values.length === 0) return null;
  return { kind: 'categorical', field, values: [...new Set(values)], mode };
}

export function legacyFiltersToConditions(filters = {}, now = new Date()) {
  const conditions = [];
  Object.entries(RANGE_FILTER_TO_FIELD).forEach(([key, field]) => {
    const condition = rangeCondition(field, filters[key]);
    if (condition) conditions.push(condition);
  });
  Object.entries(BOOLEAN_FILTER_TO_FIELD).forEach(([key, field]) => {
    if (filters[key] != null) {
      if (typeof filters[key] !== 'boolean') {
        throw new TypeError(`Legacy boolean filter ${key} must be a boolean`);
      }
      conditions.push({ kind: 'boolean', field, value: filters[key] });
    }
  });
  if (filters.symbolSearch?.trim()) {
    conditions.push({ kind: 'text', field: 'listing_search', pattern: filters.symbolSearch.trim() });
  }
  if (filters.stage != null) {
    conditions.push({ kind: 'range', field: 'stage', min: filters.stage, max: filters.stage });
  }
  [
    ['rating', filters.ratings, 'include'],
    ['ibd_industry_group', filters.ibdIndustries?.values, filters.ibdIndustries?.mode],
    ['gics_sector', filters.gicsSectors?.values, filters.gicsSectors?.mode],
    ['market', filters.markets, 'include'],
    ['se_pattern_primary', filters.sePatternPrimary, 'include'],
  ].forEach(([field, values, mode]) => {
    const condition = categoricalCondition(field, values, mode);
    if (condition) conditions.push(condition);
  });
  if (filters.passesTemplate === true) {
    conditions.push({
      kind: 'categorical',
      field: 'rating',
      values: ['Strong Buy', 'Buy'],
      mode: 'include',
    });
  }
  if (filters.minVolume != null) {
    conditions.push(filters.symbolSearch?.trim()
      ? { kind: 'range', field: 'listing_aware_volume', min: filters.minVolume, max: null }
      : { kind: 'range', field: 'volume', min: filters.minVolume, max: null });
  }
  if (filters.minMarketCap != null) {
    conditions.push({ kind: 'range', field: 'market_cap', min: filters.minMarketCap, max: null });
  }
  const ipoCutoff = resolveIpoCutoff(filters.ipoAfter, now);
  if (ipoCutoff) {
    conditions.push({ kind: 'range', field: 'ipo_date', min: ipoCutoff, max: null });
  }
  return conditions;
}

function isQuickFilterCondition(condition) {
  if (condition.kind === 'range') {
    if (FIELD_TO_RANGE_FILTER[condition.field]) return true;
    if (condition.field === 'stage') return condition.min === condition.max;
    return condition.min != null
      && condition.max == null
      && ['volume', 'listing_aware_volume', 'market_cap', 'ipo_date']
        .includes(condition.field);
  }
  if (condition.kind === 'boolean') {
    return Boolean(FIELD_TO_BOOLEAN_FILTER[condition.field]);
  }
  if (condition.kind === 'text') {
    return ['symbol', 'listing_search'].includes(condition.field);
  }
  if (condition.kind !== 'categorical') return false;
  if (['ibd_industry_group', 'gics_sector'].includes(condition.field)) return true;
  return condition.mode !== 'exclude'
    && ['rating', 'market', 'se_pattern_primary'].includes(condition.field);
}

export function patchExpressionQuickFilters(expression, filters, now = new Date()) {
  const base = expression ? structuredClone(expression) : createEmptyExpression();
  const preserved = (base.required?.conditions || []).filter(
    (condition) => !isQuickFilterCondition(condition),
  );
  base.expression_version = 1;
  base.required = {
    id: 'required',
    name: 'Always require',
    match: 'all',
    enabled: true,
    conditions: [...preserved, ...legacyFiltersToConditions(filters, now)],
  };
  base.group_join = base.group_join === 'all' ? 'all' : 'any';
  base.groups = Array.isArray(base.groups) ? base.groups : [];
  return base;
}

export function legacyFiltersToExpression(filters = {}, previousExpression = null, now = new Date()) {
  return patchExpressionQuickFilters(previousExpression, filters, now);
}

export function expressionToLegacyFilters(expression, defaults) {
  const result = structuredClone(defaults);
  const conditions = expression?.required?.conditions ?? [];
  conditions.forEach((condition) => {
    if (condition.kind === 'range') {
      const key = FIELD_TO_RANGE_FILTER[condition.field];
      if (key) result[key] = { min: condition.min ?? null, max: condition.max ?? null };
      if (condition.field === 'stage' && condition.min === condition.max) result.stage = condition.min;
      if (condition.field === 'volume' && condition.max == null) result.minVolume = condition.min;
      if (condition.field === 'listing_aware_volume' && condition.max == null) {
        result.minVolume = condition.min;
      }
      if (condition.field === 'market_cap' && condition.max == null) result.minMarketCap = condition.min;
      if (condition.field === 'ipo_date' && condition.max == null) result.ipoAfter = condition.min;
    } else if (condition.kind === 'boolean') {
      const key = FIELD_TO_BOOLEAN_FILTER[condition.field];
      if (key) result[key] = condition.value;
    } else if (condition.kind === 'text' && ['symbol', 'listing_search'].includes(condition.field)) {
      result.symbolSearch = condition.pattern;
    } else if (condition.kind === 'categorical') {
      if (condition.field === 'rating' && condition.mode !== 'exclude') {
        result.ratings = [...condition.values];
      }
      if (condition.field === 'ibd_industry_group') {
        result.ibdIndustries = { values: [...condition.values], mode: condition.mode };
      }
      if (condition.field === 'gics_sector') {
        result.gicsSectors = { values: [...condition.values], mode: condition.mode };
      }
      if (condition.field === 'market' && condition.mode !== 'exclude') {
        result.markets = [...condition.values];
      }
      if (condition.field === 'se_pattern_primary' && condition.mode !== 'exclude') {
        result.sePatternPrimary = [...condition.values];
      }
    }
  });
  return result;
}
