import { createEmptyExpression } from './filterExpressionModel';
import {
  BOOLEAN_FILTER_TO_FIELD,
  FIELD_TO_BOOLEAN_FILTER,
  FIELD_TO_RANGE_FILTER,
  RANGE_FILTER_TO_FIELD,
} from './scanFilterFields';

const IPO_PRESET_MONTHS = { '6m': 6, '1y': 12, '2y': 24, '3y': 36, '5y': 60 };
const PASSING_RATINGS = ['Strong Buy', 'Buy'];

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

export function quickFiltersToConditions(filters = {}, now = new Date()) {
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
      values: [...PASSING_RATINGS],
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

function sameValues(left = [], right = []) {
  return left.length === right.length && left.every((value) => right.includes(value));
}

function conditionBelongsToQuickFilter(condition, key) {
  if (condition.kind === 'range') {
    if (FIELD_TO_RANGE_FILTER[condition.field] === key) return true;
    if (key === 'stage') return condition.field === 'stage' && condition.min === condition.max;
    if (key === 'minVolume') {
      return ['volume', 'listing_aware_volume'].includes(condition.field)
        && condition.min != null && condition.max == null;
    }
    if (key === 'minMarketCap') {
      return condition.field === 'market_cap' && condition.min != null && condition.max == null;
    }
    if (key === 'ipoAfter') {
      return condition.field === 'ipo_date' && condition.min != null && condition.max == null;
    }
    return false;
  }
  if (condition.kind === 'boolean') {
    return FIELD_TO_BOOLEAN_FILTER[condition.field] === key;
  }
  if (condition.kind === 'text') {
    return key === 'symbolSearch' && ['symbol', 'listing_search'].includes(condition.field);
  }
  if (condition.kind !== 'categorical') return false;
  if (key === 'ratings') return condition.field === 'rating' && condition.mode !== 'exclude';
  if (key === 'passesTemplate') {
    return condition.field === 'rating'
      && condition.mode !== 'exclude'
      && sameValues(condition.values, PASSING_RATINGS);
  }
  if (key === 'ibdIndustries') return condition.field === 'ibd_industry_group';
  if (key === 'gicsSectors') return condition.field === 'gics_sector';
  if (key === 'markets') return condition.field === 'market' && condition.mode !== 'exclude';
  return key === 'sePatternPrimary'
    && condition.field === 'se_pattern_primary'
    && condition.mode !== 'exclude';
}

function quickFilterConditions(key, value, filters, now) {
  const rangeField = RANGE_FILTER_TO_FIELD[key];
  if (rangeField) {
    const condition = rangeCondition(rangeField, value);
    return condition ? [condition] : [];
  }
  const booleanField = BOOLEAN_FILTER_TO_FIELD[key];
  if (booleanField) {
    if (value == null) return [];
    if (typeof value !== 'boolean') {
      throw new TypeError(`Legacy boolean filter ${key} must be a boolean`);
    }
    return [{ kind: 'boolean', field: booleanField, value }];
  }
  if (key === 'symbolSearch') {
    const pattern = value?.trim();
    return pattern ? [{ kind: 'text', field: 'listing_search', pattern }] : [];
  }
  if (key === 'stage') {
    return value == null ? [] : [{ kind: 'range', field: 'stage', min: value, max: value }];
  }
  if (key === 'ratings') {
    const condition = categoricalCondition('rating', value, 'include');
    return condition ? [condition] : [];
  }
  if (key === 'ibdIndustries' || key === 'gicsSectors') {
    const field = key === 'ibdIndustries' ? 'ibd_industry_group' : 'gics_sector';
    const condition = categoricalCondition(field, value?.values, value?.mode);
    return condition ? [condition] : [];
  }
  if (key === 'markets' || key === 'sePatternPrimary') {
    const field = key === 'markets' ? 'market' : 'se_pattern_primary';
    const condition = categoricalCondition(field, value, 'include');
    return condition ? [condition] : [];
  }
  if (key === 'passesTemplate') {
    return value === true
      ? [{ kind: 'categorical', field: 'rating', values: [...PASSING_RATINGS], mode: 'include' }]
      : [];
  }
  if (key === 'minVolume') {
    return value == null ? [] : [{
      kind: 'range',
      field: filters.symbolSearch?.trim() ? 'listing_aware_volume' : 'volume',
      min: value,
      max: null,
    }];
  }
  if (key === 'minMarketCap') {
    return value == null
      ? [] : [{ kind: 'range', field: 'market_cap', min: value, max: null }];
  }
  if (key === 'ipoAfter') {
    const cutoff = resolveIpoCutoff(value, now);
    return cutoff ? [{ kind: 'range', field: 'ipo_date', min: cutoff, max: null }] : [];
  }
  throw new TypeError(`Unsupported quick filter: ${key}`);
}

function replaceFirstQuickFilterCondition(expression, key, value, filters, now) {
  const conditions = [...(expression.required?.conditions || [])];
  // The quick grid owns one slot per key. Later canonical duplicates remain untouched.
  const index = conditions.findIndex((condition) => conditionBelongsToQuickFilter(condition, key));
  const replacement = quickFilterConditions(key, value, filters, now);
  if (index === -1) conditions.push(...replacement);
  else conditions.splice(index, 1, ...replacement);
  expression.required = { ...expression.required, conditions };
}

export function patchExpressionQuickFilter(
  expression,
  key,
  value,
  filters = {},
  now = new Date(),
) {
  const base = structuredClone(expression ?? createEmptyExpression());
  replaceFirstQuickFilterCondition(base, key, value, filters, now);
  if (key === 'symbolSearch' && filters.minVolume != null) {
    replaceFirstQuickFilterCondition(base, 'minVolume', filters.minVolume, filters, now);
  }
  return base;
}

export function expressionToQuickFilters(expression, defaults) {
  const result = structuredClone(defaults);
  const assigned = new Set();
  const assign = (key, value) => {
    // Mirror the same first-owned-slot rule used by targeted edits.
    if (assigned.has(key)) return;
    result[key] = value;
    assigned.add(key);
  };
  const conditions = expression?.required?.conditions ?? [];
  conditions.forEach((condition) => {
    if (condition.kind === 'range') {
      const key = FIELD_TO_RANGE_FILTER[condition.field];
      if (key) assign(key, { min: condition.min ?? null, max: condition.max ?? null });
      if (condition.field === 'stage' && condition.min === condition.max) assign('stage', condition.min);
      if (condition.field === 'volume' && condition.max == null) assign('minVolume', condition.min);
      if (condition.field === 'listing_aware_volume' && condition.max == null) {
        assign('minVolume', condition.min);
      }
      if (condition.field === 'market_cap' && condition.max == null) assign('minMarketCap', condition.min);
      if (condition.field === 'ipo_date' && condition.max == null) assign('ipoAfter', condition.min);
    } else if (condition.kind === 'boolean') {
      const key = FIELD_TO_BOOLEAN_FILTER[condition.field];
      if (key) assign(key, condition.value);
    } else if (condition.kind === 'text' && ['symbol', 'listing_search'].includes(condition.field)) {
      assign('symbolSearch', condition.pattern);
    } else if (condition.kind === 'categorical') {
      if (condition.field === 'rating' && condition.mode !== 'exclude') {
        if (sameValues(condition.values, PASSING_RATINGS)) assign('passesTemplate', true);
        else assign('ratings', [...condition.values]);
      }
      if (condition.field === 'ibd_industry_group') {
        assign('ibdIndustries', { values: [...condition.values], mode: condition.mode });
      }
      if (condition.field === 'gics_sector') {
        assign('gicsSectors', { values: [...condition.values], mode: condition.mode });
      }
      if (condition.field === 'market' && condition.mode !== 'exclude') {
        assign('markets', [...condition.values]);
      }
      if (condition.field === 'se_pattern_primary' && condition.mode !== 'exclude') {
        assign('sePatternPrimary', [...condition.values]);
      }
    }
  });
  return result;
}
