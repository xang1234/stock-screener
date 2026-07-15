import { createEmptyExpression } from './filterExpressionModel';
import {
  BOOLEAN_FILTER_TO_FIELD,
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

function restoreIpoPreset(cutoff, now) {
  return Object.keys(IPO_PRESET_MONTHS).find(
    (preset) => resolveIpoCutoff(preset, now) === cutoff,
  ) ?? cutoff;
}

function rangeCondition(field, range) {
  if (!range || (range.min == null && range.max == null)) return [];
  return [{ kind: 'range', field, min: range.min ?? null, max: range.max ?? null }];
}

function categoricalCondition(field, values, mode = 'include') {
  if (!Array.isArray(values) || values.length === 0) return [];
  return [{ kind: 'categorical', field, values: [...new Set(values)], mode }];
}

function greatestBound(conditions, property) {
  return conditions.reduce((greatest, condition) => {
    const value = condition[property];
    return value != null && (greatest == null || value > greatest) ? value : greatest;
  }, null);
}

function leastBound(conditions, property) {
  return conditions.reduce((least, condition) => {
    const value = condition[property];
    return value != null && (least == null || value < least) ? value : least;
  }, null);
}

function effectiveRange(conditions) {
  return {
    min: greatestBound(conditions, 'min'),
    max: leastBound(conditions, 'max'),
  };
}

function uniqueValues(conditions) {
  return [...new Set(conditions.flatMap((condition) => condition.values || []))];
}

function intersectValues(conditions) {
  if (!conditions.length) return [];
  const [first, ...rest] = conditions;
  return [...new Set(first.values || [])].filter((value) => (
    rest.every((condition) => condition.values?.includes(value))
  ));
}

function effectiveCategoricalSelection(conditions) {
  const included = conditions.filter((condition) => condition.mode !== 'exclude');
  const excluded = conditions.filter((condition) => condition.mode === 'exclude');
  const excludedValues = new Set(uniqueValues(excluded));
  if (included.length) {
    return {
      values: intersectValues(included).filter((value) => !excludedValues.has(value)),
      mode: 'include',
    };
  }
  return { values: [...excludedValues], mode: 'exclude' };
}

function rangeSpec(field) {
  return {
    matches: (condition) => condition.kind === 'range' && condition.field === field,
    encode: (value) => rangeCondition(field, value),
    decode: effectiveRange,
  };
}

function booleanSpec(field) {
  return {
    matches: (condition) => condition.kind === 'boolean' && condition.field === field,
    encode: (value, _filters, _now, key) => {
      if (value == null) return [];
      if (typeof value !== 'boolean') {
        throw new TypeError(`Legacy boolean filter ${key} must be a boolean`);
      }
      return [{ kind: 'boolean', field, value }];
    },
    decode: (conditions) => conditions[0].value,
  };
}

function includeCategoricalSpec(field) {
  return {
    matches: (condition) => (
      condition.kind === 'categorical'
      && condition.field === field
      && condition.mode !== 'exclude'
    ),
    encode: (value) => categoricalCondition(field, value, 'include'),
    decode: intersectValues,
  };
}

function selectableCategoricalSpec(field) {
  return {
    matches: (condition) => condition.kind === 'categorical' && condition.field === field,
    encode: (value) => categoricalCondition(
      field,
      Array.isArray(value) ? value : value?.values,
      Array.isArray(value) ? 'include' : value?.mode,
    ),
    decode: effectiveCategoricalSelection,
  };
}

function sameValues(left = [], right = []) {
  return left.length === right.length && left.every((value) => right.includes(value));
}

const QUICK_FILTER_SPECS = new Map([
  ...Object.entries(RANGE_FILTER_TO_FIELD).map(([key, field]) => [key, rangeSpec(field)]),
  ...Object.entries(BOOLEAN_FILTER_TO_FIELD).map(([key, field]) => [key, booleanSpec(field)]),
  ['symbolSearch', {
    matches: (condition) => (
      condition.kind === 'text' && ['symbol', 'listing_search'].includes(condition.field)
    ),
    encode: (value) => {
      const pattern = value?.trim();
      return pattern ? [{ kind: 'text', field: 'listing_search', pattern }] : [];
    },
    decode: (conditions) => conditions[0].pattern,
  }],
  ['stage', {
    matches: (condition) => (
      condition.kind === 'range'
      && condition.field === 'stage'
      && condition.min === condition.max
    ),
    encode: (value) => (
      value == null ? [] : [{ kind: 'range', field: 'stage', min: value, max: value }]
    ),
    decode: (conditions) => conditions[0].min,
  }],
  ['ratings', {
    ...includeCategoricalSpec('rating'),
    matches: (condition) => (
      condition.kind === 'categorical'
      && condition.field === 'rating'
      && condition.mode !== 'exclude'
      && !sameValues(condition.values, PASSING_RATINGS)
    ),
    decode: intersectValues,
  }],
  ['ibdIndustries', selectableCategoricalSpec('ibd_industry_group')],
  ['gicsSectors', selectableCategoricalSpec('gics_sector')],
  ['markets', includeCategoricalSpec('market')],
  ['sePatternPrimary', includeCategoricalSpec('se_pattern_primary')],
  ['passesTemplate', {
    matches: (condition) => (
      condition.kind === 'categorical'
      && condition.field === 'rating'
      && condition.mode !== 'exclude'
      && sameValues(condition.values, PASSING_RATINGS)
    ),
    encode: (value) => (
      value === true
        ? [{ kind: 'categorical', field: 'rating', values: [...PASSING_RATINGS], mode: 'include' }]
        : []
    ),
    decode: () => true,
  }],
  ['minVolume', {
    matches: (condition) => (
      condition.kind === 'range'
      && ['volume', 'listing_aware_volume'].includes(condition.field)
      && condition.min != null
      && condition.max == null
    ),
    encode: (value, filters) => (
      value == null ? [] : [{
        kind: 'range',
        field: filters.symbolSearch?.trim() ? 'listing_aware_volume' : 'volume',
        min: value,
        max: null,
      }]
    ),
    decode: (conditions) => greatestBound(conditions, 'min'),
  }],
  ['minMarketCap', {
    matches: (condition) => (
      condition.kind === 'range'
      && condition.field === 'market_cap'
      && condition.min != null
      && condition.max == null
    ),
    encode: (value) => (
      value == null ? [] : [{ kind: 'range', field: 'market_cap', min: value, max: null }]
    ),
    decode: (conditions) => greatestBound(conditions, 'min'),
  }],
  ['ipoAfter', {
    matches: (condition) => (
      condition.kind === 'range'
      && condition.field === 'ipo_date'
      && condition.min != null
      && condition.max == null
    ),
    encode: (value, _filters, now) => {
      const cutoff = resolveIpoCutoff(value, now);
      return cutoff ? [{ kind: 'range', field: 'ipo_date', min: cutoff, max: null }] : [];
    },
    decode: (conditions, now) => restoreIpoPreset(
      greatestBound(conditions, 'min'),
      now,
    ),
  }],
]);

export function quickFiltersToConditions(filters = {}, now = new Date()) {
  return [...QUICK_FILTER_SPECS.entries()].flatMap(([key, spec]) => (
    spec.encode(filters[key], filters, now, key)
  ));
}

function replaceQuickFilterConditions(expression, key, value, filters, now) {
  const spec = QUICK_FILTER_SPECS.get(key);
  if (!spec) throw new TypeError(`Unsupported quick filter: ${key}`);

  const replacement = spec.encode(value, filters, now, key);
  const conditions = expression.required?.conditions || [];
  // Ownership is semantic and key-wide: unrelated edits preserve duplicates,
  // while editing this key normalizes every representable condition atomically.
  const firstOwnedIndex = conditions.findIndex(spec.matches);
  const preserved = conditions.filter((condition) => !spec.matches(condition));
  const insertionIndex = firstOwnedIndex < 0 ? preserved.length : firstOwnedIndex;
  preserved.splice(insertionIndex, 0, ...replacement);
  expression.required = { ...expression.required, conditions: preserved };
}

export function patchExpressionQuickFilter(
  expression,
  key,
  value,
  filters = {},
  now = new Date(),
) {
  const base = structuredClone(expression ?? createEmptyExpression());
  replaceQuickFilterConditions(base, key, value, filters, now);
  if (key === 'symbolSearch' && filters.minVolume != null) {
    replaceQuickFilterConditions(base, 'minVolume', filters.minVolume, filters, now);
  }
  return base;
}

export function expressionToQuickFilters(expression, defaults, now = new Date()) {
  const result = structuredClone(defaults);
  const conditions = expression?.required?.conditions ?? [];
  QUICK_FILTER_SPECS.forEach((spec, key) => {
    const owned = conditions.filter(spec.matches);
    if (!owned.length) return;
    const value = spec.decode(owned, now);
    if (!(Array.isArray(value) && value.length === 0)) {
      result[key] = value;
    }
  });
  return result;
}
