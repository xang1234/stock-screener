const LEGACY_FILTER_FIELDS = Object.freeze([
  ['compositeScore', 'composite_score', 'range'],
  ['minerviniScore', 'minervini_score', 'range'],
  ['canslimScore', 'canslim_score', 'range'],
  ['ipoScore', 'ipo_score', 'range'],
  ['customScore', 'custom_score', 'range'],
  ['volBreakthroughScore', 'volume_breakthrough_score', 'range'],
  ['seSetupScore', 'se_setup_score', 'range'],
  ['seDistanceToPivot', 'se_distance_to_pivot_pct', 'range'],
  ['seBbSqueeze', 'se_bb_width_pctile_252', 'range'],
  ['seVolumeVs50d', 'se_volume_vs_50d', 'range'],
  ['seUpDownVolume', 'se_up_down_volume_ratio_10d', 'range'],
  ['rsRating', 'rs_rating', 'range'],
  ['rs1m', 'rs_rating_1m', 'range'],
  ['rs3m', 'rs_rating_3m', 'range'],
  ['rs12m', 'rs_rating_12m', 'range'],
  ['epsRating', 'eps_rating', 'range'],
  ['ibdGroupRank', 'ibd_group_rank', 'range'],
  ['price', 'price', 'range'],
  ['adrPercent', 'adr_percent', 'range'],
  ['epsGrowth', 'eps_growth_qq', 'range'],
  ['salesGrowth', 'sales_growth_qq', 'range'],
  ['vcpScore', 'vcp_score', 'range'],
  ['vcpPivot', 'vcp_pivot', 'range'],
  ['perfDay', 'price_change_1d', 'range'],
  ['perfWeek', 'perf_week', 'range'],
  ['perfMonth', 'perf_month', 'range'],
  ['perf3m', 'perf_3m', 'range'],
  ['perf6m', 'perf_6m', 'range'],
  ['gapPercent', 'gap_percent', 'range'],
  ['volumeSurge', 'volume_surge', 'range'],
  ['ema10Distance', 'ema_10_distance', 'range'],
  ['ema20Distance', 'ema_20_distance', 'range'],
  ['ema50Distance', 'ema_50_distance', 'range'],
  ['week52HighDistance', 'week_52_high_distance', 'range'],
  ['week52LowDistance', 'week_52_low_distance', 'range'],
  ['beta', 'beta', 'range'],
  ['betaAdjRs', 'beta_adj_rs', 'range'],
  ['marketCapUsd', 'market_cap_usd', 'range'],
  ['advUsd', 'adv_usd', 'range'],
  ['pctDay', 'pct_day', 'range'],
  ['pctWeek', 'pct_week', 'range'],
  ['pctMonth', 'pct_month', 'range'],
  ['seSetupReady', 'se_setup_ready', 'boolean'],
  ['seRsLineNewHigh', 'se_rs_line_new_high', 'boolean'],
  ['seRsLineBlueDot', 'se_rs_line_blue_dot', 'boolean'],
  ['rsLineBlueDotRecent', 'rs_line_blue_dot_recent', 'boolean'],
  ['vcpDetected', 'vcp_detected', 'boolean'],
  ['vcpReady', 'vcp_ready_for_breakout', 'boolean'],
  ['maAlignment', 'ma_alignment', 'boolean'],
  ['pocketPivot', 'pocket_pivot', 'boolean'],
  ['powerTrend', 'power_trend', 'boolean'],
]);

const mapLegacyFields = (kind) => Object.fromEntries(
  LEGACY_FILTER_FIELDS
    .filter(([, , fieldKind]) => fieldKind === kind)
    .map(([key, field]) => [key, field]),
);

const RANGE_FILTER_TO_FIELD = Object.freeze(mapLegacyFields('range'));
const BOOLEAN_FILTER_TO_FIELD = Object.freeze(mapLegacyFields('boolean'));

const FIELD_TO_RANGE_FILTER = Object.fromEntries(
  LEGACY_FILTER_FIELDS
    .filter(([, , kind, restore = true]) => kind === 'range' && restore)
    .map(([key, field]) => [field, key]),
);
const FIELD_TO_BOOLEAN_FILTER = Object.fromEntries(
  LEGACY_FILTER_FIELDS
    .filter(([, , kind, restore = true]) => kind === 'boolean' && restore)
    .map(([key, field]) => [field, key]),
);

const IPO_PRESET_MONTHS = { '6m': 6, '1y': 12, '2y': 24, '3y': 36, '5y': 60 };

export function createEmptyExpression(requiredConditions = []) {
  return {
    expression_version: 1,
    required: {
      id: 'required',
      name: 'Always require',
      match: 'all',
      enabled: true,
      conditions: requiredConditions,
    },
    group_join: 'any',
    groups: [],
  };
}

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
      conditions.push({ kind: 'boolean', field, value: Boolean(filters[key]) });
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
    // Legacy passes_only means passing rating categories, not the persisted
    // passes_template boolean that happens to have a similar name.
    conditions.push({
      kind: 'categorical',
      field: 'rating',
      values: ['Strong Buy', 'Buy'],
      mode: 'include',
    });
  }
  if (filters.minVolume != null) {
    conditions.push(filters.symbolSearch?.trim()
      ? { kind: 'listing_discovery', min_volume: filters.minVolume }
      : {
          kind: 'range',
          field: 'volume',
          min: filters.minVolume,
          max: null,
        });
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

export function legacyFiltersToExpression(filters = {}, previousExpression = null, now = new Date()) {
  const base = previousExpression ? structuredClone(previousExpression) : createEmptyExpression();
  base.expression_version = 1;
  base.required = {
    id: 'required',
    name: 'Always require',
    match: 'all',
    enabled: true,
    conditions: legacyFiltersToConditions(filters, now),
  };
  base.group_join = base.group_join === 'all' ? 'all' : 'any';
  base.groups = Array.isArray(base.groups) ? base.groups : [];
  return base;
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
      if (condition.field === 'market_cap' && condition.max == null) result.minMarketCap = condition.min;
      if (condition.field === 'ipo_date' && condition.max == null) result.ipoAfter = condition.min;
    } else if (condition.kind === 'boolean') {
      const key = FIELD_TO_BOOLEAN_FILTER[condition.field];
      if (key) result[key] = condition.value;
    } else if (condition.kind === 'text' && ['symbol', 'listing_search'].includes(condition.field)) {
      result.symbolSearch = condition.pattern;
    } else if (condition.kind === 'categorical') {
      if (condition.field === 'rating') result.ratings = [...condition.values];
      if (condition.field === 'ibd_industry_group') {
        result.ibdIndustries = { values: [...condition.values], mode: condition.mode };
      }
      if (condition.field === 'gics_sector') {
        result.gicsSectors = { values: [...condition.values], mode: condition.mode };
      }
      if (condition.field === 'market') result.markets = [...condition.values];
      if (condition.field === 'se_pattern_primary') result.sePatternPrimary = [...condition.values];
    } else if (condition.kind === 'listing_discovery') {
      result.minVolume = condition.min_volume;
    }
  });
  return result;
}

export function canonicalizeExpression(expression) {
  const normalized = structuredClone(expression ?? createEmptyExpression());
  const normalizeGroup = (group) => ({
    id: group.id,
    name: String(group.name || '').trim(),
    match: group.match === 'any' ? 'any' : 'all',
    enabled: group.enabled !== false,
    conditions: (group.conditions || []).map((condition) => (
      condition.kind === 'categorical'
        ? { ...condition, values: [...new Set(condition.values || [])].sort() }
        : { ...condition }
    )),
  });
  return {
    expression_version: 1,
    required: normalizeGroup(normalized.required),
    group_join: normalized.group_join === 'all' ? 'all' : 'any',
    groups: (normalized.groups || []).map(normalizeGroup),
  };
}

export function stableExpressionKey(expression) {
  return JSON.stringify(canonicalizeExpression(expression));
}

export function buildScanQueryRequest(expression, {
  page = null,
  perPage = null,
  sortBy = 'composite_score',
  sortOrder = 'desc',
  includeSparklines = true,
  detailLevel = 'table',
} = {}) {
  return {
    ...canonicalizeExpression(expression),
    sort: { field: sortBy, order: sortOrder },
    page: page == null ? null : { number: page, size: perPage ?? 50 },
    options: { detail_level: detailLevel, include_sparklines: includeSparklines },
    passes_only: false,
  };
}

function rowValue(row, field) {
  if (field === 'price') return row.price ?? row.current_price;
  if (field === 'listing_search') return `${row.symbol || ''} ${row.company_name || ''}`.trim();
  if (field === 'price_change_1d') return row.price_change_1d ?? row.pct_day;
  if (field === 'perf_week') return row.perf_week ?? row.pct_week;
  if (field === 'perf_month') return row.perf_month ?? row.pct_month;
  if (field === 'market_cap_usd') {
    const currency = String(row.currency || '').toUpperCase();
    return row.market_cap_usd
      ?? ((currency === 'USD' || row.market === 'US') ? row.market_cap : null);
  }
  return row?.[field];
}

export function evaluateCondition(row, condition) {
  if (condition.kind === 'listing_discovery') {
    if (row?.scan_mode === 'listing_only') return true;
    return row?.volume != null && row.volume >= condition.min_volume;
  }
  const value = rowValue(row, condition.field);
  if (condition.kind === 'range') {
    if (value == null) return false;
    if (condition.min != null && value < condition.min) return false;
    if (condition.max != null && value > condition.max) return false;
    return true;
  }
  if (condition.kind === 'categorical') {
    const included = value != null && condition.values.includes(value);
    return condition.mode === 'exclude' ? value == null || !included : included;
  }
  if (condition.kind === 'boolean') {
    return typeof value === 'boolean' && value === condition.value;
  }
  if (condition.kind === 'text') {
    return value != null && String(value).toLowerCase().includes(condition.pattern.toLowerCase());
  }
  return false;
}

export function evaluateGroup(row, group) {
  if (group.enabled === false) return false;
  const matches = (group.conditions || []).map((condition) => evaluateCondition(row, condition));
  return group.match === 'any' ? matches.some(Boolean) : matches.every(Boolean);
}

export function evaluateExpression(row, expression) {
  if (!evaluateGroup(row, expression.required)) return false;
  const groups = (expression.groups || []).filter((group) => group.enabled !== false);
  if (!groups.length) return true;
  const matches = groups.map((group) => evaluateGroup(row, group));
  return expression.group_join === 'all' ? matches.every(Boolean) : matches.some(Boolean);
}

export function matchedGroupNames(row, expression) {
  return (expression.groups || [])
    .filter((group) => group.enabled !== false && evaluateGroup(row, group))
    .map((group) => ({ id: group.id, name: group.name }));
}

export function annotateExpressionMatches(rows, expression) {
  return rows
    .filter((row) => evaluateExpression(row, expression))
    .map((row) => ({ ...row, matched_groups: matchedGroupNames(row, expression) }));
}

export function conditionLabel(condition, catalog = []) {
  if (condition.kind === 'listing_discovery') {
    return `Listing discovery or dollar volume ≥ ${condition.min_volume}`;
  }
  const label = fieldMeta(condition.field, catalog, condition.kind).label;
  if (condition.kind === 'range') {
    if (condition.min != null && condition.max != null) return `${label} ${condition.min}–${condition.max}`;
    if (condition.min != null) return `${label} ≥ ${condition.min}`;
    return `${label} ≤ ${condition.max}`;
  }
  if (condition.kind === 'categorical') {
    const verb = condition.mode === 'exclude' ? 'excludes' : 'is';
    return `${label} ${verb} ${condition.values.join(', ')}`;
  }
  if (condition.kind === 'boolean') return `${label} is ${condition.value ? 'Yes' : 'No'}`;
  return `${label} contains “${condition.pattern}”`;
}

export function expressionSummary(expression) {
  const requiredCount = expression?.required?.conditions?.length ?? 0;
  const enabledGroups = (expression?.groups || []).filter((group) => group.enabled !== false);
  if (!enabledGroups.length) {
    return requiredCount ? `${requiredCount} always-required ${requiredCount === 1 ? 'rule' : 'rules'}` : 'No filters applied';
  }
  const join = expression.group_join === 'all' ? 'all' : 'any';
  return `${requiredCount} required · match ${join} of ${enabledGroups.length} named ${enabledGroups.length === 1 ? 'setup' : 'setups'}`;
}

export function newCondition(field = 'composite_score', catalog = []) {
  const meta = fieldMeta(field, catalog);
  if (meta.type === 'categorical') return { kind: 'categorical', field: meta.field, values: [], mode: 'include' };
  if (meta.type === 'boolean') return { kind: 'boolean', field: meta.field, value: true };
  if (meta.type === 'text') return { kind: 'text', field: meta.field, pattern: '' };
  return { kind: 'range', field: meta.field, min: null, max: null };
}

export function fieldMeta(field, catalog = [], fallbackType = 'range') {
  return catalog.find((item) => item.field === field) ?? {
    field,
    label: String(field || 'Unknown field').replaceAll('_', ' '),
    type: fallbackType,
    category: 'Other',
    sortable: false,
  };
}

function validateCondition(condition, catalog) {
  const errors = [];
  const label = fieldMeta(condition.field, catalog, condition.kind).label;
  if (condition.kind === 'range') {
    if (condition.min == null && condition.max == null) {
      errors.push(`${label} needs a minimum or maximum.`);
      return errors;
    }
    const isDate = condition.field === 'ipo_date';
    const normalize = (value) => {
      if (value == null) return null;
      if (isDate) {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return Number.NaN;
        const parsed = new Date(`${value}T00:00:00Z`);
        return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value
          ? value : Number.NaN;
      }
      return Number(value);
    };
    const minimum = normalize(condition.min);
    const maximum = normalize(condition.max);
    if ((minimum != null && (isDate ? Number.isNaN(minimum) : !Number.isFinite(minimum)))
      || (maximum != null && (isDate ? Number.isNaN(maximum) : !Number.isFinite(maximum)))) {
      errors.push(`${label} needs ${isDate ? 'valid ISO dates' : 'finite numeric values'}.`);
    } else if (minimum != null && maximum != null && minimum > maximum) {
      errors.push(`${label} minimum cannot exceed maximum.`);
    }
  } else if (condition.kind === 'categorical' && !condition.values?.length) {
    errors.push(`${label} needs at least one value.`);
  } else if (condition.kind === 'boolean' && typeof condition.value !== 'boolean') {
    errors.push(`${label} needs a Yes or No value.`);
  } else if (condition.kind === 'text' && !condition.pattern?.trim()) {
    errors.push(`${label} needs search text.`);
  } else if (
    condition.kind === 'listing_discovery'
    && (!Number.isFinite(Number(condition.min_volume)) || Number(condition.min_volume) <= 0)
  ) {
    errors.push('Listing discovery needs a positive finite dollar-volume value.');
  }
  return errors;
}

export function validateExpression(expression, catalog = []) {
  const errors = [];
  const groups = expression?.groups || [];
  if (groups.length > 8) errors.push('Use at most 8 setup groups.');
  const ids = new Set();
  const requiredConditions = expression?.required?.conditions ?? [];
  let total = requiredConditions.length;
  requiredConditions.forEach((condition) => {
    errors.push(...validateCondition(condition, catalog));
  });
  groups.forEach((group, index) => {
    total += group.conditions?.length ?? 0;
    if (!group.name?.trim()) errors.push(`Setup ${index + 1} needs a name.`);
    if (ids.has(group.id)) errors.push('Setup group IDs must be unique.');
    ids.add(group.id);
    if (group.enabled !== false && !group.conditions?.length) {
      errors.push(`${group.name || `Setup ${index + 1}`} needs at least one rule or must be disabled.`);
    }
    (group.conditions || []).forEach((condition) => {
      errors.push(...validateCondition(condition, catalog));
    });
  });
  if (total > 100) errors.push('Use at most 100 rules in one filter.');
  return [...new Set(errors)];
}

export { RANGE_FILTER_TO_FIELD, BOOLEAN_FILTER_TO_FIELD };
