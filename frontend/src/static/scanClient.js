const RANGE_FILTER_TO_FIELD = {
  compositeScore: 'composite_score',
  minerviniScore: 'minervini_score',
  canslimScore: 'canslim_score',
  ipoScore: 'ipo_score',
  customScore: 'custom_score',
  volBreakthroughScore: 'volume_breakthrough_score',
  seSetupScore: 'se_setup_score',
  seDistanceToPivot: 'se_distance_to_pivot_pct',
  seBbSqueeze: 'se_bb_width_pctile_252',
  seVolumeVs50d: 'se_volume_vs_50d',
  rsRating: 'rs_rating',
  rs1m: 'rs_rating_1m',
  rs3m: 'rs_rating_3m',
  rs12m: 'rs_rating_12m',
  epsRating: 'eps_rating',
  price: 'current_price',
  adrPercent: 'adr_percent',
  epsGrowth: 'eps_growth_qq',
  salesGrowth: 'sales_growth_qq',
  vcpScore: 'vcp_score',
  vcpPivot: 'vcp_pivot',
  perfDay: 'price_change_1d',
  perfWeek: 'perf_week',
  perfMonth: 'perf_month',
  perf3m: 'perf_3m',
  perf6m: 'perf_6m',
  gapPercent: 'gap_percent',
  volumeSurge: 'volume_surge',
  ema10Distance: 'ema_10_distance',
  ema20Distance: 'ema_20_distance',
  ema50Distance: 'ema_50_distance',
  week52HighDistance: 'week_52_high_distance',
  week52LowDistance: 'week_52_low_distance',
  beta: 'beta',
  betaAdjRs: 'beta_adj_rs',
  pctDay: 'pct_day',
  pctWeek: 'pct_week',
  pctMonth: 'pct_month',
};

const BOOLEAN_FILTER_TO_FIELD = {
  seSetupReady: 'se_setup_ready',
  seRsLineNewHigh: 'se_rs_line_new_high',
  vcpDetected: 'vcp_detected',
  vcpReady: 'vcp_ready_for_breakout',
  maAlignment: 'ma_alignment',
  passesTemplate: 'passes_template',
};

const RATING_SORT_ORDER = {
  'Strong Buy': 5,
  Buy: 4,
  Watch: 3,
  Pass: 2,
  Error: 1,
  'Insufficient Data': 0,
};

const IPO_PRESET_MONTHS = { '6m': 6, '1y': 12, '2y': 24, '3y': 36, '5y': 60 };

const resolveIpoCutoff = (preset, now = new Date()) => {
  if (!preset) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(preset)) return preset;
  const months = IPO_PRESET_MONTHS[preset];
  if (months == null) return null;
  const cutoff = new Date(Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate(),
  ));
  cutoff.setUTCMonth(cutoff.getUTCMonth() - months);
  return cutoff.toISOString().slice(0, 10);
};

const isEmptyRange = (range) => !range || (range.min == null && range.max == null);

const valueMatchesRange = (value, range) => {
  if (isEmptyRange(range)) return true;
  if (value == null) return false;
  if (range.min != null && value < range.min) return false;
  if (range.max != null && value > range.max) return false;
  return true;
};

const matchesCategoricalFilter = (value, filter) => {
  const selectedValues = filter?.values || [];
  if (selectedValues.length === 0) return true;
  const included = selectedValues.includes(value);
  return filter.mode === 'exclude' ? !included : included;
};

const compareValues = (left, right) => {
  if (left == null && right == null) return 0;
  if (left == null) return 1;
  if (right == null) return -1;
  if (typeof left === 'string' || typeof right === 'string') {
    return String(left).localeCompare(String(right));
  }
  return Number(left) - Number(right);
};

const getScanModeSortPriority = (row) => {
  const scanMode = row?.scan_mode;
  if (!scanMode || scanMode === 'full') {
    return 0;
  }
  if (scanMode === 'ipo_weighted') {
    return 1;
  }
  if (scanMode === 'listing_only') {
    return 2;
  }
  return 3;
};

const getSortValue = (row, sortBy) => {
  if (sortBy === 'rating') {
    return RATING_SORT_ORDER[row.rating] ?? 0;
  }
  return row?.[sortBy];
};

export const filterStaticScanRows = (rows, filters) => {
  const ipoCutoff = resolveIpoCutoff(filters.ipoAfter);
  return rows.filter((row) => {
    const hasSymbolSearch = Boolean(filters.symbolSearch);
    if (hasSymbolSearch) {
      const needle = filters.symbolSearch.toLowerCase();
      const haystack = `${row.symbol || ''} ${row.company_name || ''}`.toLowerCase();
      if (!haystack.includes(needle)) {
        return false;
      }
    }

    if (filters.stage != null && row.stage !== filters.stage) {
      return false;
    }

    if (filters.ratings?.length && !filters.ratings.includes(row.rating)) {
      return false;
    }

    if (!matchesCategoricalFilter(row.ibd_industry_group, filters.ibdIndustries)) {
      return false;
    }

    if (!matchesCategoricalFilter(row.gics_sector, filters.gicsSectors)) {
      return false;
    }

    const bypassMinVolumeForListingSearch = hasSymbolSearch && row.scan_mode === 'listing_only';
    if (
      !bypassMinVolumeForListingSearch &&
      filters.minVolume != null &&
      (row.volume == null || row.volume < filters.minVolume)
    ) {
      return false;
    }

    if (filters.minMarketCap != null && (row.market_cap == null || row.market_cap < filters.minMarketCap)) {
      return false;
    }

    if (ipoCutoff && (!row.ipo_date || row.ipo_date < ipoCutoff)) {
      return false;
    }

    for (const [filterKey, fieldName] of Object.entries(RANGE_FILTER_TO_FIELD)) {
      if (!valueMatchesRange(row[fieldName], filters[filterKey])) {
        return false;
      }
    }

    for (const [filterKey, fieldName] of Object.entries(BOOLEAN_FILTER_TO_FIELD)) {
      if (filters[filterKey] != null && Boolean(row[fieldName]) !== filters[filterKey]) {
        return false;
      }
    }

    return true;
  });
};

export const sortStaticScanRows = (rows, sortBy, sortOrder = 'desc') => {
  const direction = sortOrder === 'asc' ? 1 : -1;
  return [...rows].sort((left, right) => {
    if (sortBy === 'composite_score' && sortOrder === 'desc') {
      const modeComparison = compareValues(
        getScanModeSortPriority(left),
        getScanModeSortPriority(right),
      );
      if (modeComparison !== 0) {
        return modeComparison;
      }
    }
    const leftValue = getSortValue(left, sortBy);
    const rightValue = getSortValue(right, sortBy);
    if (sortBy === 'composite_score' && sortOrder === 'desc') {
      if (leftValue == null && rightValue != null) {
        return 1;
      }
      if (leftValue != null && rightValue == null) {
        return -1;
      }
    }
    const comparison = compareValues(leftValue, rightValue);
    if (comparison !== 0) {
      return comparison * direction;
    }
    if (sortBy === 'composite_score') {
      const modeComparison = compareValues(
        getScanModeSortPriority(left),
        getScanModeSortPriority(right),
      );
      if (modeComparison !== 0) {
        return modeComparison;
      }
    }
    return compareValues(left.symbol, right.symbol);
  });
};

export const paginateStaticScanRows = (rows, page, perPage) => {
  const offset = (page - 1) * perPage;
  return rows.slice(offset, offset + perPage);
};
