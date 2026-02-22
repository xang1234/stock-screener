// Field shapes match backend/app/schemas/scanning.py ScanResultResponse
// and the se_explain / se_candidates structures from the Setup Engine scanner.

/**
 * Full payload — all scores present, pattern detected, 2 candidates,
 * passed/failed checks, key_levels populated.
 */
export const fullPayloadStock = {
  symbol: 'NVDA',
  se_setup_score: 82.5,
  se_quality_score: 75.3,
  se_readiness_score: 68.9,
  se_pattern_primary: 'three_weeks_tight',
  se_pattern_confidence: 85,
  se_setup_ready: true,
  se_pivot_price: 142.50,
  se_pivot_type: 'primary_pattern_selected',
  se_pivot_date: '2026-02-10',
  se_distance_to_pivot_pct: -2.3,
  se_timeframe: 'weekly',
  se_atr14_pct: 3.45,
  se_explain: {
    passed_checks: [
      'setup_score_ok',
      'quality_floor_ok',
      'readiness_floor_ok',
      'volume_sufficient',
      'rs_leadership_ok',
      'stage_ok',
    ],
    failed_checks: ['atr14_pct_exceeds_limit'],
    key_levels: {
      pivot: 142.50,
      stop_loss: 135.20,
      sma_50: 138.75,
    },
    invalidation_flags: [],
  },
  se_candidates: [
    {
      pattern: 'three_weeks_tight',
      confidence_pct: 85,
      timeframe: 'weekly',
      setup_score: 82.5,
      quality_score: 75.3,
      readiness_score: 68.9,
      pivot_price: 142.50,
      pivot_type: 'primary_pattern_selected',
      pivot_date: '2026-02-10',
      checks: {
        setup_score_ok: true,
        quality_floor_ok: true,
        volume_sufficient: true,
      },
      notes: ['Tight weekly closes within 1.5% range'],
    },
    {
      pattern: 'flat_base',
      confidence_pct: 62,
      timeframe: 'daily',
      setup_score: 71.0,
      quality_score: 68.2,
      readiness_score: 55.1,
      pivot_price: 145.00,
      pivot_type: 'primary_pattern_fallback_selected',
      pivot_date: '2026-02-08',
      checks: {
        setup_score_ok: true,
        quality_floor_ok: true,
        volume_sufficient: false,
      },
      notes: [],
    },
  ],
};

/**
 * Partial/degraded payload — some scores null, data_policy:degraded flag, 1 candidate.
 */
export const partialPayloadStock = {
  symbol: 'PLTR',
  se_setup_score: 55.0,
  se_quality_score: null,
  se_readiness_score: 48.2,
  se_pattern_primary: 'cup_with_handle',
  se_pattern_confidence: 60,
  se_setup_ready: false,
  se_pivot_price: 28.75,
  se_pivot_type: 'primary_pattern_selected',
  se_pivot_date: '2026-01-28',
  se_distance_to_pivot_pct: 5.1,
  se_timeframe: 'daily',
  se_atr14_pct: 4.12,
  se_explain: {
    passed_checks: ['setup_score_ok', 'stage_ok'],
    failed_checks: [],
    key_levels: { pivot: 28.75 },
    invalidation_flags: ['data_policy:degraded'],
  },
  se_candidates: [
    {
      pattern: 'cup_with_handle',
      confidence_pct: 60,
      timeframe: 'daily',
      setup_score: 55.0,
      quality_score: null,
      readiness_score: 48.2,
      pivot_price: 28.75,
      pivot_type: 'primary_pattern_selected',
      pivot_date: '2026-01-28',
      checks: { setup_score_ok: true },
      notes: [],
    },
  ],
};

/**
 * Insufficient data — all scores null, insufficient data flags, empty candidates.
 */
export const insufficientDataStock = {
  symbol: 'NEWIPO',
  se_setup_score: null,
  se_quality_score: null,
  se_readiness_score: null,
  se_pattern_primary: null,
  se_pattern_confidence: null,
  se_setup_ready: null,
  se_pivot_price: null,
  se_pivot_type: null,
  se_pivot_date: null,
  se_distance_to_pivot_pct: null,
  se_timeframe: null,
  se_atr14_pct: null,
  se_explain: {
    passed_checks: [],
    failed_checks: [],
    key_levels: {},
    invalidation_flags: ['data_policy:insufficient', 'insufficient_data'],
  },
  se_candidates: [],
};

/**
 * No explain — se_explain is null. Drawer shows "not available" message.
 */
export const noExplainStock = {
  symbol: 'MYSTERY',
  se_setup_score: 45.0,
  se_quality_score: 40.0,
  se_readiness_score: 35.0,
  se_pattern_primary: null,
  se_pattern_confidence: null,
  se_setup_ready: false,
  se_pivot_price: null,
  se_pivot_type: null,
  se_pivot_date: null,
  se_distance_to_pivot_pct: null,
  se_timeframe: null,
  se_atr14_pct: null,
  se_explain: null,
  se_candidates: null,
};

/**
 * Malformed explain — se_explain is an array instead of object.
 * Tests the SE-F4 type guard: typeof se_explain === 'object' && !Array.isArray(se_explain).
 */
export const malformedExplainStock = {
  symbol: 'BROKEN',
  se_setup_score: 30.0,
  se_quality_score: 25.0,
  se_readiness_score: 20.0,
  se_pattern_primary: null,
  se_pattern_confidence: null,
  se_setup_ready: false,
  se_pivot_price: null,
  se_pivot_type: null,
  se_pivot_date: null,
  se_distance_to_pivot_pct: null,
  se_timeframe: null,
  se_atr14_pct: null,
  se_explain: [1, 2, 3],
  se_candidates: null,
};

// ── Table row fixtures for ResultsTable tests ──────────────────────────
// These match the flat scan_results API shape (no se_explain/se_candidates).

/** Common non-SE fields required to render a ResultsTable row without errors. */
const baseRow = {
  composite_score: 85.2,
  rs_rating: 92,
  current_price: 195.40,
  stage: 2,
  rating: 'Leader',
  ma_alignment: true,
  vcp_detected: false,
  vcp_ready_for_breakout: false,
  passes_template: true,
  rs_sparkline_data: null,
  price_sparkline_data: null,
  minervini_score: null,
  canslim_score: null,
  ipo_score: null,
  custom_score: null,
  volume_breakthrough_score: null,
  vcp_score: null,
  vcp_pivot: null,
  rs_rating_1m: null,
  rs_rating_3m: null,
  rs_rating_12m: null,
  beta: null,
  beta_adj_rs: null,
  eps_rating: null,
  volume: null,
  market_cap: null,
  ipo_date: null,
  eps_growth_qq: null,
  sales_growth_qq: null,
  adr_percent: null,
  gics_sector: null,
  ibd_group_rank: null,
  price_change_1d: null,
  rs_trend: null,
  price_trend: null,
  ibd_industry_group: null,
};

/** All 7 SE table columns populated. */
export const fullSeRow = {
  ...baseRow,
  symbol: 'FULL',
  se_setup_score: 78.3,
  se_pattern_primary: 'cup_with_handle',
  se_distance_to_pivot_pct: -3.2,
  se_bb_width_pctile_252: 15,
  se_volume_vs_50d: 1.8,
  se_rs_line_new_high: true,
  se_pivot_price: 198.50,
};

/** All 7 SE columns null — each should render as '-'. */
export const nullSeRow = {
  ...baseRow,
  symbol: 'NULL',
  se_setup_score: null,
  se_pattern_primary: null,
  se_distance_to_pivot_pct: null,
  se_bb_width_pctile_252: null,
  se_volume_vs_50d: null,
  se_rs_line_new_high: null,
  se_pivot_price: null,
};

/** Partial SE data + false boolean. */
export const mixedSeRow = {
  ...baseRow,
  symbol: 'MIX',
  se_setup_score: 62.1,
  se_pattern_primary: null,
  se_distance_to_pivot_pct: 4.7,
  se_bb_width_pctile_252: null,
  se_volume_vs_50d: 2.3,
  se_rs_line_new_high: false,
  se_pivot_price: null,
};

/**
 * Operational flags stock — includes operational invalidation flags
 * (too_extended + low_liquidity) for severity-aware rendering tests.
 */
export const operationalFlagsStock = {
  ...fullPayloadStock,
  symbol: 'EXTENDED',
  se_distance_to_pivot_pct: 15.2,
  se_explain: {
    ...fullPayloadStock.se_explain,
    invalidation_flags: ['too_extended', 'low_liquidity', 'breaks_50d_support'],
  },
};
