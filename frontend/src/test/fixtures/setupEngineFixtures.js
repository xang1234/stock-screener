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
