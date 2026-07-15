import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { buildDefaultScanFilters } from './defaultFilters';
import {
  annotateExpressionMatches,
  evaluateCondition,
  evaluateExpression,
} from './filterExpressionEvaluator';
import {
  buildScanQueryRequest,
  canonicalizeExpression,
  createEmptyExpression,
  stableExpressionKey,
} from './filterExpressionModel';
import { validateExpression } from './filterExpressionBuilder';
import {
  legacyFiltersToExpression,
} from './legacyFilterExpression';
import {
  expressionToQuickFilters,
  quickFiltersToConditions,
} from './quickFilterExpression';
import { EXPRESSION_LIMITS, fieldValueOptions } from './scanFilterFields';

function groupedExpression(join = 'any') {
  return {
    ...createEmptyExpression([
      { kind: 'range', field: 'price', min: 10, max: null },
    ]),
    group_join: join,
    groups: [
      {
        id: 'breakout',
        name: 'Breakout ready',
        match: 'all',
        enabled: true,
        conditions: [
          { kind: 'range', field: 'rs_rating', min: 90, max: null },
          { kind: 'boolean', field: 'vcp_detected', value: true },
        ],
      },
      {
        id: 'growth',
        name: 'Growth leader',
        match: 'any',
        enabled: true,
        conditions: [
          { kind: 'range', field: 'eps_growth_qq', min: 30, max: null },
          { kind: 'range', field: 'sales_growth_qq', min: 30, max: null },
        ],
      },
    ],
  };
}

const sharedTruthTableUrl = new URL(
  '../../../../contracts/scan_filter_truth_table.json',
  import.meta.url,
);
const sharedTruthTablePath = decodeURIComponent(sharedTruthTableUrl.pathname)
  .replace(/^\/@fs\//, '/');
const sharedTruthTable = JSON.parse(readFileSync(sharedTruthTablePath, 'utf8'));
const legacyCompatibilityUrl = new URL(
  '../../../../contracts/scan_filter_legacy_compatibility.json',
  import.meta.url,
);
const legacyCompatibilityPath = decodeURIComponent(legacyCompatibilityUrl.pathname)
  .replace(/^\/@fs\//, '/');
const legacyCompatibility = JSON.parse(readFileSync(legacyCompatibilityPath, 'utf8'));

describe('scan filter expressions', () => {
  it('canonicalizes malformed persisted collection shapes without crashing', () => {
    expect(canonicalizeExpression({ required: null, groups: [] }).required).toEqual(
      createEmptyExpression().required,
    );
    expect(canonicalizeExpression({
      required: { conditions: [{ kind: 'categorical', field: 'rating', values: {} }] },
      groups: {},
    })).toMatchObject({
      required: {
        conditions: [{ kind: 'categorical', field: 'rating', values: [] }],
      },
      groups: [],
    });
    expect(canonicalizeExpression({
      required: { conditions: [null] },
      groups: [null],
    })).toMatchObject({
      required: { conditions: [{}] },
      groups: [{ name: '', match: 'all', enabled: true, conditions: [] }],
    });
  });

  it('requires the base group and matches any named setup', () => {
    const row = {
      symbol: 'AAA',
      current_price: 20,
      rs_rating: 95,
      vcp_detected: true,
      eps_growth_qq: 10,
    };
    expect(evaluateExpression(row, groupedExpression())).toBe(true);
    expect(annotateExpressionMatches([row], groupedExpression())[0].matched_groups).toEqual([
      { id: 'breakout', name: 'Breakout ready' },
    ]);
    expect(evaluateExpression(row, groupedExpression('all'))).toBe(false);
  });

  it('treats missing boolean as unknown and missing categorical as passing exclusion', () => {
    expect(evaluateCondition({}, { kind: 'boolean', field: 'ma_alignment', value: false })).toBe(false);
    expect(evaluateCondition({}, {
      kind: 'categorical', field: 'rating', values: ['Pass'], mode: 'exclude',
    })).toBe(true);
  });

  it('round-trips quick filters through the always-required group', () => {
    const filters = buildDefaultScanFilters();
    filters.rsRating = { min: 80, max: 99 };
    filters.gicsSectors = { values: ['Technology'], mode: 'include' };
    filters.maAlignment = false;
    const expression = legacyFiltersToExpression(filters);
    const restored = expressionToQuickFilters(expression, buildDefaultScanFilters());

    expect(restored.rsRating).toEqual({ min: 80, max: 99 });
    expect(restored.gicsSectors).toEqual({ values: ['Technology'], mode: 'include' });
    expect(restored.maAlignment).toBe(false);
  });

  it('caps categorical quick filters at the shared request limit', () => {
    const values = Array.from(
      { length: EXPRESSION_LIMITS.maxCategoricalValues + 1 },
      (_, index) => `Industry ${index + 1}`,
    );

    expect(quickFiltersToConditions({
      ibdIndustries: { values, mode: 'include' },
    })).toContainEqual({
      kind: 'categorical',
      field: 'ibd_industry_group',
      values: values.slice(0, EXPRESSION_LIMITS.maxCategoricalValues),
      mode: 'include',
    });
  });

  it('restores an IPO chip preset from its exact current cutoff', () => {
    const now = new Date(Date.UTC(2026, 6, 15));
    const expression = legacyFiltersToExpression({ ipoAfter: '1y' }, now);

    expect(expression.required.conditions).toContainEqual({
      kind: 'range', field: 'ipo_date', min: '2025-07-15', max: null,
    });
    expect(expressionToQuickFilters(
      expression,
      buildDefaultScanFilters(),
      now,
    ).ipoAfter).toBe('1y');

    const absoluteDate = createEmptyExpression([{
      kind: 'range', field: 'ipo_date', min: '2024-01-02', max: null,
    }]);
    expect(expressionToQuickFilters(
      absoluteDate,
      buildDefaultScanFilters(),
      now,
    ).ipoAfter).toBe('2024-01-02');
  });

  it.each(legacyCompatibility.cases)(
    'preserves legacy category compatibility: $name',
    ({ filters, required_conditions: requiredConditions }) => {
      expect(legacyFiltersToExpression(filters).required.conditions)
        .toEqual(requiredConditions);
    },
  );

  it('rejects string booleans instead of silently treating them as true', () => {
    expect(() => legacyFiltersToExpression({ maAlignment: 'false' }))
      .toThrow('must be a boolean');
  });

  it('resolves fixed and runtime categorical options from shared field metadata', () => {
    expect(fieldValueOptions('market')).toContain('US');
    expect(fieldValueOptions('rating', { ratings: ['Strong Buy', 'Buy'] }))
      .toEqual(['Strong Buy', 'Buy']);
  });

  it('preserves legacy static performance aliases and company discovery search', () => {
    const expression = legacyFiltersToExpression({
      pctDay: { min: 5 },
      symbolSearch: 'nvidia',
    });
    const rows = [
      {
        symbol: 'NVDA', company_name: 'Nvidia Corporation', pct_day: 7, price_change_1d: 1,
      },
      { symbol: 'LOW', company_name: 'Nvidia Supplier', pct_day: 1 },
      { symbol: 'AMD', company_name: 'Advanced Micro Devices', pct_day: 8 },
    ];

    expect(rows.filter((row) => evaluateExpression(row, expression)).map((row) => row.symbol))
      .toEqual(['NVDA']);
  });

  it('bypasses the volume floor only for listing-only discovery rows', () => {
    const expression = legacyFiltersToExpression({
      symbolSearch: 'new',
      minVolume: 1_000_000,
    });
    const rows = [
      { symbol: 'NEW', scan_mode: 'listing_only', volume: null },
      { symbol: 'NEWLOW', scan_mode: 'full', volume: 100 },
      { symbol: 'NEWHIGH', scan_mode: 'full', volume: 2_000_000 },
    ];

    expect(rows.filter((row) => evaluateExpression(row, expression)).map((row) => row.symbol))
      .toEqual(['NEW', 'NEWHIGH']);
    expect(expression.required.conditions).toContainEqual(
      { kind: 'range', field: 'listing_aware_volume', min: 1_000_000, max: null },
    );
    expect(evaluateCondition(
      rows[0],
      { kind: 'range', field: 'listing_aware_volume', min: null, max: 5_000_000 },
    )).toBe(false);
  });

  it('maps the legacy passes toggle to passing ratings', () => {
    const enabled = legacyFiltersToExpression({ passesTemplate: true });
    const disabled = legacyFiltersToExpression({ passesTemplate: false });
    const rows = [
      { symbol: 'BUY', rating: 'Buy', passes_template: false },
      { symbol: 'WATCH', rating: 'Watch', passes_template: true },
    ];

    expect(rows.filter((row) => evaluateExpression(row, enabled)).map((row) => row.symbol))
      .toEqual(['BUY']);
    expect(rows.filter((row) => evaluateExpression(row, disabled)).map((row) => row.symbol))
      .toEqual(['BUY', 'WATCH']);
    expect(expressionToQuickFilters(enabled, buildDefaultScanFilters()).passesTemplate)
      .toBe(true);
  });

  it('keeps explicit ratings and the passing-rating alias as separate quick-filter owners', () => {
    const expression = createEmptyExpression([
      { kind: 'categorical', field: 'rating', values: ['Watch'], mode: 'include' },
      { kind: 'categorical', field: 'rating', values: ['Strong Buy', 'Buy'], mode: 'include' },
    ]);

    expect(expressionToQuickFilters(expression, buildDefaultScanFilters())).toMatchObject({
      ratings: ['Watch'],
      passesTemplate: true,
    });
  });

  it('builds a page-independent stable expression key and versioned request', () => {
    const expression = groupedExpression();
    const first = buildScanQueryRequest(expression, { page: 1, perPage: 50 });
    const second = buildScanQueryRequest(expression, { page: 2, perPage: 50 });

    expect(stableExpressionKey(expression)).toBe(stableExpressionKey(expression));
    expect(first.expression_version).toBe(1);
    expect(first.page).toEqual({ number: 1, size: 50 });
    expect(second.page.number).toBe(2);
  });

  it('rejects invalid ranges in required rules and named setups before querying', () => {
    const expression = groupedExpression();
    expression.required.conditions = [
      { kind: 'range', field: 'price', min: 100, max: 10 },
    ];
    expression.groups[0].conditions = [
      { kind: 'range', field: 'rs_rating', min: Number.POSITIVE_INFINITY, max: null },
    ];

    expect(validateExpression(expression)).toEqual(expect.arrayContaining([
      'Price minimum cannot exceed maximum.',
      'RS rating needs finite numeric values.',
    ]));
  });

  it('rejects named setups above the shared per-group rule limit', () => {
    const expression = groupedExpression();
    expression.groups[0].conditions = Array.from(
      { length: EXPRESSION_LIMITS.maxGroupConditions + 1 },
      () => ({ kind: 'range', field: 'price', min: 10, max: null }),
    );

    expect(validateExpression(expression)).toContain(
      `Breakout ready can contain at most ${EXPRESSION_LIMITS.maxGroupConditions} rules.`,
    );
  });

  it('rejects categorical rules above the shared value limit', () => {
    const expression = groupedExpression();
    expression.groups[0].conditions = [{
      kind: 'categorical',
      field: 'ibd_industry_group',
      values: Array.from(
        { length: EXPRESSION_LIMITS.maxCategoricalValues + 1 },
        (_, index) => `Industry ${index + 1}`,
      ),
      mode: 'include',
    }];

    expect(validateExpression(expression)).toContain(
      `IBD industry allows at most ${EXPRESSION_LIMITS.maxCategoricalValues} values.`,
    );
  });

  it('rejects calendar dates that JavaScript would otherwise roll forward', () => {
    const expression = createEmptyExpression([
      { kind: 'range', field: 'ipo_date', min: '2026-02-31', max: null },
    ]);

    expect(validateExpression(expression)).toContain('ipo date needs valid ISO dates.');
  });

  it('matches the shared browser and backend truth table', () => {
    sharedTruthTable.rows.forEach((item) => {
      expect(evaluateExpression(item.row, sharedTruthTable.expression)).toBe(item.matches);
      if (item.matches) {
        expect(annotateExpressionMatches([item.row], sharedTruthTable.expression)[0]
          .matched_groups.map((group) => group.id)).toEqual(item.matched_groups);
      }
    });
  });
});
