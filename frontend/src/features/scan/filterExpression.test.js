import { describe, expect, it } from 'vitest';

import { buildDefaultScanFilters } from './defaultFilters';
import {
  annotateExpressionMatches,
  buildScanQueryRequest,
  createEmptyExpression,
  evaluateCondition,
  evaluateExpression,
  expressionToLegacyFilters,
  legacyFiltersToExpression,
  stableExpressionKey,
} from './filterExpression';

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

describe('scan filter expressions', () => {
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
    const restored = expressionToLegacyFilters(expression, buildDefaultScanFilters());

    expect(restored.rsRating).toEqual({ min: 80, max: 99 });
    expect(restored.gicsSectors).toEqual({ values: ['Technology'], mode: 'include' });
    expect(restored.maAlignment).toBe(false);
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
});
