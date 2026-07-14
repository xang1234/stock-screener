import { describe, expect, it } from 'vitest';

import { createEmptyExpression, stableExpressionKey } from '../filterExpressionModel';
import { buildDefaultScanFilters } from '../defaultFilters';
import { selectQuickFilters } from '../filterState';
import {
  createScanFilterQueryState,
  scanFilterQueryReducer,
} from './useScanFilterQueryState';

const withMinimumPrice = (minimum) => createEmptyExpression([
  { kind: 'range', field: 'price', min: minimum, max: null },
]);
const createState = (expression) => createScanFilterQueryState({
  defaultFilters: buildDefaultScanFilters(),
  expression,
});

describe('scan filter query state', () => {
  it('updates requested query dimensions without owning server response data', () => {
    const original = createState(withMinimumPrice(10));
    const pendingSort = scanFilterQueryReducer(original, {
      type: 'request-sort',
      sortBy: 'price',
      sortOrder: 'asc',
    });

    expect(pendingSort.requested.sortBy).toBe('price');
    expect(pendingSort).not.toHaveProperty('appliedSnapshot');
  });

  it('applies a complete expression as one query transition', () => {
    const original = createState(withMinimumPrice(10));
    const pending = scanFilterQueryReducer(original, {
      type: 'request-expression',
      expression: withMinimumPrice(20),
    });

    expect(pending.requested.expression).toEqual(withMinimumPrice(20));
    expect(pending.requested.page).toBe(1);
  });

  it('resets pagination when page-size, sort, or the complete query changes', () => {
    const pageThree = scanFilterQueryReducer(
      createState(withMinimumPrice(10)),
      { type: 'request-page', page: 3 },
    );
    const resized = scanFilterQueryReducer(pageThree, {
      type: 'request-per-page',
      perPage: 100,
    });
    const replaced = scanFilterQueryReducer(pageThree, {
      type: 'request-query',
      query: { sortBy: 'price', sortOrder: 'asc' },
    });

    expect(resized.requested).toMatchObject({ page: 1, perPage: 100 });
    expect(replaced.requested).toMatchObject({ page: 1, sortBy: 'price', sortOrder: 'asc' });
  });

  it('keeps quick-filter drafts local until the matching debounced commit', () => {
    const original = createState(withMinimumPrice(10));
    const nextFilters = {
      ...selectQuickFilters(original.filterState),
      symbolSearch: 'NVDA',
    };
    const edited = scanFilterQueryReducer(original, {
      type: 'edit-quick-filters',
      filters: nextFilters,
    });

    expect(selectQuickFilters(edited.filterState).symbolSearch).toBe('NVDA');
    expect(edited.requestedKey).toBe(original.requestedKey);

    const staleCommit = scanFilterQueryReducer(edited, {
      type: 'commit-quick-filters',
      expressionKey: stableExpressionKey(original.filterState.draftExpression),
    });
    expect(staleCommit).toBe(edited);

    const committed = scanFilterQueryReducer(edited, {
      type: 'commit-quick-filters',
      expressionKey: stableExpressionKey(edited.filterState.draftExpression),
    });
    expect(committed.requested.expression.required.conditions).toContainEqual(
      expect.objectContaining({ kind: 'text', pattern: 'NVDA' }),
    );
  });

  it('preserves required conditions that the quick-filter grid cannot represent', () => {
    const expression = createEmptyExpression([
      {
        kind: 'categorical',
        field: 'rating',
        values: ['Sell'],
        mode: 'exclude',
      },
      { kind: 'range', field: 'volume', min: null, max: 5_000_000 },
    ]);
    expression.groups = [{
      id: 'breakout',
      name: 'Breakout',
      match: 'all',
      enabled: true,
      conditions: [{ kind: 'boolean', field: 'vcp_detected', value: true }],
    }];
    const original = createState(expression);
    const edited = scanFilterQueryReducer(original, {
      type: 'edit-quick-filters',
      filters: {
        ...selectQuickFilters(original.filterState),
        symbolSearch: 'NVDA',
      },
    });
    const committed = scanFilterQueryReducer(edited, {
      type: 'commit-quick-filters',
      expressionKey: stableExpressionKey(edited.filterState.draftExpression),
    });

    expect(committed.requested.expression.required.conditions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'categorical',
          field: 'rating',
          mode: 'exclude',
        }),
        expect.objectContaining({ kind: 'range', field: 'volume', max: 5_000_000 }),
        expect.objectContaining({ kind: 'text', pattern: 'NVDA' }),
      ]),
    );
    expect(committed.requested.expression.groups).toEqual(expression.groups);
  });
});
