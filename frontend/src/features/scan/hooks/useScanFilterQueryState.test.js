import { describe, expect, it } from 'vitest';

import { createEmptyExpression } from '../filterExpressionModel';
import { buildDefaultScanFilters } from '../defaultFilters';
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

function succeed(state, data, scanId = 'scan-1') {
  return scanFilterQueryReducer(state, {
    type: 'request-succeeded',
    requestKey: state.requestedKey,
    scanId,
    data,
  });
}

describe('scan filter query state', () => {
  it('promotes the complete request and result as one applied snapshot', () => {
    const original = createState(withMinimumPrice(10));
    const originalData = { results: [{ symbol: 'NVDA' }] };
    const appliedOriginal = succeed(original, originalData);
    const pendingSort = scanFilterQueryReducer(appliedOriginal, {
      type: 'request-sort',
      sortBy: 'price',
      sortOrder: 'asc',
    });

    expect(pendingSort.requested.sortBy).toBe('price');
    expect(pendingSort.appliedSnapshot).toEqual({
      request: original.requested,
      requestKey: original.requestedKey,
      scanId: 'scan-1',
      data: originalData,
    });
  });

  it('ignores stale success and atomically applies the matching query', () => {
    const original = createState(withMinimumPrice(10));
    const pending = scanFilterQueryReducer(original, {
      type: 'request-expression',
      expression: withMinimumPrice(20),
    });
    const stale = scanFilterQueryReducer(pending, {
      type: 'request-succeeded',
      requestKey: original.requestedKey,
      scanId: 'scan-1',
      data: { results: [{ symbol: 'STALE' }] },
    });

    expect(stale.appliedSnapshot).toBeNull();

    const data = { results: [{ symbol: 'NVDA' }] };
    const applied = succeed(pending, data);
    expect(applied.appliedSnapshot.request).toEqual(pending.requested);
    expect(applied.appliedSnapshot.data).toBe(data);
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
    const nextFilters = { ...original.filterState.filters, symbolSearch: 'NVDA' };
    const edited = scanFilterQueryReducer(original, {
      type: 'edit-quick-filters',
      filters: nextFilters,
    });

    expect(edited.filterState.filters.symbolSearch).toBe('NVDA');
    expect(edited.requestedKey).toBe(original.requestedKey);

    const staleCommit = scanFilterQueryReducer(edited, {
      type: 'commit-quick-filters',
      filterKey: original.filterState.filterKey,
    });
    expect(staleCommit).toBe(edited);

    const committed = scanFilterQueryReducer(edited, {
      type: 'commit-quick-filters',
      filterKey: edited.filterState.filterKey,
    });
    expect(committed.requested.expression.required.conditions).toContainEqual(
      expect.objectContaining({ kind: 'text', pattern: 'NVDA' }),
    );
  });
});
