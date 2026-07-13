import { describe, expect, it } from 'vitest';

import { createEmptyExpression, stableExpressionKey } from '../filterExpression';
import {
  createScanFilterQueryState,
  scanFilterQueryReducer,
} from './useScanFilterQueryState';

const withMinimumPrice = (minimum) => createEmptyExpression([
  { kind: 'range', field: 'price', min: minimum, max: null },
]);

describe('scan filter query state', () => {
  it('promotes a requested expression only after its own request succeeds', () => {
    const original = withMinimumPrice(10);
    const requested = withMinimumPrice(20);
    const initial = createScanFilterQueryState(original);
    const pending = scanFilterQueryReducer(initial, {
      type: 'request-expression',
      expression: requested,
    });

    expect(pending.appliedExpression).toEqual(original);
    expect(scanFilterQueryReducer(pending, {
      type: 'request-succeeded',
      key: stableExpressionKey(original),
    }).appliedExpression).toEqual(original);

    const applied = scanFilterQueryReducer(pending, {
      type: 'request-succeeded',
      key: stableExpressionKey(requested),
      data: { results: [{ symbol: 'NVDA' }] },
      scanId: 'scan-1',
    });
    expect(applied.appliedExpression).toEqual(requested);
    expect(applied.appliedResultsData.results[0].symbol).toBe('NVDA');
  });
});
