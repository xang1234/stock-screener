import { act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { renderHookWithProviders } from '../../../test/renderWithProviders';
import { DEFAULT_FILTER_KEY } from '../constants';
import { useScanResultsPrefetch } from './useScanResultsPrefetch';

const priceHistory = vi.hoisted(() => ({
  fetchPriceHistory: vi.fn(),
  prefetchPriceHistoryBatch: vi.fn(),
}));
vi.mock('../../../api/priceHistory', async (importOriginal) => ({
  ...(await importOriginal()),
  ...priceHistory,
}));

const defaultQuery = { page: 1, perPage: 50, sortBy: 'composite_score', sortOrder: 'desc' };
const results = [{ symbol: 'NVDA' }, { symbol: 'MSFT' }, { symbol: null }];

const renderPrefetch = (props) => renderHookWithProviders(() => useScanResultsPrefetch({
  results,
  viewScanId: 'scan-1',
  bootstrappedScanId: null,
  displayedQuery: defaultQuery,
  stableFilterKey: DEFAULT_FILTER_KEY,
  ...props,
}));

describe('useScanResultsPrefetch', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    priceHistory.fetchPriceHistory.mockResolvedValue({ bars: [] });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('loads only the row the pointer rests on', async () => {
    const { result } = renderPrefetch();

    act(() => {
      result.current('NVDA');
      vi.advanceTimersByTime(100);
      result.current('MSFT');
      vi.advanceTimersByTime(150);
    });

    expect(priceHistory.fetchPriceHistory.mock.calls).toEqual([['MSFT', '6mo']]);
  });

  it('prefetches the visible symbols once the page is idle', () => {
    const { queryClient } = renderPrefetch();

    act(() => { vi.runAllTimers(); });

    expect(priceHistory.prefetchPriceHistoryBatch).toHaveBeenCalledWith(queryClient, ['NVDA', 'MSFT'], '6mo');
  });

  it('skips the default view of the bootstrapped scan, which arrives prefetched', () => {
    renderPrefetch({ bootstrappedScanId: 'scan-1' });

    act(() => { vi.runAllTimers(); });

    expect(priceHistory.prefetchPriceHistoryBatch).not.toHaveBeenCalled();
  });
});
