import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { queryScanResults } from '../../../api/scans';
import { buildDefaultScanFilters } from '../defaultFilters';
import { legacyFiltersToExpression } from '../legacyFilterExpression';
import { useScanResultsController } from './useScanResultsController';

vi.mock('../../../api/scans', () => ({
  queryScanResults: vi.fn(),
}));

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: Infinity },
    },
  });
}

describe('useScanResultsController', () => {
  beforeEach(() => {
    queryScanResults.mockReset();
  });

  it('falls back only to the exact query that was last displayed', async () => {
    const firstResults = { results: [{ symbol: 'FIRST' }], total: 1 };
    queryScanResults
      .mockResolvedValueOnce(firstResults)
      .mockRejectedValueOnce(new Error('next request failed'));
    const queryClient = createQueryClient();
    const wrapper = ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
    const defaults = buildDefaultScanFilters();
    const { result } = renderHook(() => useScanResultsController({
      currentScanId: 'scan-1',
      scanStatus: 'completed',
      initialFilters: defaults,
      initialExpression: legacyFiltersToExpression(defaults),
    }), { wrapper });

    await waitFor(() => {
      expect(result.current.displayedResultsData).toEqual(firstResults);
    });

    queryClient.setQueryData(['scanResultsQuery', 'scan-1', 'unrelated-request'], {
      data: { results: [{ symbol: 'WRONG' }], total: 1 },
      request: { expression: legacyFiltersToExpression(defaults) },
      requestKey: 'unrelated-request',
      scanId: 'scan-1',
    });
    act(() => result.current.requestSort('price', 'asc'));

    await waitFor(() => {
      expect(result.current.resultsError).toBeInstanceOf(Error);
    });
    expect(result.current.displayedResultsData).toEqual(firstResults);
  });
});
