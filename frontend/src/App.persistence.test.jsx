import { QueryClient } from '@tanstack/react-query';
import { describe, expect, it } from 'vitest';

import { shouldDehydratePersistedQuery } from './appQueryPersistence';

const successfulQuery = (queryKey) => ({
  queryKey,
  state: { status: 'success' },
});

const failedQuery = (queryKey) => ({
  queryKey,
  state: { status: 'error' },
});

describe('app query persistence', () => {
  it('does not persist volatile runtime or auth state', () => {
    expect(shouldDehydratePersistedQuery(successfulQuery(['appCapabilities']))).toBe(false);
    expect(shouldDehydratePersistedQuery(successfulQuery(['runtimeActivity']))).toBe(false);
  });

  it('never restores a scannable watchlist symbol list from storage', () => {
    expect(shouldDehydratePersistedQuery(successfulQuery(['userWatchlistSymbols', 7]))).toBe(false);
    expect(shouldDehydratePersistedQuery(successfulQuery(['userWatchlists']))).toBe(true);
  });

  it('persists ordinary successful data queries only', () => {
    expect(shouldDehydratePersistedQuery(successfulQuery(['scanHistory', 'US']))).toBe(true);
    expect(shouldDehydratePersistedQuery(failedQuery(['scanHistory', 'US']))).toBe(false);
  });

  it('persists only the most recently loaded scan results page', () => {
    const queryClient = new QueryClient();
    const queryCache = queryClient.getQueryCache();
    queryClient.setQueryData(['scanResultsQuery', 'scan-1', 'page-1'], { rows: 1 }, { updatedAt: 1000 });
    queryClient.setQueryData(['scanResultsQuery', 'scan-1', 'page-2'], { rows: 2 }, { updatedAt: 2000 });
    queryClient.setQueryData(['scanHistory', 'US'], { scans: [] }, { updatedAt: 500 });

    const find = (queryKey) => queryCache.find({ queryKey, exact: true });

    expect(shouldDehydratePersistedQuery(find(['scanResultsQuery', 'scan-1', 'page-1']), queryCache)).toBe(false);
    expect(shouldDehydratePersistedQuery(find(['scanResultsQuery', 'scan-1', 'page-2']), queryCache)).toBe(true);
    expect(shouldDehydratePersistedQuery(find(['scanHistory', 'US']), queryCache)).toBe(true);
  });
});
