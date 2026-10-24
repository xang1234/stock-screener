import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider, focusManager } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useWatchlistUniverseSelections } from './useWatchlistUniverseSelections';

const api = vi.hoisted(() => ({ getWatchlists: vi.fn(), getWatchlistSymbols: vi.fn() }));
vi.mock('../../../api/userWatchlists', () => api);

const renderSelections = (queryClient) => renderHook(
  () => useWatchlistUniverseSelections({
    universeSelections: { markets: [], scopesByMarket: {} },
    universeMarket: 'WATCHLIST',
    universeScope: '7',
    setUniverseScope: vi.fn(),
  }),
  {
    wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
  },
);

const scopeSymbols = (result) => result.current.scopesByMarket.WATCHLIST[0].universe_def?.symbols;

describe('useWatchlistUniverseSelections', () => {
  let queryClient;

  beforeEach(() => {
    api.getWatchlists.mockResolvedValue({ watchlists: [{ id: 7, name: 'Leaders' }] });
    api.getWatchlistSymbols.mockResolvedValue({ symbols: ['NVDA', 'MSFT'] });
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false, staleTime: 5 * 60 * 1000, refetchOnWindowFocus: false },
      },
    });
  });

  afterEach(() => {
    focusManager.setFocused(undefined);
    vi.clearAllMocks();
  });

  it('refetches symbols the page cached a moment ago', async () => {
    queryClient.setQueryData(['userWatchlistSymbols', 7], { symbols: ['NVDA'] });

    const { result } = renderSelections(queryClient);

    await waitFor(() => expect(scopeSymbols(result)).toEqual(['NVDA', 'MSFT']));
  });

  it('refetches symbols when the window regains focus', async () => {
    const { result } = renderSelections(queryClient);
    await waitFor(() => expect(scopeSymbols(result)).toEqual(['NVDA', 'MSFT']));
    api.getWatchlistSymbols.mockResolvedValue({ symbols: ['NVDA', 'MSFT', 'AAPL'] });

    act(() => {
      focusManager.setFocused(false);
      focusManager.setFocused(true);
    });

    await waitFor(() => expect(scopeSymbols(result)).toEqual(['NVDA', 'MSFT', 'AAPL']));
  });
});
