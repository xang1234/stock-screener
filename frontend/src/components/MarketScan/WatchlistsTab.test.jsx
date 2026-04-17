import { screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import WatchlistsTab from './WatchlistsTab';
import { renderWithProviders } from '../../test/renderWithProviders';

const getWatchlists = vi.fn();
const getWatchlistData = vi.fn();
const getWatchlistStewardship = vi.fn();

vi.mock('../../api/userWatchlists', () => ({
  getWatchlists: (...args) => getWatchlists(...args),
  getWatchlistData: (...args) => getWatchlistData(...args),
  getWatchlistStewardship: (...args) => getWatchlistStewardship(...args),
}));

vi.mock('../../contexts/StrategyProfileContext', () => ({
  useStrategyProfile: () => ({
    activeProfile: 'default',
  }),
}));

vi.mock('./WatchlistTable', () => ({
  default: ({ watchlistData }) => (
    <div data-testid="watchlist-table">{watchlistData?.name ?? 'watchlist-table'}</div>
  ),
}));

vi.mock('./UserWatchlistManager', () => ({
  default: () => null,
}));

vi.mock('./WatchlistChartModal', () => ({
  default: () => null,
}));

function createDeferred() {
  let resolve;
  const promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe('WatchlistsTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getWatchlists.mockResolvedValue({
      watchlists: [{ id: 1, name: 'Leaders' }],
      total: 1,
    });
    getWatchlistStewardship.mockResolvedValue({
      items: [],
      summary_counts: {},
    });
  });

  it('waits for watchlist data before requesting stewardship context', async () => {
    const deferred = createDeferred();
    getWatchlistData.mockReturnValue(deferred.promise);

    renderWithProviders(<WatchlistsTab />);

    await waitFor(() => {
      expect(getWatchlists).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(getWatchlistData).toHaveBeenCalledWith(1);
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(getWatchlistStewardship).not.toHaveBeenCalled();

    deferred.resolve({
      id: 1,
      name: 'Leaders',
      description: null,
      color: null,
      items: [],
      price_change_bounds: {},
    });

    await waitFor(() => {
      expect(getWatchlistStewardship).toHaveBeenCalled();
    });
    expect(getWatchlistStewardship.mock.calls[0]?.slice(0, 2)).toEqual([1, 'default']);
    expect(await screen.findByTestId('watchlist-table')).toHaveTextContent('Leaders');
  });
});
