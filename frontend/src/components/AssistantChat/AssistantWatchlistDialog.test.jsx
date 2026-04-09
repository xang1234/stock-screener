import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderWithProviders } from '../../test/renderWithProviders';
import AssistantWatchlistDialog from './AssistantWatchlistDialog';

const previewWatchlistAdd = vi.fn();
const getWatchlists = vi.fn();
const bulkAddItems = vi.fn();

vi.mock('../../api/assistant', () => ({
  previewWatchlistAdd: (...args) => previewWatchlistAdd(...args),
}));

vi.mock('../../api/userWatchlists', () => ({
  getWatchlists: (...args) => getWatchlists(...args),
  bulkAddItems: (...args) => bulkAddItems(...args),
}));

describe('AssistantWatchlistDialog', () => {
  beforeEach(() => {
    previewWatchlistAdd.mockReset();
    getWatchlists.mockReset();
    bulkAddItems.mockReset();

    getWatchlists.mockResolvedValue({
      watchlists: [{ id: 7, name: 'Leaders' }],
    });
    previewWatchlistAdd.mockResolvedValue({
      watchlist: { id: 7, name: 'Leaders' },
      requested_symbols: ['NVDA', 'AVGO'],
      addable_symbols: ['NVDA'],
      existing_symbols: ['AVGO'],
      invalid_symbols: [],
      reason: null,
      summary: '1 symbol can be added to Leaders.',
    });
    bulkAddItems.mockResolvedValue({
      added: [{ symbol: 'NVDA' }],
    });
  });

  it('previews the diff and confirms the add through the existing watchlist API', async () => {
    const onClose = vi.fn();
    const { queryClient } = renderWithProviders(
      <AssistantWatchlistDialog
        open
        symbols={['NVDA', 'AVGO']}
        onClose={onClose}
      />,
    );
    const invalidateQueriesSpy = vi.spyOn(queryClient, 'invalidateQueries');

    await waitFor(() => {
      expect(getWatchlists).toHaveBeenCalledTimes(1);
      expect(previewWatchlistAdd).toHaveBeenCalledWith({
        watchlist: 'Leaders',
        symbols: ['NVDA', 'AVGO'],
      });
      expect(screen.getByText('1 symbol can be added to Leaders.')).toBeInTheDocument();
    });

    expect(screen.getByText('Addable: NVDA')).toBeInTheDocument();
    expect(screen.getByText('Already present: AVGO')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Confirm add' }));

    await waitFor(() => {
      expect(bulkAddItems).toHaveBeenCalledWith(7, ['NVDA']);
      expect(onClose).toHaveBeenCalledTimes(1);
    });

    expect(invalidateQueriesSpy).toHaveBeenCalledWith({ queryKey: ['userWatchlists'] });
    expect(invalidateQueriesSpy).toHaveBeenCalledWith({ queryKey: ['userWatchlistData', 7] });
  });
});
