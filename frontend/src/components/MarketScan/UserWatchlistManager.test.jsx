import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import UserWatchlistManager from './UserWatchlistManager';

const api = vi.hoisted(() => ({
  getWatchlists: vi.fn(),
  createWatchlist: vi.fn(),
  updateWatchlist: vi.fn(),
  deleteWatchlist: vi.fn(),
  getWatchlistData: vi.fn(),
  addItem: vi.fn(),
  removeItem: vi.fn(),
  reorderWatchlists: vi.fn(),
  reorderItems: vi.fn(),
  importItems: vi.fn(),
}));

vi.mock('../../api/userWatchlists', () => api);

describe('UserWatchlistManager import flow', () => {
  beforeEach(() => {
    Object.values(api).forEach((mockFn) => mockFn.mockReset());
    api.getWatchlists.mockResolvedValue({
      watchlists: [{ id: 1, name: 'Leaders' }],
      total: 1,
    });
    api.getWatchlistData.mockResolvedValue({
      id: 1,
      name: 'Leaders',
      items: [],
      price_change_bounds: {},
    });
    api.importItems.mockResolvedValue({
      requested_count: 3,
      added: ['NVDA', 'MSFT'],
      skipped_existing: ['AAPL'],
      invalid_symbols: [],
      added_items: [],
    });
  });

  it('opens the import dialog and submits pasted content', async () => {
    renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    const user = userEvent.setup();

    expect(await screen.findByText('Leaders')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Leaders' }));

    await user.click(screen.getByRole('button', { name: 'Import' }));
    const importDialog = await screen.findByRole('dialog', { name: 'Import Symbols' });
    const importTextbox = within(importDialog).getByRole('textbox');

    await user.type(importTextbox, 'NVDA\nMSFT\nAAPL');
    await user.click(within(importDialog).getByRole('button', { name: 'Import' }));

    await waitFor(() => {
      expect(api.importItems).toHaveBeenCalledWith(1, {
        content: 'NVDA\nMSFT\nAAPL',
        format: 'auto',
      });
    });
  });
});
