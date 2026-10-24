import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';

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
    const { queryClient } = renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    queryClient.setQueryData(['userWatchlistSymbols', 1], { symbols: ['AAPL'] });

    expect(await screen.findByText('Leaders')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Leaders'));

    fireEvent.click(screen.getByRole('button', { name: 'Import' }));
    const importDialog = await screen.findByRole('dialog', { name: 'Import Symbols' });
    const importTextbox = within(importDialog).getByRole('textbox', { name: 'Symbols to import' });

    fireEvent.change(importTextbox, { target: { value: 'NVDA\nMSFT\nAAPL' } });
    fireEvent.click(within(importDialog).getByRole('button', { name: 'Import' }));

    await waitFor(() => {
      expect(api.importItems).toHaveBeenCalledWith(1, {
        content: 'NVDA\nMSFT\nAAPL',
        format: 'auto',
      });
    });
    await waitFor(() => {
      expect(queryClient.getQueryState(['userWatchlistSymbols', 1]).isInvalidated).toBe(true);
    });
  }, 20000);

  it('imports a TradingView export chosen as a file', async () => {
    const user = userEvent.setup();
    const exportText = '###Leaders,NASDAQ:NVDA,NYSE:ANET,###Watch,HKEX:700';
    renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );

    fireEvent.click(await screen.findByText('Leaders'));
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));
    const importDialog = await screen.findByRole('dialog', { name: 'Import Symbols' });

    await user.upload(
      within(importDialog).getByLabelText('Choose file'),
      new File([exportText], 'leaders.txt', { type: 'text/plain' })
    );

    await waitFor(() => {
      expect(within(importDialog).getByRole('textbox', { name: 'Symbols to import' })).toHaveValue(exportText);
    });
    fireEvent.click(within(importDialog).getByRole('button', { name: 'Import' }));

    await waitFor(() => {
      expect(api.importItems).toHaveBeenCalledWith(1, { content: exportText, format: 'auto' });
    });
  }, 20000);

  const openImportDialog = async () => {
    fireEvent.click(await screen.findByText('Leaders'));
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));
    return screen.findByRole('dialog', { name: 'Import Symbols' });
  };

  it('rejects a file over 1 MB without reading it', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    const importDialog = await openImportDialog();
    const file = new File(['x'.repeat(1_000_001)], 'huge.csv', { type: 'text/csv' });
    file.text = vi.fn();

    await user.upload(within(importDialog).getByLabelText('Choose file'), file);

    expect(await within(importDialog).findByText('huge.csv is over 1 MB.')).toBeInTheDocument();
    expect(file.text).not.toHaveBeenCalled();
    expect(within(importDialog).getByRole('textbox', { name: 'Symbols to import' })).toHaveValue('');
  }, 20000);

  it('reports a file that cannot be read', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    const importDialog = await openImportDialog();
    const file = new File(['NVDA'], 'locked.txt', { type: 'text/plain' });
    file.text = () => Promise.reject(new Error('NotReadableError'));

    await user.upload(within(importDialog).getByLabelText('Choose file'), file);

    expect(await within(importDialog).findByText('Could not read locked.txt.')).toBeInTheDocument();
  }, 20000);

  it('shows a readable message when the server rejects the import with a validation list', async () => {
    api.importItems.mockRejectedValue({
      response: {
        status: 422,
        data: {
          detail: [{
            type: 'string_too_long',
            loc: ['body', 'content'],
            msg: 'String should have at most 1000000 characters',
            input: 'NVDA,NVDA',
            ctx: { max_length: 1000000 },
          }],
        },
      },
    });
    renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    const importDialog = await openImportDialog();

    fireEvent.change(
      within(importDialog).getByRole('textbox', { name: 'Symbols to import' }),
      { target: { value: 'NVDA' } },
    );
    fireEvent.click(within(importDialog).getByRole('button', { name: 'Import' }));

    expect(await within(importDialog).findByRole('alert')).toHaveTextContent(
      'String should have at most 1000000 characters',
    );
  }, 20000);

  it('shows the watchlist size limit the server reports', async () => {
    const limit = 'A watchlist holds at most 2,000 symbols. This one has 1, and adding 2,000 more would make 2,001.';
    api.importItems.mockRejectedValue({ response: { status: 400, data: { detail: limit } } });
    renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    const importDialog = await openImportDialog();

    fireEvent.change(
      within(importDialog).getByRole('textbox', { name: 'Symbols to import' }),
      { target: { value: 'NVDA' } },
    );
    fireEvent.click(within(importDialog).getByRole('button', { name: 'Import' }));

    expect(await within(importDialog).findByRole('alert')).toHaveTextContent(limit);

    fireEvent.click(within(importDialog).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Import Symbols' })).toBeNull());
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));
    const reopened = await screen.findByRole('dialog', { name: 'Import Symbols' });
    expect(within(reopened).queryByRole('alert')).toBeNull();
  }, 20000);

  it('keeps an import error visible after the import dialog is closed while it was pending', async () => {
    let reject;
    api.importItems.mockReturnValue(new Promise((_, fail) => { reject = fail; }));
    renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    const importDialog = await openImportDialog();

    fireEvent.change(
      within(importDialog).getByRole('textbox', { name: 'Symbols to import' }),
      { target: { value: 'NVDA' } },
    );
    fireEvent.click(within(importDialog).getByRole('button', { name: 'Import' }));
    await waitFor(() => expect(api.importItems).toHaveBeenCalled());
    fireEvent.click(within(importDialog).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Import Symbols' })).toBeNull());
    reject({ response: { status: 400, data: { detail: 'Too many symbols.' } } });

    expect(await screen.findByRole('alert')).toHaveTextContent('Too many symbols.');
  }, 20000);

  it('blocks pasted text over the import limit before sending it', async () => {
    renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    const importDialog = await openImportDialog();

    fireEvent.change(
      within(importDialog).getByRole('textbox', { name: 'Symbols to import' }),
      { target: { value: 'N'.repeat(1_000_001) } },
    );

    expect(within(importDialog).getByText('This text is over 1,000,000 characters.')).toBeInTheDocument();
    expect(within(importDialog).getByRole('button', { name: 'Import' })).toBeDisabled();
    expect(api.importItems).not.toHaveBeenCalled();
  }, 20000);
});

describe('UserWatchlistManager item mutations', () => {
  const deferred = () => {
    let resolve;
    const promise = new Promise((done) => { resolve = done; });
    return { promise, resolve };
  };

  beforeEach(() => {
    Object.values(api).forEach((mockFn) => mockFn.mockReset());
    api.getWatchlists.mockResolvedValue({
      watchlists: [{ id: 7, name: 'Leaders' }, { id: 8, name: 'Laggards' }],
      total: 2,
    });
    api.getWatchlistData.mockImplementation(async (id) => ({
      id,
      name: id === 7 ? 'Leaders' : 'Laggards',
      items: id === 7 ? [{ id: 70, symbol: 'AAPL', position: 0 }] : [],
      price_change_bounds: {},
    }));
  });

  const renderWithSymbolCaches = () => {
    const { queryClient } = renderWithProviders(
      <UserWatchlistManager open={true} onClose={vi.fn()} onUpdate={vi.fn()} />
    );
    queryClient.setQueryData(['userWatchlistSymbols', 7], { symbols: ['AAPL'] });
    queryClient.setQueryData(['userWatchlistSymbols', 8], { symbols: [] });
    const invalidated = (id) => queryClient.getQueryState(['userWatchlistSymbols', id]).isInvalidated;
    return invalidated;
  };

  it('refreshes the watchlist an add targeted after the selection moves on', async () => {
    const add = deferred();
    api.addItem.mockReturnValue(add.promise);
    const invalidated = renderWithSymbolCaches();

    fireEvent.click(await screen.findByText('Leaders'));
    fireEvent.change(screen.getByPlaceholderText('Add symbol (e.g., NVDA)'), { target: { value: 'nvda' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));
    await waitFor(() => expect(api.addItem).toHaveBeenCalledWith(7, { symbol: 'NVDA' }));
    fireEvent.click(screen.getByText('Laggards'));
    add.resolve({ id: 71, symbol: 'NVDA' });

    await waitFor(() => expect(invalidated(7)).toBe(true));
    expect(invalidated(8)).toBe(false);
  }, 20000);

  it('refreshes the watchlist a removal targeted after the selection moves on', async () => {
    const remove = deferred();
    api.removeItem.mockReturnValue(remove.promise);
    const invalidated = renderWithSymbolCaches();

    fireEvent.click(await screen.findByText('Leaders'));
    fireEvent.click(await screen.findByRole('button', { name: 'Remove AAPL' }));
    await waitFor(() => expect(api.removeItem).toHaveBeenCalledWith(70));
    fireEvent.click(screen.getByText('Laggards'));
    remove.resolve({ status: 'deleted' });

    await waitFor(() => expect(invalidated(7)).toBe(true));
    expect(invalidated(8)).toBe(false);
  }, 20000);
});
