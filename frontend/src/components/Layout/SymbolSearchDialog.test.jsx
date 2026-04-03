import { useLocation } from 'react-router-dom';
import { Routes, Route, MemoryRouter } from 'react-router-dom';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import SymbolSearchDialog from './SymbolSearchDialog';

const searchStocks = vi.fn();

vi.mock('../../api/stocks', () => ({
  searchStocks: (...args) => searchStocks(...args),
}));

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderDialog() {
  return renderWithProviders(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="*" element={
          <>
            <LocationProbe />
            <SymbolSearchDialog open={true} onClose={vi.fn()} />
          </>
        }
        />
      </Routes>
    </MemoryRouter>
  );
}

describe('SymbolSearchDialog', () => {
  beforeEach(() => {
    searchStocks.mockResolvedValue([
      { symbol: 'NVDA', name: 'NVIDIA', sector: 'Technology', industry: 'Semiconductors' },
      { symbol: 'NVDS', name: 'Inverse NVIDIA', sector: 'ETF', industry: 'Leveraged' },
    ]);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('debounces search and navigates with keyboard selection', async () => {
    renderDialog();
    const user = userEvent.setup();

    await user.type(screen.getByLabelText('Search symbols'), 'nvd');

    expect(await screen.findByText(/NVDA · NVIDIA/)).toBeInTheDocument();
    expect(searchStocks).toHaveBeenCalledWith('nvd', 8);

    await user.keyboard('{ArrowDown}{Enter}');

    await waitFor(() => {
      expect(screen.getByTestId('location')).toHaveTextContent('/stock/NVDS');
    });
  });
});
