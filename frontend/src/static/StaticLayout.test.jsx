import { act, fireEvent, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderWithProviders } from '../test/renderWithProviders';
import { STATIC_MARKET_STORAGE_KEY, StaticMarketProvider } from './StaticMarketContext';
import StaticLayout from './StaticLayout';
import { STATIC_GENERATION_EXPIRED_EVENT } from './dataClient';

const manifest = {
  default_market: 'US',
  generation: 'g1',
  data_root: 'g/g1/',
  supported_markets: ['US', 'HK'],
  unavailable_markets: ['IN'],
  markets: {
    US: { display_name: 'United States', features: {}, pages: {}, assets: {} },
    HK: { display_name: 'Hong Kong', features: {}, pages: {}, assets: {} },
  },
};

const refetch = vi.fn();

vi.mock('./dataClient', async (importOriginal) => ({
  ...(await importOriginal()),
  useStaticManifest: () => ({ data: manifest, refetch, isFetching: false }),
}));

const renderLayout = (initialEntry = '/') => renderWithProviders(
  <MemoryRouter initialEntries={[initialEntry]}>
    <StaticMarketProvider
      supportedMarkets={manifest.supported_markets}
      unavailableMarkets={manifest.unavailable_markets}
      defaultMarket="US"
    >
      <StaticLayout><div data-testid="page">page</div></StaticLayout>
    </StaticMarketProvider>
  </MemoryRouter>,
);

describe('StaticLayout market selector', () => {
  afterEach(() => {
    window.localStorage.clear();
    refetch.mockClear();
  });

  it('lists unavailable markets as disabled options', () => {
    renderLayout();
    fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Static market selector' }));
    const option = within(screen.getByRole('listbox')).getByRole('option', { name: /IN — unavailable/ });
    expect(option).toHaveAttribute('aria-disabled', 'true');
  });

  it('keeps an unavailable market from the URL selected and says so (#504)', () => {
    renderLayout('/?market=IN');

    expect(screen.getByRole('combobox', { name: 'Static market selector' })).toHaveTextContent('IN — unavailable');
    expect(screen.queryByTestId('page')).not.toBeInTheDocument();
    expect(screen.getByText(/IN data is not available in this publish/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(refetch).toHaveBeenCalled();
  });

  it('keeps an unavailable saved selection instead of switching markets', () => {
    window.localStorage.setItem(STATIC_MARKET_STORAGE_KEY, 'IN');
    renderLayout('/');

    expect(screen.getByRole('combobox', { name: 'Static market selector' })).toHaveTextContent('IN — unavailable');
    expect(screen.queryByTestId('page')).not.toBeInTheDocument();
  });

  it('offers a reload when the tab is left on a replaced data generation', async () => {
    renderLayout('/');
    expect(screen.queryByRole('button', { name: 'Reload' })).not.toBeInTheDocument();

    await act(async () => {
      window.dispatchEvent(new CustomEvent(STATIC_GENERATION_EXPIRED_EVENT, { detail: { dataRoot: 'g/g1/' } }));
    });

    expect(screen.getByText(/newer data has been published/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reload' })).toBeInTheDocument();
    expect(screen.getByTestId('page')).toBeInTheDocument();
  });
});
