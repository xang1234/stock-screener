import { fireEvent, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { renderWithProviders } from '../test/renderWithProviders';
import { StaticMarketProvider } from './StaticMarketContext';
import StaticLayout from './StaticLayout';

const manifest = {
  default_market: 'US',
  supported_markets: ['US', 'HK'],
  unavailable_markets: ['IN'],
  markets: {
    US: { display_name: 'United States', features: {}, pages: {}, assets: {} },
    HK: { display_name: 'Hong Kong', features: {}, pages: {}, assets: {} },
  },
};

vi.mock('./dataClient', async (importOriginal) => ({
  ...(await importOriginal()),
  useStaticManifest: () => ({ data: manifest }),
}));

const renderLayout = (initialEntry = '/') => renderWithProviders(
  <MemoryRouter initialEntries={[initialEntry]}>
    <StaticMarketProvider supportedMarkets={manifest.supported_markets} defaultMarket="US">
      <StaticLayout><div /></StaticLayout>
    </StaticMarketProvider>
  </MemoryRouter>,
);

describe('StaticLayout market selector', () => {
  it('lists unavailable markets as disabled options', () => {
    renderLayout();
    fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Static market selector' }));
    const option = within(screen.getByRole('listbox')).getByRole('option', { name: /IN — unavailable/ });
    expect(option).toHaveAttribute('aria-disabled', 'true');
  });

  it('falls back to the default market when the URL names an unavailable market', () => {
    renderLayout('/?market=IN');
    expect(screen.getByRole('combobox', { name: 'Static market selector' })).toHaveTextContent('United States');
  });
});
