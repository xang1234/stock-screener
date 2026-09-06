import { screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import StaticOptionsSymbolPage from './StaticOptionsSymbolPage';
import { optionsManifestFixture } from '../../features/options/__fixtures__/optionsResponses';

vi.mock('../StaticMarketContext', () => ({ useStaticMarket: () => ({ selectedMarket: 'US' }) }));
vi.mock('../dataClient', () => ({
  useStaticManifest: () => ({ data: { markets: { US: { pages: { options: { path: 'options/manifest.json' } } } } } }),
  resolveStaticMarketEntry: (manifest) => manifest.markets.US,
}));
vi.mock('../optionsClient', () => ({
  getStaticOptionsManifest: vi.fn(() => Promise.resolve(optionsManifestFixture)),
  staticOptionsSymbolQueryOptions: vi.fn(() => { throw new Error('Options symbol NVDA is not advertised'); }),
}));

describe('StaticOptionsSymbolPage', () => {
  it('does not derive a path for an unavailable deep link', async () => {
    renderWithProviders(
      <MemoryRouter initialEntries={['/options/NVDA']}>
        <Routes><Route path="/options/:symbol" element={<StaticOptionsSymbolPage />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole('heading', { name: /NVDA is not in the published options cohort/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to Command Center/i })).toHaveAttribute('href', '/options');
  });
});
