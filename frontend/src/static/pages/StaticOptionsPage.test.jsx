import { screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import StaticOptionsPage from './StaticOptionsPage';
import { commandCenterFixture, optionsManifestFixture } from '../../features/options/__fixtures__/optionsResponses';

vi.mock('../StaticMarketContext', () => ({ useStaticMarket: () => ({ selectedMarket: 'US' }) }));
vi.mock('../dataClient', () => ({
  useStaticManifest: () => ({ data: { markets: { US: { pages: { options: { path: 'options/manifest.json' } } } } } }),
  resolveStaticMarketEntry: (manifest) => manifest.markets.US,
}));
vi.mock('../optionsClient', () => ({
  getStaticOptionsManifest: vi.fn(() => Promise.resolve(optionsManifestFixture)),
  staticOptionsCommandCenterQueryOptions: vi.fn(() => ({
    queryKey: ['static-options-command'],
    queryFn: () => Promise.resolve(commandCenterFixture),
    staleTime: Infinity,
    gcTime: Infinity,
  })),
}));
vi.mock('../../features/options/OptionsCommandCenterView', () => ({
  default: ({ data }) => <div>Static command run {data.run_id}</div>,
}));

describe('StaticOptionsPage', () => {
  beforeEach(() => vi.clearAllMocks());

  it('loads advertised static data and exposes no refresh action', async () => {
    renderWithProviders(<MemoryRouter><StaticOptionsPage /></MemoryRouter>);
    expect(await screen.findByText('Static command run 7')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /refresh/i })).not.toBeInTheDocument();
  });
});
