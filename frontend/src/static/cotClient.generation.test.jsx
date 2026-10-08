import { act, render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';

import StaticCotSection from './components/StaticCotSection';
import { makeCotHistory, staticCotIndexFixture } from '../features/cot/__fixtures__/cotResponses';

// After a generation switch the index is the previous generation's placeholder;
// its publication id must not drive a request for the new history (#504).
describe('static COT across a data generation switch', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('requests the new history only after the new index has loaded', async () => {
    const site = { generation: 'g1', requests: [], pending: [] };
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url.endsWith('static-data/manifest.json')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            generation: site.generation,
            data_root: `g/${site.generation}/`,
            assets: { cot: { path: 'cot/index.json' } },
          }),
        });
      }
      site.requests.push(url.replace(/^.*static-data\//, ''));
      const body = url.endsWith('index.json')
        ? staticCotIndexFixture
        : makeCotHistory({ weekCount: 260, range: '5y' });
      return new Promise((resolve) => {
        site.pending.push(() => resolve({ ok: true, status: 200, json: async () => body }));
      });
    }));
    const release = () => site.pending.splice(0).forEach((resolve) => resolve());
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={queryClient}>
        <StaticCotSection manifest={{ assets: { cot: { path: 'cot/index.json' } } }} />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(site.requests).toEqual(['g/g1/cot/index.json']));
    await act(async () => release());
    await waitFor(() => expect(site.requests).toContain('g/g1/cot/sp-500.json'));
    await act(async () => release());

    site.generation = 'g2';
    site.requests.length = 0;
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ['staticManifest'] });
    });
    await waitFor(() => expect(site.requests).toEqual(['g/g2/cot/index.json']));
    // The g2 index is still in flight: no g2 history request yet.
    expect(site.requests).not.toContain('g/g2/cot/sp-500.json');

    await act(async () => release());
    await waitFor(() => expect(site.requests).toContain('g/g2/cot/sp-500.json'));
  });
});
