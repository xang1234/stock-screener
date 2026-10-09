import { act, render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { afterEach, describe, expect, it, vi } from 'vitest';

import StaticChartViewerModal from './StaticChartViewerModal';
import { useStaticChartIndex } from './chartClient';

vi.mock('../components/Charts/CandlestickChart', () => ({ default: () => <div /> }));
vi.mock('../components/Scan/StockMetricsSidebar', () => ({ default: () => <div /> }));

function OpenChart() {
  const indexQuery = useStaticChartIndex('markets/us/charts/index.json');
  return (
    <StaticChartViewerModal open onClose={() => {}} initialSymbol="NVDA" chartIndex={indexQuery.data} />
  );
}

// While the chart index is the previous generation's placeholder, its entries
// must not be fetched from the new generation's data root (#504).
describe('static charts across a data generation switch', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('fetches payloads only from the index of the current generation', async () => {
    const site = { generation: 'g1', requests: [], pending: [] };
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url.endsWith('static-data/manifest.json')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ generation: site.generation, data_root: `g/${site.generation}/` }),
        });
      }
      const path = url.replace(/^.*static-data\//, '');
      site.requests.push(path);
      const body = path.endsWith('index.json')
        ? { symbols: [{ symbol: 'NVDA', path: `markets/us/charts/NVDA-${site.generation}.json` }] }
        : { symbol: 'NVDA', bars: [] };
      return new Promise((resolve) => {
        site.pending.push(() => resolve({ ok: true, status: 200, json: async () => body }));
      });
    }));
    const release = () => site.pending.splice(0).forEach((resolve) => resolve());
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={createTheme()}>
          <OpenChart />
        </ThemeProvider>
      </QueryClientProvider>,
    );

    await waitFor(() => expect(site.requests).toEqual(['g/g1/markets/us/charts/index.json']));
    await act(async () => release());
    await waitFor(() => expect(site.requests).toContain('g/g1/markets/us/charts/NVDA-g1.json'));
    await act(async () => release());

    site.generation = 'g2';
    site.requests.length = 0;
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ['staticManifest'] });
    });
    await waitFor(() => expect(site.requests).toEqual(['g/g2/markets/us/charts/index.json']));
    // The old index's entry would point at a file g2 does not have.
    expect(site.requests).not.toContain('g/g2/markets/us/charts/NVDA-g1.json');

    await act(async () => release());
    await waitFor(() => expect(site.requests).toContain('g/g2/markets/us/charts/NVDA-g2.json'));
    expect(site.requests).not.toContain('g/g2/markets/us/charts/NVDA-g1.json');
  });
});
