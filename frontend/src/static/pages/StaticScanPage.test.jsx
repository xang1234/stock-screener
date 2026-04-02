import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import StaticScanPage from './StaticScanPage';

const filterPanelSpy = vi.fn();
const resultsTableSpy = vi.fn();
const staticChartModalSpy = vi.fn();

vi.mock('../../components/Scan/FilterPanel', () => ({
  default: (props) => {
    filterPanelSpy(props);
    return <div data-testid="filter-panel" />;
  },
}));

vi.mock('../../components/Scan/ResultsTable', () => ({
  default: (props) => {
    resultsTableSpy(props);
    return (
      <div>
        <div data-testid="results-table-page">{props.page}</div>
        <div data-testid="results-table-total">{props.total}</div>
        <div data-testid="results-table-rows">{props.results.map((row) => row.symbol).join(',')}</div>
        <div data-testid="results-table-actions">{props.showActions ? 'actions-visible' : 'actions-hidden'}</div>
        <button type="button" onClick={() => props.onPageChange(3)}>
          go-to-page-3
        </button>
        <button type="button" onClick={() => props.onSortChange('rating', 'asc')}>
          resort
        </button>
        <button type="button" onClick={() => props.onOpenChart?.('NVDA')}>
          open-chart
        </button>
      </div>
    );
  },
}));

vi.mock('../StaticChartViewerModal', () => ({
  default: (props) => {
    staticChartModalSpy(props);
    return props.open ? <div data-testid="static-chart-modal">{props.initialSymbol}</div> : null;
  },
}));

const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={createTheme()}>
        <StaticScanPage />
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('StaticScanPage', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_STATIC_SITE', 'true');
    filterPanelSpy.mockClear();
    resultsTableSpy.mockClear();
    staticChartModalSpy.mockClear();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('renders the exported first page before background hydration completes', async () => {
    const chunkRequest = deferred();

    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            pages: {
              scan: {
                path: 'scan/manifest.json',
              },
            },
          }),
        };
      }

      if (path === 'scan/manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            generated_at: '2026-04-01T00:00:00Z',
            as_of_date: '2026-03-31',
            run_id: 9,
            sort: { field: 'composite_score', order: 'desc' },
            default_page_size: 50,
            rows_total: 2,
            filter_options: {
              ibd_industries: ['Semiconductors'],
              gics_sectors: ['Technology'],
              ratings: ['Strong Buy'],
            },
            initial_rows: [
              { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5 },
            ],
            chunks: [{ path: 'scan/chunks/chunk-0001.json', count: 2 }],
            charts: {
              path: 'charts/index.json',
              limit: 200,
              symbols_total: 1,
              available: true,
            },
          }),
        };
      }

      if (path === 'charts/index.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            symbols: [{ symbol: 'NVDA', rank: 1, path: 'charts/NVDA.json' }],
          }),
        };
      }

      if (path === 'scan/chunks/chunk-0001.json') {
        return {
          ok: true,
          status: 200,
          json: () => chunkRequest.promise,
        };
      }

      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    });

    renderPage();

    expect(await screen.findByText(/Loading full scan dataset: 1 \/ 2 rows/i)).toBeInTheDocument();
    expect(screen.getByTestId('results-table-rows')).toHaveTextContent('NVDA');
    expect(screen.queryByTestId('filter-panel')).not.toBeInTheDocument();

    await act(async () => {
      chunkRequest.resolve({
        rows: [
          { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5 },
          { symbol: 'MSFT', company_name: 'Microsoft Corporation', composite_score: 89.2 },
        ],
      });
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.queryByText(/Loading full scan dataset/i)).not.toBeInTheDocument();
      expect(screen.getByTestId('filter-panel')).toBeInTheDocument();
      expect(screen.getByTestId('results-table-total')).toHaveTextContent('2');
      expect(screen.getByTestId('results-table-actions')).toHaveTextContent('actions-visible');
    });
  });

  it('normalizes exported filter options before rendering the filter panel', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            pages: {
              scan: {
                path: 'scan/manifest.json',
              },
            },
          }),
        };
      }

      if (path === 'scan/manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            generated_at: '2026-04-01T00:00:00Z',
            as_of_date: '2026-03-31',
            run_id: 9,
            sort: { field: 'composite_score', order: 'desc' },
            default_page_size: 50,
            rows_total: 1,
            filter_options: {
              ibd_industries: ['Semiconductors'],
              gics_sectors: ['Technology'],
              ratings: ['Strong Buy'],
            },
            initial_rows: [
              { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5 },
            ],
            chunks: [],
            charts: {
              path: 'charts/index.json',
              limit: 200,
              symbols_total: 1,
              available: true,
            },
          }),
        };
      }

      if (path === 'charts/index.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            symbols: [{ symbol: 'NVDA', rank: 1, path: 'charts/NVDA.json' }],
          }),
        };
      }

      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    });

    renderPage();

    await waitFor(() => {
      expect(filterPanelSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          filterOptions: {
            ibdIndustries: ['Semiconductors'],
            gicsSectors: ['Technology'],
            ratings: ['Strong Buy'],
          },
        })
      );
    });
  });

  it('resets back to page 1 when the sort changes after hydration completes', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            pages: {
              scan: {
                path: 'scan/manifest.json',
              },
            },
          }),
        };
      }

      if (path === 'scan/manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            generated_at: '2026-04-01T00:00:00Z',
            as_of_date: '2026-03-31',
            run_id: 9,
            sort: { field: 'composite_score', order: 'desc' },
            default_page_size: 50,
            rows_total: 1,
            filter_options: {
              ibd_industries: ['Semiconductors'],
              gics_sectors: ['Technology'],
              ratings: ['Strong Buy'],
            },
            initial_rows: [
              { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5 },
            ],
            chunks: [],
            charts: {
              path: 'charts/index.json',
              limit: 200,
              symbols_total: 1,
              available: true,
            },
          }),
        };
      }

      if (path === 'charts/index.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            symbols: [{ symbol: 'NVDA', rank: 1, path: 'charts/NVDA.json' }],
          }),
        };
      }

      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    });

    renderPage();
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getByTestId('results-table-page')).toHaveTextContent('1');
    });

    await user.click(screen.getByRole('button', { name: 'go-to-page-3' }));

    await waitFor(() => {
      expect(screen.getByTestId('results-table-page')).toHaveTextContent('3');
    });

    await user.click(screen.getByRole('button', { name: 'resort' }));

    await waitFor(() => {
      expect(screen.getByTestId('results-table-page')).toHaveTextContent('1');
    });

    expect(resultsTableSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        page: 1,
        sortBy: 'rating',
        sortOrder: 'asc',
      })
    );
  });

  it('opens the static chart modal for exported chart symbols', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            pages: {
              scan: {
                path: 'scan/manifest.json',
              },
            },
          }),
        };
      }

      if (path === 'scan/manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            generated_at: '2026-04-01T00:00:00Z',
            as_of_date: '2026-03-31',
            run_id: 9,
            sort: { field: 'composite_score', order: 'desc' },
            default_page_size: 50,
            rows_total: 1,
            filter_options: {
              ibd_industries: ['Semiconductors'],
              gics_sectors: ['Technology'],
              ratings: ['Strong Buy'],
            },
            initial_rows: [
              { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5 },
            ],
            chunks: [],
            charts: {
              path: 'charts/index.json',
              limit: 200,
              symbols_total: 1,
              available: true,
            },
          }),
        };
      }

      if (path === 'charts/index.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            symbols: [{ symbol: 'NVDA', rank: 1, path: 'charts/NVDA.json' }],
          }),
        };
      }

      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    });

    renderPage();
    const user = userEvent.setup();

    expect(await screen.findByRole('heading', { name: 'Daily Scan' })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByTestId('results-table-actions')).toHaveTextContent('actions-visible');
    });

    await user.click(screen.getByRole('button', { name: 'open-chart' }));

    expect(await screen.findByTestId('static-chart-modal')).toHaveTextContent('NVDA');
  });

  it('passes the current sorted chart navigation order into the static modal', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            pages: {
              scan: {
                path: 'scan/manifest.json',
              },
            },
          }),
        };
      }

      if (path === 'scan/manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            generated_at: '2026-04-01T00:00:00Z',
            as_of_date: '2026-03-31',
            run_id: 9,
            sort: { field: 'composite_score', order: 'desc' },
            default_page_size: 50,
            rows_total: 2,
            filter_options: {
              ibd_industries: ['Semiconductors'],
              gics_sectors: ['Technology'],
              ratings: ['Strong Buy', 'Buy'],
            },
            initial_rows: [
              { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5, rating: 'Strong Buy' },
              { symbol: 'MSFT', company_name: 'Microsoft Corporation', composite_score: 89.2, rating: 'Buy' },
            ],
            chunks: [],
            charts: {
              path: 'charts/index.json',
              limit: 200,
              symbols_total: 2,
              available: true,
            },
          }),
        };
      }

      if (path === 'charts/index.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            symbols: [
              { symbol: 'NVDA', rank: 1, path: 'charts/NVDA.json' },
              { symbol: 'MSFT', rank: 2, path: 'charts/MSFT.json' },
            ],
          }),
        };
      }

      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    });

    renderPage();
    const user = userEvent.setup();

    expect(await screen.findByRole('heading', { name: 'Daily Scan' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'resort' }));
    await user.click(screen.getByRole('button', { name: 'open-chart' }));

    await waitFor(() => {
      expect(staticChartModalSpy).toHaveBeenLastCalledWith(
        expect.objectContaining({
          open: true,
          initialSymbol: 'NVDA',
          navigationSymbols: ['MSFT', 'NVDA'],
        })
      );
    });
  });
});
