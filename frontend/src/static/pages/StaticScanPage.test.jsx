import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import StaticScanPage from './StaticScanPage';

const filterPanelSpy = vi.fn();
const resultsTableSpy = vi.fn();

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
        <button type="button" onClick={() => props.onPageChange(3)}>
          go-to-page-3
        </button>
        <button type="button" onClick={() => props.onSortChange('rating', 'asc')}>
          resort
        </button>
      </div>
    );
  },
}));

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
            chunks: [{ path: 'scan/chunks/chunk-0001.json', count: 1 }],
          }),
        };
      }

      if (path === 'scan/chunks/chunk-0001.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            rows: [{ symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5 }],
          }),
        };
      }

      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('normalizes exported filter options before rendering the filter panel', async () => {
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

  it('resets back to page 1 when the sort changes', async () => {
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
});
