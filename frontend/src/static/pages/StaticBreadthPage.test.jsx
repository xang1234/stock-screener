import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { MemoryRouter } from 'react-router-dom';

import StaticBreadthPage from './StaticBreadthPage';
import { StaticMarketProvider } from '../StaticMarketContext';

vi.mock('../../components/Charts/BreadthChart', () => ({
  default: ({ benchmarkLabel, spyData = [] }) => (
    <div data-testid="breadth-chart">
      {benchmarkLabel}:{spyData.length}
    </div>
  ),
}));

const renderPage = (initialEntry = '/breadth') => {
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
        <MemoryRouter initialEntries={[initialEntry]}>
          <StaticMarketProvider supportedMarkets={['US', 'HK']} defaultMarket="US">
            <StaticBreadthPage />
          </StaticMarketProvider>
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('StaticBreadthPage', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_STATIC_SITE', 'true');
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('renders an info alert when the exported breadth bundle is unavailable', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            pages: {
              breadth: {
                path: 'breadth.json',
              },
            },
          }),
        };
      }

      if (path === 'breadth.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            available: false,
            message: 'No breadth snapshot is available for static-site export date 2026-04-02.',
            payload: {},
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

    expect(
      await screen.findByText('No breadth snapshot is available for static-site export date 2026-04-02.')
    ).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Market Breadth' })).not.toBeInTheDocument();
  });

  it('passes non-US benchmark labels and overlays to the chart', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            default_market: 'US',
            supported_markets: ['US', 'HK'],
            markets: {
              HK: {
                display_name: 'Hong Kong',
                pages: {
                  breadth: {
                    path: 'markets/hk/breadth.json',
                  },
                },
              },
            },
            pages: {},
          }),
        };
      }

      if (path === 'markets/hk/breadth.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            available: true,
            generated_at: '2026-04-24T22:00:00Z',
            payload: {
              benchmark_symbol: '^HSI',
              benchmark_overlay: [{ date: '2026-04-24', close: 18400 }],
              current: {
                market: 'HK',
                date: '2026-04-24',
                stocks_up_4pct: 22,
                stocks_down_4pct: 8,
                ratio_10day: 2.4,
              },
              chart_data: [{ market: 'HK', date: '2026-04-24', stocks_up_4pct: 22, stocks_down_4pct: 8 }],
              history_90d: [],
            },
          }),
        };
      }

      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    });

    renderPage('/breadth?market=HK');

    expect(await screen.findByRole('heading', { name: 'Hong Kong Breadth' })).toBeInTheDocument();
    expect(screen.getByTestId('breadth-chart')).toHaveTextContent('^HSI:1');
  });
});
