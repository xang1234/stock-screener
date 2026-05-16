import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, fireEvent } from '@testing-library/react';
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

  it('renders the By Group tab with drill-down stock list when attribution data is present', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            default_market: 'US',
            supported_markets: ['US'],
            markets: {
              US: {
                display_name: 'United States',
                pages: { breadth: { path: 'markets/us/breadth.json' } },
              },
            },
            pages: {},
          }),
        };
      }

      if (path === 'markets/us/breadth.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            available: true,
            generated_at: '2026-05-15T22:00:00Z',
            payload: {
              benchmark_symbol: 'SPY',
              benchmark_overlay: [],
              current: { market: 'US', date: '2026-05-15', stocks_up_4pct: 4, stocks_down_4pct: 1 },
              chart_data: [],
              history_90d: [],
              group_attribution: {
                available: true,
                market: 'US',
                threshold_pct: 4.0,
                lookback_days: 10,
                latest_date: '2026-05-15',
                history: [
                  {
                    date: '2026-05-15',
                    stocks_up_4pct: 4,
                    stocks_down_4pct: 1,
                    groups: [
                      {
                        group: 'Computer Software-Database',
                        up_count: 2,
                        down_count: 0,
                        net: 2,
                        up_stocks: [
                          { symbol: 'PLTR', name: 'Palantir', pct_change: 7.2, close: 28.5 },
                          { symbol: 'AAPL', name: 'Apple', pct_change: 4.5, close: 215.0 },
                        ],
                        down_stocks: [],
                      },
                      {
                        group: 'No Group',
                        up_count: 1,
                        down_count: 1,
                        net: 0,
                        up_stocks: [
                          { symbol: 'NEWCO', name: 'New Co', pct_change: 5.1, close: 12.4 },
                        ],
                        down_stocks: [
                          { symbol: 'OLDCO', name: 'Old Co', pct_change: -4.3, close: 7.2 },
                        ],
                      },
                    ],
                  },
                ],
              },
            },
          }),
        };
      }

      return { ok: false, status: 404, json: async () => ({}) };
    });

    renderPage();

    expect(await screen.findByRole('heading', { name: 'United States Breadth' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: /by group/i }));

    expect(await screen.findByText('Computer Software-Database')).toBeInTheDocument();
    expect(screen.getByText('No Group')).toBeInTheDocument();

    // Drill-down: clicking the group row should reveal the stock list.
    fireEvent.click(screen.getByText('Computer Software-Database'));
    expect(await screen.findByText('PLTR')).toBeInTheDocument();
    expect(screen.getByText('AAPL')).toBeInTheDocument();
  });

  it('shows an info message inside the By Group tab when attribution is unavailable', async () => {
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
                pages: { breadth: { path: 'markets/hk/breadth.json' } },
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
            generated_at: '2026-05-15T22:00:00Z',
            payload: {
              benchmark_symbol: '^HSI',
              benchmark_overlay: [],
              current: { market: 'HK', date: '2026-05-15', stocks_up_4pct: 6, stocks_down_4pct: 2 },
              chart_data: [],
              history_90d: [],
              group_attribution: {
                available: false,
                reason: 'Group attribution is not yet supported for market HK.',
              },
            },
          }),
        };
      }

      return { ok: false, status: 404, json: async () => ({}) };
    });

    renderPage('/breadth?market=HK');

    expect(await screen.findByRole('heading', { name: 'Hong Kong Breadth' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: /by group/i }));
    expect(
      await screen.findByText('Group attribution is not yet supported for market HK.')
    ).toBeInTheDocument();
  });
});
