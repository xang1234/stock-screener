import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, fireEvent } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import StaticGroupsPage from './StaticGroupsPage';

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
        <StaticGroupsPage />
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('StaticGroupsPage', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_STATIC_SITE', 'true');
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            pages: {
              groups: {
                path: 'groups.json',
              },
            },
          }),
        };
      }

      if (path === 'groups.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            available: true,
            payload: {
              movers_period: '1w',
              rankings: {
                date: '2026-03-31',
                rankings: [
                  {
                    industry_group: 'Semiconductors',
                    rank: 1,
                    avg_rs_rating: 92.5,
                    num_stocks: 14,
                    rank_change_1w: 2,
                    rank_change_1m: 4,
                    rank_change_3m: 7,
                  },
                ],
              },
              movers: {
                gainers: [{ industry_group: 'Semiconductors', rank: 1, rank_change_1w: 3 }],
                losers: [{ industry_group: 'Retail', rank: 197, rank_change_1w: -5 }],
              },
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
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('renders 1W movers and the 1W rank-change column', async () => {
    renderPage();

    expect(await screen.findByRole('heading', { name: 'US Group Rankings' })).toBeInTheDocument();
    expect(screen.getByText('Top Gainers (1W)')).toBeInTheDocument();
    expect(screen.getByText('Top Losers (1W)')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '1W' })).toBeInTheDocument();
    expect(screen.getAllByText('Semiconductors').length).toBeGreaterThan(0);
    expect(screen.getByText('+3')).toBeInTheDocument();
  });

  it('renders the RRG chart from the baked bundle when the toggle is selected', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];
      if (path === 'manifest.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            pages: { groups: { path: 'groups.json' } },
            assets: { groups_rrg: { path: 'groups_rrg.json' } },
          }),
        };
      }
      if (path === 'groups.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            available: true,
            payload: {
              movers_period: '1w',
              rankings: { date: '2026-03-31', rankings: [] },
              movers: { gainers: [], losers: [] },
            },
          }),
        };
      }
      if (path === 'groups_rrg.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            available: true,
            available_scopes: ['groups'],
            payload: {
              groups: {
                date: '2026-03-31',
                scope: 'groups',
                groups: [
                  {
                    industry_group: 'Semiconductors',
                    rank: 1,
                    num_stocks: 14,
                    avg_rs_rating: 92.5,
                    quadrant: 'Leading',
                    is_provisional: false,
                    current: { date: '2026-03-31', x: 108.3, y: 106.1 },
                    tail: [
                      { date: '2026-02-01', x: 104.0, y: 98.0 },
                      { date: '2026-03-31', x: 108.3, y: 106.1 },
                    ],
                  },
                ],
              },
              sectors: { date: '2026-03-31', scope: 'sectors', groups: [] },
            },
          }),
        };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    });

    renderPage();

    expect(await screen.findByRole('heading', { name: 'US Group Rankings' })).toBeInTheDocument();
    // Switch from the table view to the Relative Rotation Graph.
    fireEvent.click(screen.getByRole('button', { name: 'RRG' }));
    expect(await screen.findByText(/Relative Rotation Graph/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Sectors' })).not.toBeInTheDocument();
  });
});
