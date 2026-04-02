import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
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
              movers_period: '3m',
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
                gainers: [{ industry_group: 'Semiconductors', rank: 1 }],
                losers: [{ industry_group: 'Retail', rank: 197 }],
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

  it('renders 3M movers and the 3M rank-change column', async () => {
    renderPage();

    expect(await screen.findByRole('heading', { name: 'Industry Group Rankings' })).toBeInTheDocument();
    expect(screen.getByText('Top Gainers (3M)')).toBeInTheDocument();
    expect(screen.getByText('Top Losers (3M)')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: '3M' })).toBeInTheDocument();
    expect(screen.getAllByText('Semiconductors').length).toBeGreaterThan(0);
    expect(screen.getByText('7')).toBeInTheDocument();
  });
});
