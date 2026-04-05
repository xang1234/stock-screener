import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import StaticBreadthPage from './StaticBreadthPage';

vi.mock('../../components/Charts/BreadthChart', () => ({
  default: () => <div data-testid="breadth-chart" />,
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
        <StaticBreadthPage />
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
});
