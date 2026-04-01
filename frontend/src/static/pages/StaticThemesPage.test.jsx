import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';

import StaticThemesPage from './StaticThemesPage';

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
        <StaticThemesPage />
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('StaticThemesPage', () => {
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
              themes: {
                path: 'themes/index.json',
              },
            },
          }),
        };
      }

      if (path === 'themes/index.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            available: true,
            variants: {
              'technical:grouped': { available: false, path: 'themes/technical-grouped.json' },
              'technical:flat': { available: true, path: 'themes/technical-flat.json' },
              'fundamental:grouped': { available: false, path: 'themes/fundamental-grouped.json' },
              'fundamental:flat': { available: false, path: 'themes/fundamental-flat.json' },
            },
          }),
        };
      }

      if (path === 'themes/technical-flat.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            generated_at: '2026-04-01T00:00:00Z',
            payload: {
              emerging: { count: 1, themes: [{ theme: 'AI Infrastructure', mentions_7d: 18, velocity: 1.6 }] },
              pending_merge_count: 0,
              failed_items_count: { failed_count: 0 },
              rankings: {
                rankings: [{ id: 1, theme: 'AI Infrastructure', rank: 1, momentum_score: 94, mentions_7d: 18, num_constituents: 2 }],
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

  it('keeps the controls visible and falls forward to the nearest available variant', async () => {
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Themes')).toBeInTheDocument();
    });

    expect(screen.getByRole('button', { name: 'Technical' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Grouped' })).toBeInTheDocument();
    expect(
      screen.getByText('The selected theme view is unavailable in this export. Showing Technical / Flat instead.')
    ).toBeInTheDocument();
    expect(screen.getByText('Flat Rankings')).toBeInTheDocument();
    expect(screen.getAllByText('AI Infrastructure').length).toBeGreaterThan(0);
  });
});
