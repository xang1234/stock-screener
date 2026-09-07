import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ColorModeContext } from '../contexts/ColorModeContext';
import StaticAppShell from './StaticAppShell';

vi.mock('./pages/StaticHomePage', () => ({
  default: () => <div>Static daily home</div>,
}));

const staticRoot = dirname(fileURLToPath(import.meta.url));
const forbidden = /social(?:[_-]?signal)?|x[_-]?post|tweet|source[_-]?metrics/i;

function productionFiles(directory) {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) return productionFiles(path);
    return /\.[jt]sx?$/.test(name) && !/\.test\.[jt]sx?$/.test(name) ? [path] : [];
  });
}

describe('static Social isolation', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.location.hash = '';
  });

  it('keeps Social routes, imports, and request paths out of the static module graph', () => {
    const violations = productionFiles(staticRoot).flatMap((path) => {
      const source = readFileSync(path, 'utf8');
      return source.split('\n').flatMap((line, index) => (
        forbidden.test(line) ? [`${path}:${index + 1}:${line.trim()}`] : []
      ));
    });

    expect(violations).toEqual([]);
  });

  it('renders the static shell without Social navigation or API requests', async () => {
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        default_market: 'US',
        supported_markets: ['US'],
        markets: {
          US: {
            display_name: 'United States',
            pages: {},
            assets: {},
            features: {},
          },
        },
      }),
    }));
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={createTheme()}>
          <ColorModeContext.Provider value={{ toggleColorMode: vi.fn() }}>
            <StaticAppShell />
          </ColorModeContext.Provider>
        </ThemeProvider>
      </QueryClientProvider>
    );

    expect(await screen.findByText('Static daily home')).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: /social/i })).not.toBeInTheDocument();
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    expect(
      globalThis.fetch.mock.calls.every(([url]) => !forbidden.test(String(url)))
    ).toBe(true);
  });
});
