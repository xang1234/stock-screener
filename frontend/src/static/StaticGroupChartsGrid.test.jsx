import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import StaticGroupChartsGrid from './StaticGroupChartsGrid';

vi.mock('../components/Charts/CandlestickChart', () => ({
  default: (props) => (
    <div
      data-testid="static-candlestick-chart"
      data-symbol={props.symbol}
      data-interactive={String(Boolean(props.interactive))}
      data-hide-timeframe-toggle={String(Boolean(props.hideTimeframeToggle))}
    >
      {props.symbol}:{props.priceData?.length || 0}
    </div>
  ),
}));

const renderGrid = (props = {}) => {
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
        <StaticGroupChartsGrid
          symbols={['NVDA']}
          chartIndex={{ symbols: [{ symbol: 'NVDA', path: 'charts/NVDA.json' }] }}
          {...props}
        />
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('StaticGroupChartsGrid', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows a no-data message when a loaded chart payload has no bars array', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        symbol: 'NVDA',
        stock_data: { symbol: 'NVDA', company_name: 'NVIDIA Corporation' },
      }),
    });

    renderGrid();

    expect(await screen.findByText('No price data available for NVDA.')).toBeInTheDocument();
    expect(screen.queryByTestId('static-candlestick-chart')).not.toBeInTheDocument();
  });

  it('locks zoom and hides Daily/Weekly toggle until a card is clicked, and unlocks the clicked card', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (path) => {
      const sym = String(path).includes('AAPL') ? 'AAPL' : 'NVDA';
      return {
        ok: true,
        status: 200,
        json: async () => ({
          symbol: sym,
          stock_data: { symbol: sym, company_name: `${sym} Corp` },
          bars: [{ date: '2026-01-02', open: 1, high: 2, low: 0.5, close: 1.5, volume: 100 }],
          generated_at: '2026-01-02T00:00:00Z',
        }),
      };
    });

    renderGrid({
      symbols: ['NVDA', 'AAPL'],
      chartIndex: {
        symbols: [
          { symbol: 'NVDA', path: 'charts/NVDA.json' },
          { symbol: 'AAPL', path: 'charts/AAPL.json' },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getAllByTestId('static-candlestick-chart')).toHaveLength(2);
    });

    const findChart = (symbol) =>
      screen.getAllByTestId('static-candlestick-chart').find((el) => el.dataset.symbol === symbol);

    // Daily/Weekly toggle is always hidden in the static groups grid.
    screen.getAllByTestId('static-candlestick-chart').forEach((chart) => {
      expect(chart.dataset.hideTimeframeToggle).toBe('true');
      expect(chart.dataset.interactive).toBe('false');
    });

    const user = userEvent.setup();

    // Click the NVDA card → only NVDA becomes interactive.
    await user.click(findChart('NVDA'));
    expect(findChart('NVDA').dataset.interactive).toBe('true');
    expect(findChart('AAPL').dataset.interactive).toBe('false');

    // Click AAPL card → selection moves over.
    await user.click(findChart('AAPL'));
    expect(findChart('NVDA').dataset.interactive).toBe('false');
    expect(findChart('AAPL').dataset.interactive).toBe('true');

    // Click the wrapper Box that holds the click-away handler → deselects.
    const gridContainer = document.querySelector('.MuiGrid-container');
    expect(gridContainer).toBeTruthy();
    await user.click(gridContainer.parentElement);
    screen.getAllByTestId('static-candlestick-chart').forEach((chart) => {
      expect(chart.dataset.interactive).toBe('false');
    });
  });

  it('exposes button semantics on the chart cards and supports keyboard selection', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (path) => {
      const sym = String(path).includes('AAPL') ? 'AAPL' : 'NVDA';
      return {
        ok: true,
        status: 200,
        json: async () => ({
          symbol: sym,
          stock_data: { symbol: sym, company_name: `${sym} Corp` },
          bars: [{ date: '2026-01-02', open: 1, high: 2, low: 0.5, close: 1.5, volume: 100 }],
          generated_at: '2026-01-02T00:00:00Z',
        }),
      };
    });

    renderGrid({
      symbols: ['NVDA', 'AAPL'],
      chartIndex: {
        symbols: [
          { symbol: 'NVDA', path: 'charts/NVDA.json' },
          { symbol: 'AAPL', path: 'charts/AAPL.json' },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getAllByTestId('static-candlestick-chart')).toHaveLength(2);
    });

    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThanOrEqual(2);
    const nvdaButton = buttons.find((b) => /NVDA chart/.test(b.getAttribute('aria-label') || ''));
    const aaplButton = buttons.find((b) => /AAPL chart/.test(b.getAttribute('aria-label') || ''));
    expect(nvdaButton).toBeTruthy();
    expect(aaplButton).toBeTruthy();
    expect(nvdaButton).toHaveAttribute('aria-pressed', 'false');
    expect(nvdaButton).toHaveAttribute('tabindex', '0');

    const user = userEvent.setup();
    nvdaButton.focus();
    await user.keyboard('{Enter}');

    expect(nvdaButton).toHaveAttribute('aria-pressed', 'true');
    expect(aaplButton).toHaveAttribute('aria-pressed', 'false');

    aaplButton.focus();
    await user.keyboard(' ');

    expect(nvdaButton).toHaveAttribute('aria-pressed', 'false');
    expect(aaplButton).toHaveAttribute('aria-pressed', 'true');
  });
});
