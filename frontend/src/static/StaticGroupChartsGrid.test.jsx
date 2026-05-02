import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import StaticGroupChartsGrid from './StaticGroupChartsGrid';

vi.mock('../components/Charts/CandlestickChart', () => ({
  default: (props) => (
    <div data-testid="static-candlestick-chart">
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
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        symbol: 'NVDA',
        stock_data: { symbol: 'NVDA', company_name: 'NVIDIA Corporation' },
      }),
    }));

    renderGrid();

    expect(await screen.findByText('No price data available for NVDA.')).toBeInTheDocument();
    expect(screen.queryByTestId('static-candlestick-chart')).not.toBeInTheDocument();
  });
});
