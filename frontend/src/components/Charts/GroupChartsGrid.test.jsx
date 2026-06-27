import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import GroupChartsGrid from './GroupChartsGrid';

vi.mock('./CandlestickChart', () => ({
  default: ({ symbol, priceData }) => (
    <div data-testid="group-candlestick-chart" data-symbol={symbol}>
      {symbol}:{priceData?.length || 0}
    </div>
  ),
}));

vi.mock('../../api/priceHistory', () => ({
  fetchPriceHistoryBatch: vi.fn(async (symbols) => ({
    data: Object.fromEntries(
      symbols.map((symbol) => [
        symbol,
        [{ date: '2026-06-26', open: 1, high: 2, low: 0.5, close: 1.5, volume: 100 }],
      ]),
    ),
    missing: [],
  })),
  priceHistoryKeys: {
    batch: (symbols, period = '6mo') => ['priceHistory', 'batch', period, symbols.join(',')],
  },
  PRICE_HISTORY_STALE_TIME: 300000,
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
        <GroupChartsGrid
          symbols={['NVDA', 'AAPL', 'MSFT', 'META']}
          {...props}
        />
      </ThemeProvider>
    </QueryClientProvider>,
  );
};

describe('GroupChartsGrid', () => {
  it('lays out chart cards as two columns on desktop widths', async () => {
    renderGrid();

    await waitFor(() => {
      expect(screen.getAllByTestId('group-candlestick-chart')).toHaveLength(4);
    });

    const chartGrid = screen.getByTestId('group-charts-grid');
    expect(chartGrid).toHaveStyle({
      display: 'grid',
      gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
    });
  });
});
