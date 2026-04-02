import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import StaticChartViewerModal from './StaticChartViewerModal';

const chartSpy = vi.fn();
const sidebarSpy = vi.fn();

vi.mock('../components/Charts/CandlestickChart', () => ({
  default: (props) => {
    chartSpy(props);
    return <div data-testid="static-candlestick-chart">{props.symbol}:{props.priceData?.length || 0}</div>;
  },
}));

vi.mock('../components/Scan/StockMetricsSidebar', () => ({
  default: (props) => {
    sidebarSpy(props);
    return (
      <div data-testid="static-stock-sidebar">
        {props.stockData?.symbol}:{props.fundamentals?.symbol}
      </div>
    );
  },
}));

const renderModal = (props) => {
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
        <StaticChartViewerModal {...props} />
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('StaticChartViewerModal', () => {
  beforeEach(() => {
    chartSpy.mockClear();
    sidebarSpy.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders exported bars and sidebar metadata without live API calls', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = String(url).split('/static-data/')[1];

      if (path === 'charts/NVDA.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            generated_at: '2026-04-03T20:10:00Z',
            symbol: 'NVDA',
            bars: [
              { date: '2026-04-01', open: 100, high: 105, low: 99, close: 104, volume: 1000000 },
              { date: '2026-04-02', open: 104, high: 107, low: 103, close: 106, volume: 1200000 },
            ],
            stock_data: {
              symbol: 'NVDA',
              company_name: 'NVIDIA Corporation',
              ibd_group_rank: 1,
              ibd_industry_group: 'Semiconductors',
              gics_sector: 'Technology',
              gics_industry: 'Semiconductors',
              adr_percent: 3.7,
              eps_rating: 94,
            },
            fundamentals: {
              symbol: 'NVDA',
              description: 'AI chip leader',
              pe_ratio: 45.2,
            },
          }),
        };
      }

      if (path === 'charts/MSFT.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            generated_at: '2026-04-03T20:10:00Z',
            symbol: 'MSFT',
            bars: [],
            stock_data: { symbol: 'MSFT', company_name: 'Microsoft Corporation' },
            fundamentals: { symbol: 'MSFT' },
          }),
        };
      }

      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    });

    renderModal({
      open: true,
      onClose: vi.fn(),
      initialSymbol: 'NVDA',
      chartIndex: {
        symbols: [
          { symbol: 'NVDA', rank: 1, path: 'charts/NVDA.json' },
          { symbol: 'MSFT', rank: 2, path: 'charts/MSFT.json' },
        ],
      },
    });

    expect(await screen.findByText('Stock 1 of 2', {}, { timeout: 10000 })).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByTestId('static-candlestick-chart')).toHaveTextContent('NVDA:2');
      expect(screen.getByTestId('static-stock-sidebar')).toHaveTextContent('NVDA:NVDA');
    });

    expect(chartSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        symbol: 'NVDA',
        priceData: expect.arrayContaining([
          expect.objectContaining({ date: '2026-04-01', close: 104 }),
        ]),
      })
    );
    expect(sidebarSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        stockData: expect.objectContaining({ symbol: 'NVDA', company_name: 'NVIDIA Corporation' }),
        fundamentals: expect.objectContaining({ symbol: 'NVDA', pe_ratio: 45.2 }),
      })
    );
    const requestedUrls = globalThis.fetch.mock.calls.map(([url]) => String(url));
    expect(requestedUrls.every((url) => url.includes('/static-data/') && !url.includes('/api'))).toBe(true);
  }, 10000);
});
