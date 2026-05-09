import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import userEvent from '@testing-library/user-event';

import BreadthPage from './BreadthPage';
import * as breadthApi from '../api/breadth';
import * as stocksApi from '../api/stocks';
import { renderWithProviders } from '../test/renderWithProviders';

const runtimeState = {
  runtimeReady: true,
  uiSnapshots: { breadth: false },
  primaryMarket: 'HK',
  enabledMarkets: ['US', 'HK'],
  supportedMarkets: ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA'],
};

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../components/Charts/BreadthChart', () => ({
  default: ({ breadthData, benchmarkLabel, spyData, error }) => (
    <div data-testid="breadth-chart" data-error={error?.message || ''}>
      {benchmarkLabel}:{breadthData?.length ?? 0}:{spyData?.length ?? 0}
    </div>
  ),
}));

vi.mock('../api/breadth', () => ({
  getBreadthBootstrap: vi.fn(),
  getCurrentBreadth: vi.fn(),
  getHistoricalBreadth: vi.fn(),
  getBreadthSummary: vi.fn(),
}));

vi.mock('../api/stocks', () => ({
  getPriceHistory: vi.fn(),
}));

function breadthRow(market = 'HK') {
  return {
    market,
    date: '2026-04-24',
    stocks_up_4pct: market === 'HK' ? 22 : 10,
    stocks_down_4pct: market === 'HK' ? 8 : 4,
    ratio_5day: 2.75,
    ratio_10day: 2.5,
    stocks_up_25pct_quarter: 30,
    stocks_down_25pct_quarter: 12,
    stocks_up_25pct_month: 24,
    stocks_down_25pct_month: 10,
    stocks_up_50pct_month: 6,
    stocks_down_50pct_month: 2,
    stocks_up_13pct_34days: 18,
    stocks_down_13pct_34days: 7,
    total_stocks_scanned: 30,
    calculation_duration_seconds: 1.25,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  runtimeState.runtimeReady = true;
  runtimeState.uiSnapshots = { breadth: false };
  runtimeState.primaryMarket = 'HK';
  runtimeState.enabledMarkets = ['US', 'HK'];
  runtimeState.supportedMarkets = ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA'];

  breadthApi.getCurrentBreadth.mockImplementation((market = 'US') => Promise.resolve(breadthRow(market)));
  breadthApi.getHistoricalBreadth.mockImplementation((startDate, endDate, limit, market = 'US') => (
    Promise.resolve([breadthRow(market)])
  ));
  breadthApi.getBreadthSummary.mockImplementation((market = 'US') => Promise.resolve({
    market,
    latest_date: '2026-04-24',
    total_records: 1,
    date_range_start: '2026-04-24',
    date_range_end: '2026-04-24',
  }));
  breadthApi.getBreadthBootstrap.mockRejectedValue(new Error('no snapshot'));
  stocksApi.getPriceHistory.mockResolvedValue([]);
});

describe('BreadthPage', () => {
  it('defaults breadth requests to the runtime primary market', async () => {
    renderWithProviders(<BreadthPage />);

    expect(await screen.findByText('Latest Breadth Data')).toBeInTheDocument();

    await waitFor(() => {
      expect(breadthApi.getCurrentBreadth).toHaveBeenCalledWith('HK');
      expect(breadthApi.getBreadthSummary).toHaveBeenCalledWith('HK');
      expect(stocksApi.getPriceHistory).toHaveBeenCalledWith('2800.HK', '1mo');
    });
  });

  it('supports Korea as a runtime primary market', async () => {
    runtimeState.primaryMarket = 'KR';
    runtimeState.enabledMarkets = ['KR', 'US'];

    renderWithProviders(<BreadthPage />);

    expect(await screen.findByText('Latest Breadth Data')).toBeInTheDocument();

    await waitFor(() => {
      expect(breadthApi.getCurrentBreadth).toHaveBeenCalledWith('KR');
      expect(breadthApi.getBreadthSummary).toHaveBeenCalledWith('KR');
      expect(stocksApi.getPriceHistory).toHaveBeenCalledWith('069500.KS', '1mo');
    });
    expect(screen.getByRole('combobox', { name: /market/i })).toHaveTextContent('South Korea');
  });

  it('resyncs the default market when runtime primary market data loads late', async () => {
    runtimeState.primaryMarket = 'US';
    runtimeState.enabledMarkets = ['US'];

    const { rerender } = renderWithProviders(<BreadthPage />);

    expect(await screen.findByText('Latest Breadth Data')).toBeInTheDocument();
    await waitFor(() => {
      expect(breadthApi.getCurrentBreadth).toHaveBeenCalledWith('US');
    });

    runtimeState.primaryMarket = 'HK';
    runtimeState.enabledMarkets = ['US', 'HK'];
    rerender(<BreadthPage />);

    await waitFor(() => {
      expect(breadthApi.getCurrentBreadth).toHaveBeenCalledWith('HK');
      expect(stocksApi.getPriceHistory).toHaveBeenCalledWith('2800.HK', '1mo');
    });
  });

  it('resyncs to the runtime primary market when market options load late', async () => {
    runtimeState.primaryMarket = 'HK';
    runtimeState.enabledMarkets = ['US'];

    const { rerender } = renderWithProviders(<BreadthPage />);

    expect(await screen.findByText('Latest Breadth Data')).toBeInTheDocument();
    await waitFor(() => {
      expect(breadthApi.getCurrentBreadth).toHaveBeenCalledWith('US');
    });

    runtimeState.enabledMarkets = ['US', 'HK'];
    rerender(<BreadthPage />);

    await waitFor(() => {
      expect(breadthApi.getCurrentBreadth).toHaveBeenCalledWith('HK');
      expect(stocksApi.getPriceHistory).toHaveBeenCalledWith('2800.HK', '1mo');
    });
  });

  it('seeds bootstrap benchmark overlay under the live benchmark query key', async () => {
    runtimeState.uiSnapshots = { breadth: true };
    breadthApi.getBreadthBootstrap.mockResolvedValue({
      is_stale: false,
      payload: {
        current: breadthRow('HK'),
        history_90d: [breadthRow('HK')],
        summary: { market: 'HK', total_records: 1 },
        chart_range: '1M',
        chart_data: [breadthRow('HK')],
        benchmark_symbol: '^HSI',
        benchmark_overlay: [{ date: '2026-04-24', close: 18400 }],
      },
    });

    renderWithProviders(<BreadthPage />);

    const chart = await screen.findByTestId('breadth-chart');
    await waitFor(() => {
      expect(chart).toHaveTextContent('2800.HK:1:1');
    });
    expect(stocksApi.getPriceHistory).not.toHaveBeenCalledWith('^HSI', '1mo');
  });

  it('renders breadth data when the optional benchmark overlay request fails', async () => {
    stocksApi.getPriceHistory.mockRejectedValue(new Error('benchmark unavailable'));

    renderWithProviders(<BreadthPage />);

    const chart = await screen.findByTestId('breadth-chart');
    expect(chart).toHaveTextContent('2800.HK:1');
    expect(chart).toHaveAttribute('data-error', '');
  });

  it('refetches breadth data when the selected market changes', async () => {
    const user = userEvent.setup();

    renderWithProviders(<BreadthPage />);

    const marketSelect = await screen.findByRole('combobox', { name: /market/i });
    fireEvent.mouseDown(marketSelect);
    await user.click(await screen.findByRole('option', { name: /United States/i }));

    await waitFor(() => {
      expect(breadthApi.getCurrentBreadth).toHaveBeenCalledWith('US');
      expect(breadthApi.getHistoricalBreadth).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(String),
        730,
        'US',
      );
      expect(stocksApi.getPriceHistory).toHaveBeenCalledWith('SPY', '1mo');
    });
  });

  it('keeps the market selector available when the selected market has no breadth rows', async () => {
    const user = userEvent.setup();
    breadthApi.getCurrentBreadth.mockImplementation((market = 'US') => (
      market === 'HK'
        ? Promise.reject(new Error('No breadth data available for market HK.'))
        : Promise.resolve(breadthRow(market))
    ));

    renderWithProviders(<BreadthPage />);

    expect(
      await screen.findByText('Error loading HK breadth data: No breadth data available for market HK.')
    ).toBeInTheDocument();

    const marketSelect = screen.getByRole('combobox', { name: /market/i });
    fireEvent.mouseDown(marketSelect);
    await user.click(await screen.findByRole('option', { name: /United States/i }));

    await waitFor(() => {
      expect(breadthApi.getCurrentBreadth).toHaveBeenCalledWith('US');
    });
    expect(await screen.findByText('Latest Breadth Data')).toBeInTheDocument();
  });
});
