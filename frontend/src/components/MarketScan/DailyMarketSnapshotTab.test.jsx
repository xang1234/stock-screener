import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import DailyMarketSnapshotTab from './DailyMarketSnapshotTab';

const getWatchlist = vi.fn();
const getScanBootstrap = vi.fn();
const getScanResults = vi.fn();
const getCurrentRankings = vi.fn();
const fetchPriceHistory = vi.fn();
const chartModalSpy = vi.fn();

vi.mock('../../api/marketScan', () => ({
  getWatchlist: (...args) => getWatchlist(...args),
}));

vi.mock('../../api/scans', () => ({
  getScanBootstrap: (...args) => getScanBootstrap(...args),
  getScanResults: (...args) => getScanResults(...args),
}));

vi.mock('../../api/groups', () => ({
  getCurrentRankings: (...args) => getCurrentRankings(...args),
}));

vi.mock('../../api/priceHistory', () => ({
  fetchPriceHistory: (...args) => fetchPriceHistory(...args),
  priceHistoryKeys: {
    symbol: (symbol, range) => ['priceHistory', symbol, range],
  },
  PRICE_HISTORY_STALE_TIME: 60_000,
}));

vi.mock('../Scan/PriceSparkline', () => ({
  default: () => <span data-testid="price-sparkline" />,
}));

vi.mock('../Scan/RSSparkline', () => ({
  default: () => <span data-testid="rs-sparkline" />,
}));

vi.mock('../Scan/ChartViewerModal', () => ({
  default: (props) => {
    chartModalSpy(props);
    return null;
  },
}));

const liveRow = {
  symbol: 'LIVE',
  company_name: 'Live Row Holdings',
  composite_score: 91.2,
  current_price: 10.25,
  rating: 'Buy',
  volume: 150_000_000,
  market_cap: 2_500_000_000,
  currency: 'USD',
  price_sparkline_data: null,
  rs_sparkline_data: null,
  ibd_industry_group: 'Software',
  ibd_group_rank: 7,
};

describe('DailyMarketSnapshotTab', () => {
  beforeEach(() => {
    getWatchlist.mockReset();
    getScanBootstrap.mockReset();
    getScanResults.mockReset();
    getCurrentRankings.mockReset();
    fetchPriceHistory.mockReset();
    chartModalSpy.mockReset();
    getWatchlist.mockResolvedValue({ symbols: [] });
    getCurrentRankings.mockResolvedValue({ date: '2026-04-24', rankings: [] });
    getScanBootstrap.mockResolvedValue({
      payload: {
        selected_scan: {
          scan_id: 'scan-1',
          as_of_date: '2026-04-24',
        },
        results_page: {
          results: [
            {
              symbol: 'BOOT',
              company_name: 'Bootstrap Preview',
              dollar_volume: 999_000_000,
              market_cap: 25_000_000_000,
            },
          ],
        },
      },
    });
    getScanResults.mockResolvedValue({
      total: 1,
      results: [liveRow],
    });
  });

  it('renders the market-cap column from live results instead of the bootstrap preview slice', async () => {
    renderWithProviders(<DailyMarketSnapshotTab />);

    await waitFor(() => {
      expect(getScanResults).toHaveBeenCalledWith(
        'scan-1',
        expect.objectContaining({
          page: 1,
          per_page: 20,
          sort_by: 'composite_score',
          sort_order: 'desc',
          min_volume: 100_000_000,
        })
      );
    });
    expect(await screen.findByText('LIVE', {}, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.getByText('MCap')).toBeInTheDocument();
    expect(screen.getByText('$2.5B')).toBeInTheDocument();
    expect(screen.queryByText('BOOT')).not.toBeInTheDocument();
  });

  it('refetches the live snapshot with a market-cap bucket filter', async () => {
    getScanResults
      .mockResolvedValueOnce({
        total: 2,
        results: [
          { ...liveRow, symbol: 'BIG', market_cap: 5_000_000_000, market_cap_usd: 5_000_000_000 },
          { ...liveRow, symbol: 'SMALL', market_cap: 5_000_000_000, market_cap_usd: 500_000_000 },
        ],
      })
      .mockResolvedValueOnce({
        total: 1,
        results: [
          { ...liveRow, symbol: 'BIG', market_cap: 5_000_000_000, market_cap_usd: 5_000_000_000 },
        ],
      });

    renderWithProviders(<DailyMarketSnapshotTab />);

    expect(await screen.findByText('BIG')).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('combobox', { name: 'Mkt Cap' }));
    await user.click(await screen.findByRole('option', { name: '>$1B' }));

    await waitFor(() => {
      expect(getScanResults).toHaveBeenLastCalledWith(
        'scan-1',
        expect.objectContaining({
          page: 1,
          per_page: 20,
          sort_by: 'composite_score',
          sort_order: 'desc',
          min_volume: 100_000_000,
          min_market_cap_usd: 1_000_000_000,
        })
      );
    });
  });

  it('shows an inline error state when top candidates fail to load', async () => {
    getScanResults.mockRejectedValueOnce(new Error('scan fetch failed'));

    renderWithProviders(<DailyMarketSnapshotTab />);

    expect(await screen.findByText('Failed to load scan candidates.')).toBeInTheDocument();
    expect(screen.queryByText('No scan candidates match the current filters.')).not.toBeInTheDocument();
  });

  it('uses USD-normalized market cap in the table and keeps modal navigation scoped to visible rows', async () => {
    getScanResults.mockResolvedValue({
      total: 2,
      results: [
        {
          ...liveRow,
          symbol: '0700.HK',
          currency: 'HKD',
          market_cap: 3_900_000_000_000,
          market_cap_usd: 500_000_000_000,
        },
        {
          ...liveRow,
          symbol: 'AAPL',
          currency: 'USD',
          market_cap: 2_000_000_000_000,
          market_cap_usd: 2_000_000_000_000,
        },
      ],
    });

    renderWithProviders(<DailyMarketSnapshotTab />);

    expect(await screen.findByText('0700.HK')).toBeInTheDocument();
    expect(screen.getByText('$500.0B')).toBeInTheDocument();
    expect(screen.queryByText('HK$3.9T')).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByText('0700.HK'));

    await waitFor(() => {
      expect(chartModalSpy).toHaveBeenLastCalledWith(
        expect.objectContaining({
          open: true,
          initialSymbol: '0700.HK',
          navigationSymbolsOverride: ['0700.HK', 'AAPL'],
        })
      );
    });
  });
});
