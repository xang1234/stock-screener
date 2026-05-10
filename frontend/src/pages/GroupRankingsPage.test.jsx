import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import GroupRankingsPage from './GroupRankingsPage';

const getGroupsBootstrap = vi.fn();
const getCurrentRankings = vi.fn();
const getRankMovers = vi.fn();
const getGroupDetail = vi.fn();
const triggerCalculation = vi.fn();
const getCalculationStatus = vi.fn();
const fetchPriceHistoryBatch = vi.fn();
const runtimeState = {
  features: { tasks: false },
  runtimeReady: true,
  uiSnapshots: { groups: false },
  primaryMarket: 'HK',
  enabledMarkets: ['HK', 'US'],
  supportedMarkets: ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA', 'DE'],
};

vi.mock('../api/groups', () => ({
  getGroupsBootstrap: (...args) => getGroupsBootstrap(...args),
  getCurrentRankings: (...args) => getCurrentRankings(...args),
  getRankMovers: (...args) => getRankMovers(...args),
  getGroupDetail: (...args) => getGroupDetail(...args),
  triggerCalculation: (...args) => triggerCalculation(...args),
  getCalculationStatus: (...args) => getCalculationStatus(...args),
}));

vi.mock('../api/priceHistory', () => ({
  fetchPriceHistoryBatch: (...args) => fetchPriceHistoryBatch(...args),
  priceHistoryKeys: {
    batch: (symbols, period = '6mo') => ['priceHistory', 'batch', period, symbols.join(',')],
  },
  PRICE_HISTORY_STALE_TIME: 300000,
}));

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../components/Charts/CandlestickChart', () => ({
  default: ({ symbol, priceData, compact }) => (
    <div data-testid="candlestick-chart">{`${symbol}:${priceData?.length ?? 0}:${compact ? 'compact' : 'full'}`}</div>
  ),
}));

const rankingRowFor = (market) => ({
  industry_group: `${market} Internet Services`,
  date: '2026-04-18',
  rank: 3,
  avg_rs_rating: 82.1,
  median_rs_rating: 81.5,
  weighted_avg_rs_rating: 84.0,
  rs_std_dev: 3.2,
  num_stocks: 7,
  num_stocks_rs_above_80: 4,
  pct_rs_above_80: 57.14,
  top_symbol: market === 'HK' ? '0700.HK' : 'META',
  top_rs_rating: 97.0,
  rank_change_1w: 2,
  rank_change_1m: 4,
  rank_change_3m: null,
  rank_change_6m: null,
});

describe('GroupRankingsPage', () => {
  beforeEach(() => {
    runtimeState.features = { tasks: false };
    runtimeState.runtimeReady = true;
    runtimeState.uiSnapshots = { groups: false };
    runtimeState.primaryMarket = 'HK';
    runtimeState.enabledMarkets = ['HK', 'US'];
    runtimeState.supportedMarkets = ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA', 'DE'];
    getGroupsBootstrap.mockReset();
    getCurrentRankings.mockReset();
    getRankMovers.mockReset();
    getGroupDetail.mockReset();
    triggerCalculation.mockReset();
    getCalculationStatus.mockReset();
    fetchPriceHistoryBatch.mockReset();

    getCurrentRankings.mockImplementation(async (_limit, market = 'US') => ({
      date: '2026-04-18',
      total_groups: 1,
      market_scope: market,
      rankings: [rankingRowFor(market)],
    }));
    getRankMovers.mockImplementation(async (period = '1w', _limit = 10, market = 'US') => ({
      period,
      market_scope: market,
      gainers: [rankingRowFor(market)],
      losers: [],
    }));
    getGroupDetail.mockResolvedValue({
      industry_group: 'Internet Services',
      current_rank: 3,
      current_avg_rs: 82.1,
      num_stocks: 7,
      top_symbol: '0700.HK',
      top_rs_rating: 97,
      history: [],
      stocks: [],
    });
    getCalculationStatus.mockResolvedValue({ status: 'queued' });
  });

  it('defaults to the runtime primary market and refetches when the market filter changes', async () => {
    renderWithProviders(<GroupRankingsPage />);

    await waitFor(() => {
      expect(getCurrentRankings).toHaveBeenCalledWith(197, 'HK');
    });
    expect(getRankMovers).toHaveBeenCalledWith('1w', 10, 'HK');
    expect(await screen.findByText('HK Internet Services')).toBeInTheDocument();
    expect(screen.getByText('HK | 1 groups | 2026-04-18')).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'US' }));

    await waitFor(() => {
      expect(getCurrentRankings).toHaveBeenCalledWith(197, 'US');
    });
    expect(getRankMovers).toHaveBeenCalledWith('1w', 10, 'US');
    expect(await screen.findByText('US Internet Services')).toBeInTheDocument();
    expect(screen.getByText('US | 1 groups | 2026-04-18')).toBeInTheDocument();
  });

  it('keeps the market filter visible on non-US load errors and hides the US-only calculation action', async () => {
    runtimeState.features = { tasks: true };
    getCurrentRankings.mockImplementation(async (_limit, market = 'US') => {
      if (market === 'HK') {
        throw new Error('HK rankings unavailable');
      }
      return {
        date: '2026-04-18',
        total_groups: 1,
        market_scope: market,
        rankings: [rankingRowFor(market)],
      };
    });

    renderWithProviders(<GroupRankingsPage />);

    expect(await screen.findByText('Error loading rankings: HK rankings unavailable')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'HK' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'US' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Calculate Rankings' })).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'US' }));

    await waitFor(() => {
      expect(getCurrentRankings).toHaveBeenCalledWith(197, 'US');
    });
    expect(await screen.findByText('US Internet Services')).toBeInTheDocument();
  });

  it('loads batch price history when the Charts tab is opened in the group modal', async () => {
    getGroupDetail.mockResolvedValueOnce({
      industry_group: 'HK Internet Services',
      current_rank: 3,
      current_avg_rs: 82.1,
      num_stocks: 2,
      top_symbol: '0700.HK',
      top_rs_rating: 97,
      history: [],
      stocks: [{ symbol: '0700.HK' }, { symbol: '9988.HK' }],
    });
    fetchPriceHistoryBatch.mockResolvedValueOnce({
      data: {
        '0700.HK': [{ date: '2026-04-18', open: 500, high: 510, low: 495, close: 505, volume: 1000 }],
        '9988.HK': [{ date: '2026-04-18', open: 100, high: 101, low: 99, close: 100.5, volume: 2000 }],
      },
      missing: [],
    });

    renderWithProviders(<GroupRankingsPage />);

    const user = userEvent.setup();
    await user.click(await screen.findByText('HK Internet Services'));

    expect(await screen.findByRole('tab', { name: 'Charts (2)' })).toBeInTheDocument();
    await user.click(screen.getByRole('tab', { name: 'Charts (2)' }));

    await waitFor(() => {
      expect(fetchPriceHistoryBatch).toHaveBeenCalledWith(['0700.HK', '9988.HK'], '6mo');
    });
    expect(await screen.findByText('0700.HK:1:compact')).toBeInTheDocument();
    expect(await screen.findByText('9988.HK:1:compact')).toBeInTheDocument();
  });

  it('shows an inline error state when the Charts tab batch fetch fails', async () => {
    getGroupDetail.mockResolvedValueOnce({
      industry_group: 'HK Internet Services',
      current_rank: 3,
      current_avg_rs: 82.1,
      num_stocks: 1,
      top_symbol: '0700.HK',
      top_rs_rating: 97,
      history: [],
      stocks: [{ symbol: '0700.HK' }],
    });
    fetchPriceHistoryBatch.mockRejectedValueOnce(new Error('batch failed'));

    renderWithProviders(<GroupRankingsPage />);

    const user = userEvent.setup();
    await user.click(await screen.findByText('HK Internet Services'));
    await user.click(await screen.findByRole('tab', { name: 'Charts (1)' }));

    expect(await screen.findByText('Failed to load group charts')).toBeInTheDocument();
    expect(screen.getByText('batch failed')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });
});
