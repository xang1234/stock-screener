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

vi.mock('../api/groups', () => ({
  getGroupsBootstrap: (...args) => getGroupsBootstrap(...args),
  getCurrentRankings: (...args) => getCurrentRankings(...args),
  getRankMovers: (...args) => getRankMovers(...args),
  getGroupDetail: (...args) => getGroupDetail(...args),
  triggerCalculation: (...args) => triggerCalculation(...args),
  getCalculationStatus: (...args) => getCalculationStatus(...args),
}));

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => ({
    features: { tasks: false },
    runtimeReady: true,
    uiSnapshots: { groups: false },
    primaryMarket: 'HK',
    enabledMarkets: ['HK', 'US'],
    supportedMarkets: ['US', 'HK', 'JP', 'TW'],
  }),
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
    getGroupsBootstrap.mockReset();
    getCurrentRankings.mockReset();
    getRankMovers.mockReset();
    getGroupDetail.mockReset();
    triggerCalculation.mockReset();
    getCalculationStatus.mockReset();

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
});
