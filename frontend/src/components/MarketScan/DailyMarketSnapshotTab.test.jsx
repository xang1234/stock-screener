import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import DailyMarketSnapshotTab from './DailyMarketSnapshotTab';

const getDailySnapshot = vi.fn();
const getScanResults = vi.fn();
const chartModalSpy = vi.fn();

vi.mock('../../api/marketScan', () => ({
  getDailySnapshot: (...args) => getDailySnapshot(...args),
}));

vi.mock('../../api/scans', () => ({
  getScanResults: (...args) => getScanResults(...args),
}));

vi.mock('../Scan/PriceSparkline', () => ({
  default: () => <span data-testid="price-sparkline" />,
}));

vi.mock('../Scan/RSSparkline', () => ({
  default: () => <span data-testid="rs-sparkline" />,
}));

vi.mock('../Scan/ChartViewerModalLazy', () => ({
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
  rs_rating: 88,
  volume: 150_000_000,
  market_cap: 2_500_000_000,
  currency: 'USD',
  price_sparkline_data: null,
  rs_sparkline_data: null,
  ibd_industry_group: 'Software',
  ibd_group_rank: 7,
};

function snapshotPayload(overrides = {}) {
  return {
    market: 'US',
    market_display_name: 'United States',
    scan_id: 'scan-1',
    freshness: {
      scan_id: 'scan-1',
      scan_as_of_date: '2026-04-24',
      snapshot_as_of_date: '2026-04-24',
      market_timezone: 'America/New_York',
      breadth_latest_date: '2026-04-24',
      groups_latest_date: '2026-04-24',
      exposure_latest_date: '2026-04-24',
      key_markets_latest_date: '2026-04-24',
      date_coherence_status: 'coherent',
    },
    key_markets: [
      {
        symbol: 'BITSTAMP:BTCUSD',
        display_name: 'Bitcoin',
        currency: 'USD',
        latest_close: 61429.69,
        latest_date: '2026-04-24',
        change_1d: -0.35,
        history: [
          { date: '2026-04-23', close: 61645.5 },
          { date: '2026-04-24', close: 61429.69 },
        ],
      },
      {
        symbol: 'TVC:VIX',
        display_name: 'Volatility Index',
        currency: 'USD',
        latest_close: null,
        latest_date: null,
        change_1d: null,
        history: [],
      },
    ],
    top_candidates: {
      min_dollar_volume: 100_000_000,
      rows: [liveRow],
    },
    leaders: {
      criteria: {
        max_group_rank: 40,
        min_rs_rating: 80,
        min_dollar_volume: 100_000_000,
      },
      rows: [{ ...liveRow, symbol: 'LEAD', company_name: 'Leader Corp' }],
    },
    top_groups: [
      {
        industry_group: 'Computer-Data Storage',
        rank: 1,
        rank_change_1w: 0,
        rank_change_1m: 4,
        top_symbol: 'SNDK',
        top_symbol_name: 'Sandisk Corp',
      },
    ],
    ...overrides,
  };
}

describe('DailyMarketSnapshotTab', () => {
  beforeEach(() => {
    getDailySnapshot.mockReset();
    getScanResults.mockReset();
    chartModalSpy.mockReset();
    getDailySnapshot.mockResolvedValue(snapshotPayload());
    getScanResults.mockResolvedValue({ total: 1, results: [liveRow] });
  });

  it('renders all sections from one aggregated snapshot request', async () => {
    renderWithProviders(<DailyMarketSnapshotTab />);

    expect(await screen.findByText('LIVE')).toBeInTheDocument();
    expect(getDailySnapshot).toHaveBeenCalledTimes(1);
    // No per-section fan-out: scan results are only fetched for filters.
    expect(getScanResults).not.toHaveBeenCalled();

    // Freshness header collapses coherent section dates into one snapshot date.
    expect(screen.getByText(/As of 2026-04-24/)).toBeInTheDocument();
    expect(screen.getByText(/America\/New_York/)).toBeInTheDocument();
    expect(screen.queryByText(/Breadth 2026-04-24/)).not.toBeInTheDocument();

    // Key market card renders the aliased instrument with data...
    expect(screen.getByText('BITSTAMP:BTCUSD')).toBeInTheDocument();
    // ...while instruments without history are hidden instead of showing "-"
    expect(screen.queryByText('TVC:VIX')).not.toBeInTheDocument();

    // Leaders in Leading Groups section
    expect(screen.getByText('Leaders in Leading Groups')).toBeInTheDocument();
    expect(screen.getByText('LEAD')).toBeInTheDocument();

    // Top 10 Groups with the 1M column and the top-stock company name
    expect(screen.getByText('1M')).toBeInTheDocument();
    expect(screen.getByText('SNDK')).toBeInTheDocument();
    expect(screen.getByText('Sandisk Corp')).toBeInTheDocument();
  });

  it('refetches live scan results only when a market-cap bucket is selected', async () => {
    getScanResults.mockResolvedValue({
      total: 1,
      results: [{ ...liveRow, symbol: 'BIG', market_cap_usd: 5_000_000_000 }],
    });

    renderWithProviders(<DailyMarketSnapshotTab />);

    expect(await screen.findByText('LIVE')).toBeInTheDocument();

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
    expect(await screen.findByText('BIG')).toBeInTheDocument();
  });

  it('shows an error state when the snapshot fails to load', async () => {
    getDailySnapshot.mockRejectedValue(new Error('snapshot failed'));

    renderWithProviders(<DailyMarketSnapshotTab />);

    expect(await screen.findByText('Failed to load the daily snapshot.')).toBeInTheDocument();
  });

  it('uses USD-normalized market cap and keeps modal navigation scoped to visible rows', async () => {
    getDailySnapshot.mockResolvedValue(snapshotPayload({
      top_candidates: {
        min_dollar_volume: 100_000_000,
        rows: [
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
      },
    }));

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
