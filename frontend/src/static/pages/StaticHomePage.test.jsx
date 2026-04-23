import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import StaticHomePage from './StaticHomePage';

const fetchStaticJson = vi.fn();
const useStaticManifest = vi.fn();
const useStaticChartIndex = vi.fn();
const useStaticMarket = vi.fn();
const modalSpy = vi.fn();

vi.mock('../dataClient', () => ({
  fetchStaticJson: (...args) => fetchStaticJson(...args),
  useStaticManifest: (...args) => useStaticManifest(...args),
  resolveStaticMarketEntry: (manifest, selectedMarket) => ({
    market: selectedMarket,
    display_name: manifest.markets[selectedMarket].display_name,
    pages: manifest.markets[selectedMarket].pages,
    assets: manifest.markets[selectedMarket].assets,
  }),
}));

vi.mock('../chartClient', () => ({
  useStaticChartIndex: (...args) => useStaticChartIndex(...args),
}));

vi.mock('../StaticMarketContext', () => ({
  useStaticMarket: (...args) => useStaticMarket(...args),
}));

vi.mock('../StaticChartViewerModal', () => ({
  default: (props) => {
    modalSpy(props);
    return <div data-testid="static-chart-modal" data-open={props.open ? 'yes' : 'no'} />;
  },
}));

vi.mock('../../components/Scan/PriceSparkline', () => ({
  default: () => <span data-testid="price-sparkline" />,
}));

vi.mock('../../components/Scan/RSSparkline', () => ({
  default: () => <span data-testid="rs-sparkline" />,
}));

const manifest = {
  markets: {
    US: {
      display_name: 'United States',
      pages: {
        home: { path: 'markets/us/home.json' },
        scan: { path: 'markets/us/scan/manifest.json' },
      },
      assets: {
        charts: { path: 'markets/us/charts/index.json' },
      },
    },
  },
};

describe('StaticHomePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    modalSpy.mockClear();
    useStaticManifest.mockReturnValue({
      data: manifest,
      isLoading: false,
      isError: false,
    });
    useStaticMarket.mockReturnValue({ selectedMarket: 'US' });
    useStaticChartIndex.mockReturnValue({
      data: {
        symbols: [
          { symbol: 'LOW', rank: 1, path: 'charts/LOW.json' },
          { symbol: 'NVDA', rank: 2, path: 'charts/NVDA.json' },
          { symbol: 'AAPL', rank: 3, path: 'charts/AAPL.json' },
        ],
      },
    });
    fetchStaticJson.mockImplementation(async (path) => {
      if (path === 'markets/us/home.json') {
        return {
          market_display_name: 'United States',
          freshness: {
            scan_as_of_date: '2026-04-24',
            breadth_latest_date: '2026-04-24',
            groups_latest_date: '2026-04-24',
          },
          key_markets: [],
          scan_summary: {
            top_results: [
              { symbol: 'SUMMARYONLY', company_name: 'Home Summary Only', composite_score: 99.9 },
            ],
          },
          top_groups: [],
        };
      }

      if (path === 'markets/us/scan/manifest.json') {
        return {
          initial_rows: [
            {
              symbol: 'NVDA',
              company_name: 'NVIDIA Corporation',
              composite_score: 98.0,
              current_price: 100,
              rating: 'Strong Buy',
              volume: 180_000_000,
              market_cap: 2_000_000_000,
              currency: 'USD',
              price_sparkline_data: null,
              rs_sparkline_data: null,
            },
          ],
          chunks: [
            { path: 'markets/us/scan/chunks/chunk-0001.json' },
          ],
        };
      }

      if (path === 'markets/us/scan/chunks/chunk-0001.json') {
        return {
          rows: [
            {
              symbol: '0700.HK',
              company_name: 'Tencent Holdings',
              composite_score: 99.0,
              current_price: 25,
              rating: 'Buy',
              volume: 170_000_000,
              market_cap: 3_900_000_000_000,
              market_cap_usd: 500_000_000,
              currency: 'HKD',
              price_sparkline_data: null,
              rs_sparkline_data: null,
            },
            {
              symbol: 'AAPL',
              company_name: 'Apple Inc.',
              composite_score: 97.0,
              current_price: 200,
              rating: 'Buy',
              volume: 160_000_000,
              market_cap: 3_000_000_000,
              currency: 'USD',
              price_sparkline_data: null,
              rs_sparkline_data: null,
            },
          ],
        };
      }

      throw new Error(`Unexpected static path: ${path}`);
    });
  });

  it('loads top candidates from the static scan bundle, filters by market cap, and keeps chart navigation aligned', async () => {
    renderWithProviders(<StaticHomePage />);

    expect(await screen.findByText('0700.HK')).toBeInTheDocument();
    expect(screen.getByText('MCap')).toBeInTheDocument();
    expect(screen.getByText('$500.0M')).toBeInTheDocument();
    expect(screen.queryByText('HK$3.9T')).not.toBeInTheDocument();
    expect(fetchStaticJson).toHaveBeenCalledWith('markets/us/scan/manifest.json');
    expect(fetchStaticJson).toHaveBeenCalledWith('markets/us/scan/chunks/chunk-0001.json');
    expect(screen.queryByText('SUMMARYONLY')).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('combobox', { name: 'Mkt Cap' }));
    await user.click(await screen.findByRole('option', { name: '>$1B' }));

    await waitFor(() => {
      expect(screen.queryByText('0700.HK')).not.toBeInTheDocument();
    });
    expect(screen.getByText('NVDA')).toBeInTheDocument();
    expect(screen.getByText('AAPL')).toBeInTheDocument();

    await user.click(screen.getByText('NVDA'));

    await waitFor(() => {
      const props = modalSpy.mock.calls.at(-1)?.[0];
      expect(props).toMatchObject({
        open: true,
        initialSymbol: 'NVDA',
        navigationSymbols: ['NVDA', 'AAPL'],
      });
    });
  });
});
