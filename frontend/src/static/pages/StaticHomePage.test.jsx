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
const priceSparklineSpy = vi.fn();

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
  default: (props) => {
    priceSparklineSpy(props);
    return <span data-testid="price-sparkline" />;
  },
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
  let homePayload;
  let scanManifestPayload;
  let scanChunkPayload;

  beforeEach(() => {
    vi.clearAllMocks();
    modalSpy.mockClear();
    priceSparklineSpy.mockClear();
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
    homePayload = {
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
    scanManifestPayload = {
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
    scanChunkPayload = {
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

    fetchStaticJson.mockImplementation(async (path) => {
      if (path === 'markets/us/home.json') {
        return homePayload;
      }

      if (path === 'markets/us/scan/manifest.json') {
        return scanManifestPayload;
      }

      if (path === 'markets/us/scan/chunks/chunk-0001.json') {
        return scanChunkPayload;
      }

      throw new Error(`Unexpected static path: ${path}`);
    });
  });

  it('filters key market cards to entries with renderable close history', async () => {
    homePayload.key_markets = [
      {
        symbol: 'VALID',
        display_name: 'Valid Market',
        currency: 'USD',
        latest_close: 102,
        change_1d: 2,
        history: [{ close: 100 }, { close: null }, { close: 102 }],
      },
      {
        symbol: 'NULLS',
        display_name: 'Null History',
        currency: 'USD',
        latest_close: 10,
        change_1d: null,
        history: [{ close: null }],
      },
      {
        symbol: 'SINGLE',
        display_name: 'Single Close',
        currency: 'USD',
        latest_close: 20,
        change_1d: null,
        history: [{ close: 20 }],
      },
      {
        symbol: 'MISSING',
        display_name: 'Missing Price',
        currency: 'USD',
        latest_close: null,
        change_1d: null,
        history: [{ close: 19 }, { close: 20 }],
      },
    ];

    renderWithProviders(<StaticHomePage />);

    expect(await screen.findByText('VALID')).toBeInTheDocument();
    expect(screen.queryByText('NULLS')).not.toBeInTheDocument();
    expect(screen.queryByText('SINGLE')).not.toBeInTheDocument();
    expect(screen.queryByText('MISSING')).not.toBeInTheDocument();
    expect(priceSparklineSpy).toHaveBeenCalledWith(expect.objectContaining({
      data: [100, 102],
      showChange: false,
    }));
  });

  it('keeps top candidate price sparklines within the compact table width', async () => {
    scanChunkPayload.rows[0].price_sparkline_data = [20, 22, 24];
    scanChunkPayload.rows[0].price_trend = 1;
    scanChunkPayload.rows[0].price_change_1d = 12.3;

    renderWithProviders(<StaticHomePage />);

    expect(await screen.findByText('0700.HK')).toBeInTheDocument();
    expect(priceSparklineSpy).toHaveBeenCalledWith(expect.objectContaining({
      data: [20, 22, 24],
      width: 137,
      sparklineWidth: 86,
      change1d: 12.3,
    }));
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
