import { screen, waitFor, within } from '@testing-library/react';
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

const makeLeadersPresetScreen = (minVolume = 100_000_000) => ({
  id: 'leaders_in_leading_groups',
  name: 'Leaders in Leading Groups',
  short_name: 'Leaders',
  description: 'Strong report-card stocks in top 40 IBD groups',
  tier: 2,
  filters: {
    minVolume,
    ibdGroupRank: { min: null, max: 40 },
    rsRating: { min: 80, max: null },
  },
  sort_by: 'composite_score',
  sort_order: 'desc',
});

const makeLeaderRow = (index, overrides = {}) => ({
  symbol: `LEAD${String(index).padStart(2, '0')}`,
  company_name: `Leader ${index}`,
  composite_score: 100 - index,
  rs_rating: 95,
  current_price: 100 + index,
  rating: 'Strong Buy',
  volume: 150_000_000,
  market_cap: 2_000_000_000,
  currency: 'USD',
  ibd_industry_group: 'Semiconductors',
  ibd_group_rank: 10,
  price_sparkline_data: null,
  rs_sparkline_data: null,
  ...overrides,
});

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
      default_filters: { minVolume: 100_000_000 },
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
      preset_screens: [makeLeadersPresetScreen()],
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
    expect(screen.getAllByText('MCap').length).toBeGreaterThan(0);
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

  it('uses the static scan manifest default volume for Daily top candidates', async () => {
    scanManifestPayload.default_filters = { minVolume: 1_300_000 };
    scanManifestPayload.preset_screens = [makeLeadersPresetScreen(1_300_000)];
    scanManifestPayload.initial_rows = [
      {
        symbol: 'LOCALPASS',
        company_name: 'Local Liquid',
        composite_score: 88.0,
        current_price: 12,
        rating: 'Buy',
        volume: 5_000_000,
        market_cap: 2_000_000_000,
        currency: 'SGD',
        price_sparkline_data: null,
        rs_sparkline_data: null,
      },
      {
        symbol: 'TOOTHIN',
        company_name: 'Too Thin',
        composite_score: 99.0,
        current_price: 8,
        rating: 'Buy',
        volume: 900_000,
        market_cap: 2_000_000_000,
        currency: 'SGD',
        price_sparkline_data: null,
        rs_sparkline_data: null,
      },
    ];
    scanManifestPayload.chunks = [];

    renderWithProviders(<StaticHomePage />);

    expect(await screen.findByText('LOCALPASS')).toBeInTheDocument();
    expect(screen.queryByText('TOOTHIN')).not.toBeInTheDocument();
  });

  it('uses market liquidity defaults and composite ranking for leaders in leading groups', async () => {
    scanManifestPayload.default_filters = { minVolume: 1_300_000 };
    scanManifestPayload.preset_screens = [makeLeadersPresetScreen(1_300_000)];
    scanManifestPayload.initial_rows = [
      makeLeaderRow(1, {
        symbol: 'LOCALLEAD',
        composite_score: 64.23,
        rs_rating: 94.94,
        volume: 2_000_000,
        ibd_group_rank: 26,
      }),
      makeLeaderRow(2, {
        symbol: 'THINLEAD',
        composite_score: 65.0,
        rs_rating: 99,
        volume: 900_000,
        ibd_group_rank: 10,
      }),
    ];
    scanManifestPayload.chunks = [];

    renderWithProviders(<StaticHomePage />);

    const leadersSection = await screen.findByTestId('leaders-in-leading-groups-section');
    expect(
      within(leadersSection).getByText('Top 20 by report card: group rank <=40, RS >=80, dollar volume >= 1,300,000.')
    ).toBeInTheDocument();
    expect(within(leadersSection).getByText('LOCALLEAD')).toBeInTheDocument();
    expect(within(leadersSection).queryByText('THINLEAD')).not.toBeInTheDocument();
  });

  it('omits the leaders liquidity subtitle when the resolved preset has no volume floor', async () => {
    scanManifestPayload.default_filters = { minVolume: null };
    scanManifestPayload.preset_screens = [makeLeadersPresetScreen(null)];
    scanManifestPayload.initial_rows = [
      makeLeaderRow(1, {
        symbol: 'NOFLOOR',
        volume: 1,
        ibd_group_rank: 10,
        rs_rating: 90,
      }),
    ];
    scanManifestPayload.chunks = [];

    renderWithProviders(<StaticHomePage />);

    const leadersSection = await screen.findByTestId('leaders-in-leading-groups-section');
    expect(
      within(leadersSection).getByText('Top 20 by report card: group rank <=40, RS >=80.')
    ).toBeInTheDocument();
    expect(within(leadersSection).queryByText(/dollar volume >=/i)).not.toBeInTheDocument();
    expect(within(leadersSection).getByText('NOFLOOR')).toBeInTheDocument();
  });

  it('shows top 20 leaders in leading groups after top scan candidates with leader-scoped chart navigation', async () => {
    const leaderRows = Array.from({ length: 21 }, (_, index) => makeLeaderRow(index + 1));
    // Verifies preset sorting stays exact and does not demote IPO-weighted rows behind lower-scoring full rows.
    leaderRows[1] = makeLeaderRow(2, { scan_mode: 'ipo_weighted' });
    const rejectedRows = [
      makeLeaderRow(31, { symbol: 'WEAKRS', rs_rating: 79 }),
      makeLeaderRow(32, { symbol: 'LATEGROUP', ibd_group_rank: 41 }),
      makeLeaderRow(33, { symbol: 'LOWSCORE', composite_score: 69 }),
      makeLeaderRow(34, { symbol: 'THINVOL', volume: 99_999_999 }),
    ];
    useStaticChartIndex.mockReturnValue({
      data: {
        symbols: [
          { symbol: 'LEAD01', rank: 1, path: 'charts/LEAD01.json' },
          { symbol: 'LEAD02', rank: 2, path: 'charts/LEAD02.json' },
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
          scan_summary: { top_results: [] },
          top_groups: [],
        };
      }

      if (path === 'markets/us/scan/manifest.json') {
        return {
          initial_rows: [],
          chunks: [
            { path: 'markets/us/scan/chunks/chunk-0001.json' },
          ],
          preset_screens: [makeLeadersPresetScreen()],
        };
      }

      if (path === 'markets/us/scan/chunks/chunk-0001.json') {
        return {
          rows: [...leaderRows, ...rejectedRows],
        };
      }

      throw new Error(`Unexpected static path: ${path}`);
    });

    renderWithProviders(<StaticHomePage />);

    const topCandidatesHeading = await screen.findByText('Top Scan Candidates');
    const leadersHeading = await screen.findByText('Leaders in Leading Groups');
    const topGroupsHeading = await screen.findByText('Top 10 Groups');
    expect(
      topCandidatesHeading.compareDocumentPosition(leadersHeading) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
    expect(
      leadersHeading.compareDocumentPosition(topGroupsHeading) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
    const leadersSection = screen.getByTestId('leaders-in-leading-groups-section');
    expect(within(leadersSection).getByText('LEAD01')).toBeInTheDocument();
    expect(within(leadersSection).getByText('LEAD02')).toBeInTheDocument();
    expect(within(leadersSection).queryByText('LEAD21')).not.toBeInTheDocument();
    expect(within(leadersSection).queryByText('WEAKRS')).not.toBeInTheDocument();
    expect(within(leadersSection).queryByText('LATEGROUP')).not.toBeInTheDocument();
    expect(within(leadersSection).queryByText('LOWSCORE')).not.toBeInTheDocument();
    expect(within(leadersSection).queryByText('THINVOL')).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(within(leadersSection).getByText('LEAD01'));

    await waitFor(() => {
      const props = modalSpy.mock.calls.at(-1)?.[0];
      expect(props).toMatchObject({
        open: true,
        initialSymbol: 'LEAD01',
        navigationSymbols: ['LEAD01', 'LEAD02'],
      });
    });
  });
});
