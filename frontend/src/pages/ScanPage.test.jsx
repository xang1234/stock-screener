import { act, fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import userEvent from '@testing-library/user-event';

import ScanPage from './ScanPage';
import { DEFAULT_SCAN_DEFAULTS } from '../constants/scanDefaults';
import * as scanApi from '../api/scans';
import { renderWithProviders } from '../test/renderWithProviders';

const runtimeState = {
  runtimeReady: false,
  uiSnapshots: {
    scan: false,
  },
  scanDefaults: DEFAULT_SCAN_DEFAULTS,
  universeOptions: null,
  features: {},
};
const useRuntimeActivityMock = vi.hoisted(() => vi.fn());
const filterPresetsState = vi.hoisted(() => ({ presets: [] }));
const CAPABILITY_TRANSITION_TIMEOUT = 10_000;
const STALE_TAIL_WARNING = {
  code: 'market_data_stale_tail_omitted',
  message: 'Omitted 1 stale symbol from this broad scan (99.00% fresh).',
  omitted_symbols: ['LHSW'],
  omitted_count: 1,
};

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

// null by default, like the context without a provider; tests set a market.
const marketState = vi.hoisted(() => ({ selectedMarket: null }));
vi.mock('../contexts/MarketContext', async (importOriginal) => ({
  ...(await importOriginal()),
  useMarket: () => ({
    selectedMarket: marketState.selectedMarket,
    setSelectedMarket: () => {},
    selectableMarkets: [],
    marketLabel: (code) => code,
  }),
}));

vi.mock('../hooks/useRuntimeActivity', () => ({
  useRuntimeActivity: (...args) => useRuntimeActivityMock(...args),
}));

vi.mock('../contexts/StrategyProfileContext', () => ({
  useStrategyProfileData: () => ({
    activeProfileDetail: null,
  }),
}));

vi.mock('../hooks/useFilterPresets', () => ({
  useFilterPresets: () => ({
    presets: filterPresetsState.presets,
    isLoading: false,
    createPresetAsync: vi.fn(),
    updatePresetAsync: vi.fn(),
    deletePreset: vi.fn(),
    isCreating: false,
    isUpdating: false,
  }),
}));

vi.mock('../api/scans', () => ({
  createScan: vi.fn(),
  getScanBootstrap: vi.fn(),
  getScanStatus: vi.fn(),
  queryScanResults: vi.fn(),
  getUniverseStats: vi.fn().mockResolvedValue({
    active: 321,
    sp500: 500,
    by_exchange: {
      NYSE: 100,
      NASDAQ: 200,
      AMEX: 21,
    },
  }),
  exportScanResultsQuery: vi.fn(),
  getScans: vi.fn().mockResolvedValue({ scans: [] }),
  cancelScan: vi.fn(),
  getFilterOptions: vi.fn(),
  refreshScanCache: vi.fn(),
}));

beforeEach(() => {
  window.history.replaceState(null, '', '/scan');
  vi.clearAllMocks();
  runtimeState.runtimeReady = false;
  runtimeState.uiSnapshots = { scan: false };
  runtimeState.scanDefaults = DEFAULT_SCAN_DEFAULTS;
  runtimeState.universeOptions = null;
  runtimeState.features = {};
  filterPresetsState.presets = [];
  marketState.selectedMarket = null;
  useRuntimeActivityMock.mockReset();
  useRuntimeActivityMock.mockReturnValue({
    data: {
      bootstrap: {},
      summary: { active_market_count: 0, active_markets: [], status: 'idle' },
      markets: [],
    },
  });
  scanApi.getScanBootstrap.mockResolvedValue(null);
  scanApi.getScanStatus.mockResolvedValue({ status: 'completed' });
  scanApi.queryScanResults.mockResolvedValue({ total: 0, results: [] });
  scanApi.getFilterOptions.mockResolvedValue({
    ibd_industries: [],
    gics_sectors: [],
    ratings: [],
  });
  scanApi.getScans.mockResolvedValue({ scans: [] });
});

describe('ScanPage', () => {
  it('turns a URL-safe Social selection into a custom scan universe', async () => {
    runtimeState.runtimeReady = true;
    window.history.replaceState(null, '', '/scan?market=HK&symbols=0700%2C9988');
    scanApi.createScan.mockResolvedValueOnce({ scan_id: 'social-scan', status: 'queued' });

    renderWithProviders(<ScanPage />);

    expect(await screen.findByText('Social selection: 2 symbols')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Scan' }));
    await waitFor(() => expect(scanApi.createScan.mock.calls[0][0]).toEqual(
      expect.objectContaining({
        universe_def: { type: 'custom', symbols: ['0700', '9988'] },
      })
    ));
  });

  it('renders without a temporal-dead-zone crash before runtime bootstrap completes', () => {
    renderWithProviders(<ScanPage />);

    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('hydrates scan controls from runtime scan defaults', async () => {
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      universe: 'all',
      screeners: ['custom'],
      composite_method: 'maximum',
      criteria: {
        include_vcp: false,
        custom_filters: {
          ...DEFAULT_SCAN_DEFAULTS.criteria.custom_filters,
          price_min: 123,
        },
      },
    };

    renderWithProviders(<ScanPage />);

    await waitFor(() => {
      expect(screen.getByDisplayValue('123')).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByRole('checkbox', { name: /vcp/i })).not.toBeChecked();
    });
  });

  describe('market-scoped scan bootstrap', () => {
    const marketBootstrap = (market, { scanId = `${market}-scan`, recentScans } = {}) => ({
      is_stale: false,
      published_at: new Date().toISOString(),
      payload: {
        market,
        universe_stats: { active: 1 },
        recent_scans: recentScans ?? { scans: [{ scan_id: scanId, status: 'completed' }] },
        selected_scan: { scan_id: scanId, status: 'completed' },
        selected_scan_status: { status: 'completed' },
        filter_options: { ibd_industries: [], gics_sectors: [], ratings: [] },
        results_page: {
          scan_id: scanId,
          total: 1,
          results: [{ symbol: 'NVDA', company_name: 'NVIDIA', composite_score: 98, current_price: 900 }],
        },
      },
    });

    beforeEach(() => {
      runtimeState.runtimeReady = true;
      runtimeState.uiSnapshots = { scan: true };
    });

    it('requests the selected market and seeds its scan history from the snapshot', async () => {
      marketState.selectedMarket = 'HK';
      scanApi.getScanBootstrap.mockResolvedValue(marketBootstrap('HK'));

      const { queryClient } = renderWithProviders(<ScanPage />);

      await waitFor(() => {
        expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument();
      });
      expect(scanApi.getScanBootstrap).toHaveBeenCalledWith(null, 'HK');
      expect(queryClient.getQueryData(['scanHistory', 'HK'])).toEqual({
        scans: [{ scan_id: 'HK-scan', status: 'completed' }],
      });
      // A freshly published list is used as is: no duplicate history request.
      expect(scanApi.getScans).not.toHaveBeenCalled();
    });

    it('does not seed history from an explicit scan snapshot', async () => {
      marketState.selectedMarket = 'US';
      const liveHistory = { scans: [{ scan_id: 'us-old', status: 'completed' }, { scan_id: 'us-new', status: 'queued' }] };
      scanApi.getScans.mockResolvedValue(liveHistory);
      scanApi.getScanBootstrap.mockImplementation(async (scanId) => (
        scanId
          // Explicit variants never go stale, so their lists can be old.
          ? marketBootstrap('US', { scanId, recentScans: { scans: [{ scan_id: 'us-old', status: 'completed' }] } })
          : { ...marketBootstrap('US'), is_stale: true }
      ));

      const { queryClient } = renderWithProviders(<ScanPage />);

      await waitFor(() => {
        expect(scanApi.getScanBootstrap).toHaveBeenCalledWith('us-old');
      });
      await waitFor(() => {
        expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument();
      });
      expect(queryClient.getQueryData(['scanHistory', 'US'])).toEqual(liveHistory);
    });
  });

  it('renders completed scan results flow with filter panel', async () => {
    runtimeState.runtimeReady = true;
    runtimeState.uiSnapshots = { scan: true };
    scanApi.getScanBootstrap.mockResolvedValue({
      is_stale: false,
      payload: {
        universe_stats: {
          active: 321,
          sp500: 500,
          by_exchange: { NYSE: 100, NASDAQ: 200, AMEX: 21 },
        },
        recent_scans: {
          scans: [
            {
              scan_id: 'scan-1',
              status: 'completed',
              created_at: '2026-04-09T00:00:00Z',
            },
          ],
        },
        selected_scan: {
          scan_id: 'scan-1',
          status: 'completed',
        },
        selected_scan_status: {
          status: 'completed',
        },
        filter_options: {
          ibd_industries: ['Semiconductors'],
          gics_sectors: ['Technology'],
          ratings: ['Buy'],
        },
        results_page: {
          scan_id: 'scan-1',
          total: 1,
          results: [
            {
              symbol: 'NVDA',
              company_name: 'NVIDIA',
              composite_score: 98,
              minervini_score: 92,
              current_price: 900,
              stage: 2,
            },
          ],
        },
      },
    });
    scanApi.getFilterOptions.mockResolvedValue({
      ibd_industries: ['Semiconductors'],
      gics_sectors: ['Technology'],
      ratings: ['Buy'],
    });

    const { queryClient } = renderWithProviders(<ScanPage />);

    await waitFor(() => {
      expect(scanApi.getScanBootstrap).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument();
    });
    expect(screen.getByText('Filters')).toBeInTheDocument();
    // History is read per market (['scanHistory', market]); the global
    // bootstrap list must not be cached (and persisted) under an unread key.
    expect(queryClient.getQueryCache().find({ queryKey: ['scanHistory'], exact: true })).toBeUndefined();
  });

  it('sanitizes a capable survivor query before showing results from a legacy scan', async () => {
    const user = userEvent.setup();
    runtimeState.runtimeReady = true;
    filterPresetsState.presets = [{
      id: 'correction-survivors-live',
      name: 'Correction Survivors',
      filters: { correctionSurvivor: true },
      sort_by: 'resilience_score',
      sort_order: 'desc',
    }];
    scanApi.getScans.mockResolvedValue({
      scans: [
        {
          scan_id: 'scan-capable',
          status: 'completed',
          started_at: '2026-08-21T20:00:00Z',
          trigger_source: 'manual',
          passed_stocks: 1,
          total_stocks: 1,
          universe_def: { type: 'market', market: 'US' },
        },
        {
          scan_id: 'scan-legacy',
          status: 'completed',
          started_at: '2026-08-20T20:00:00Z',
          trigger_source: 'manual',
          passed_stocks: 1,
          total_stocks: 1,
          universe_def: { type: 'market', market: 'US' },
        },
      ],
    });
    let resolveLegacyUnsafe;
    let resolveLegacySafe;
    const legacyUnsafe = new Promise((resolve) => {
      resolveLegacyUnsafe = resolve;
    });
    const legacySafe = new Promise((resolve) => {
      resolveLegacySafe = resolve;
    });
    scanApi.queryScanResults.mockImplementation((scanId, request) => {
      const usesSurvivorFilter = request.required.conditions.some(
        (condition) => condition.field === 'correction_survivor',
      );
      if (scanId === 'scan-capable') {
        return Promise.resolve(usesSurvivorFilter
          ? {
              total: 0,
              unfiltered_total: 1,
              results: [],
              capabilities: { opportunity_state: true },
            }
          : {
              total: 1,
              unfiltered_total: 1,
              results: [{ symbol: 'CURRENT', company_name: 'Current Corp' }],
              capabilities: { opportunity_state: true },
            });
      }
      return usesSurvivorFilter ? legacyUnsafe : legacySafe;
    });

    renderWithProviders(<ScanPage />);

    expect(await screen.findByText(
      /Results:\s*1 stocks/i,
      {},
      { timeout: CAPABILITY_TRANSITION_TIMEOUT },
    )).toBeInTheDocument();
    await user.click(screen.getByText('Select Preset'));
    await user.click(await screen.findByRole(
      'option',
      { name: 'Correction Survivors' },
      { timeout: CAPABILITY_TRANSITION_TIMEOUT },
    ));
    expect(await screen.findByText(
      'No stocks match the applied logic',
      {},
      { timeout: CAPABILITY_TRANSITION_TIMEOUT },
    )).toBeInTheDocument();

    const capableRequests = scanApi.queryScanResults.mock.calls.filter(
      ([scanId]) => scanId === 'scan-capable',
    );
    expect(capableRequests).toHaveLength(2);
    expect(capableRequests[1][1]).toMatchObject({
      sort: { field: 'resilience_score', order: 'desc' },
      page: { number: 1, size: 50 },
    });

    await user.click(screen.getByRole('combobox', { name: 'Previous Scans' }));
    const scanOptions = await screen.findAllByRole('option');
    await user.click(scanOptions.at(-1));
    await waitFor(() => {
      expect(scanApi.queryScanResults.mock.calls.some(
        ([scanId, request]) => (
          scanId === 'scan-legacy'
          && request.required.conditions.some(
            (condition) => condition.field === 'correction_survivor',
          )
        ),
      )).toBe(true);
    }, { timeout: CAPABILITY_TRANSITION_TIMEOUT });

    await act(async () => {
      resolveLegacyUnsafe({
        total: 0,
        unfiltered_total: 1,
        results: [],
      });
    });

    await waitFor(() => {
      const legacyRequests = scanApi.queryScanResults.mock.calls.filter(
        ([scanId]) => scanId === 'scan-legacy',
      );
      expect(legacyRequests).toHaveLength(2);
      expect(legacyRequests[1][1]).toMatchObject({
        required: { conditions: [] },
        groups: [],
        sort: { field: 'composite_score', order: 'desc' },
        page: { number: 1, size: 50 },
      });
    }, { timeout: CAPABILITY_TRANSITION_TIMEOUT });
    expect(screen.queryByText('No stocks match the applied logic')).not.toBeInTheDocument();
    expect(screen.getByText('Loading results...')).toBeInTheDocument();

    await act(async () => {
      resolveLegacySafe({
        total: 1,
        unfiltered_total: 1,
        results: [{ symbol: 'LEGACY', company_name: 'Legacy Corp' }],
      });
    });

    expect(await screen.findByText(
      /Results:\s*1 stocks/i,
      {},
      { timeout: CAPABILITY_TRANSITION_TIMEOUT },
    )).toBeInTheDocument();
    expect(screen.getByText('Select Preset')).toBeInTheDocument();
    expect(scanApi.queryScanResults.mock.calls.filter(
      ([scanId]) => scanId === 'scan-legacy',
    )).toHaveLength(2);
  }, 40_000);

  it('keeps the applied rows and sort indicator atomic when a grouped query fails', async () => {
    runtimeState.runtimeReady = true;
    runtimeState.uiSnapshots = { scan: true };
    runtimeState.features = { grouped_scan_filters: true };
    scanApi.getScanBootstrap.mockResolvedValue({
      is_stale: false,
      payload: {
        recent_scans: { scans: [] },
        selected_scan: { scan_id: 'scan-atomic', status: 'completed' },
        selected_scan_status: { status: 'completed' },
        filter_options: {
          ibd_industries: [],
          gics_sectors: [],
          ratings: [],
        },
      },
    });
    scanApi.queryScanResults.mockImplementation((_scanId, request) => {
      if (request.sort.field !== 'composite_score') {
        return Promise.reject(new Error('Sort request failed'));
      }
      return Promise.resolve({
        total: 1,
        unfiltered_total: 1,
        results: [{
          symbol: 'NVDA',
          company_name: 'NVIDIA',
          composite_score: 98,
          current_price: 900,
        }],
      });
    });

    renderWithProviders(<ScanPage />);

    await waitFor(() => {
      expect(scanApi.queryScanResults).toHaveBeenCalled();
    });
    expect(scanApi.queryScanResults.mock.calls[0][1].sort).toEqual({
      field: 'composite_score',
      order: 'desc',
    });
    expect(await screen.findByText(/Results:\s*1 stocks/i)).toBeInTheDocument();
    expect(screen.getByText('Comp')).toHaveClass('Mui-active');

    fireEvent.click(screen.getByText('SE'));

    expect(await screen.findByText('Sort request failed')).toBeInTheDocument();
    expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument();
    expect(screen.getByText('Comp')).toHaveClass('Mui-active');
    expect(screen.getByText('SE')).not.toHaveClass('Mui-active');
    expect(scanApi.queryScanResults).toHaveBeenCalledWith(
      'scan-atomic',
      expect.objectContaining({
        sort: { field: 'se_setup_score', order: 'asc' },
      }),
      expect.any(Object),
    );
  });

  it('renders stale-tail warning returned by scan creation', async () => {
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      ...DEFAULT_SCAN_DEFAULTS,
      universe: 'market:us',
    };
    scanApi.createScan.mockResolvedValueOnce({
      scan_id: 'scan-warning',
      status: 'queued',
      total_stocks: 99,
      warnings: [STALE_TAIL_WARNING],
    });

    renderWithProviders(<ScanPage />);

    fireEvent.click(await screen.findByRole('button', { name: 'Scan' }));

    expect(await screen.findByText(STALE_TAIL_WARNING.message)).toBeInTheDocument();
  });

  it('renders bootstrap stale-tail warning and clears it for a new scan', async () => {
    const user = userEvent.setup();
    runtimeState.runtimeReady = true;
    runtimeState.uiSnapshots = { scan: true };
    scanApi.getScanBootstrap.mockResolvedValue({
      is_stale: false,
      payload: {
        universe_stats: {
          active: 321,
          sp500: 500,
          by_exchange: { NYSE: 100, NASDAQ: 200, AMEX: 21 },
        },
        recent_scans: {
          scans: [
            {
              scan_id: 'scan-warning',
              status: 'completed',
              created_at: '2026-04-09T00:00:00Z',
              warnings: [STALE_TAIL_WARNING],
            },
          ],
        },
        selected_scan: {
          scan_id: 'scan-warning',
          status: 'completed',
          warnings: [STALE_TAIL_WARNING],
        },
        selected_scan_status: {
          status: 'completed',
          warnings: [STALE_TAIL_WARNING],
        },
        filter_options: {
          ibd_industries: [],
          gics_sectors: [],
          ratings: [],
        },
        results_page: {
          scan_id: 'scan-warning',
          total: 0,
          results: [],
        },
      },
    });

    renderWithProviders(<ScanPage />);

    expect(await screen.findByText(STALE_TAIL_WARNING.message)).toBeInTheDocument();

    await user.click(screen.getByRole('combobox', { name: 'Previous Scans' }));
    await user.click(await screen.findByRole('option', { name: 'New Scan' }));

    await waitFor(() => {
      expect(screen.queryByText(STALE_TAIL_WARNING.message)).not.toBeInTheDocument();
    });
  });

  it('auto-loads the latest completed scan after scan history refreshes from running-only state', async () => {
    runtimeState.runtimeReady = true;
    scanApi.getScans
      .mockResolvedValueOnce({
        scans: [
          {
            scan_id: 'scan-running',
            status: 'running',
            created_at: '2026-04-09T00:00:00Z',
          },
        ],
      })
      .mockResolvedValueOnce({
        scans: [
          {
            scan_id: 'scan-complete',
            status: 'completed',
            created_at: '2026-04-09T00:05:00Z',
          },
        ],
      });
    scanApi.getScanStatus.mockResolvedValue({ status: 'completed' });
    scanApi.queryScanResults.mockResolvedValue({
      total: 1,
      results: [
        {
          symbol: 'NVDA',
          company_name: 'NVIDIA',
          composite_score: 98,
          minervini_score: 92,
          current_price: 900,
          stage: 2,
        },
      ],
    });

    const { queryClient } = renderWithProviders(<ScanPage />);

    await waitFor(() => {
      expect(scanApi.getScans).toHaveBeenCalledTimes(1);
    });

    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: ['scanHistory'] });
    });

    await waitFor(() => {
      expect(scanApi.getScans).toHaveBeenCalledTimes(2);
    });
    await waitFor(() => {
      expect(scanApi.getScanStatus).toHaveBeenCalledWith('scan-complete');
    });
    await waitFor(() => {
      expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument();
    });
  });

  it('disables scan creation with a hover warning when prices refresh is active for the selected market', async () => {
    const user = userEvent.setup();
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      ...DEFAULT_SCAN_DEFAULTS,
      universe: 'market:us',
    };
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {},
        summary: { active_market_count: 1, active_markets: ['US'], status: 'active' },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            lifecycle: 'daily_refresh',
            progress_mode: 'determinate',
            percent: 30,
            current: 300,
            total: 1000,
            message: 'Refreshing prices',
          },
        ],
      },
    });

    renderWithProviders(<ScanPage />);

    const scanButton = await screen.findByRole('button', { name: 'Scan' });
    expect(scanButton).toBeDisabled();

    await user.hover(scanButton.parentElement);

    expect(
      await screen.findByText('US price refresh is running. Wait for it to finish before starting a scan.')
    ).toBeInTheDocument();
  });

  it('adds keyboard focus semantics to the disabled scan control wrapper when refresh is blocked', async () => {
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      ...DEFAULT_SCAN_DEFAULTS,
      universe: 'market:us',
    };
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {},
        summary: { active_market_count: 1, active_markets: ['US'], status: 'active' },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            lifecycle: 'daily_refresh',
            progress_mode: 'determinate',
            percent: 30,
            current: 300,
            total: 1000,
            message: 'Refreshing prices',
          },
        ],
      },
    });

    renderWithProviders(<ScanPage />);

    const scanButton = await screen.findByRole('button', { name: 'Scan' });
    expect(scanButton.parentElement).toHaveAttribute('tabindex', '0');
    expect(scanButton.parentElement).toHaveAttribute(
      'aria-label',
      'US price refresh is running. Wait for it to finish before starting a scan.'
    );
  });

  it('disables scan creation with a hover warning when fundamentals refresh is active for the selected market', async () => {
    const user = userEvent.setup();
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      ...DEFAULT_SCAN_DEFAULTS,
      universe: 'market:us',
    };
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {},
        summary: { active_market_count: 1, active_markets: ['US'], status: 'active' },
        markets: [
          {
            market: 'US',
            stage_key: 'fundamentals',
            stage_label: 'Fundamentals Refresh',
            status: 'queued',
            lifecycle: 'bootstrap',
            progress_mode: 'indeterminate',
            percent: null,
            current: null,
            total: null,
            message: 'Queued fundamentals refresh',
          },
        ],
      },
    });

    renderWithProviders(<ScanPage />);

    const scanButton = await screen.findByRole('button', { name: 'Scan' });
    expect(scanButton).toBeDisabled();

    await user.hover(scanButton.parentElement);

    expect(
      await screen.findByText('US fundamentals refresh is queued. Wait for it to finish before starting a scan.')
    ).toBeInTheDocument();
  });

  it('shows the backend market-refresh conflict message if a scan request loses the polling race', async () => {
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      ...DEFAULT_SCAN_DEFAULTS,
      universe: 'market:us',
    };
    scanApi.createScan.mockRejectedValueOnce({
      response: {
        status: 409,
        data: {
          detail: {
            code: 'market_refresh_active',
            message: 'US fundamentals refresh is running. Wait for it to finish before starting a scan.',
          },
        },
      },
      message: 'Request failed with status code 409',
    });

    renderWithProviders(<ScanPage />);

    const scanButton = await screen.findByRole('button', { name: 'Scan' });
    fireEvent.click(scanButton);

    expect(
      await screen.findByText('Error: US fundamentals refresh is running. Wait for it to finish before starting a scan.')
    ).toBeInTheDocument();
  });

  it('uses runtime capability listing-tier options when starting a scan', async () => {
    const user = userEvent.setup();
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      ...DEFAULT_SCAN_DEFAULTS,
      universe: 'market:hk',
    };
    runtimeState.universeOptions = {
      markets: [
        {
          code: 'HK',
          label: 'Hong Kong',
          enabled: true,
          market: {
            value: 'market:HK',
            label: 'All Hong Kong',
            universe_def: { type: 'market', market: 'HK' },
          },
          mics: [
            {
              value: 'market:HK:mic:XHKG',
              label: 'XHKG',
              mic: 'XHKG',
              aliases: ['HKEX'],
              universe_def: { type: 'market', market: 'HK', mic: 'XHKG' },
            },
          ],
          indexes: [],
          listing_tiers: [
            {
              value: 'market:HK:mic:XHKG:tier:main_board',
              label: 'Main Board',
              key: 'main_board',
              mic: 'XHKG',
              aliases: [],
              universe_def: {
                type: 'market',
                market: 'HK',
                mic: 'XHKG',
                listing_tier: 'main_board',
              },
            },
          ],
        },
      ],
    };

    renderWithProviders(<ScanPage />);

    await user.click(await screen.findByRole('combobox', { name: 'Universe' }));
    await user.click(await screen.findByRole('option', { name: 'Main Board' }));
    await user.click(screen.getByRole('button', { name: 'Scan' }));

    await waitFor(() => {
      expect(scanApi.createScan).toHaveBeenCalled();
      expect(scanApi.createScan.mock.calls[0][0]).toEqual(
        expect.objectContaining({
          universe_def: {
            type: 'market',
            market: 'HK',
            mic: 'XHKG',
            listing_tier: 'main_board',
          },
        })
      );
    });
  });

  it('disables non-enabled markets from runtime capability options', async () => {
    const user = userEvent.setup();
    runtimeState.runtimeReady = true;
    runtimeState.universeOptions = {
      markets: [
        {
          code: 'US',
          label: 'United States',
          enabled: true,
          market: {
            value: 'market:US',
            label: 'All United States',
            universe_def: { type: 'market', market: 'US' },
          },
          mics: [],
          indexes: [],
          listing_tiers: [],
        },
        {
          code: 'HK',
          label: 'Hong Kong',
          enabled: false,
          market: {
            value: 'market:HK',
            label: 'All Hong Kong',
            universe_def: { type: 'market', market: 'HK' },
          },
          mics: [],
          indexes: [],
          listing_tiers: [],
        },
      ],
    };

    renderWithProviders(<ScanPage />);

    await user.click(await screen.findByRole('combobox', { name: 'Market' }));

    expect(await screen.findByRole('option', { name: 'United States' })).not.toHaveAttribute(
      'aria-disabled',
      'true'
    );
    expect(await screen.findByRole('option', { name: /Hong Kong/ })).toHaveAttribute(
      'aria-disabled',
      'true'
    );
  });

  it('lets the user refresh stale market data from a scan failure', async () => {
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      ...DEFAULT_SCAN_DEFAULTS,
      universe: 'market:hk',
    };
    scanApi.createScan.mockRejectedValueOnce({
      response: {
        status: 409,
        data: {
          detail: {
            code: 'market_data_stale',
            message: 'Price data is stale for HK.',
            stale_markets: [
              {
                market: 'HK',
                total_symbols: 10,
                covered_symbols: 9,
                uncovered_symbols: 1,
                oldest_last_cached_date: '2026-04-22',
                expected_date: '2026-04-24',
              },
            ],
          },
        },
      },
      message: 'Request failed with status code 409',
    });
    scanApi.refreshScanCache.mockResolvedValueOnce({
      status: 'queued',
      task_id: 'refresh-hk',
      message: 'Smart refresh started',
    });

    renderWithProviders(<ScanPage />);

    const scanButton = await screen.findByRole('button', { name: 'Scan' });
    fireEvent.click(scanButton);

    const refreshButton = await screen.findByRole('button', { name: /refresh hk data/i });
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(scanApi.refreshScanCache).toHaveBeenCalledWith({ market: 'HK', mode: 'full' });
    });
  });

  it('renders structured refresh failure details without crashing', async () => {
    runtimeState.runtimeReady = true;
    runtimeState.scanDefaults = {
      ...DEFAULT_SCAN_DEFAULTS,
      universe: 'market:hk',
    };
    scanApi.createScan.mockRejectedValueOnce({
      response: {
        status: 409,
        data: {
          detail: {
            code: 'market_data_stale',
            message: 'Price data is stale for HK.',
            stale_markets: [{ market: 'HK' }],
          },
        },
      },
      message: 'Request failed with status code 409',
    });
    scanApi.refreshScanCache.mockRejectedValueOnce({
      response: {
        status: 400,
        data: {
          detail: {
            code: 'refresh_rejected',
            message: 'Refresh blocked for HK.',
          },
        },
      },
      message: 'Request failed with status code 400',
    });

    renderWithProviders(<ScanPage />);

    const scanButton = await screen.findByRole('button', { name: 'Scan' });
    fireEvent.click(scanButton);

    const refreshButton = await screen.findByRole('button', { name: /refresh hk data/i });
    fireEvent.click(refreshButton);

    expect(await screen.findByText('Error: Refresh blocked for HK.')).toBeInTheDocument();
  });

  describe('last-published data (#492)', () => {
    const US_PRICE_REFRESH = {
      data: {
        bootstrap: {},
        summary: { active_market_count: 1, active_markets: ['US'], status: 'active' },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            lifecycle: 'daily_refresh',
            progress_mode: 'determinate',
            percent: 30,
            current: 300,
            total: 1000,
            message: 'Refreshing prices',
          },
        ],
      },
    };
    const OLD_SOURCE = {
      data_mode: 'last_published',
      as_of_date: '2026-10-01',
      expected_session: '2026-10-02',
      is_current: false,
      feature_run_id: 9,
    };
    const NVDA_PAGE = {
      total: 1,
      results: [{ symbol: 'NVDA', company_name: 'NVIDIA', composite_score: 98, current_price: 900, stage: 2 }],
    };

    beforeEach(() => {
      runtimeState.runtimeReady = true;
      runtimeState.scanDefaults = { ...DEFAULT_SCAN_DEFAULTS, universe: 'market:us' };
    });

    it('offers last-published data while a refresh blocks scanning and shows its age', async () => {
      useRuntimeActivityMock.mockReturnValue(US_PRICE_REFRESH);
      scanApi.createScan.mockResolvedValueOnce({
        scan_id: 'snap-1',
        status: 'completed',
        total_stocks: 1,
        published_source: OLD_SOURCE,
      });
      scanApi.getScanStatus.mockResolvedValue({ status: 'completed', published_source: OLD_SOURCE });
      scanApi.queryScanResults.mockResolvedValue(NVDA_PAGE);

      renderWithProviders(<ScanPage />);

      expect(await screen.findByRole('button', { name: 'Scan' })).toBeDisabled();
      fireEvent.click(await screen.findByRole('button', { name: 'Use last published data' }));

      await waitFor(() => expect(scanApi.createScan).toHaveBeenCalledTimes(1));
      expect(scanApi.createScan.mock.calls[0][0]).toEqual(
        expect.objectContaining({ data_mode: 'last_published' }),
      );
      expect(
        await screen.findByText(
          'Last published data as of 2026-10-01. When this scan was created, the latest completed session was 2026-10-02.',
        ),
      ).toBeInTheDocument();
      await waitFor(
        () => expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument(),
        { timeout: 3000 },
      );
    });

    it('offers last-published data when the backend reports another active scan', async () => {
      scanApi.createScan.mockRejectedValueOnce({
        response: {
          status: 409,
          data: { detail: { code: 'scan_already_active', message: 'Another scan is already queued or running.' } },
        },
      });

      renderWithProviders(<ScanPage />);

      expect(screen.queryByRole('button', { name: 'Use last published data' })).not.toBeInTheDocument();
      fireEvent.click(await screen.findByRole('button', { name: 'Scan' }));

      expect(await screen.findByRole('button', { name: 'Use last published data' })).toBeEnabled();
    });

    it('shows the source of a reloaded snapshot scan from history', async () => {
      scanApi.getScans.mockResolvedValue({
        scans: [{ scan_id: 'snap-old', status: 'completed', published_source: OLD_SOURCE }],
      });
      scanApi.queryScanResults.mockResolvedValue(NVDA_PAGE);

      renderWithProviders(<ScanPage />);

      expect(
        await screen.findByText(/^Last published data as of 2026-10-01/),
      ).toBeInTheDocument();
    });

    it('keeps the source notice with the snapshot scan after a later rejected request', async () => {
      window.history.replaceState(null, '', '/scan?symbols=NVDA');
      const activeConflict = {
        response: {
          status: 409,
          data: { detail: { code: 'scan_already_active', message: 'Another scan is already queued or running.' } },
        },
      };
      scanApi.createScan
        .mockRejectedValueOnce(activeConflict)
        .mockResolvedValueOnce({ scan_id: 'snap-social', status: 'completed', published_source: OLD_SOURCE })
        .mockRejectedValueOnce(activeConflict);
      scanApi.queryScanResults.mockResolvedValue(NVDA_PAGE);

      renderWithProviders(<ScanPage />);

      fireEvent.click(await screen.findByRole('button', { name: 'Scan' }));
      fireEvent.click(await screen.findByRole('button', { name: 'Use last published data' }));
      expect(await screen.findByText(/^Last published data as of 2026-10-01/)).toBeInTheDocument();

      fireEvent.click(screen.getByRole('button', { name: 'Scan' }));
      await waitFor(() => expect(scanApi.createScan).toHaveBeenCalledTimes(3));
      await screen.findByText('Error: Another scan is already queued or running.');

      expect(screen.getByText(/^Last published data as of 2026-10-01/)).toBeInTheDocument();
    });

    it('stops offering last-published data once the backend says none qualifies', async () => {
      useRuntimeActivityMock.mockReturnValue(US_PRICE_REFRESH);
      scanApi.createScan.mockRejectedValueOnce({
        response: {
          status: 409,
          data: {
            detail: {
              code: 'snapshot_unavailable',
              reason: 'incomplete_coverage',
              message: 'The published snapshot is missing rows for some requested symbols.',
            },
          },
        },
      });

      renderWithProviders(<ScanPage />);

      fireEvent.click(await screen.findByRole('button', { name: 'Use last published data' }));

      expect(
        await screen.findByText('Error: The published snapshot is missing rows for some requested symbols.'),
      ).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Use last published data' })).not.toBeInTheDocument();
    });

    it('ignores a scan creation response after the user picked another scan', async () => {
      const user = userEvent.setup();
      scanApi.getScans.mockResolvedValue({
        scans: [
          { scan_id: 'scan-a', status: 'completed' },
          { scan_id: 'scan-b', status: 'completed' },
        ],
      });
      scanApi.queryScanResults.mockResolvedValue(NVDA_PAGE);
      let resolveCreate;
      scanApi.createScan.mockReturnValueOnce(new Promise((resolve) => { resolveCreate = resolve; }));

      renderWithProviders(<ScanPage />);

      await waitFor(
        () => expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument(),
        { timeout: 3000 },
      );
      fireEvent.click(screen.getByRole('button', { name: 'Scan' }));
      await user.click(screen.getByRole('combobox', { name: 'Previous Scans' }));
      const options = await screen.findAllByRole('option');
      await user.click(options.at(-1));
      await waitFor(() => expect(scanApi.getScanStatus).toHaveBeenCalledWith('scan-b'));

      await act(async () => {
        resolveCreate({ scan_id: 'scan-late', status: 'queued', total_stocks: 10 });
      });

      expect(scanApi.getScanStatus).not.toHaveBeenCalledWith('scan-late');
      expect(screen.queryByText('Showing your previous results until the new scan finishes.')).not.toBeInTheDocument();
    });

    it('drops the retained results when the global market changes', async () => {
      marketState.selectedMarket = 'US';
      scanApi.getScans.mockImplementation(async ({ market } = {}) => (
        market === 'US' ? { scans: [{ scan_id: 'us-done', status: 'completed' }] } : { scans: [] }
      ));
      scanApi.queryScanResults.mockResolvedValue(NVDA_PAGE);
      scanApi.createScan.mockResolvedValueOnce({ scan_id: 'scan-new', status: 'queued', total_stocks: 500 });
      scanApi.getScanStatus.mockImplementation(async (scanId) => (
        scanId === 'scan-new'
          ? { scan_id: 'scan-new', status: 'running', progress: 10, total_stocks: 500, completed_stocks: 50 }
          : { status: 'completed' }
      ));

      const { rerender } = renderWithProviders(<ScanPage />);

      await waitFor(
        () => expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument(),
        { timeout: 3000 },
      );
      fireEvent.click(screen.getByRole('button', { name: 'Scan' }));
      await screen.findByText('Showing your previous results until the new scan finishes.');

      marketState.selectedMarket = 'HK';
      rerender(<ScanPage />);

      await waitFor(() => {
        expect(screen.queryByText('Showing your previous results until the new scan finishes.')).not.toBeInTheDocument();
      });
      expect(screen.queryByText(/Results:\s*1 stocks/i)).not.toBeInTheDocument();
    });

    it('keeps the previous completed results visible while a new scan runs', async () => {
      scanApi.getScans.mockResolvedValue({
        scans: [{ scan_id: 'scan-done', status: 'completed' }],
      });
      scanApi.queryScanResults.mockResolvedValue(NVDA_PAGE);
      scanApi.createScan.mockResolvedValueOnce({ scan_id: 'scan-new', status: 'queued', total_stocks: 500 });
      scanApi.getScanStatus.mockImplementation(async (scanId) => (
        scanId === 'scan-new'
          ? { scan_id: 'scan-new', status: 'running', progress: 10, total_stocks: 500, completed_stocks: 50 }
          : { status: 'completed' }
      ));

      renderWithProviders(<ScanPage />);

      await waitFor(
        () => expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument(),
        { timeout: 3000 },
      );
      fireEvent.click(screen.getByRole('button', { name: 'Scan' }));

      expect(
        await screen.findByText('Showing your previous results until the new scan finishes.'),
      ).toBeInTheDocument();
      expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument();
      expect(scanApi.queryScanResults).not.toHaveBeenCalledWith('scan-new', expect.anything(), expect.anything());
    });
  });
});
