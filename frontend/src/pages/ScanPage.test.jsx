import { createTheme, ThemeProvider } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import ScanPage from './ScanPage';
import { DEFAULT_SCAN_DEFAULTS } from '../constants/scanDefaults';
import * as scanApi from '../api/scans';

const runtimeState = {
  runtimeReady: false,
  uiSnapshots: {
    scan: false,
  },
  scanDefaults: DEFAULT_SCAN_DEFAULTS,
};

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../contexts/StrategyProfileContext', () => ({
  useStrategyProfile: () => ({
    activeProfileDetail: null,
  }),
}));

vi.mock('../hooks/useFilterPresets', () => ({
  useFilterPresets: () => ({
    presets: [],
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
  getScanResults: vi.fn(),
  getUniverseStats: vi.fn().mockResolvedValue({
    active: 321,
    sp500: 500,
    by_exchange: {
      NYSE: 100,
      NASDAQ: 200,
      AMEX: 21,
    },
  }),
  exportScanResults: vi.fn(),
  getScans: vi.fn().mockResolvedValue({ scans: [] }),
  cancelScan: vi.fn(),
  getFilterOptions: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
  runtimeState.runtimeReady = false;
  runtimeState.uiSnapshots = { scan: false };
  runtimeState.scanDefaults = DEFAULT_SCAN_DEFAULTS;
  scanApi.getScanBootstrap.mockResolvedValue(null);
  scanApi.getScanStatus.mockResolvedValue({ status: 'completed' });
  scanApi.getScanResults.mockResolvedValue({ total: 0, results: [] });
  scanApi.getFilterOptions.mockResolvedValue({
    ibd_industries: [],
    gics_sectors: [],
    ratings: [],
  });
  scanApi.getScans.mockResolvedValue({ scans: [] });
});

describe('ScanPage', () => {
  it('renders without a temporal-dead-zone crash before runtime bootstrap completes', () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={createTheme()}>
          <ScanPage />
        </ThemeProvider>
      </QueryClientProvider>
    );

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

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={createTheme()}>
          <ScanPage />
        </ThemeProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByDisplayValue('123')).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByRole('checkbox', { name: /vcp/i })).not.toBeChecked();
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

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={createTheme()}>
          <ScanPage />
        </ThemeProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/Results:\s*1 stocks/i)).toBeInTheDocument();
    });
    expect(screen.getByText('Filters')).toBeInTheDocument();
  });
});
