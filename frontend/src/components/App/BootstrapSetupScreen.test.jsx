import { fireEvent, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import BootstrapSetupScreen from './BootstrapSetupScreen';
import { renderWithProviders } from '../../test/renderWithProviders';

const useRuntimeActivityMock = vi.hoisted(() => vi.fn());

vi.mock('../../hooks/useRuntimeActivity', () => ({
  useRuntimeActivity: (...args) => useRuntimeActivityMock(...args),
}));

describe('BootstrapSetupScreen', () => {
  beforeEach(() => {
    useRuntimeActivityMock.mockReset();
  });

  it('renders bootstrap progress and background-loading warning while running', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Price Refresh',
          progress_mode: 'determinate',
          percent: 25,
          current: 250,
          total: 1000,
          message: 'Refreshing prices',
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            progress_mode: 'determinate',
            percent: 42,
            current: 420,
            total: 1000,
            message: 'Refreshing prices',
          },
          {
            market: 'HK',
            stage_label: 'Universe Refresh',
            status: 'queued',
            progress_mode: 'indeterminate',
            message: 'Queued for background bootstrap',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US', 'HK']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('Price Refresh')).toBeInTheDocument();
    expect(screen.getByText('25%')).toBeInTheDocument();
    expect(screen.getByText('250 / 1,000 stocks')).toBeInTheDocument();
    expect(screen.getAllByText(/Refreshing prices/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Additional data loading continues in the background/)).toBeInTheDocument();
    expect(screen.getByText('Enabled market queue')).toBeInTheDocument();
    expect(screen.getAllByText('US (primary)').length).toBeGreaterThan(0);
    expect(screen.getAllByText('HK').length).toBeGreaterThan(0);
    expect(screen.getByRole('progressbar', { name: 'Bootstrap progress' })).toBeInTheDocument();
  });

  it('does not invent a background warning when the API omits one', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Price Refresh',
          progress_mode: 'determinate',
          percent: 25,
          message: 'Refreshing prices',
          background_warning: null,
        },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            progress_mode: 'determinate',
            percent: 25,
            current: 250,
            total: 1000,
            message: 'Refreshing prices',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.queryByText(/Additional enabled markets/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/continue loading after the workspace opens/i)).not.toBeInTheDocument();
  });

  it('renders indeterminate bootstrap progress when no real percent is available yet', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Fundamentals Refresh',
          progress_mode: 'indeterminate',
          percent: null,
          message: 'Refreshing fundamentals',
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'fundamentals',
            stage_label: 'Fundamentals Refresh',
            status: 'running',
            progress_mode: 'indeterminate',
            percent: null,
            message: 'Refreshing fundamentals',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    const progressBar = screen.getByRole('progressbar', { name: 'Bootstrap progress' });
    expect(progressBar).not.toHaveAttribute('aria-valuenow');
    expect(screen.getByText('Fundamentals Refresh')).toBeInTheDocument();
    expect(screen.getByText('Refreshing fundamentals')).toBeInTheDocument();
    expect(screen.queryByText('0%')).not.toBeInTheDocument();
  });

  it('renders determinate fundamentals progress from the bootstrap summary', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Fundamentals Refresh',
          progress_mode: 'determinate',
          percent: 40,
          current: 400,
          total: 1000,
          message: 'Refreshing fundamentals',
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'fundamentals',
            stage_label: 'Fundamentals Refresh',
            status: 'running',
            progress_mode: 'determinate',
            percent: 75,
            current: 750,
            total: 1000,
            message: 'Refreshing fundamentals',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('40%')).toBeInTheDocument();
    expect(screen.getByText('400 / 1,000 stocks')).toBeInTheDocument();
    expect(screen.getByText('Fundamentals Refresh')).toBeInTheDocument();
  });

  it('does not derive bootstrap summary percent from counts when percent is absent', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Price Refresh',
          progress_mode: 'determinate',
          percent: null,
          current: 250,
          total: 1000,
          message: 'Refreshing prices',
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            progress_mode: 'determinate',
            percent: null,
            current: 550,
            total: 1000,
            message: 'Batch 2/4 · refreshing prices',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.queryByText('25%')).not.toBeInTheDocument();
    expect(screen.getByText('250 / 1,000 stocks')).toBeInTheDocument();
    expect(screen.getAllByText(/Batch 2\/4 · refreshing prices/).length).toBeGreaterThan(0);
  });

  it('does not override backend indeterminate progress with market row counts', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Price Refresh',
          progress_mode: 'indeterminate',
          percent: null,
          message: 'Refreshing prices',
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            progress_mode: 'indeterminate',
            percent: null,
            current: 550,
            total: 1000,
            message: 'Batch 2/4 · refreshing prices',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.queryByText('55%')).not.toBeInTheDocument();
    expect(screen.queryByText('550 / 1,000 stocks')).not.toBeInTheDocument();
  });

  it('keeps bootstrap progress independent from market-row stage labels', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Refreshing market data',
          progress_mode: 'determinate',
          percent: 30,
          current: 300,
          total: 1000,
          message: 'Batch 2/4 · refreshing prices',
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Refreshing market data',
            status: 'running',
            progress_mode: 'determinate',
            percent: 55,
            current: 550,
            total: 1000,
            message: 'Batch 2/4 · refreshing prices',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('30%')).toBeInTheDocument();
    expect(screen.getByText('300 / 1,000 stocks')).toBeInTheDocument();
    expect(screen.getAllByText(/Batch 2\/4 · refreshing prices/).length).toBeGreaterThan(0);
  });

  it('uses the bootstrap summary message instead of replacing it from market rows', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Price Refresh',
          progress_mode: 'determinate',
          percent: 30,
          message: 'Preparing bootstrap',
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            progress_mode: 'determinate',
            percent: 55,
            current: 550,
            total: 1000,
            message: 'Batch 2/4 · refreshing prices',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('Preparing bootstrap')).toBeInTheDocument();
  });

  it('keeps bootstrap percent sourced from a complete tuple', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Price Refresh',
          progress_mode: 'determinate',
          percent: null,
          current: 25,
          total: null,
          message: 'Refreshing prices',
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            progress_mode: 'determinate',
            percent: null,
            current: 550,
            total: 1000,
            message: 'Batch 2/4 · refreshing prices',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.queryByText('55%')).not.toBeInTheDocument();
    expect(screen.queryByText('25%')).not.toBeInTheDocument();
    expect(screen.queryByText('0%')).not.toBeInTheDocument();
  });

  it('uses backend stage-weighted secondary bootstrap progress after the primary market completes', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Price Refresh',
          progress_mode: 'determinate',
          percent: 22,
          current: 1200,
          total: 3750,
          message: 'Refreshing market prices',
          background_warning: 'Additional enabled markets are still loading in the background.',
        },
        markets: [
          {
            market: 'US',
            stage_key: 'scan',
            stage_label: 'Scan',
            status: 'completed',
            progress_mode: 'determinate',
            percent: 100,
            current: 1000,
            total: 1000,
            message: 'Primary bootstrap complete',
          },
          {
            market: 'JP',
            stage_key: 'prices',
            stage_label: 'Price Refresh',
            status: 'running',
            progress_mode: 'determinate',
            percent: 32,
            current: 1200,
            total: 3750,
            message: 'Refreshing market prices',
          },
        ],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US', 'JP']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('22%')).toBeInTheDocument();
    expect(screen.getByText('1,200 / 3,750 stocks')).toBeInTheDocument();
    expect(screen.queryByText('100%')).not.toBeInTheDocument();
    expect(screen.getAllByText(/Refreshing market prices/).length).toBeGreaterThan(0);
  });

  it('does not synthesize queued market rows when runtime activity has no markets', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Preparing bootstrap',
          progress_mode: 'indeterminate',
          percent: null,
          message: 'Bootstrap queued.',
          background_warning: null,
        },
        markets: [],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US', 'HK']}
        deploymentMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.queryByText('Enabled market queue')).not.toBeInTheDocument();
    expect(screen.queryByText(/Waiting for bootstrap task/)).not.toBeInTheDocument();
  });

  it('renders bootstrap order from runtime activity stage metadata', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Preparing bootstrap',
          progress_mode: 'indeterminate',
          percent: null,
          message: 'Bootstrap queued.',
          background_warning: null,
          stages: [
            { key: 'seed', label: 'Seed Import' },
            { key: 'top_up', label: 'Live Top-Up' },
          ],
        },
        markets: [],
      },
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US']}
        bootstrapState="not_started"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('1. Seed Import')).toBeInTheDocument();
    expect(screen.getByText('2. Live Top-Up')).toBeInTheDocument();
    expect(screen.queryByText('1. Universe refresh')).not.toBeInTheDocument();
  });

  it('renders Market Catalog labels while submitting Market codes', () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    const onStartBootstrap = vi.fn().mockResolvedValue();

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="HK"
        enabledMarkets={['HK', 'US']}
        deploymentMarkets={['US', 'HK']}
        marketCatalog={{
          version: 'test.v1',
          markets: [
            {
              code: 'US',
              label: 'United States',
              currency: 'USD',
              timezone: 'America/New_York',
              calendar_id: 'XNYS',
              exchanges: ['NYSE', 'NASDAQ'],
              indexes: ['SP500'],
              capabilities: {},
            },
            {
              code: 'HK',
              label: 'Hong Kong',
              currency: 'HKD',
              timezone: 'Asia/Hong_Kong',
              calendar_id: 'XHKG',
              exchanges: ['HKEX'],
              indexes: ['HSI'],
              capabilities: {},
            },
          ],
        }}
        bootstrapState="not_started"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={onStartBootstrap}
      />
    );

    expect(screen.getAllByText('Hong Kong').length).toBeGreaterThan(0);
    expect(screen.getByText('Hong Kong (primary)')).toBeInTheDocument();
    expect(screen.getByText('United States')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Start bootstrap' }));

    expect(onStartBootstrap).toHaveBeenCalledWith({
      primaryMarket: 'HK',
      enabledMarkets: ['HK', 'US'],
    });
  });

  it('falls back to the Market code when a catalog label is missing', () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['US']}
        marketCatalog={{
          version: 'partial.v1',
          markets: [
            {
              code: 'US',
              label: '',
              currency: 'USD',
              timezone: 'America/New_York',
              calendar_id: 'XNYS',
              exchanges: ['NYSE'],
              indexes: ['SP500'],
              capabilities: {},
            },
          ],
        }}
        bootstrapState="not_started"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('US (primary)')).toBeInTheDocument();
  });

  it("offers only the deployment's markets, in catalog order", () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    const catalogMarket = (code, label) => ({
      code, label, currency: 'X', timezone: 'UTC', calendar_id: code, exchanges: [], indexes: [], capabilities: {},
    });

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US']}
        deploymentMarkets={['HK', 'US']}
        marketCatalog={{
          version: 'test.v1',
          markets: [catalogMarket('US', 'United States'), catalogMarket('HK', 'Hong Kong'), catalogMarket('JP', 'Japan')],
        }}
        bootstrapState="not_started"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getAllByRole('checkbox').map((box) => box.closest('label').textContent))
      .toEqual(['United States (primary)', 'Hong Kong']);
    expect(screen.queryByText('Japan')).not.toBeInTheDocument();
    expect(screen.getByText("This deployment's ENABLED_MARKETS decides which markets are listed."))
      .toBeInTheDocument();
  });

  it('starts from markets the deployment runs when the saved choice is outside it', () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    const onStartBootstrap = vi.fn().mockResolvedValue();

    renderWithProviders(
      <BootstrapSetupScreen
        primaryMarket="US"
        enabledMarkets={['US', 'JP', 'HK']}
        deploymentMarkets={['HK', 'TW']}
        bootstrapState="not_started"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={onStartBootstrap}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: 'Start bootstrap' }));

    expect(onStartBootstrap).toHaveBeenCalledWith({
      primaryMarket: 'HK',
      enabledMarkets: ['HK'],
    });
  });
});
