import { screen } from '@testing-library/react';
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
        supportedMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('Price Refresh')).toBeInTheDocument();
    expect(screen.getByText('42%')).toBeInTheDocument();
    expect(screen.getByText('420 / 1,000 stocks')).toBeInTheDocument();
    expect(screen.getAllByText(/Refreshing prices/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Additional data loading continues in the background/)).toBeInTheDocument();
    expect(screen.getByText('Enabled market queue')).toBeInTheDocument();
    expect(screen.getAllByText('US (primary)').length).toBeGreaterThan(0);
    expect(screen.getAllByText('HK').length).toBeGreaterThan(0);
    expect(screen.getByRole('progressbar', { name: 'Bootstrap progress' })).toBeInTheDocument();
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
        supportedMarkets={['US', 'HK', 'JP', 'TW']}
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

  it('renders determinate fundamentals progress from the primary market activity row', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Fundamentals Refresh',
          progress_mode: 'determinate',
          percent: 40,
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
        supportedMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('75%')).toBeInTheDocument();
    expect(screen.getByText('750 / 1,000 stocks')).toBeInTheDocument();
    expect(screen.getByText('Fundamentals Refresh')).toBeInTheDocument();
  });

  it('derives stage-local bootstrap percent from counts when percent is absent', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Price Refresh',
          progress_mode: 'determinate',
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
        supportedMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('55%')).toBeInTheDocument();
    expect(screen.getByText('550 / 1,000 stocks')).toBeInTheDocument();
    expect(screen.getAllByText(/Batch 2\/4 · refreshing prices/).length).toBeGreaterThan(0);
  });

  it('derives stage-local progress from counts even when progress_mode is stale', () => {
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
        supportedMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('55%')).toBeInTheDocument();
    expect(screen.getByText('550 / 1,000 stocks')).toBeInTheDocument();
  });

  it('uses stage_key instead of the display label to unlock primary-market stage progress', () => {
    useRuntimeActivityMock.mockReturnValue({
      data: {
        bootstrap: {
          primary_market: 'US',
          current_stage: 'Refreshing market data',
          progress_mode: 'determinate',
          percent: 30,
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
        supportedMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('55%')).toBeInTheDocument();
    expect(screen.getByText('550 / 1,000 stocks')).toBeInTheDocument();
    expect(screen.getAllByText(/Batch 2\/4 · refreshing prices/).length).toBeGreaterThan(0);
  });

  it('prefers the primary market activity message over the bootstrap summary when stage-local progress is active', () => {
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
        supportedMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getAllByText(/Batch 2\/4 · refreshing prices/).length).toBeGreaterThan(0);
    expect(screen.queryByText('Preparing bootstrap')).not.toBeInTheDocument();
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
        supportedMarkets={['US', 'HK', 'JP', 'TW']}
        bootstrapState="running"
        isStartingBootstrap={false}
        bootstrapError={null}
        onStartBootstrap={vi.fn()}
      />
    );

    expect(screen.getByText('55%')).toBeInTheDocument();
    expect(screen.queryByText('25%')).not.toBeInTheDocument();
  });
});
