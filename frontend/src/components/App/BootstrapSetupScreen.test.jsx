import { screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import BootstrapSetupScreen from './BootstrapSetupScreen';
import { renderWithProviders } from '../../test/renderWithProviders';

vi.mock('../../hooks/useRuntimeActivity', () => ({
  useRuntimeActivity: () => ({
    data: {
      bootstrap: {
        primary_market: 'US',
        current_stage: 'Price Refresh',
        percent: 42,
        message: 'Refreshing prices',
        background_warning: 'Additional data loading continues in the background.',
      },
      markets: [
        {
          market: 'US',
          stage_label: 'Price Refresh',
          status: 'running',
          message: 'Refreshing prices',
        },
        {
          market: 'HK',
          stage_label: 'Universe Refresh',
          status: 'queued',
          message: 'Queued for background bootstrap',
        },
      ],
    },
  }),
}));

describe('BootstrapSetupScreen', () => {
  it('renders bootstrap progress and background-loading warning while running', () => {
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
    expect(screen.getAllByText(/Refreshing prices/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Additional data loading continues in the background/)).toBeInTheDocument();
    expect(screen.getByText('Enabled market queue')).toBeInTheDocument();
    expect(screen.getAllByText('US (primary)').length).toBeGreaterThan(0);
    expect(screen.getAllByText('HK').length).toBeGreaterThan(0);
    expect(screen.getByRole('progressbar', { name: 'Bootstrap progress' })).toBeInTheDocument();
  });
});
