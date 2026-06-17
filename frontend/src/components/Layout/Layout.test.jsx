import { MemoryRouter } from 'react-router-dom';
import { screen } from '@testing-library/react';
import { act } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import Layout from './Layout';
import { renderWithProviders } from '../../test/renderWithProviders';

const runtimeState = {
  auth: { required: false, authenticated: true },
  features: { chatbot: false, themes: false, tasks: false },
  isLoggingOut: false,
  logout: vi.fn(),
};

const strategyState = {
  activeProfile: 'default',
  setActiveProfile: vi.fn(),
};
const useRuntimeActivityMock = vi.hoisted(() => vi.fn());

vi.mock('../../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../../contexts/StrategyProfileContext', () => ({
  useStrategyProfile: () => strategyState,
  useStrategyProfileData: () => ({
    ...strategyState,
    activeProfileDetail: { label: 'Default' },
    profiles: [{ profile: 'default', label: 'Default' }],
    isLoadingProfiles: false,
  }),
}));

vi.mock('../../hooks/useRuntimeActivity', () => ({
  useRuntimeActivity: (...args) => useRuntimeActivityMock(...args),
}));

describe('Layout', () => {
  beforeEach(() => {
    useRuntimeActivityMock.mockReset();
  });

  it('defers the header runtime activity request until after initial paint', () => {
    vi.useFakeTimers();
    useRuntimeActivityMock.mockReturnValue({ data: null });

    try {
      renderWithProviders(
        <MemoryRouter initialEntries={['/']}>
          <Layout>
            <div>content</div>
          </Layout>
        </MemoryRouter>
      );

      expect(useRuntimeActivityMock).toHaveBeenLastCalledWith({ enabled: false });
      expect(screen.getByText('Checking activity')).toBeInTheDocument();
      expect(screen.getByText('View operations')).toBeInTheDocument();
      expect(screen.queryByText('Markets ready')).not.toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(1500);
      });

      expect(useRuntimeActivityMock).toHaveBeenLastCalledWith({ enabled: true });
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows the runtime activity header summary and removes Digest navigation', () => {
    useRuntimeActivityMock.mockReturnValue({
      dataUpdatedAt: 1,
      data: {
        bootstrap: { state: 'ready' },
        summary: { active_market_count: 1, status: 'active' },
        markets: [
          {
            market: 'HK',
            status: 'running',
            stage_label: 'Fundamentals Refresh',
          },
        ],
      },
    });

    renderWithProviders(
      <MemoryRouter initialEntries={['/scan']}>
        <Layout>
          <div>content</div>
        </Layout>
      </MemoryRouter>
    );

    expect(screen.getByText('1 market active')).toBeInTheDocument();
    expect(screen.getByText('HK · Fundamentals Refresh')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /digest/i })).not.toBeInTheDocument();
  });

  it('does not show a fake 0 percent for indeterminate bootstrap progress', () => {
    useRuntimeActivityMock.mockReturnValue({
      dataUpdatedAt: 1,
      data: {
        bootstrap: {
          state: 'running',
          primary_market: 'US',
          current_stage: 'Universe Refresh',
          progress_mode: 'indeterminate',
          percent: null,
          message: 'Refreshing official market universe',
        },
        summary: { active_market_count: 1, status: 'active' },
        markets: [
          {
            market: 'US',
            status: 'running',
            stage_label: 'Universe Refresh',
            progress_mode: 'indeterminate',
            percent: null,
          },
        ],
      },
    });

    renderWithProviders(
      <MemoryRouter initialEntries={['/scan']}>
        <Layout>
          <div>content</div>
        </Layout>
      </MemoryRouter>
    );

    expect(screen.getByText('Bootstrapping US')).toBeInTheDocument();
    expect(screen.getByText('Universe Refresh')).toBeInTheDocument();
    expect(screen.queryByText('0%')).not.toBeInTheDocument();
  });
});
