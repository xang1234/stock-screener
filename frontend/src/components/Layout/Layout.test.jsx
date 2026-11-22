import { MemoryRouter } from 'react-router-dom';
import { fireEvent, screen, within } from '@testing-library/react';
import { act } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Layout from './Layout';
import { MarketProvider } from '../../contexts/MarketContext';
import { renderWithProviders } from '../../test/renderWithProviders';

const runtimeState = {
  auth: { required: false, authenticated: true },
  features: { chatbot: false, themes: false, tasks: false },
  marketCatalog: { markets: [] },
  primaryMarket: 'US',
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

const loadedPaths = vi.hoisted(() => []);
const api = vi.hoisted(() => ({
  getBreadthBootstrap: vi.fn(),
  getDailySnapshot: vi.fn(),
  getGroupsBootstrap: vi.fn(),
  getScanBootstrap: vi.fn(),
}));

vi.mock('../../pageLoaders', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    // Record the prefetch without running it: a real cold page import runs
    // beside later tests and can outlast their waits on a busy CI runner.
    loaderForPath: (path) => actual.loaderForPath(path) && (() => {
      loadedPaths.push(path);
      return Promise.resolve();
    }),
    loadStockDetails: () => Promise.resolve(),
  };
});
vi.mock('../../pages/BreadthPage', () => ({ default: () => null }));
vi.mock('../../pages/MarketScanPage', () => ({ default: () => null }));
vi.mock('../../pages/GroupRankingsPage', () => ({ default: () => null }));
vi.mock('../../pages/ScanPage', () => ({ default: () => null }));
vi.mock('../../api/scans', async (importOriginal) => ({
  ...(await importOriginal()),
  getScanBootstrap: api.getScanBootstrap,
}));
vi.mock('../../api/breadth', async (importOriginal) => ({
  ...(await importOriginal()),
  getBreadthBootstrap: api.getBreadthBootstrap,
}));
vi.mock('../../api/marketScan', async (importOriginal) => ({
  ...(await importOriginal()),
  getDailySnapshot: api.getDailySnapshot,
}));
vi.mock('../../api/groups', async (importOriginal) => ({
  ...(await importOriginal()),
  getGroupsBootstrap: api.getGroupsBootstrap,
}));

function renderInApp(path = '/scan') {
  return renderWithProviders(
    <MemoryRouter initialEntries={[path]}>
      <MarketProvider>
        <Layout><div>content</div></Layout>
      </MarketProvider>
    </MemoryRouter>,
  );
}

describe('Layout', () => {
  beforeEach(() => {
    useRuntimeActivityMock.mockReset();
    Object.values(api).forEach((mock) => mock.mockReset().mockResolvedValue(null));
    loadedPaths.length = 0;
    window.localStorage.clear();
    runtimeState.features = { chatbot: false, themes: false, tasks: false };
    runtimeState.marketCatalog = { markets: [] };
    runtimeState.primaryMarket = 'US';
    runtimeState.enabledMarkets = undefined;
  });

  it("loads a page's code and snapshot when its nav link is hovered", async () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    runtimeState.enabledMarkets = ['US', 'HK'];
    runtimeState.primaryMarket = 'HK';
    renderInApp();

    fireEvent.mouseEnter(screen.getByRole('link', { name: 'Breadth' }));

    expect(loadedPaths).toEqual(['/breadth']);
    await vi.waitFor(() => expect(api.getBreadthBootstrap).toHaveBeenCalledWith('HK'));
  });

  it('prefetches the scan bootstrap for the selected market from another page', async () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    runtimeState.enabledMarkets = ['US', 'HK'];
    runtimeState.primaryMarket = 'HK';
    renderInApp('/');

    fireEvent.mouseEnter(screen.getByRole('link', { name: 'Scan' }));

    await vi.waitFor(() => expect(api.getScanBootstrap).toHaveBeenCalledWith(null, 'HK'));
  });

  it.each(['/scan', '/scan/', '/SCAN'])("leaves the open page's data to that page at %s", async (openPath) => {
    // The mounted Scan page re-applies any new latest-scan bootstrap, which
    // would replace the scan the user picked or just started. The router
    // renders it for a trailing slash or other casing too.
    useRuntimeActivityMock.mockReturnValue({ data: null });
    renderInApp(openPath);

    fireEvent.mouseEnter(screen.getByRole('link', { name: 'Scan' }));
    await act(async () => {});

    expect(api.getScanBootstrap).not.toHaveBeenCalled();
  });

  it('resolves capability-limited pages to the market they would show', async () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    // SG is selected, but group rankings and breadth fall back to US for it.
    runtimeState.enabledMarkets = ['SG', 'US'];
    runtimeState.primaryMarket = 'SG';
    renderInApp();

    fireEvent.mouseEnter(screen.getByRole('link', { name: 'Breadth' }));
    fireEvent.mouseEnter(screen.getByRole('link', { name: 'Groups' }));
    fireEvent.mouseEnter(screen.getByRole('link', { name: 'Daily' }));

    await vi.waitFor(() => expect(api.getDailySnapshot).toHaveBeenCalledWith('SG'));
    expect(api.getBreadthBootstrap).toHaveBeenCalledWith('US');
    expect(api.getGroupsBootstrap).toHaveBeenCalledWith('US');
  });

  it('does not retry a failed prefetch on every hover', async () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    api.getScanBootstrap.mockRejectedValue(new Error('404'));
    renderInApp('/');
    const scanLink = screen.getByRole('link', { name: 'Scan' });

    fireEvent.mouseEnter(scanLink);
    await vi.waitFor(() => expect(api.getScanBootstrap).toHaveBeenCalledTimes(1));
    await act(async () => {});
    fireEvent.mouseEnter(scanLink);
    await act(async () => {});

    expect(api.getScanBootstrap).toHaveBeenCalledTimes(1);
  });

  it('prefetches the daily snapshot for the selected market on focus', async () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    runtimeState.enabledMarkets = ['US', 'HK'];
    runtimeState.primaryMarket = 'HK';
    renderInApp();

    fireEvent.focus(screen.getByRole('link', { name: 'Daily' }));

    await vi.waitFor(() => expect(api.getDailySnapshot).toHaveBeenCalledWith('HK'));
  });

  it('seeds the same caches as the page when it prefetches a groups snapshot', async () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    const rankings = { date: '2026-10-09', rankings: [] };
    api.getGroupsBootstrap.mockResolvedValue({
      is_stale: false,
      payload: { rankings, movers: { gainers: [] } },
    });
    const { queryClient } = renderInApp();

    fireEvent.mouseEnter(screen.getByRole('link', { name: 'Groups' }));

    await vi.waitFor(() => (
      expect(queryClient.getQueryData(['groupRankings', 'US', '2026-10-09'])).toEqual(rankings)
    ));
    expect(queryClient.getQueryData(['groupMovers', '1w', 'US', '2026-10-09'])).toEqual({ gainers: [] });
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

  it('shows stale runtime activity as a header warning', () => {
    useRuntimeActivityMock.mockReturnValue({
      dataUpdatedAt: 1,
      data: {
        bootstrap: { state: 'ready' },
        summary: { active_market_count: 0, status: 'warning' },
        markets: [
          {
            market: 'US',
            status: 'stale',
            stage_label: 'Price Refresh',
            message: 'Refreshing market prices - stale: No live data-fetch lock owns task old-task.',
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

    expect(screen.getByText('Refresh warning')).toBeInTheDocument();
    expect(screen.getByText('US · Price Refresh')).toBeInTheDocument();
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

  it('shows Options only for an enabled US market capability and keeps detail active', () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    runtimeState.features = { ...runtimeState.features, options_analytics: true };
    runtimeState.marketCatalog = {
      markets: [{ code: 'US', capabilities: { options_analytics: true } }],
    };

    const { rerender } = renderWithProviders(
      <MemoryRouter initialEntries={['/options/AAPL']}>
        <Layout><div>content</div></Layout>
      </MemoryRouter>,
    );
    expect(screen.getByRole('link', { name: 'Options' })).toHaveAttribute('aria-current', 'page');

    runtimeState.primaryMarket = 'HK';
    rerender(
      <MemoryRouter initialEntries={['/options/AAPL']}>
        <Layout><div>content</div></Layout>
      </MemoryRouter>,
    );
    expect(screen.queryByRole('link', { name: 'Options' })).not.toBeInTheDocument();
  });

  it('has no scheduled-tasks gear: scheduled jobs live on the Operations page', () => {
    useRuntimeActivityMock.mockReturnValue({ data: null });
    runtimeState.features = { ...runtimeState.features, tasks: true };
    renderInApp('/scan');

    expect(screen.queryByRole('link', { name: /scheduled tasks/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /scheduled tasks/i })).not.toBeInTheDocument();
    expect(screen.queryByTitle(/scheduled tasks/i)).not.toBeInTheDocument();
  });

  describe('on a phone-width screen', () => {
    beforeEach(() => {
      // useMediaQuery(breakpoints.down('md')) matches.
      window.matchMedia = vi.fn().mockReturnValue({
        matches: true,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      });
      useRuntimeActivityMock.mockReturnValue({
        dataUpdatedAt: 1,
        data: { bootstrap: { state: 'ready' }, summary: { active_market_count: 0 }, markets: [] },
      });
    });

    afterEach(() => {
      delete window.matchMedia;
    });

    const renderAt = (path) => renderWithProviders(
      <MemoryRouter initialEntries={[path]}>
        <Layout><div>content</div></Layout>
      </MemoryRouter>,
    );

    it('moves navigation into a bottom bar with a More sheet', () => {
      renderAt('/scan');

      const nav = screen.getByRole('navigation', { name: 'Main' });
      expect(within(nav).getAllByRole('link').map((link) => link.textContent))
        .toEqual(['Daily', 'Scan', 'Breadth', 'Groups']);
      expect(within(nav).getByRole('link', { name: 'Scan' })).toHaveAttribute('aria-current', 'page');
      expect(screen.queryByRole('link', { name: 'Backtest' })).not.toBeInTheDocument();

      fireEvent.click(within(nav).getByRole('button', { name: 'More' }));
      const sheet = screen.getByRole('list', { name: 'More' });
      expect(within(sheet).getByRole('link', { name: 'Backtest' })).toBeInTheDocument();
      expect(within(sheet).getByRole('button', { name: 'Default' })).toBeInTheDocument();
      expect(within(sheet).getByRole('button', { name: /mode$/ })).toBeInTheDocument();
    });

    it('highlights More when the current page lives in the sheet', () => {
      renderAt('/validation');

      const nav = screen.getByRole('navigation', { name: 'Main' });
      expect(within(nav).getByRole('button', { name: 'More' })).toHaveClass('Mui-selected');
    });

    it('collapses the runtime status to its badge but keeps the full label', () => {
      renderAt('/');

      expect(screen.queryByText('Markets ready')).not.toBeInTheDocument();
      expect(screen.getByRole('link', { name: 'Markets ready: View operations' })).toBeInTheDocument();
    });

    it('prefetches a page when its bottom-bar item is touched', async () => {
      renderInApp('/');

      const nav = screen.getByRole('navigation', { name: 'Main' });
      fireEvent.touchStart(within(nav).getByRole('link', { name: 'Breadth' }));

      expect(loadedPaths).toEqual(['/breadth']);
      await vi.waitFor(() => expect(api.getBreadthBootstrap).toHaveBeenCalledWith('US'));
    });

    it('has no Scheduled tasks item in the More sheet', () => {
      runtimeState.features = { ...runtimeState.features, tasks: true };
      renderInApp('/');

      fireEvent.click(within(screen.getByRole('navigation', { name: 'Main' })).getByRole('button', { name: 'More' }));

      expect(screen.getByRole('list', { name: 'More' })).toBeInTheDocument();
      expect(screen.queryByText('Scheduled tasks')).not.toBeInTheDocument();
    });

    it('loads a More-sheet page when its link is touched', () => {
      renderInApp('/');

      fireEvent.click(within(screen.getByRole('navigation', { name: 'Main' })).getByRole('button', { name: 'More' }));
      fireEvent.touchStart(within(screen.getByRole('list', { name: 'More' })).getByRole('link', { name: 'Backtest' }));

      expect(loadedPaths).toEqual(['/validation']);
    });
  });
});
