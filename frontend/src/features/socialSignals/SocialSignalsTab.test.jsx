import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createTheme, ThemeProvider } from '@mui/material';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import SocialSignalsTab from './SocialSignalsTab';
import * as socialApi from '../../api/socialSignals';

const marketState = vi.hoisted(() => ({ selectedMarket: 'HK' }));

vi.mock('../../api/socialSignals', async () => {
  const actual = await vi.importActual('../../api/socialSignals');
  return {
    ...actual,
    getSocialQueue: vi.fn(), getSocialContext: vi.fn(), getSocialEvidence: vi.fn(),
  };
});
vi.mock('../../contexts/MarketContext', () => ({
  useMarket: () => marketState,
}));
vi.mock('../../components/common/AddToWatchlistMenu', () => ({
  default: ({ symbols }) => <button type="button">Watch {symbols}</button>,
}));
vi.mock('../../components/Scan/ChartViewerModalLazy', () => ({
  default: ({ open, initialSymbol }) => open ? <div>chart:{initialSymbol}</div> : null,
}));

const item = {
  candidate_key: 'HK:0700', canonical_symbol: '0700', market: 'HK',
  state: 'actionable', social_score: 88.6, confirmation_score: 73,
  queue_score: 82, mention_count: 12, observed_list_count: 1,
  enabled_list_count: 2, coverage: ['limited_history'],
  latest_mention: '2026-09-07T12:00:00Z', formula_version: 'social-signal-v1',
  normalization_scope: 'market', explanation: {
    setup_score: 76, readiness: 'ready', rs_rating_3m: 91,
    group_rank: 3, theme: 'AI infrastructure',
    state_reasons: ['setup_not_ready'], security_kind: 'stock',
    acceleration: null,
    social_components: { authors: { value: 82, available_weight: 15, total_weight: 15 } },
    confirmation_components: { setup: { value: 76, available_weight: 20, total_weight: 20 } },
  },
};

const queue = (overrides = {}) => ({
  supported: true, available: true, reason_code: null, market: 'HK', window: '7d',
  view: 'actionable', rank_mode: 'blended', page: 1, page_size: 50,
  total: 1, items: [item], run_id: 'run-1', stale: false,
  published_at: '2026-09-07T12:00:00Z', ...overrides,
});

const byView = (views) => async ({ view }) => views[view] || queue({ view, total: 0, items: [] });

const watchItem = (explanation) => ({
  ...item, candidate_key: 'HK:0005', canonical_symbol: '0005', state: 'watch',
  explanation: { ...item.explanation, ...explanation },
});

function LocationProbe() {
  const { pathname, hash } = useLocation();
  return <span data-testid="location">{pathname + hash}</span>;
}

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ThemeProvider theme={createTheme()}><SocialSignalsTab /></ThemeProvider>
        <LocationProbe />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

describe('SocialSignalsTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    marketState.selectedMarket = 'HK';
    socialApi.getSocialQueue.mockResolvedValue(queue());
    socialApi.getSocialContext.mockResolvedValue(queue({ total: 0, items: [] }));
    socialApi.getSocialEvidence.mockResolvedValue({
      supported: true, available: true, candidate_key: 'HK:0700', window: '7d',
      item, related_listings: [{ market: 'HK', canonical_symbol: '0700' }],
      posts: [{ post_id: '1', author_handle: 'analyst',
        created_at: '2026-09-07T11:00:00Z', excerpt: 'Fresh thesis',
        url: 'https://x.com/analyst/status/1', source_names: ['Asia Growth'],
        engagement: { likes: 10 } }],
    });
  });
  afterEach(() => vi.useRealTimers());

  it('shows an unsupported-market state without calling Social APIs', async () => {
    marketState.selectedMarket = 'KR';

    renderTab();

    expect(await screen.findByText(/available for US, HK, CN, JP, and TW markets/)).toBeInTheDocument();
    expect(socialApi.getSocialQueue).not.toHaveBeenCalled();
    expect(socialApi.getSocialContext).not.toHaveBeenCalled();
    expect(socialApi.getSocialEvidence).not.toHaveBeenCalled();
  });

  it('renders a dense queue with one plain freshness chip and no operator detail', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-09-07T15:00:00Z'));
    renderTab();
    expect(await screen.findByText('0700')).toBeInTheDocument();
    expect(screen.getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual([
      'Symbol', 'Market', 'Queue', 'Social', 'Confirmation', 'Mentions', 'Setup',
      'RS', 'Group', 'Theme', 'State',
    ]);
    expect(screen.getByText('82')).toBeInTheDocument();
    expect(screen.getByText('89')).toBeInTheDocument();
    expect(screen.getByText('Updated 3h ago')).toBeInTheDocument();
    expect(screen.queryByText(/Formula/)).not.toBeInTheDocument();
    expect(screen.queryByText(/lists ·/)).not.toBeInTheDocument();
    expect(socialApi.getSocialQueue.mock.calls.map(([controls]) => controls.view)).toEqual(['actionable']);

    socialApi.getSocialQueue.mockResolvedValue(queue({ stale: true }));
    fireEvent.click(screen.getByRole('button', { name: '14D' }));
    expect(await screen.findByText(/^Delayed, last update /)).toBeInTheDocument();
  });

  it('changes ordering controls without hiding confirmation and shows context under All Signals', async () => {
    socialApi.getSocialContext.mockResolvedValue(queue({ items: [{ ...item,
      candidate_key: 'HK:HSI', canonical_symbol: 'HSI', state: 'context',
      social_score: null, confirmation_score: null, queue_score: null }], total: 1 }));
    renderTab();
    await screen.findByText('0700');
    fireEvent.click(screen.getByRole('button', { name: 'Pure Social' }));
    await waitFor(() => expect(socialApi.getSocialQueue).toHaveBeenLastCalledWith(
      expect.objectContaining({ rankMode: 'pure_social' }),
    ));
    expect(await screen.findByText('73')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'All Signals' }));
    expect(await screen.findByRole('heading', { name: 'Context' })).toBeInTheDocument();
    expect(screen.getByText('HSI')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Needs resolution' })).not.toBeInTheDocument();
  });

  it('opens evidence before the chart and keeps posts plain linked text', async () => {
    renderTab();
    fireEvent.click(await screen.findByText('0700'));
    expect(await screen.findByText('Fresh thesis')).toBeInTheDocument();
    expect(screen.queryByText('chart:0700')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open on X' })).toHaveAttribute(
      'href', 'https://x.com/analyst/status/1',
    );
    expect(screen.getByText('Setup not ready')).toBeInTheDocument();
    expect(screen.getByText('Acceleration —')).toBeInTheDocument();
    expect(screen.getByText('Social components')).toBeInTheDocument();
    expect(screen.getByText('82 · 15/15 weight')).toBeInTheDocument();
    expect(screen.getByText('Confirmation components')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open chart & setup' }));
    expect(await screen.findByText('chart:0700')).toBeInTheDocument();
  });

  it('renders typed unavailable and empty states', async () => {
    socialApi.getSocialQueue.mockResolvedValueOnce(queue({
      available: false, reason_code: 'no_published_run', total: 0, items: [],
    }));
    const first = renderTab();
    expect(await screen.findByText(/first Social Signal run/)).toBeInTheDocument();
    first.unmount();
    socialApi.getSocialQueue.mockImplementation(byView({}));
    renderTab();
    expect(await screen.findByText('No Actionable signals in this window.')).toBeInTheDocument();
  });

  it('opens on Watch with the market exposure reason when nothing is actionable', async () => {
    socialApi.getSocialQueue.mockImplementation(byView({ watch: queue({ view: 'watch', items: [
      watchItem({ market_exposure: '42', state_reasons: ['setup_not_ready'] }),
    ] }) }));
    renderTab();
    expect(await screen.findByText(
      'No actionable signals: market exposure is 42, below 50. Showing Watch.',
    )).toBeInTheDocument();
    expect(screen.getByText('0005')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Actionable' })).toHaveClass('MuiButton-contained');
  });

  it('names the session stale price features are waiting on', async () => {
    socialApi.getSocialQueue.mockImplementation(byView({ watch: queue({ view: 'watch', items: [
      watchItem({ market_exposure: '65', feature_required_session: '2026-10-30',
        state_reasons: ['stale_features', 'missing_setup_score'] }),
      { ...watchItem({ market_exposure: '65', state_reasons: ['stale_features'] }),
        candidate_key: 'HK:0011', canonical_symbol: '0011' },
    ], total: 2 }) }));
    renderTab();
    expect(await screen.findByText(
      'No actionable signals: price features not yet updated for 2026-10-30. Showing Watch.',
    )).toBeInTheDocument();
  });

  it('labels the queue when price features are one session old', async () => {
    socialApi.getSocialQueue.mockResolvedValue(queue({ items: [
      { ...item, explanation: { ...item.explanation, feature_sessions_behind: 1 } },
    ] }));
    const first = renderTab();
    expect(await screen.findByText('Price features 1 session old')).toBeInTheDocument();
    first.unmount();
    socialApi.getSocialQueue.mockResolvedValue(queue({ items: [
      { ...item, explanation: { ...item.explanation, feature_sessions_behind: 0 } },
    ] }));
    renderTab();
    expect(await screen.findByText('0700')).toBeInTheDocument();
    expect(screen.queryByText(/session old/)).not.toBeInTheDocument();
  });

  it('offers Watch and Risk-off as views', async () => {
    renderTab();
    await screen.findByText('0700');
    fireEvent.click(screen.getByRole('button', { name: 'Risk-off' }));
    await waitFor(() => expect(socialApi.getSocialQueue).toHaveBeenLastCalledWith(
      expect.objectContaining({ view: 'risk_off' }),
    ));
    fireEvent.click(screen.getByRole('button', { name: 'Watch' }));
    await waitFor(() => expect(socialApi.getSocialQueue).toHaveBeenLastCalledWith(
      expect.objectContaining({ view: 'watch' }),
    ));
  });

  it('says in one line that collection failed and points to Operations', async () => {
    socialApi.getSocialQueue.mockResolvedValueOnce(queue({
      available: false, reason_code: 'no_published_run', total: 0, items: [],
      latest_attempt: { status: 'collection_failed' },
    }));

    renderTab();

    expect(await screen.findByText('Latest collection did not publish.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open Operations' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/operations#sources');
  });

  it('says analysis failed and points to Operations', async () => {
    socialApi.getSocialQueue.mockResolvedValueOnce(queue({
      available: false, reason_code: 'no_published_run', total: 0, items: [],
      latest_attempt: { status: 'failed' },
    }));

    renderTab();

    expect(await screen.findByText('Latest analysis failed.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Operations' })).toBeInTheDocument();
  });
});
