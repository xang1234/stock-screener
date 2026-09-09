import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createTheme, ThemeProvider } from '@mui/material';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import SocialSignalsTab from './SocialSignalsTab';
import * as socialApi from '../../api/socialSignals';

vi.mock('../../api/socialSignals', async () => {
  const actual = await vi.importActual('../../api/socialSignals');
  return {
    ...actual,
    getSocialQueue: vi.fn(), getSocialContext: vi.fn(),
    getSocialUnresolved: vi.fn(), getSocialEvidence: vi.fn(),
  };
});
vi.mock('../../contexts/MarketContext', () => ({
  useMarket: () => ({ selectedMarket: 'HK' }),
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
    author_count: 5, setup_score: 76, readiness: 'ready', rs_rating_3m: 91,
    group_rank: 3, theme: 'AI infrastructure', theme_id: 7,
    state_reasons: ['setup_ready', 'market_confirmed'], security_kind: 'stock',
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

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ThemeProvider theme={createTheme()}><SocialSignalsTab /></ThemeProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

describe('SocialSignalsTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    socialApi.getSocialQueue.mockResolvedValue(queue());
    socialApi.getSocialContext.mockResolvedValue(queue({ total: 0, items: [] }));
    socialApi.getSocialUnresolved.mockResolvedValue(queue({ total: 0, items: [] }));
    socialApi.getSocialEvidence.mockResolvedValue({
      supported: true, available: true, candidate_key: 'HK:0700', window: '7d',
      item, related_listings: [{ market: 'HK', canonical_symbol: '0700' }],
      posts: [{ post_id: '1', author_handle: 'analyst',
        created_at: '2026-09-07T11:00:00Z', excerpt: 'Fresh thesis',
        url: 'https://x.com/analyst/status/1', source_names: ['Asia Growth'],
        engagement: { likes: 10 } }],
    });
  });

  it('renders a dense truthful queue with warming-up and stale states', async () => {
    renderTab();
    expect(await screen.findByText('0700')).toBeInTheDocument();
    expect(screen.getByText('82')).toBeInTheDocument();
    expect(screen.getByText('89')).toBeInTheDocument();
    expect(screen.getByText('1/2 lists · 7D')).toBeInTheDocument();
    expect(screen.getByText(/Warming up/)).toBeInTheDocument();

    socialApi.getSocialQueue.mockResolvedValue(queue({ stale: true }));
    fireEvent.click(screen.getByRole('button', { name: '14D' }));
    expect(await screen.findByText('Stale')).toBeInTheDocument();
  });

  it('changes ordering controls without hiding confirmation and separates all sections', async () => {
    socialApi.getSocialContext.mockResolvedValue(queue({ items: [{ ...item,
      candidate_key: 'HK:HSI', canonical_symbol: 'HSI', state: 'context',
      social_score: null, confirmation_score: null, queue_score: null }], total: 1 }));
    socialApi.getSocialUnresolved.mockResolvedValue(queue({ items: [{ ...item,
      candidate_key: 'unresolved:x', canonical_symbol: '$XYZ', market: null,
      state: 'unresolved', social_score: null, confirmation_score: null,
      queue_score: null }], total: 1 }));
    renderTab();
    await screen.findByText('0700');
    fireEvent.click(screen.getByRole('button', { name: 'Pure Social' }));
    await waitFor(() => expect(socialApi.getSocialQueue).toHaveBeenLastCalledWith(
      expect.objectContaining({ rankMode: 'pure_social' }),
    ));
    expect(await screen.findByText('73')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'All Signals' }));
    expect(await screen.findByRole('heading', { name: 'Context' })).toBeInTheDocument();
    expect(await screen.findByRole('heading', { name: 'Needs resolution' })).toBeInTheDocument();
    expect(screen.getByText('HSI')).toBeInTheDocument();
    expect(screen.getByText('$XYZ')).toBeInTheDocument();
  });

  it('opens evidence before the chart and keeps posts plain linked text', async () => {
    renderTab();
    fireEvent.click(await screen.findByText('0700'));
    expect(await screen.findByText('Fresh thesis')).toBeInTheDocument();
    expect(screen.queryByText('chart:0700')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open on X' })).toHaveAttribute(
      'href', 'https://x.com/analyst/status/1',
    );
    expect(screen.getByText('setup ready')).toBeInTheDocument();
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
    socialApi.getSocialQueue.mockResolvedValueOnce(queue({ total: 0, items: [] }));
    renderTab();
    expect(await screen.findByText('No signals match these controls.')).toBeInTheDocument();
  });

  it('explains mixed source outcomes when the first publication is blocked', async () => {
    socialApi.getSocialQueue.mockResolvedValueOnce(queue({
      available: false, reason_code: 'no_published_run', total: 0, items: [],
      latest_attempt: {
        run_id: 'failed-collection-1', status: 'collection_failed',
        started_at: '2026-09-08T05:49:14Z', completed_at: null,
        sources: [
          { name: 'Minervini', read_status: 'failed', received_count: 0,
            history_status: 'limited', reason_codes: ['provider_unavailable'] },
          { name: 'AI Investing', read_status: 'success', received_count: 50,
            history_status: 'warming_up', reason_codes: ['bounded_provider_read'] },
        ],
      },
    }));

    renderTab();

    expect(await screen.findByText('Latest collection did not publish')).toBeInTheDocument();
    expect(screen.getByText('Minervini')).toBeInTheDocument();
    expect(screen.getByText('Failed · 0 posts')).toBeInTheDocument();
    expect(screen.getByText('Provider unavailable')).toBeInTheDocument();
    expect(screen.getByText('AI Investing')).toBeInTheDocument();
    expect(screen.getByText('Collected · 50 posts')).toBeInTheDocument();
    expect(screen.getByText('Limited history')).toBeInTheDocument();
    expect(screen.getByText(/Analysis did not start because every enabled list/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Operations' })).toBeInTheDocument();
  });

  it('explains terminal analysis failure and directs the admin to retry', async () => {
    socialApi.getSocialQueue.mockResolvedValueOnce(queue({
      available: false, reason_code: 'no_published_run', total: 0, items: [],
      latest_attempt: {
        run_id: 'failed-analysis-1', status: 'failed',
        started_at: '2026-09-08T05:49:14Z', completed_at: '2026-09-08T05:50:14Z',
        sources: [
          { name: 'Minervini', read_status: 'success', received_count: 20,
            history_status: 'limited', reason_codes: [] },
          { name: 'AI Investing', read_status: 'success', received_count: 50,
            history_status: 'limited', reason_codes: [] },
        ],
      },
    }));

    renderTab();

    expect(await screen.findByText('Latest analysis failed')).toBeInTheDocument();
    expect(screen.getByText(/Review and retry failed work in Operations/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Operations' })).toBeInTheDocument();
  });
});
