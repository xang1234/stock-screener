import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createTheme, ThemeProvider } from '@mui/material';
import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import SocialEvidenceDrawer from './SocialEvidenceDrawer';

function renderDrawer(data) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ThemeProvider theme={createTheme()}>
        <SocialEvidenceDrawer
          open
          onClose={() => {}}
          selected={{ canonical_symbol: 'CEG' }}
          query={{ data }}
          onOpenChart={() => {}}
          onScan={() => {}}
        />
      </ThemeProvider>
    </QueryClientProvider>,
  );
}

const claim = (overrides) => ({
  post_id: 'p-1', author_handle: 'analyst', created_at: '2026-09-07T10:00:00Z',
  url: 'https://x.com/analyst/status/p-1', theme: 'Nuclear power',
  relationship: 'generator', excerpt: 'signed a 20-year nuclear PPA', support: 'supported',
  ...overrides,
});

describe('SocialEvidenceDrawer', () => {
  it('says when the latest run no longer holds the candidate', () => {
    renderDrawer({ available: false, reason_code: 'candidate_not_found', item: null, posts: [] });

    expect(screen.getByText('This signal is not in the latest published run.')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'CEG' })).toBeInTheDocument();
  });

  it('keeps the not-in-run message for a missing candidate only', () => {
    renderDrawer({ available: false, reason_code: 'no_published_run', item: null, posts: [] });

    expect(screen.queryByText('This signal is not in the latest published run.')).toBeNull();
    expect(screen.getByRole('heading', { name: 'CEG' })).toBeInTheDocument();
  });

  it('shows each extracted claim with its theme, relationship, support and source', () => {
    renderDrawer({
      available: true, item: { canonical_symbol: 'CEG' }, posts: [],
      claims: [claim(), claim({ post_id: 'p-2', theme: 'Data center power', support: 'uncertain',
        excerpt: 'maybe adds a second PPA', url: 'https://x.com/analyst/status/p-2' })],
    });

    const list = screen.getByRole('list', { name: 'Extracted claims' });
    const [first, second] = within(list).getAllByRole('listitem');
    expect(within(first).getByText('“signed a 20-year nuclear PPA”')).toBeInTheDocument();
    expect(within(first).getByText('Nuclear power · generator')).toBeInTheDocument();
    expect(within(first).getByText('Supported')).toBeInTheDocument();
    expect(within(first).getByRole('link', { name: '@analyst' }))
      .toHaveAttribute('href', 'https://x.com/analyst/status/p-1');
    expect(within(second).getByText('Uncertain')).toBeInTheDocument();
  });

  it('credits attachment wording to its source, not to the poster', () => {
    renderDrawer({
      available: true, item: { canonical_symbol: 'CEG' }, posts: [],
      claims: [claim({ source: 'article', source_url: 'https://publisher.example.com/ceg' })],
    });

    const [item] = within(screen.getByRole('list', { name: 'Extracted claims' })).getAllByRole('listitem');
    expect(within(item).getByRole('link', { name: 'linked article' }))
      .toHaveAttribute('href', 'https://publisher.example.com/ceg');
    expect(within(item).getByText(/shared by/)).toBeInTheDocument();
    expect(within(item).getByRole('link', { name: '@analyst' }))
      .toHaveAttribute('href', 'https://x.com/analyst/status/p-1');
  });

  it('labels source posts with the model post kind and stance hints', () => {
    const post = {
      post_id: 'p-1', author_handle: 'analyst', created_at: '2026-09-07T10:00:00Z',
      excerpt: '$CEG breakout over the pivot', url: 'https://x.com/analyst/status/p-1',
      source_names: [], engagement: { likes: 3 },
    };
    renderDrawer({
      available: true, item: { canonical_symbol: 'CEG' }, claims: [],
      posts: [{ ...post, post_kind: 'setup', stance: 'bullish' }, { ...post, post_id: 'p-2' }],
    });

    expect(screen.getByText('Setup · Bullish').closest('[title]'))
      .toHaveAttribute('title', 'Model hint, not verified');
    expect(screen.getAllByText(/Setup ·/)).toHaveLength(1);
  });

  it('offers only the actions it can complete', () => {
    renderDrawer({
      available: true, posts: [], claims: [],
      item: { canonical_symbol: 'CEG', explanation: { theme: 'nuclear_power', theme_id: 7 } },
    });
    expect(screen.getAllByRole('button').map((button) => button.textContent)).toEqual([
      'Close', 'Open chart & setup', 'Send visible to Scan',
    ]);
  });

  it('explains why it surfaced in plain words, including one-session-old features', () => {
    renderDrawer({
      available: true, posts: [], claims: [],
      item: { canonical_symbol: 'CEG', explanation: {
        state_reasons: ['setup_not_ready', 'missing_liquidity'], feature_sessions_behind: 1,
      } },
    });
    expect(screen.getByText('Setup not ready')).toBeInTheDocument();
    expect(screen.getByText('Trading volume unknown')).toBeInTheDocument();
    expect(screen.getByText('Price features 1 session old')).toBeInTheDocument();
  });

  it('says so when no claims were extracted', () => {
    renderDrawer({ available: true, item: { canonical_symbol: 'CEG' }, posts: [], claims: [] });
    expect(screen.getByText('No company claims extracted in this window.')).toBeInTheDocument();
  });

  it('omits the section for an API that does not report claims', () => {
    renderDrawer({ available: true, item: { canonical_symbol: 'CEG' }, posts: [] });
    expect(screen.queryByText('What X is claiming')).not.toBeInTheDocument();
  });
});
