import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createTheme, ThemeProvider } from '@mui/material';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import * as api from '../../api/socialSignals';
import SocialSignalHealthPanel from './SocialSignalHealthPanel';

vi.mock('../../api/socialSignals', () => ({
  getSocialAdminRuntime: vi.fn(), updateSocialAdminRuntime: vi.fn(),
  getSocialAdminHealth: vi.fn(), getSocialAdminSources: vi.fn(),
  createSocialSource: vi.fn(), renameSocialSource: vi.fn(), testSocialSource: vi.fn(),
  transitionSocialSource: vi.fn(), refreshSocialSignals: vi.fn(),
  getSocialAnalysis: vi.fn(), retrySocialAnalysis: vi.fn(),
  getSocialAssociations: vi.fn(), decideSocialAssociation: vi.fn(),
  decideEconomicSocialAssociation: vi.fn(),
  getSocialCompanyIdentities: vi.fn(), updateSocialCompanyIdentities: vi.fn(),
  getSocialRuns: vi.fn(), getSocialUnresolved: vi.fn(), getSocialValidation: vi.fn(),
}));
vi.mock('../../contexts/MarketContext', () => ({ useMarket: () => ({ selectedMarket: 'HK' }) }));

const unresolvedRow = {
  candidate_key: 'unresolved:$xyz', canonical_symbol: null, market: null,
  state: 'unresolved', mention_count: 4, observed_list_count: 1, enabled_list_count: 2,
  latest_mention: null,
};

const sources = [{
  source_id: '9', name: 'Asia Growth', list_id: '1986290701492232693',
  canonical_url: 'https://x.com/i/lists/1986290701492232693', lifecycle: 'pending',
  version: 3, test_progress: null, test_outcome: null, collected_at: null,
  audit: [{ action: 'created', actor: 'server-admin', occurred_at: '2026-09-07T01:00:00Z' }],
}];

function seedMocks() {
  api.getSocialAdminRuntime.mockResolvedValue({ mode: 'validation', provider: 'xui', version: 7 });
  api.getSocialAdminHealth.mockResolvedValue({
    mode: 'validation', provider: 'xui', enabled_source_count: 2, source_count: 3,
    participating_source_count: 1, unknown_company_identity_count: 1,
    collection_status: 'incomplete', processing_status: 'pending', social_fresh: false,
    last_collection_at: '2026-09-07T01:00:00Z', reason_codes: ['reauthentication_required'],
    budget: { limit_usd: '2', spent_usd: '0.75', reserved_usd: '0.25',
      remaining_usd: '1', timezone: 'Asia/Singapore', next_reset_at: '2026-09-07T16:00:00Z',
      pricing_status: 'configured', pricing_version: '2026-09' },
    backlog: { waiting: 4, failed: 1, outside_window: 2, no_signal: 9 },
    formula_version: 'social-signal-v1',
    latest_live_attempt: {
      run_id: 'failed-collection-1', status: 'collection_failed',
      started_at: '2026-09-08T05:49:14Z', completed_at: null,
      sources: [
        { name: 'Minervini', read_status: 'failed', received_count: 0,
          history_status: 'limited', reason_codes: ['provider_unavailable'] },
        { name: 'AI Investing', read_status: 'failed', received_count: 0,
          history_status: 'warming_up', reason_codes: ['provider_timeout'] },
      ],
    },
  });
  api.getSocialUnresolved.mockResolvedValue({ available: true, total: 1, items: [unresolvedRow] });
  api.getSocialAdminSources.mockResolvedValue(sources);
  api.getSocialAnalysis.mockResolvedValue([{ work_id: 4, state: 'outside_window' },
    { work_id: 5, state: 'succeeded', error_code: 'skipped:no_signal' }]);
  api.getSocialAssociations.mockResolvedValue([{
    association_id: 12, theme_name: 'AI Infrastructure', market: 'US',
    canonical_symbol: 'NVDA', state: 'proposed', version: 2,
  }]);
  api.getSocialCompanyIdentities.mockResolvedValue({ registry_version: 7, version: 1, entries: [] });
  api.getSocialRuns.mockResolvedValue([{ run_id: 'validation-1', mode: 'validation', status: 'staged' }]);
}

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const rendered = render(<QueryClientProvider client={client}><ThemeProvider theme={createTheme()}>
    <SocialSignalHealthPanel />
  </ThemeProvider></QueryClientProvider>);
  return { ...rendered, client };
}

async function unlock() {
  fireEvent.change(screen.getByLabelText('Social admin key'), { target: { value: 'secret' } });
  fireEvent.click(screen.getByRole('button', { name: 'Load Social administration' }));
  await screen.findByLabelText('Rename Asia Growth');
}

describe('SocialSignalHealthPanel', () => {
  beforeEach(() => { vi.clearAllMocks(); seedMocks(); });

  it('keeps admin data locked until a key is supplied and then shows truthful health', async () => {
    renderPanel();
    expect(screen.getByText(/Enter the admin key/)).toBeInTheDocument();
    expect(api.getSocialAdminHealth).not.toHaveBeenCalled();
    await unlock();
    expect(screen.getByText(/2\/3 enabled sources/)).toBeInTheDocument();
    expect(screen.getByText(/1\/2 source coverage/)).toBeInTheDocument();
    expect(screen.getByText(/US\$2 per day/)).toBeInTheDocument();
    expect(screen.getByText(/US\$1 remaining/)).toBeInTheDocument();
    expect(screen.getByText(/9 skipped \(repost, or no ticker or company\)/)).toBeInTheDocument();
    expect(screen.getByText('Work 5 · skipped: repost, or no ticker or company')).toBeInTheDocument();
    expect(screen.getByText(/Reauthenticate the private worker/)).toBeInTheDocument();
    expect(screen.getByText(/1 company links need administrator verification/)).toBeInTheDocument();
    expect(screen.getByText(/at most five posts/)).toBeInTheDocument();
  });

  it('shows the operator detail the Social Signals tab leaves out', async () => {
    renderPanel();
    await unlock();
    expect(screen.getByText('Formula social-signal-v1')).toBeInTheDocument();
    expect(screen.getByText('Latest collection did not publish')).toBeInTheDocument();
    expect(screen.getByText(/^Live run started /)).toBeInTheDocument();
    expect(screen.getByText('Minervini')).toBeInTheDocument();
    expect(screen.getAllByText('Failed · 0 posts')).toHaveLength(2);
    expect(screen.getByText('Provider unavailable')).toBeInTheDocument();
    expect(screen.getByText('Provider timed out after one retry')).toBeInTheDocument();
    expect(screen.getByText('Limited history')).toBeInTheDocument();
    expect(screen.getByText(/Analysis did not start because no enabled list collected/)).toBeInTheDocument();
    const unresolved = await screen.findByRole('table', { name: 'Tickers that need resolution' });
    expect(within(unresolved).getAllByRole('row').map((row) => row.textContent)).toEqual([
      'TickerMentionsListsLatest mention', '$xyz41/2—',
    ]);
    expect(api.getSocialUnresolved).toHaveBeenCalledWith(
      { window: '7d', page: 1, pageSize: 50 },
    );
  });

  const withAttempt = async (attempt) => {
    const health = await api.getSocialAdminHealth();
    api.getSocialAdminHealth.mockResolvedValue({
      ...health, latest_live_attempt: { ...health.latest_live_attempt, sources: [], ...attempt },
    });
  };

  it.each([
    ['analysis_failed', /could be scored, so nothing was published\. Later runs retry queued posts\. Retry failed work/],
    ['publication_failed', /Analysis finished, but publishing failed/],
    ['not_published', /Analysis finished, but the run was not published/],
  ])('says why a collected run did not publish (%s)', async (reason, copy) => {
    await withAttempt({ status: 'failed', failure_reason: reason, unanalyzed_posts: 3 });
    renderPanel();
    await unlock();
    expect(screen.getByText(copy)).toBeInTheDocument();
    expect(screen.queryByText(/left out of this run's scores/)).not.toBeInTheDocument();
  });

  it('counts the posts a published run left out of its scores', async () => {
    await withAttempt({ status: 'published', unanalyzed_posts: 400 });
    renderPanel();
    await unlock();
    expect(screen.getByText('Latest run published')).toBeInTheDocument();
    expect(screen.getByText(
      "400 collected posts were not analysed and left out of this run's scores. Later runs retry the ones still queued.",
    )).toBeInTheDocument();
  });

  it('refreshes health while the latest run is still in progress', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await withAttempt({ status: 'processing' });
      renderPanel();
      await unlock();
      const calls = api.getSocialAdminHealth.mock.calls.length;
      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
      expect(api.getSocialAdminHealth.mock.calls.length).toBeGreaterThan(calls);
    } finally {
      vi.useRealTimers();
    }
  });

  it('still checks health every 30 seconds while no run is in progress', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await withAttempt({ status: 'published' });
      renderPanel();
      await unlock();
      const calls = api.getSocialAdminHealth.mock.calls.length;
      await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
      expect(api.getSocialAdminHealth.mock.calls.length).toBe(calls + 1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('drops the validation preview when another key is loaded in the same panel', async () => {
    api.getSocialValidation.mockResolvedValue({ candidates: [{}, {}], associations: [{}] });
    renderPanel();
    await unlock();
    fireEvent.click(await screen.findByRole('button', { name: 'Preview validation-1' }));
    expect(await screen.findByText(/2 staged candidates/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Social admin key'), { target: { value: 'other' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load Social administration' }));
    expect(screen.queryByText(/2 staged candidates/)).not.toBeInTheDocument();
  });

  it('clears the new-list fields when another key is loaded', async () => {
    renderPanel();
    await unlock();
    fireEvent.change(screen.getByLabelText('List name'), { target: { value: 'Old admin list' } });
    fireEvent.change(screen.getByLabelText(/List ID or URL/), { target: { value: '123' } });
    fireEvent.change(screen.getByLabelText('Social admin key'), { target: { value: 'other' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load Social administration' }));
    expect(screen.getByLabelText('List name')).toHaveValue('');
    expect(screen.getByLabelText(/List ID or URL/)).toHaveValue('');
  });

  it('ignores a validation preview that finishes after another key is loaded', async () => {
    let finish;
    api.getSocialValidation.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    renderPanel();
    await unlock();
    fireEvent.click(await screen.findByRole('button', { name: 'Preview validation-1' }));
    fireEvent.change(screen.getByLabelText('Social admin key'), { target: { value: 'other' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load Social administration' }));
    await act(async () => { finish({ candidates: [{}, {}], associations: [] }); });
    expect(screen.queryByText(/2 staged candidates/)).not.toBeInTheDocument();
  });

  it('does not show an earlier administrator\'s health after a new key is loaded', async () => {
    const { client, unmount } = renderPanel();
    await unlock();
    expect(screen.getByText('Formula social-signal-v1')).toBeInTheDocument();
    unmount();
    api.getSocialAdminHealth.mockRejectedValue(new Error('forbidden'));
    render(<QueryClientProvider client={client}><ThemeProvider theme={createTheme()}>
      <SocialSignalHealthPanel />
    </ThemeProvider></QueryClientProvider>);
    await unlock();
    expect(screen.queryByText('Formula social-signal-v1')).not.toBeInTheDocument();
    expect(screen.queryByText('Minervini')).not.toBeInTheDocument();
  });

  it.each([
    ['loading', () => new Promise(() => {}), 'Loading unresolved tickers…'],
    ['failed', () => Promise.reject(new Error('down')), 'Unresolved tickers could not be loaded.'],
    ['unavailable', () => Promise.resolve({
      available: false, reason_code: 'no_published_run', total: 0, items: [],
    }), 'Unresolved tickers are unavailable (no_published_run).'],
    ['empty', () => Promise.resolve({ available: true, total: 0, items: [] }),
      'No unresolved tickers in the latest run.'],
  ])('shows the %s unresolved state on its own', async (_, respond, text) => {
    api.getSocialUnresolved.mockImplementation(respond);
    renderPanel();
    await unlock();
    expect(await screen.findByText(text)).toBeInTheDocument();
    expect(screen.getAllByText(/unresolved tickers/i).map((node) => node.textContent)).toEqual([text]);
  });

  it('pages through unresolved tickers', async () => {
    api.getSocialUnresolved.mockResolvedValue({ available: true, total: 120, items: [unresolvedRow] });
    renderPanel();
    await unlock();
    expect(await screen.findByText('1–50 of 120')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Go to next page' }));
    expect(screen.getByRole('table', { name: 'Tickers that need resolution' })).toBeInTheDocument();
    expect(screen.queryByText('Loading unresolved tickers…')).not.toBeInTheDocument();
    await waitFor(() => expect(api.getSocialUnresolved).toHaveBeenLastCalledWith(
      { window: '7d', page: 2, pageSize: 50 },
    ));
    expect(await screen.findByText('51–100 of 120')).toBeInTheDocument();
  });

  it('returns to the last page when a newer run has fewer unresolved tickers', async () => {
    let total = 51;
    api.getSocialUnresolved.mockImplementation(async ({ page }) => ({
      available: true, total, items: (page - 1) * 50 < total ? [unresolvedRow] : [],
    }));
    const { client } = renderPanel();
    await unlock();
    fireEvent.click(await screen.findByRole('button', { name: 'Go to next page' }));
    expect(await screen.findByText('51–51 of 51')).toBeInTheDocument();

    total = 1;
    await act(() => client.invalidateQueries({ queryKey: ['socialAdmin', 'unresolved'] }));

    expect(await screen.findByText('1–1 of 1')).toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Tickers that need resolution' });
    expect(within(table).getAllByRole('row').map((row) => row.textContent)).toEqual([
      'TickerMentionsListsLatest mention', '$xyz41/2—',
    ]);
    expect(api.getSocialUnresolved).toHaveBeenLastCalledWith(
      { window: '7d', page: 1, pageSize: 50 },
    );
  });

  it('creates pending named lists and tests before enablement', async () => {
    api.createSocialSource.mockResolvedValue({ source_id: '10', lifecycle: 'pending' });
    api.testSocialSource.mockResolvedValue({ status: 'queued' });
    api.transitionSocialSource.mockResolvedValue({ lifecycle: 'enabled' });
    renderPanel();
    await unlock();
    fireEvent.change(screen.getByLabelText('List name'), { target: { value: 'Japan Leaders' } });
    fireEvent.change(screen.getByLabelText('List ID or URL'), { target: { value: '12345' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add pending list' }));
    await waitFor(() => expect(api.createSocialSource).toHaveBeenCalledWith(
      'secret', { name: 'Japan Leaders', list_ref: '12345' },
    ));
    fireEvent.click(screen.getByRole('button', { name: 'Test Asia Growth' }));
    await waitFor(() => expect(api.testSocialSource).toHaveBeenCalledWith('secret', '9', 3));
    expect(screen.queryByRole('button', { name: 'Enable Asia Growth' })).not.toBeInTheDocument();
  });

  it('refreshes installation capabilities after changing the runtime', async () => {
    api.updateSocialAdminRuntime.mockResolvedValue({ mode: 'live', provider: 'xui', version: 8 });
    const { client } = renderPanel();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    await unlock();

    fireEvent.click(screen.getByRole('button', { name: 'Save runtime' }));

    await waitFor(() => expect(api.updateSocialAdminRuntime).toHaveBeenCalledWith(
      'secret', { mode: 'validation', provider: 'xui', expected_version: 7 },
    ));
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith(
      { queryKey: ['appCapabilities'] },
    ));
  });

  it('supports refresh cooldown, outside-window retry and reasoned decisions', async () => {
    api.refreshSocialSignals.mockRejectedValue({ response: { status: 429, headers: { 'retry-after': '900' } } });
    api.retrySocialAnalysis.mockResolvedValue({ status: 'queued' });
    api.decideSocialAssociation.mockResolvedValue({ status: 'accepted' });
    renderPanel();
    await unlock();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh Social Signals' }));
    expect(await screen.findByText(/Try again in 900 seconds/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry work 4' }));
    await waitFor(() => expect(api.retrySocialAnalysis).toHaveBeenCalledWith('secret', 4));
    fireEvent.change(screen.getByLabelText('Decision reason 12'), { target: { value: 'Verified issuer relationship' } });
    fireEvent.click(screen.getByRole('button', { name: 'Accept NVDA' }));
    await waitFor(() => expect(api.decideSocialAssociation).toHaveBeenCalledWith(
      'secret', 12, { target: 'accepted', reason: 'Verified issuer relationship', expected_version: 2 },
    ));
  });

  it('decides an economic association by its economic id and revision', async () => {
    api.getSocialAssociations.mockResolvedValue([{
      association_id: null, economic_association_id: 'a1b2', theme_name: 'Semiconductors',
      market: 'US', canonical_symbol: 'AMD', state: 'proposed', version: null, economic_revision: 4,
    }]);
    api.decideEconomicSocialAssociation.mockResolvedValue({ status: 'accepted' });
    renderPanel();
    await unlock();
    fireEvent.change(await screen.findByLabelText('Decision reason a1b2'), { target: { value: 'Verified' } });
    fireEvent.click(screen.getByRole('button', { name: 'Accept AMD' }));
    await waitFor(() => expect(api.decideEconomicSocialAssociation).toHaveBeenCalledWith(
      'secret', 'a1b2', { target: 'accepted', reason: 'Verified', expected_revision: 4 },
    ));
    expect(api.decideSocialAssociation).not.toHaveBeenCalled();
  });

  it('reloads the identity draft after an optimistic version conflict', async () => {
    api.getSocialCompanyIdentities
      .mockResolvedValueOnce({
        registry_version: 7, version: 1,
        entries: [{ symbol: 'OLD', company_id: 'old-company' }],
      })
      .mockResolvedValue({
        registry_version: 8, version: 2,
        entries: [{ symbol: 'NEW', company_id: 'server-company' }],
      });
    api.updateSocialCompanyIdentities
      .mockRejectedValueOnce({ response: { data: { detail: { code: 'version_conflict' } } } })
      .mockResolvedValueOnce({ registry_version: 9, entries: [] });
    renderPanel();
    await unlock();
    const editor = await screen.findByLabelText('Identity entries JSON');
    fireEvent.change(editor, { target: { value: '[{"symbol":"LOCAL"}]' } });

    fireEvent.click(screen.getByRole('button', { name: 'Save verified identities' }));

    await waitFor(() => expect(editor.value).toContain('server-company'));
    expect(screen.getByText(/Reloaded the latest version/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Save verified identities' }));
    await waitFor(() => expect(api.updateSocialCompanyIdentities).toHaveBeenLastCalledWith(
      'secret', {
        expected_version: 8,
        entries: [{ symbol: 'NEW', company_id: 'server-company' }],
      },
    ));
  });

  it('replaces a stale source rename draft after an optimistic version conflict', async () => {
    const latestSources = [{ ...sources[0], name: 'Server Name', version: 4 }];
    api.getSocialAdminSources
      .mockResolvedValueOnce(sources)
      .mockResolvedValue(latestSources);
    api.renameSocialSource
      .mockRejectedValueOnce({ response: { data: { detail: { code: 'version_conflict' } } } })
      .mockResolvedValueOnce(latestSources[0]);
    renderPanel();
    await unlock();
    const editor = screen.getByLabelText('Rename Asia Growth').querySelector('input');
    fireEvent.change(editor, { target: { value: 'Local Name' } });

    fireEvent.click(screen.getByRole('button', { name: 'Rename' }));

    await waitFor(() => expect(
      screen.getByLabelText('Rename Server Name').querySelector('input'),
    ).toHaveValue('Server Name'));
    expect(screen.getByText(/Reloaded the latest version/)).toBeInTheDocument();
    const refreshedEditor = screen.getByLabelText('Rename Server Name').querySelector('input');
    fireEvent.change(refreshedEditor, { target: { value: 'Final Name' } });
    fireEvent.click(screen.getByRole('button', { name: 'Rename' }));
    await waitFor(() => expect(api.renameSocialSource).toHaveBeenLastCalledWith(
      'secret', '9', 'Final Name', 4,
    ));
  });
});
