import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createTheme, ThemeProvider } from '@mui/material';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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
  getSocialCompanyIdentities: vi.fn(), updateSocialCompanyIdentities: vi.fn(),
  getSocialRuns: vi.fn(), getSocialValidation: vi.fn(),
}));

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
    backlog: { waiting: 4, failed: 1, outside_window: 2 },
  });
  api.getSocialAdminSources.mockResolvedValue(sources);
  api.getSocialAnalysis.mockResolvedValue([{ work_id: 4, state: 'outside_window' }]);
  api.getSocialAssociations.mockResolvedValue([{
    association_id: 12, theme_name: 'AI Infrastructure', market: 'US',
    canonical_symbol: 'NVDA', state: 'proposed', version: 2,
  }]);
  api.getSocialCompanyIdentities.mockResolvedValue({ registry_version: 7, version: 1, entries: [] });
  api.getSocialRuns.mockResolvedValue([{ run_id: 'validation-1', mode: 'validation', status: 'staged' }]);
}

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><ThemeProvider theme={createTheme()}>
    <SocialSignalHealthPanel />
  </ThemeProvider></QueryClientProvider>);
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
    expect(screen.getByText(/Reauthenticate the private worker/)).toBeInTheDocument();
    expect(screen.getByText(/1 company links need administrator verification/)).toBeInTheDocument();
    expect(screen.getByText(/at most five posts/)).toBeInTheDocument();
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
});
