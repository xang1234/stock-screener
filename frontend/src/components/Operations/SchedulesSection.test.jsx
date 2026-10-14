import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, useQuery } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import SchedulesSection from './SchedulesSection';

const getScheduledTasks = vi.fn();
const triggerTask = vi.fn();

vi.mock('../../api/tasks', () => ({
  getScheduledTasks: (...args) => getScheduledTasks(...args),
  triggerTask: (...args) => triggerTask(...args),
}));

const lastRun = (status, extra = {}) => ({
  id: 1,
  task_id: `run-${status}`,
  status,
  started_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
  completed_at: null,
  duration_seconds: null,
  triggered_by: 'auto',
  error_message: null,
  ...extra,
});

const task = (name, displayName, run) => ({
  name,
  display_name: displayName,
  task_function: `app.tasks.${name}`,
  description: `${displayName} description`,
  schedule_description: 'Sunday 2:00 AM ET',
  is_enabled: true,
  last_run: run,
});

function payload(tasks) {
  return { tasks, total_tasks: tasks.length };
}

function rowFor(displayName) {
  return screen.getByText(displayName).closest('tr');
}

function renderSection() {
  return renderWithProviders(<SchedulesSection />);
}

describe('SchedulesSection', () => {
  beforeEach(() => {
    getScheduledTasks.mockReset();
    triggerTask.mockReset();
  });

  it('puts jobs of inactive markets behind a disclosure, keeping any that need attention', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      { ...task('daily-market-pipeline-us', 'Daily Market Pipeline (US)', null), market: 'US' },
      { ...task('daily-market-pipeline-de', 'Daily Market Pipeline (DE)', null), market: 'DE' },
      { ...task('daily-market-pipeline-in', 'Daily Market Pipeline (IN)', null), market: 'IN' },
      { ...task('weekly-universe-refresh-tw', 'Weekly Universe Refresh (TW)', lastRun('failed', { error_message: 'boom' })), market: 'TW' },
      task('weekly-global-maintenance', 'Weekly Cache & Telemetry Cleanup', null),
    ]));

    renderWithProviders(<SchedulesSection activeMarkets={new Set(['US'])} />);

    expect(await screen.findByText('Daily Market Pipeline (US)')).toBeVisible();
    expect(screen.getByText('Weekly Cache & Telemetry Cleanup')).toBeVisible();
    // A failed run is live state: never tucked away with its market.
    expect(screen.getByText('Weekly Universe Refresh (TW)')).toBeVisible();
    expect(screen.queryByText('Daily Market Pipeline (DE)')).not.toBeInTheDocument();

    const toggle = screen.getByRole('button', { name: 'Inactive markets (2): DE, IN' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(toggle);

    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('Daily Market Pipeline (DE)')).toBeVisible();
    expect(within(rowFor('Daily Market Pipeline (IN)')).getByRole('button', { name: 'Run Daily Market Pipeline (IN) now' })).toBeEnabled();
  });

  it('shows every job while the active markets are unknown', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      { ...task('daily-market-pipeline-de', 'Daily Market Pipeline (DE)', null), market: 'DE' },
    ]));

    renderSection();

    expect(await screen.findByText('Daily Market Pipeline (DE)')).toBeVisible();
    expect(screen.queryByRole('button', { name: /Inactive markets/ })).not.toBeInTheDocument();
  });

  it('shows each task with its server-reported last run', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      task('cleanup', 'Weekly Scan Cleanup', lastRun('completed', { duration_seconds: 42 })),
      task('refresh', 'Weekly Full Refresh', lastRun('failed', { error_message: 'db locked' })),
      task('prune', 'Monthly Price Data Cleanup', lastRun('lost')),
      task('universe', 'Weekly Universe Refresh (US)', null),
    ]));

    renderSection();

    expect(await screen.findByText('Weekly Scan Cleanup')).toBeInTheDocument();
    expect(within(rowFor('Weekly Scan Cleanup')).getByText('Completed')).toBeInTheDocument();
    expect(within(rowFor('Weekly Scan Cleanup')).getByText('5m ago · 42s')).toBeInTheDocument();
    expect(within(rowFor('Weekly Full Refresh')).getByText('Failed')).toBeInTheDocument();
    expect(within(rowFor('Weekly Full Refresh')).getByText('db locked')).toBeInTheDocument();
    // No recorded outcome is not a failure, and must not look like Queued.
    const unknown = within(rowFor('Monthly Price Data Cleanup')).getByText('Unknown').closest('.MuiChip-root');
    expect(unknown).toHaveClass('MuiChip-outlined');
    expect(unknown).not.toHaveClass('MuiChip-colorWarning');
    expect(within(rowFor('Weekly Universe Refresh (US)')).getByText('Never')).toBeInTheDocument();
  });

  it('says a run skipped, and why, instead of calling it completed', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      task('daily-market-pipeline-us', 'Daily Market Pipeline (US)', lastRun('skipped', { detail: 'not_trading_day' })),
      task('options', 'Daily US Options Analytics', lastRun('skipped', { detail: null })),
    ]));

    renderSection();
    await screen.findByText('Daily Market Pipeline (US)');

    const pipeline = rowFor('Daily Market Pipeline (US)');
    expect(within(pipeline).getByText('Skipped')).toBeInTheDocument();
    expect(within(pipeline).queryByText('Completed')).not.toBeInTheDocument();
    expect(within(pipeline).getByText('Not trading day')).toBeInTheDocument();
    // No reason recorded: the chip alone, no empty note.
    expect(within(rowFor('Daily US Options Analytics')).getByText('Skipped')).toBeInTheDocument();
    // Skipping did nothing, so it can run again straight away.
    expect(screen.getByRole('button', { name: 'Run Daily Market Pipeline (US) now' })).toBeEnabled();
  });

  it('says a pipeline handed its stages off instead of calling it completed', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      task('daily-market-pipeline-us', 'Daily Market Pipeline (US)', lastRun('dispatched', { duration_seconds: 1.4 })),
    ]));

    renderSection();
    await screen.findByText('Daily Market Pipeline (US)');

    const row = rowFor('Daily Market Pipeline (US)');
    expect(within(row).getByText('Handed off')).toBeInTheDocument();
    // The backend refuses a second chain while this one holds its claim
    // (a second Run reads Skipped, 'Market pipeline already active'), so
    // the note no longer needs to warn about it.
    const note = within(row).getByText('Stages queued. Follow them in Market activity and the job console.');
    expect(note).toBeInTheDocument();
    expect(within(row).queryByText(/second pipeline/)).not.toBeInTheDocument();
  });

  it('runs only the clicked task and refreshes the list', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      task('cleanup', 'Weekly Scan Cleanup', lastRun('completed')),
      task('refresh', 'Weekly Full Refresh', lastRun('completed')),
    ]));
    let finishTrigger;
    triggerTask.mockReturnValue(new Promise((resolve) => { finishTrigger = resolve; }));

    renderSection();
    await screen.findByText('Weekly Scan Cleanup');

    fireEvent.click(screen.getByRole('button', { name: 'Run Weekly Scan Cleanup now' }));

    await waitFor(() => expect(triggerTask).toHaveBeenCalledWith('cleanup'));
    expect(screen.getByRole('button', { name: 'Run Weekly Scan Cleanup now' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Run Weekly Full Refresh now' })).toBeEnabled();

    const callsBefore = getScheduledTasks.mock.calls.length;
    finishTrigger({ task_id: 'new-run', status: 'queued' });
    await waitFor(() => expect(getScheduledTasks.mock.calls.length).toBeGreaterThan(callsBefore));
  });

  it('keeps a just-triggered row blocked until the refreshed list arrives', async () => {
    getScheduledTasks.mockResolvedValueOnce(payload([
      task('cleanup', 'Weekly Scan Cleanup', lastRun('completed')),
    ]));
    // The refetch after the trigger never returns.
    getScheduledTasks.mockReturnValue(new Promise(() => {}));
    triggerTask.mockResolvedValue({ task_id: 'new-run', status: 'queued' });

    renderSection();
    fireEvent.click(await screen.findByRole('button', { name: 'Run Weekly Scan Cleanup now' }));

    expect(await within(rowFor('Weekly Scan Cleanup')).findByText('Queued')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run Weekly Scan Cleanup now' })).toBeDisabled();
  });

  it('does not keep a row busy while the slow job console refetches', async () => {
    // The job inventory queries Celery workers and can take many seconds.
    function JobConsoleQuery() {
      useQuery({ queryKey: ['operations', 'jobs'], queryFn: () => new Promise(() => {}) });
      return null;
    }
    getScheduledTasks.mockResolvedValue(payload([
      task('cleanup', 'Weekly Scan Cleanup', lastRun('completed')),
    ]));
    triggerTask.mockResolvedValue({ task_id: 'new-run', status: 'queued' });

    renderWithProviders(<><SchedulesSection /><JobConsoleQuery /></>);
    fireEvent.click(await screen.findByRole('button', { name: 'Run Weekly Scan Cleanup now' }));

    // The schedules list has refetched (completed again), so the row is free.
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run Weekly Scan Cleanup now' })).toBeEnabled());
  });

  it('keeps every in-flight row blocked when several are triggered', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      task('cleanup', 'Weekly Scan Cleanup', lastRun('completed')),
      task('refresh', 'Weekly Full Refresh', lastRun('completed')),
    ]));
    triggerTask.mockReturnValue(new Promise(() => {}));

    renderSection();
    fireEvent.click(await screen.findByRole('button', { name: 'Run Weekly Scan Cleanup now' }));
    fireEvent.click(screen.getByRole('button', { name: 'Run Weekly Full Refresh now' }));

    await waitFor(() => expect(triggerTask).toHaveBeenCalledTimes(2));
    expect(screen.getByRole('button', { name: 'Run Weekly Scan Cleanup now' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Run Weekly Full Refresh now' })).toBeDisabled();
  });

  it.each([
    ['running', 60],
    ['queued', 2],
  ])('blocks a second run while the task is %s (%i min)', async (status, ageMinutes) => {
    getScheduledTasks.mockResolvedValue(payload([
      task('refresh', 'Weekly Full Refresh', lastRun(status, {
        started_at: new Date(Date.now() - ageMinutes * 60 * 1000).toISOString(),
      })),
    ]));

    renderSection();

    const button = await screen.findByRole('button', { name: 'Run Weekly Full Refresh now' });
    expect(button).toBeDisabled();
  });

  it('keeps a long-queued run blocked and points to the job console', async () => {
    // Its broker message may still exist (e.g. no worker on that queue yet);
    // queueing another would run the job twice once a worker comes up. The
    // job console's cancel removes the message and closes the run instead.
    getScheduledTasks.mockResolvedValue(payload([
      task('refresh', 'Weekly Full Refresh', lastRun('queued', {
        started_at: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
      })),
    ]));

    renderSection();

    const button = await screen.findByRole('button', { name: 'Run Weekly Full Refresh now' });
    expect(button).toBeDisabled();
    expect(within(rowFor('Weekly Full Refresh'))
      .getByText('Not picked up for 30m. Cancel it in the job console to run it again.'))
      .toBeInTheDocument();
  });

  it('gives a long-queued Social Signals refresh a recovery path that works', async () => {
    // The job console does not read the social_ingestion queue, so it cannot
    // cancel this run; pointing there would be a dead end.
    getScheduledTasks.mockResolvedValue(payload([
      task('social-signal-refresh', 'Refresh Social Signals', lastRun('queued', {
        started_at: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
      })),
    ]));

    renderSection();
    await screen.findByText('Refresh Social Signals');

    const row = rowFor('Refresh Social Signals');
    expect(within(row).getByText(
      'Not picked up for 30m. Start the social worker to let it run.',
    )).toBeInTheDocument();
    expect(within(row).queryByText(/job console/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run Refresh Social Signals now' })).toBeDisabled();
  });

  describe('pre-flight: Run is refused before queueing when it would only skip', () => {
    it('disables Run while the market pipeline is in progress and links to its claim', async () => {
      const since = '2026-10-13T20:30:00+00:00';
      getScheduledTasks.mockResolvedValue(payload([{
        ...task('daily-market-pipeline-us', 'Daily Market Pipeline (US)', lastRun('dispatched')),
        run_blocker: { reason: 'market_pipeline_already_active', message: 'busy', active_since: since },
      }]));

      renderSection();

      expect(await screen.findByRole('button', { name: 'Run Daily Market Pipeline (US) now' })).toBeDisabled();
      const row = rowFor('Daily Market Pipeline (US)');
      expect(within(row).getByText(new RegExp(`Pipeline in progress since ${new Date(since).toLocaleString()}`)))
        .toBeInTheDocument();
      expect(within(row).getByRole('link', { name: 'See its claim in Lease status' }))
        .toHaveAttribute('href', '#lease-status');
    });

    it('says why on a non-trading day, even before the job has ever run', async () => {
      getScheduledTasks.mockResolvedValue(payload([{
        ...task('daily-market-pipeline-hk', 'Daily Market Pipeline (HK)', null),
        run_blocker: { reason: 'not_trading_day', message: 'Not a trading day in HK (2026-10-17)' },
      }]));

      renderSection();

      expect(await screen.findByRole('button', { name: 'Run Daily Market Pipeline (HK) now' })).toBeDisabled();
      const row = rowFor('Daily Market Pipeline (HK)');
      expect(within(row).getByText(/Not a trading day in HK \(2026-10-17\)/)).toBeInTheDocument();
      expect(within(row).queryByRole('link')).not.toBeInTheDocument();
    });

    it('leaves Run enabled when there is no blocker', async () => {
      getScheduledTasks.mockResolvedValue(payload([{
        ...task('daily-market-pipeline-us', 'Daily Market Pipeline (US)', lastRun('completed')),
        run_blocker: null,
      }]));

      renderSection();

      expect(await screen.findByRole('button', { name: 'Run Daily Market Pipeline (US) now' })).toBeEnabled();
    });
  });

  describe('a run with no outcome past the cutoff (Unknown)', () => {
    const lost = (extra) => lastRun('lost', { started_at: new Date(Date.now() - 26 * 3600 * 1000).toISOString(), ...extra });

    it('keeps Run blocked when it never started and its message is still queued', async () => {
      getScheduledTasks.mockResolvedValue(payload([
        task('refresh', 'Weekly Full Refresh', lost({ lost_from: 'queued', still_queued: true })),
      ]));

      renderSection();

      expect(await screen.findByRole('button', { name: 'Run Weekly Full Refresh now' })).toBeDisabled();
      const row = rowFor('Weekly Full Refresh');
      expect(within(row).getByText('Unknown')).toBeInTheDocument();
      expect(within(row).getByText(
        'Not picked up, and its message is still waiting. It runs once a worker takes it, or cancel it in the job console to run it again.',
      )).toBeInTheDocument();
    });

    it('keeps Run blocked when the queue could not be checked', async () => {
      getScheduledTasks.mockResolvedValue(payload([
        task('refresh', 'Weekly Full Refresh', lost({ lost_from: 'queued', still_queued: null })),
      ]));

      renderSection();

      expect(await screen.findByRole('button', { name: 'Run Weekly Full Refresh now' })).toBeDisabled();
      expect(within(rowFor('Weekly Full Refresh')).getByText(/Could not check whether it is still queued/))
        .toBeInTheDocument();
    });

    it('allows a re-run when it never started and its message is gone', async () => {
      getScheduledTasks.mockResolvedValue(payload([
        task('refresh', 'Weekly Full Refresh', lost({ lost_from: 'queued', still_queued: false })),
      ]));

      renderSection();

      expect(await screen.findByRole('button', { name: 'Run Weekly Full Refresh now' })).toBeEnabled();
      expect(within(rowFor('Weekly Full Refresh')).getByText(/no message is waiting/)).toBeInTheDocument();
    });

    it('allows a re-run when it started and the worker died', async () => {
      getScheduledTasks.mockResolvedValue(payload([
        task('prune', 'Monthly Price Data Cleanup', lost({ lost_from: 'running' })),
      ]));

      renderSection();

      expect(await screen.findByRole('button', { name: 'Run Monthly Price Data Cleanup now' })).toBeEnabled();
      expect(within(rowFor('Monthly Price Data Cleanup')).getByText(/no result was recorded/)).toBeInTheDocument();
    });
  });

  it('does not flag a freshly queued run as stuck', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      task('refresh', 'Weekly Full Refresh', lastRun('queued', {
        started_at: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
      })),
    ]));

    renderSection();
    await screen.findByText('Weekly Full Refresh');

    expect(screen.queryByText(/Not picked up/)).not.toBeInTheDocument();
  });

  it('warns when the list could not be refreshed but keeps the last data', async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    queryClient.setQueryData(['scheduledTasks'], payload([
      task('cleanup', 'Weekly Scan Cleanup', lastRun('completed')),
    ]));
    getScheduledTasks.mockRejectedValue(new Error('network down'));

    renderWithProviders(<SchedulesSection />, { queryClient });
    await queryClient.refetchQueries({ queryKey: ['scheduledTasks'] });

    expect(await screen.findByText(/Couldn't refresh scheduled jobs/)).toBeInTheDocument();
    expect(screen.getByText('Weekly Scan Cleanup')).toBeInTheDocument();
  });

  it('shows a dash for an unreadable run time', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      task('cleanup', 'Weekly Scan Cleanup', lastRun('completed', { started_at: 'not-a-date' })),
    ]));

    renderSection();
    await screen.findByText('Weekly Scan Cleanup');

    expect(within(rowFor('Weekly Scan Cleanup')).getByText('—')).toBeInTheDocument();
    expect(screen.queryByText('Just now')).not.toBeInTheDocument();
  });

  it('shows why a run was refused', async () => {
    getScheduledTasks.mockResolvedValue(payload([
      task('social', 'Refresh Social Signals', null),
    ]));
    triggerTask.mockRejectedValue({ response: { data: { detail: 'manual_refresh_cooldown' } } });

    renderSection();
    fireEvent.click(await screen.findByRole('button', { name: 'Run Refresh Social Signals now' }));

    expect(await screen.findByText(/manual_refresh_cooldown/)).toBeInTheDocument();
  });

  describe('on a phone-width screen', () => {
    beforeEach(() => {
      // useMediaQuery(breakpoints.down('md')) matches, as in the Layout tests.
      window.matchMedia = vi.fn().mockReturnValue({
        matches: true,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      });
    });

    afterEach(() => {
      delete window.matchMedia;
    });

    it('lists jobs as cards instead of a five-column table', async () => {
      getScheduledTasks.mockResolvedValue(payload([
        task('refresh', 'Weekly Full Refresh', lastRun('failed', { duration_seconds: 42, error_message: 'db locked' })),
        task('cleanup', 'Weekly Scan Cleanup', lastRun('running')),
      ]));

      renderSection();

      const list = await screen.findByRole('list', { name: 'Scheduled jobs' });
      expect(screen.queryByRole('table')).not.toBeInTheDocument();
      const card = within(list).getByText('Weekly Full Refresh').closest('li');
      expect(within(card).getByText('Failed')).toBeInTheDocument();
      expect(within(card).getByText('5m ago · 42s')).toBeInTheDocument();
      expect(within(card).getByText('Sunday 2:00 AM ET')).toBeInTheDocument();
      expect(within(card).getByText('db locked')).toBeInTheDocument();
      // Descriptions are dropped on phones; the name and schedule say enough.
      expect(within(card).queryByText('Weekly Full Refresh description')).not.toBeInTheDocument();
      expect(within(card).getByRole('button', { name: 'Run Weekly Full Refresh now' })).toBeEnabled();
      expect(screen.getByRole('button', { name: 'Run Weekly Scan Cleanup now' })).toBeDisabled();
    });

    it('shows the pre-flight reason and claim link on the card, not only in a tooltip', async () => {
      getScheduledTasks.mockResolvedValue(payload([{
        ...task('daily-market-pipeline-us', 'Daily Market Pipeline (US)', lastRun('dispatched')),
        run_blocker: { reason: 'market_pipeline_already_active', active_since: '2026-10-13T20:30:00+00:00' },
      }]));

      renderSection();

      const list = await screen.findByRole('list', { name: 'Scheduled jobs' });
      const card = within(list).getByText('Daily Market Pipeline (US)').closest('li');
      expect(within(card).getByText(/Pipeline in progress since/)).toBeInTheDocument();
      expect(within(card).getByRole('link', { name: 'See its claim in Lease status' })).toBeInTheDocument();
    });
  });
});
