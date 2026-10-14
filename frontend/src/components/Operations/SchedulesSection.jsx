/**
 * Scheduled jobs on the Operations page (replaces the header gear modal).
 *
 * Run status comes from the server's task history, which the worker writes
 * (#588), so a run started here or by Beat shows up here whether or not this
 * page was open; there is no client-side tracking of triggered runs.
 */
import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  Divider,
  IconButton,
  Link,
  Popover,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { getScheduledTasks, triggerTask } from '../../api/tasks';
import { useIsMobileShell } from '../Layout/shellNav';
import InactiveMarkets from './InactiveMarkets';
import { splitByMarket } from './activeMarkets';

const ACTIVE_STATUSES = new Set(['queued', 'running']);
// Last runs that need a look: never tucked away with an inactive market.
const ATTENTION_STATUSES = new Set(['queued', 'running', 'failed', 'lost']);
// A run still queued after this has likely lost its message or has no worker
// on its queue. Run stays blocked (its message may still be in the broker, so
// queueing another would run the job twice); the note points to the cancel.
const STALE_QUEUED_MS = 10 * 60 * 1000;

const STATUS_DISPLAY = {
  completed: { color: 'success', label: 'Completed' },
  // The task ran but did nothing (not a trading day, already running, …).
  skipped: { color: 'default', variant: 'outlined', label: 'Skipped' },
  // The task only queued other tasks (the daily pipeline's stage chain).
  dispatched: { color: 'info', variant: 'outlined', label: 'Handed off' },
  running: { color: 'info', label: 'Running' },
  queued: { color: 'warning', label: 'Queued' },
  failed: { color: 'error', label: 'Failed' },
  cancelled: { color: 'default', label: 'Cancelled' },
  // No outcome recorded within Celery's hard time limit: never picked up, or
  // the worker died without a stored result. Not known to have failed, so it
  // is neutral and outlined rather than sharing Queued's amber.
  lost: { color: 'default', variant: 'outlined', label: 'Unknown' },
};

function ageMs(dateString) {
  return Date.now() - new Date(dateString).getTime();
}

function ageLabel(ms) {
  const minutes = Math.floor(ms / 60000);
  if (minutes >= 1440) return `${Math.floor(minutes / 1440)}d`;
  if (minutes >= 60) return `${Math.floor(minutes / 60)}h`;
  return `${minutes}m`;
}

function formatRelativeTime(dateString) {
  const ms = ageMs(dateString);
  if (Number.isNaN(ms)) return '—';
  if (ms < 60000) return 'Just now';
  return `${ageLabel(ms)} ago`;
}

function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function isStaleQueued(run) {
  return run?.status === 'queued' && ageMs(run.started_at) >= STALE_QUEUED_MS;
}

// Jobs whose queue the job console does not read (OperationsJobService
// .broker_queue_names), so a queued run cannot be cancelled from there.
const NOT_IN_JOB_CONSOLE = new Set(['social-signal-refresh']);

// How to clear a run stuck in the queue so Run works again.
function stuckRecovery(taskName) {
  return NOT_IN_JOB_CONSOLE.has(taskName)
    ? 'Start the social worker to let it run.'
    : 'Cancel it in the job console to run it again.';
}

// A lost run that never started whose message may still be in the broker:
// queueing another would run the job twice once a worker picks both up.
// still_queued is null when the broker could not be read.
function lostButMaybeQueued(run) {
  return run?.status === 'lost' && run.lost_from === 'queued' && run.still_queued !== false;
}

// The message may sit in a queue (the console can cancel it) or be held by a
// stopped worker (it runs once a worker restores it), so name both ways out.
function lostQueuedReason(task) {
  if (task.last_run.still_queued !== true) {
    return 'Not picked up. Could not check whether it is still queued; try again shortly.';
  }
  return NOT_IN_JOB_CONSOLE.has(task.name)
    ? 'Not picked up, and its message is still waiting. Start the social worker to let it run.'
    : 'Not picked up, and its message is still waiting. It runs once a worker takes it, or cancel it in the job console to run it again.';
}

const PIPELINE_ACTIVE = 'market_pipeline_already_active';

// The server's pre-flight answer (run_blocker): why a Run would only read
// Skipped right now, e.g. not a trading day or the market's pipeline is mid-chain.
function preflightReason(task) {
  const blocker = task.run_blocker;
  if (!blocker) return null;
  if (blocker.reason === PIPELINE_ACTIVE) {
    return blocker.active_since
      ? `Pipeline in progress since ${new Date(blocker.active_since).toLocaleString()}`
      : 'Pipeline in progress';
  }
  return blocker.message || humanizeReason(blocker.reason);
}

// Why Run is unavailable for this job, or null when it may run again.
function runBlockedReason(task) {
  const run = task.last_run;
  if (run?.status === 'running') return 'Already running';
  if (isStaleQueued(run)) return `Still queued. ${stuckRecovery(task.name)}`;
  if (run?.status === 'queued') return 'Already queued';
  if (lostButMaybeQueued(run)) return lostQueuedReason(task);
  return preflightReason(task);
}

// Shown whatever the last run was (or with none): the pre-flight reason,
// and for a running pipeline where to see or release its claim.
function PreflightNote({ task }) {
  const reason = preflightReason(task);
  if (!reason) return null;
  return (
    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, maxWidth: 320 }}>
      {reason}.
      {task.run_blocker.reason === PIPELINE_ACTIVE && (
        <>
          {' '}
          <Link href="#lease-status">See its claim in Lease status</Link>.
        </>
      )}
    </Typography>
  );
}

function LastRun({ run }) {
  if (!run) return <Typography variant="body2" color="text.secondary">Never</Typography>;
  const when = run.started_at ? formatRelativeTime(run.started_at) : '—';
  const label = run.duration_seconds != null ? `${when} · ${formatDuration(run.duration_seconds)}` : when;
  return (
    <Tooltip title={run.triggered_by ? `Triggered: ${run.triggered_by}` : ''}>
      <Typography variant="body2">{label}</Typography>
    </Tooltip>
  );
}

function StatusChip({ run }) {
  const display = STATUS_DISPLAY[run.status] || { color: 'default', label: run.status };
  return (
    <Chip
      size="small"
      color={display.color}
      variant={display.variant || 'filled'}
      label={display.label}
      icon={ACTIVE_STATUSES.has(run.status) ? <CircularProgress size={12} color="inherit" /> : undefined}
    />
  );
}

// The failure message, or what to do about a run stuck in the queue.
// 'not_trading_day' -> 'Not trading day'; free-text reasons pass through.
function humanizeReason(reason) {
  const text = String(reason).replaceAll('_', ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function statusNote(task) {
  const run = task.last_run;
  if (run?.status === 'failed') return [run.error_message, 'error'];
  if (isStaleQueued(run)) {
    return [`Not picked up for ${ageLabel(ageMs(run.started_at))}. ${stuckRecovery(task.name)}`, 'warning.main'];
  }
  if (lostButMaybeQueued(run)) return [lostQueuedReason(task), 'warning.main'];
  if (run?.status === 'lost' && run.lost_from === 'queued') {
    // Usually never started; a worker whose start write failed also lands here.
    return ['No outcome recorded, and no message is waiting. Run it again.', 'text.secondary'];
  }
  if (run?.status === 'lost' && run.lost_from === 'running') {
    return ['Started, but no result was recorded (the worker may have stopped). Run it again.', 'text.secondary'];
  }
  if (run?.status === 'skipped' && run.detail) return [humanizeReason(run.detail), 'text.secondary'];
  if (run?.status === 'dispatched') {
    // While the chain holds its claim, the pre-flight disables Run and says so
    // (PreflightNote), so no duplicate warning is needed here.
    return ['Stages queued. Follow them in Market activity and the job console.', 'text.secondary'];
  }
  return [null, null];
}

// The failure message, why a run skipped, where handed-off work went, or
// what to do about a run stuck in the queue.
function StatusNote({ task }) {
  const [note, color] = statusNote(task);
  if (!note) return null;
  return (
    <Typography
      variant="caption"
      color={color}
      sx={{ display: 'block', mt: 0.5, maxWidth: 320, overflowWrap: 'anywhere' }}
    >
      {note}
    </Typography>
  );
}

function Status({ task }) {
  return (
    <Box>
      {task.last_run ? (
        <>
          <StatusChip run={task.last_run} />
          <StatusNote task={task} />
        </>
      ) : (
        <Typography variant="body2" color="text.secondary">—</Typography>
      )}
      <PreflightNote task={task} />
    </Box>
  );
}

function RunButton({ task, pending, onRun }) {
  const blockedReason = runBlockedReason(task);
  return (
    <Tooltip title={blockedReason || 'Run this job now'}>
      <span>
        <IconButton
          size="small"
          color="primary"
          aria-label={`Run ${task.display_name} now`}
          onClick={() => onRun(task.name)}
          disabled={Boolean(blockedReason) || pending}
        >
          {pending ? <CircularProgress size={18} /> : <PlayArrowIcon />}
        </IconButton>
      </span>
    </Tooltip>
  );
}

// Per-market registry entries share a job_type; the rest (and rows from an
// older server without job_type) stand alone.
function groupJobs(tasks) {
  const groups = new Map();
  tasks.forEach((task) => {
    const key = task.job_type ?? task.name;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(task);
  });
  return [...groups.values()];
}

const isPerMarket = (group) => group[0].job_type != null && group[0].job_type !== group[0].name;

const jobDescription = (task) => task.job_description ?? task.description;

const withoutMarket = (task) => task.schedule_description.replace(` (${task.market})`, '');

function groupSchedule(group) {
  const schedules = new Set(group.map(withoutMarket));
  return schedules.size === 1 ? [...schedules][0] : 'Varies by market';
}

// Paired with the chip colour so the outcome reads without colour too.
const STATUS_GLYPH = {
  completed: '✓',
  failed: '✗',
  skipped: '–',
  dispatched: '→',
  cancelled: '⊘',
  lost: '?',
};

function MarketChip({ task, pending, onOpen }) {
  const run = task.last_run;
  const display = run ? STATUS_DISPLAY[run.status] || { color: 'default', label: run.status } : null;
  const age = run ? ageMs(run.started_at) : NaN;
  const label = run
    ? [task.market, STATUS_GLYPH[run.status], Number.isNaN(age) ? null : ageLabel(age)].filter(Boolean).join(' ')
    : `${task.market} — never`;
  const busy = pending || ACTIVE_STATUSES.has(run?.status);
  return (
    <Chip
      size="small"
      color={display?.color ?? 'default'}
      variant={display ? display.variant || 'filled' : 'outlined'}
      label={label}
      icon={busy ? <CircularProgress size={12} color="inherit" /> : undefined}
      aria-label={`${task.display_name}: ${display ? `${display.label}, ${formatRelativeTime(run.started_at)}` : 'never run'}`}
      aria-haspopup="dialog"
      onClick={(event) => onOpen(task.name, event.currentTarget)}
    />
  );
}

function MarketDetail({ task, groupDescription, pending, onRun }) {
  return (
    <>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography variant="subtitle2" sx={{ flex: 1 }}>{task.display_name}</Typography>
        <RunButton task={task} pending={pending} onRun={onRun} />
      </Box>
      {task.description !== groupDescription && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>{task.description}</Typography>
      )}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>{task.schedule_description}</Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 1 }}>
        {task.last_run && <StatusChip run={task.last_run} />}
        <LastRun run={task.last_run} />
      </Box>
      <StatusNote task={task} />
      <PreflightNote task={task} />
    </>
  );
}

// One chip per market; a chip opens that market's status and Run.
function MarketChips({ group, pendingNames, onRun }) {
  const [open, setOpen] = useState(null);
  // Looked up each render so a refetch updates an open popover.
  const task = open && group.find((item) => item.name === open.name);
  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
      {group.map((item) => (
        <MarketChip
          key={item.name}
          task={item}
          pending={pendingNames.has(item.name)}
          onOpen={(name, anchorEl) => setOpen({ name, anchorEl })}
        />
      ))}
      <Popover
        open={Boolean(task)}
        anchorEl={open?.anchorEl}
        onClose={() => setOpen(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        PaperProps={{ role: 'dialog', 'aria-label': task?.display_name, sx: { p: 1.5, maxWidth: 360 } }}
      >
        {task && (
          <MarketDetail
            task={task}
            groupDescription={jobDescription(group[0])}
            pending={pendingNames.has(task.name)}
            onRun={onRun}
          />
        )}
      </Popover>
    </Box>
  );
}

function JobsTable({ tasks, pendingNames, onRun }) {
  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Job</TableCell>
          <TableCell>Schedule</TableCell>
          <TableCell>Last run</TableCell>
          <TableCell>Status</TableCell>
          <TableCell align="center">Run</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {groupJobs(tasks).map((group) => {
          const [task] = group;
          const perMarket = isPerMarket(group);
          return (
            <TableRow key={task.job_type ?? task.name} hover>
              <TableCell>
                <Typography variant="body2" fontWeight="medium">{perMarket ? task.job_name : task.display_name}</Typography>
                <Typography variant="caption" color="text.secondary">{perMarket ? jobDescription(task) : task.description}</Typography>
              </TableCell>
              <TableCell>
                <Typography variant="body2">{perMarket ? groupSchedule(group) : task.schedule_description}</Typography>
              </TableCell>
              {perMarket ? (
                <TableCell colSpan={3}>
                  <MarketChips group={group} pendingNames={pendingNames} onRun={onRun} />
                </TableCell>
              ) : (
                <>
                  <TableCell><LastRun run={task.last_run} /></TableCell>
                  <TableCell><Status task={task} /></TableCell>
                  <TableCell align="center">
                    <RunButton task={task} pending={pendingNames.has(task.name)} onRun={onRun} />
                  </TableCell>
                </>
              )}
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

// Phones: one card per job instead of a five-column table (matches #577).
function JobCards({ tasks, pendingNames, onRun }) {
  return (
    // role="list": WebKit drops list semantics from a ul styled list-style: none.
    <Box component="ul" role="list" aria-label="Scheduled jobs" sx={{ listStyle: 'none', p: 0, m: 0 }}>
      {groupJobs(tasks).map((group) => {
        const [task] = group;
        return (
          <Box
            component="li"
            key={task.job_type ?? task.name}
            sx={{ py: 1.25, borderBottom: 1, borderColor: 'divider' }}
          >
            {isPerMarket(group) ? (
              <>
                <Typography variant="body2" fontWeight="medium" sx={{ mb: 0.5 }}>{task.job_name}</Typography>
                <MarketChips group={group} pendingNames={pendingNames} onRun={onRun} />
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                  {groupSchedule(group)}
                </Typography>
              </>
            ) : (
              <>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" fontWeight="medium" sx={{ flex: 1 }}>{task.display_name}</Typography>
                  <RunButton task={task} pending={pendingNames.has(task.name)} onRun={onRun} />
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 0.5 }}>
                  {task.last_run && <StatusChip run={task.last_run} />}
                  <LastRun run={task.last_run} />
                </Box>
                <StatusNote task={task} />
                <PreflightNote task={task} />
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                  {task.schedule_description}
                </Typography>
              </>
            )}
          </Box>
        );
      })}
    </Box>
  );
}

function markQueued(data, taskName, taskId) {
  if (!data?.tasks) return data;
  return {
    ...data,
    tasks: data.tasks.map((task) => (task.name !== taskName ? task : {
      ...task,
      last_run: {
        ...task.last_run,
        task_id: taskId,
        status: 'queued',
        started_at: new Date().toISOString(),
        completed_at: null,
        duration_seconds: null,
        triggered_by: 'manual',
        error_message: null,
      },
    })),
  };
}

// activeMarkets: markets shown up front (activeMarkets.js); undefined shows all.
export default function SchedulesSection({ activeMarkets }) {
  const queryClient = useQueryClient();
  const isMobile = useIsMobileShell();
  const [error, setError] = useState(null);
  const [pendingNames, setPendingNames] = useState(() => new Set());

  const tasksQuery = useQuery({
    queryKey: ['scheduledTasks'],
    queryFn: getScheduledTasks,
    refetchInterval: (query) => (
      query.state.data?.tasks?.some((task) => ACTIVE_STATUSES.has(task.last_run?.status)) ? 5000 : 30000
    ),
  });

  const setPending = (taskName, isPending) => setPendingNames((current) => {
    const next = new Set(current);
    if (isPending) next.add(taskName);
    else next.delete(taskName);
    return next;
  });

  const runMutation = useMutation({
    mutationFn: (taskName) => triggerTask(taskName),
    onMutate: (taskName) => {
      setError(null);
      setPending(taskName, true);
    },
    onSuccess: (response, taskName) => {
      // The server has committed a queued row; show it now so Run stays
      // blocked until the refetch confirms it.
      queryClient.setQueryData(['scheduledTasks'], (data) => markQueued(data, taskName, response?.task_id));
      // Not awaited: the job inventory queries Celery workers and can take
      // seconds; the row's busy state should only wait for its own list.
      queryClient.invalidateQueries({ queryKey: ['operations', 'jobs'] });
      return queryClient.invalidateQueries({ queryKey: ['scheduledTasks'] });
    },
    onError: (err) => setError(err?.response?.data?.detail || err?.message || 'Failed to trigger task'),
    onSettled: (_data, _error, taskName) => setPending(taskName, false),
  });
  const tasks = tasksQuery.data?.tasks ?? [];
  const Jobs = isMobile ? JobCards : JobsTable;
  const { shown, hidden, hiddenMarkets } = splitByMarket(
    tasks,
    (task) => task.market,
    activeMarkets,
    // A mid-chain pipeline reads Handed off; its claim is live state. Other
    // blockers (not a trading day) would keep every idle market up front.
    (task) => ATTENTION_STATUSES.has(task.last_run?.status) || task.run_blocker?.reason === PIPELINE_ACTIVE,
  );
  const onRun = (taskName) => runMutation.mutate(taskName);
  const routine = shown.filter((task) => !task.advanced);
  const advanced = shown.filter((task) => task.advanced);

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" sx={{ mb: 1 }}>
        Scheduled jobs
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        Beat schedules and their last run. Run now queues the job right away; once a worker starts it, it shows in the job console above.
      </Typography>
      {error && (
        <Alert severity="error" sx={{ mb: 1.5 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {tasksQuery.isError && tasks.length > 0 && (
        <Alert severity="warning" sx={{ mb: 1.5 }}>
          Couldn&apos;t refresh scheduled jobs. Showing the last known state.
        </Alert>
      )}
      {tasksQuery.isLoading ? (
        <CircularProgress size={20} />
      ) : tasksQuery.isError && tasks.length === 0 ? (
        <Typography variant="body2" color="error">Failed to load scheduled tasks.</Typography>
      ) : tasks.length === 0 ? (
        <Typography variant="body2" color="text.secondary">No scheduled tasks configured.</Typography>
      ) : (
        <>
          <Jobs tasks={routine} pendingNames={pendingNames} onRun={onRun} />
          <InactiveMarkets markets={hiddenMarkets}>
            <Jobs tasks={hidden} pendingNames={pendingNames} onRun={onRun} />
          </InactiveMarkets>
          {advanced.length > 0 && (
            <Box component="section" aria-labelledby="advanced-jobs-heading" sx={{ mt: 2 }}>
              <Divider textAlign="left">
                <Typography id="advanced-jobs-heading" variant="overline" color="text.secondary">Advanced</Typography>
              </Divider>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Run on demand only, or delete data.
              </Typography>
              <Jobs tasks={advanced} pendingNames={pendingNames} onRun={onRun} />
            </Box>
          )}
        </>
      )}
    </Box>
  );
}
