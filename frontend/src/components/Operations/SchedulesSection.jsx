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
  IconButton,
  Link,
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
        {tasks.map((task) => (
          <TableRow key={task.name} hover>
            <TableCell>
              <Typography variant="body2" fontWeight="medium">{task.display_name}</Typography>
              <Typography variant="caption" color="text.secondary">{task.description}</Typography>
            </TableCell>
            <TableCell>
              <Typography variant="body2">{task.schedule_description}</Typography>
            </TableCell>
            <TableCell><LastRun run={task.last_run} /></TableCell>
            <TableCell><Status task={task} /></TableCell>
            <TableCell align="center">
              <RunButton task={task} pending={pendingNames.has(task.name)} onRun={onRun} />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// Phones: one card per job instead of a five-column table (matches #577).
function JobCards({ tasks, pendingNames, onRun }) {
  return (
    // role="list": WebKit drops list semantics from a ul styled list-style: none.
    <Box component="ul" role="list" aria-label="Scheduled jobs" sx={{ listStyle: 'none', p: 0, m: 0 }}>
      {tasks.map((task) => (
        <Box
          component="li"
          key={task.name}
          sx={{ py: 1.25, borderBottom: 1, borderColor: 'divider' }}
        >
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
        </Box>
      ))}
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
          <Jobs tasks={shown} pendingNames={pendingNames} onRun={onRun} />
          <InactiveMarkets markets={hiddenMarkets}>
            <Jobs tasks={hidden} pendingNames={pendingNames} onRun={onRun} />
          </InactiveMarkets>
        </>
      )}
    </Box>
  );
}
