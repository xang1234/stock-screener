/**
 * Operations / telemetry dashboard (bead asia.10.2).
 *
 * Polls runtime activity, live job inventory, and telemetry summaries so
 * operators can inspect queued/running/stuck jobs and cancel supported work.
 */
import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControl,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import { fetchAlerts, acknowledgeAlert } from '../api/telemetry';
import { cancelOperationsJob, fetchOperationsJobs, releasePipelineChain } from '../api/operations';
import { getBootstrapStatus } from '../api/appRuntime';
import { useRuntimeActivity } from '../hooks/useRuntimeActivity';
import { useRuntime } from '../contexts/RuntimeContext';
import SocialSignalHealthPanel from '../features/socialSignals/SocialSignalHealthPanel';
import ExposureResearchWorkspace from '../features/companyExposure/ExposureResearchWorkspace';
import SchedulesSection from '../components/Operations/SchedulesSection';
import InactiveMarkets from '../components/Operations/InactiveMarkets';
import {
  ACTIVE_JOB_STATES,
  findActiveMarkets,
  queueMarket,
  splitByMarket,
} from '../components/Operations/activeMarkets';

const POLL_MS = 30000;
const EMPTY_ARRAY = [];
const EMPTY_LEASES = { external_fetch_global: null, market_workload: {} };

const SEVERITY_COLOR = {
  warning: 'warning',
  critical: 'error',
};

const STATE_COLOR = {
  open: 'error',
  acknowledged: 'warning',
  closed: 'default',
};

const ACTIVITY_STATUS_COLOR = {
  running: 'info',
  queued: 'warning',
  completed: 'success',
  failed: 'error',
  idle: 'default',
  stale: 'warning',
  stuck: 'error',
};

const JOB_STATE_COLOR = {
  queued: 'warning',
  waiting: 'warning',
  reserved: 'info',
  running: 'info',
  stale: 'warning',
  stuck: 'error',
  cancelled: 'default',
  failed: 'error',
};

function formatLagSeconds(seconds) {
  if (seconds == null) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function formatTimestamp(value) {
  if (!value) return '—';
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

// What each cancel strategy actually does, in the words the confirm dialog uses.
const CANCEL_ACTIONS = {
  revoke_and_remove_from_queue: {
    label: 'Cancel',
    detail: 'Remove from queue — it never runs.',
  },
  revoke: {
    label: 'Revoke',
    detail: 'A worker has already picked this job up. Revoking makes the worker drop it instead of running it.',
  },
  scan_cancel: {
    label: 'Cancel scan',
    detail: 'Mark the scan cancelled — the worker stops at its next checkpoint.',
  },
  force_cancel_refresh: {
    label: 'Force cancel',
    detail: 'Force-release the refresh locks — the worker may still be running; use only if it is dead.',
    danger: true,
  },
  force_release_market_lease: {
    label: 'Force release',
    detail: 'Force-release the market lock — the worker may still be running; use only if it is dead.',
    danger: true,
  },
};

// Set by the server when no responding worker reads the job's queue. A
// solo-pool worker busy with a task does not answer, hence "responding".
const NO_WORKER_WAIT_REASON = 'no_worker_for_queue';

function Blocker({ job }) {
  if (job.wait_reason !== NO_WORKER_WAIT_REASON) return job.wait_reason || '—';
  return (
    <Typography variant="caption" color="error">
      No responding worker reads {job.queue}. Start one, or cancel this job.
    </Typography>
  );
}

function formatCancelLabel(strategy) {
  return CANCEL_ACTIONS[strategy]?.label || 'Unavailable';
}

function resolveDeterminatePercent(percent, current, total) {
  if (Number.isFinite(percent)) {
    return Math.max(0, Math.min(100, Number(percent)));
  }
  if (current != null && total != null && total > 0) {
    return Math.max(0, Math.min(100, (Number(current) / Number(total)) * 100));
  }
  return null;
}

const missingKeys = (statuses) => Object.keys(statuses ?? {}).filter((key) => statuses[key] === 'missing');

// Derived stages and optional features never gate readiness, so surface the
// ones that have never published (queued and failed both read as missing).
function UnpublishedDataAlert({ status }) {
  const lines = Object.entries(status?.market_stages ?? {})
    .map(([market, stages]) => [market, missingKeys(stages)])
    .filter(([, missing]) => missing.length > 0)
    .map(([market, missing]) => `${market}: ${missing.join(', ')}`);
  const features = missingKeys(status?.features);
  if (features.length > 0) lines.push(`Features: ${features.join(', ')}`);
  if (lines.length === 0) return null;
  return (
    <Alert severity="warning" role="status" sx={{ mb: 1.5 }}>
      <Typography variant="body2" fontWeight={700}>Not yet published</Typography>
      {lines.map((line) => (
        <Typography key={line} variant="body2">{line}</Typography>
      ))}
    </Alert>
  );
}

function MarketSummaryCard({ summary }) {
  const freshness = summary.freshness_lag;
  const benchmark = summary.benchmark_age;
  const completeness = summary.completeness_distribution;
  const drift = summary.universe_drift;

  return (
    <Card variant="outlined" sx={{ minWidth: 220 }}>
      <CardContent>
        <Typography variant="overline" color="text.secondary">
          {summary.market}
        </Typography>
        <Stack spacing={0.5} sx={{ mt: 1 }}>
          <Typography variant="body2">
            <strong>Freshness:</strong> {formatLagSeconds(freshness?.lag_seconds)}
          </Typography>
          <Typography variant="body2">
            <strong>Benchmark age:</strong> {formatLagSeconds(benchmark?.age_seconds)}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <Typography variant="body2" component="span">
              <strong>Universe:</strong> {drift?.current_size ?? '—'}
            </Typography>
            {drift?.delta != null && drift.delta !== 0 && (
              <Chip
                label={`${drift.delta > 0 ? '+' : ''}${drift.delta}`}
                size="small"
                color={Math.abs(drift.delta) > (drift.prior_size || 1) * 0.05 ? 'warning' : 'default'}
              />
            )}
          </Box>
          <Typography variant="body2">
            <strong>Low completeness (0-25):</strong>{' '}
            {completeness?.bucket_counts && completeness?.symbols_total
              ? `${Math.round((completeness.bucket_counts['0-25'] / completeness.symbols_total) * 100)}%`
              : '—'}
          </Typography>
        </Stack>
      </CardContent>
    </Card>
  );
}

function MarketActivityCard({ activity }) {
  const showProgress = activity.status === 'running' || activity.status === 'queued';
  const progressValue = resolveDeterminatePercent(activity.percent, activity.current, activity.total);

  return (
    <Card variant="outlined" sx={{ minWidth: 260, flex: '1 1 280px' }}>
      <CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
          <Typography variant="overline" color="text.secondary">
            {activity.market}
          </Typography>
          <Stack direction="row" spacing={1}>
            <Chip label={activity.lifecycle || 'idle'} size="small" variant="outlined" />
            <Chip
              label={activity.status || 'idle'}
              size="small"
              color={ACTIVITY_STATUS_COLOR[activity.status] || 'default'}
            />
          </Stack>
        </Stack>
        <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
          {activity.stage_label || 'Idle'}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ minHeight: 40 }}>
          {activity.message || 'No active work for this market.'}
        </Typography>
        {showProgress && (
          <Box sx={{ mt: 1.5 }}>
            <LinearProgress
              variant={progressValue != null ? 'determinate' : 'indeterminate'}
              value={progressValue != null ? progressValue : undefined}
              aria-label={`${activity.market} progress`}
            />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
              {progressValue != null
                ? `${Math.round(progressValue)}%`
                : 'Progress pending'}{activity.current != null && activity.total != null
                ? ` · ${activity.current}/${activity.total}`
                : ''}
            </Typography>
          </Box>
        )}
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5 }}>
          Task: {activity.task_name || '—'}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
          Updated: {formatTimestamp(activity.updated_at)}
        </Typography>
      </CardContent>
    </Card>
  );
}

function AlertsTable({ alerts, onAcknowledge }) {
  if (alerts.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
        No active alerts.
      </Box>
    );
  }
  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Severity</TableCell>
          <TableCell>State</TableCell>
          <TableCell>Market</TableCell>
          <TableCell>Metric</TableCell>
          <TableCell>Owner</TableCell>
          <TableCell>Opened</TableCell>
          <TableCell>Description</TableCell>
          <TableCell>Action</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {alerts.map((alert) => (
          <TableRow key={alert.id}>
            <TableCell>
              <Chip label={alert.severity} size="small" color={SEVERITY_COLOR[alert.severity] || 'default'} />
            </TableCell>
            <TableCell>
              <Chip label={alert.state} size="small" color={STATE_COLOR[alert.state] || 'default'} />
            </TableCell>
            <TableCell>{alert.market}</TableCell>
            <TableCell>{alert.metric_key}</TableCell>
            <TableCell>{alert.owner || '—'}</TableCell>
            <TableCell>{alert.opened_at?.slice(0, 19).replace('T', ' ') || '—'}</TableCell>
            <TableCell>{alert.description}</TableCell>
            <TableCell>
              {alert.state === 'open' ? (
                <Button size="small" onClick={() => onAcknowledge(alert.id)}>
                  Ack
                </Button>
              ) : (
                <Typography variant="caption" color="text.secondary">
                  {alert.acknowledged_by || ''}
                </Typography>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function MarketLease({ market, holder }) {
  return (
    <Box>
      <Typography variant="body2">
        <strong>{market}:</strong> {holder?.task_name || 'idle'}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {holder?.task_id ? `Task ID: ${holder.task_id}` : 'No active workload lease'}
      </Typography>
    </Box>
  );
}

function LeaseSummary({ leases, onReleaseChain, releasingToken, activeMarkets }) {
  const externalHolder = leases?.external_fetch_global;
  const marketLeases = splitByMarket(Object.entries(leases?.market_workload || {}), ([market]) => market, activeMarkets);
  const chainClaims = Object.entries(leases?.pipeline_chain || {});

  return (
    <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
      <Card variant="outlined" sx={{ minWidth: 260, flex: '1 1 280px' }}>
        <CardContent>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            External Fetch Lease
          </Typography>
          {externalHolder ? (
            <>
              <Typography variant="body2">{externalHolder.task_name || 'unknown task'}</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                Task ID: {externalHolder.task_id || '—'}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                Started: {formatTimestamp(externalHolder.started_at)}
              </Typography>
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No task currently holds the external fetch lease.
            </Typography>
          )}
        </CardContent>
      </Card>
      <Card variant="outlined" sx={{ minWidth: 260, flex: '1 1 320px' }}>
        <CardContent>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Market Workload Leases
          </Typography>
          <Stack spacing={1}>
            {marketLeases.shown.map(([market, holder]) => (
              <MarketLease key={market} market={market} holder={holder} />
            ))}
          </Stack>
          <InactiveMarkets markets={marketLeases.hiddenMarkets}>
            <Stack spacing={1}>
              {marketLeases.hidden.map(([market, holder]) => (
                <MarketLease key={market} market={market} holder={holder} />
              ))}
            </Stack>
          </InactiveMarkets>
        </CardContent>
      </Card>
      <Card variant="outlined" sx={{ minWidth: 260, flex: '1 1 320px' }}>
        <CardContent>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Pipeline Chain Claims
          </Typography>
          {chainClaims.length ? (
            <Stack spacing={1}>
              {chainClaims.map(([market, claim]) => (
                <Stack key={market} direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="body2">
                      <strong>{market}:</strong> daily pipeline
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      Since {formatTimestamp(claim.started_at)} · expires in {formatLagSeconds(claim.ttl_seconds)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      Claim: {claim.task_id}
                    </Typography>
                  </Box>
                  <Button
                    size="small"
                    variant="outlined"
                    color="error"
                    onClick={() => onReleaseChain({ market, token: claim.task_id })}
                    disabled={!claim.task_id || releasingToken === claim.task_id}
                    aria-label={`Release ${market} pipeline chain claim`}
                  >
                    {releasingToken === claim.task_id ? 'Working…' : 'Release'}
                  </Button>
                </Stack>
              ))}
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No pipeline chain claims are held.
            </Typography>
          )}
        </CardContent>
      </Card>
    </Stack>
  );
}

function QueueSummaryTable({ queues }) {
  if (!queues?.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No queue data available.
      </Typography>
    );
  }

  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Queue</TableCell>
          <TableCell>Depth</TableCell>
          <TableCell>Oldest</TableCell>
          <TableCell>Workers</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {queues.map((queue) => (
          <TableRow key={queue.queue}>
            <TableCell>{queue.queue}</TableCell>
            <TableCell>{queue.depth}</TableCell>
            <TableCell>{formatLagSeconds(queue.oldest_age_seconds)}</TableCell>
            <TableCell>
              {/* null: no worker answered, so unknown rather than unserved. */}
              {queue.consumers === 0 && queue.depth > 0 ? (
                <Chip label="No worker" size="small" color="error" />
              ) : (
                queue.consumers ?? '—'
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function WorkerStatusTable({ workers }) {
  if (!workers?.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No worker inspect data available.
      </Typography>
    );
  }

  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Worker</TableCell>
          <TableCell>Status</TableCell>
          <TableCell>Queues</TableCell>
          <TableCell>Active</TableCell>
          <TableCell>Reserved</TableCell>
          <TableCell>Scheduled</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {workers.map((worker) => (
          <TableRow key={worker.worker}>
            <TableCell>{worker.worker}</TableCell>
            <TableCell>
              <Chip label={worker.status} size="small" color={worker.status === 'online' ? 'success' : 'warning'} />
            </TableCell>
            <TableCell>{worker.queues?.join(', ') || '—'}</TableCell>
            <TableCell>{worker.active}</TableCell>
            <TableCell>{worker.reserved}</TableCell>
            <TableCell>{worker.scheduled}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// One rule for every section built from the jobs request. With no data, a
// failed request says so: empty fallbacks would read as "nothing holds the
// lease". After a success, a failed refetch keeps the last data (a blip should
// not hide lease state) and says it may be stale.
function JobsData({ query, children }) {
  if (!query.data) {
    return query.isError ? (
      <Typography variant="body2" color="error">
        Failed to load Operations job inventory.
      </Typography>
    ) : (
      <CircularProgress size={20} />
    );
  }
  return (
    <>
      {query.isError && (
        <Typography variant="caption" color="warning.main" sx={{ display: 'block', mb: 1 }}>
          Couldn&apos;t refresh. Showing the last known status from{' '}
          {/* Date and time: the last data can be from yesterday (an open tab). */}
          {formatTimestamp(new Date(query.dataUpdatedAt).toISOString())}.
        </Typography>
      )}
      {children}
    </>
  );
}

function JobsTable({ jobs, onCancel, cancellingTaskId, cancelInFlight }) {
  if (!jobs.length) {
    return (
      <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
        No queued or running jobs match the current filters.
      </Box>
    );
  }

  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Task</TableCell>
          <TableCell>State</TableCell>
          <TableCell>Queue</TableCell>
          <TableCell>Market</TableCell>
          <TableCell>Worker</TableCell>
          <TableCell>Progress</TableCell>
          <TableCell>Age</TableCell>
          <TableCell>Blocker</TableCell>
          <TableCell>Heartbeat</TableCell>
          <TableCell>Action</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {jobs.map((job) => {
          const canCancel = Boolean(CANCEL_ACTIONS[job.cancel_strategy]);
          // Stays until the job leaves the inventory, unless a different action now applies.
          const cancelRequested = job.cancel_requested_at
            && job.cancel_requested_strategy === job.cancel_strategy;
          const isCancelling = cancelInFlight && cancellingTaskId === job.task_id;
          const progressValue = resolveDeterminatePercent(job.percent, job.current, job.total);
          const hasDeterminateProgress = job.progress_mode === 'determinate' && progressValue != null;
          return (
            <TableRow key={job.task_id}>
              <TableCell>
                <Typography variant="body2">{job.task_name}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {job.task_id}
                </Typography>
              </TableCell>
              <TableCell>
                <Chip
                  label={job.state}
                  size="small"
                  color={JOB_STATE_COLOR[job.state] || 'default'}
                />
              </TableCell>
              <TableCell>{job.queue || '—'}</TableCell>
              <TableCell>{job.market || '—'}</TableCell>
              <TableCell>{job.worker || '—'}</TableCell>
              <TableCell sx={{ minWidth: 220 }}>
                {hasDeterminateProgress ? (
                  <Box sx={{ minWidth: 180 }}>
                    <LinearProgress
                      variant="determinate"
                      value={progressValue}
                      aria-label={`${job.task_name} progress`}
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
                      {`${Math.round(progressValue)}%`}{job.current != null && job.total != null
                        ? ` · ${job.current}/${job.total}`
                        : ''}
                    </Typography>
                    {job.message && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                        {job.message}
                      </Typography>
                    )}
                  </Box>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    {job.message || 'Progress pending'}
                  </Typography>
                )}
              </TableCell>
              <TableCell>{formatLagSeconds(job.age_seconds)}</TableCell>
              <TableCell><Blocker job={job} /></TableCell>
              <TableCell>{formatLagSeconds(job.heartbeat_lag_seconds)}</TableCell>
              <TableCell>
                {cancelRequested ? (
                  <Typography variant="caption" color="warning.main" title={formatTimestamp(job.cancel_requested_at)}>
                    Cancel requested
                  </Typography>
                ) : canCancel ? (
                  <Button
                    size="small"
                    variant="outlined"
                    color={CANCEL_ACTIONS[job.cancel_strategy].danger ? 'error' : 'primary'}
                    onClick={() => onCancel(job)}
                    disabled={isCancelling}
                  >
                    {isCancelling ? 'Working…' : formatCancelLabel(job.cancel_strategy)}
                  </Button>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    {formatCancelLabel(job.cancel_strategy)}
                  </Typography>
                )}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

export default function OperationsPage() {
  const queryClient = useQueryClient();
  const [ackError, setAckError] = useState(null);
  const [jobActionError, setJobActionError] = useState(null);
  const [stateFilter, setStateFilter] = useState('all');
  const [queueFilter, setQueueFilter] = useState('all');
  const [marketFilter, setMarketFilter] = useState('all');
  const [taskSearch, setTaskSearch] = useState('');
  const activityQuery = useRuntimeActivity();
  // Exposure routes are registered only with the themes feature.
  const { features, enabledMarkets, supportedMarkets } = useRuntime();

  const alertsQuery = useQuery({
    queryKey: ['telemetry', 'alerts'],
    queryFn: () => fetchAlerts({ evaluate: true }),
    refetchInterval: POLL_MS,
  });

  const bootstrapStatusQuery = useQuery({
    queryKey: ['runtime', 'bootstrapStatus'],
    queryFn: getBootstrapStatus,
    refetchInterval: POLL_MS,
  });

  const jobsQuery = useQuery({
    queryKey: ['operations', 'jobs'],
    queryFn: fetchOperationsJobs,
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs || [];
      return jobs.some((job) => ACTIVE_JOB_STATES.has(job.state)) ? 5000 : POLL_MS;
    },
    retry: 1,
  });

  const ackMutation = useMutation({
    mutationFn: (alertId) => acknowledgeAlert(alertId, 'operator'),
    onSuccess: () => {
      setAckError(null);
      queryClient.invalidateQueries({ queryKey: ['telemetry', 'alerts'] });
    },
    onError: (err) => setAckError(err?.response?.data?.detail || err.message),
  });

  const cancelMutation = useMutation({
    mutationFn: ({ taskId, expectedStrategy }) => cancelOperationsJob(taskId, expectedStrategy),
    onSuccess: (payload) => {
      setJobActionError(null);
      if (payload.status !== 'accepted') {
        setJobActionError(payload.message);
      }
    },
    // A 409 means the job moved since it was shown: say so and refresh.
    onError: (err) => setJobActionError(err?.response?.data?.detail || err.message),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['operations', 'jobs'] });
      queryClient.invalidateQueries({ queryKey: ['runtimeActivity'] });
    },
  });
  const releaseChainMutation = useMutation({
    mutationFn: ({ market, token }) => releasePipelineChain(market, token),
    onSuccess: (payload) => setJobActionError(payload.status === 'accepted' ? null : payload.message),
    onError: (err) => setJobActionError(err?.response?.data?.detail || err.message),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['operations', 'jobs'] });
      // A released claim re-enables that pipeline's Run (its run_blocker).
      queryClient.invalidateQueries({ queryKey: ['scheduledTasks'] });
    },
  });
  const [confirmChain, setConfirmChain] = useState(null);
  const [confirmJob, setConfirmJob] = useState(null);
  const confirmAction = confirmJob ? CANCEL_ACTIONS[confirmJob.cancel_strategy] : null;

  const summaries = alertsQuery.data?.summaries || [];
  const alerts = alertsQuery.data?.alerts || [];
  const marketActivity = activityQuery.data?.markets || [];
  const bootstrap = activityQuery.data?.bootstrap;
  const hasMarketActivity = marketActivity.length > 0;
  const jobsPayload = jobsQuery.data;
  const jobs = jobsPayload?.jobs ?? EMPTY_ARRAY;
  const queues = jobsPayload?.queues ?? EMPTY_ARRAY;
  const workers = jobsPayload?.workers ?? EMPTY_ARRAY;
  const leases = jobsPayload?.leases ?? EMPTY_LEASES;
  // Until the jobs data loads (or while it fails) this is the enabled markets
  // and open alerts alone, so a worker-served market such as CN waits behind
  // the disclosure until it does.
  const activeMarkets = findActiveMarkets({ enabledMarkets, supportedMarkets, jobsPayload, alerts });
  const queueSplit = splitByMarket(queues, (queue) => queueMarket(queue.queue, supportedMarkets), activeMarkets);
  const summarySplit = splitByMarket(summaries, (summary) => summary.market, activeMarkets);

  const queueOptions = useMemo(
    () => Array.from(new Set(jobs.map((job) => job.queue).filter(Boolean))).sort(),
    [jobs]
  );
  const marketOptions = useMemo(
    () => Array.from(new Set(jobs.map((job) => job.market).filter(Boolean))).sort(),
    [jobs]
  );

  const filteredJobs = useMemo(() => {
    const normalizedSearch = taskSearch.trim().toLowerCase();
    return jobs.filter((job) => {
      if (stateFilter !== 'all' && job.state !== stateFilter) return false;
      if (queueFilter !== 'all' && job.queue !== queueFilter) return false;
      if (marketFilter !== 'all' && job.market !== marketFilter) return false;
      if (!normalizedSearch) return true;
      return [
        job.task_name,
        job.task_id,
        job.queue,
        job.market,
        job.worker,
        job.wait_reason,
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(normalizedSearch));
    });
  }, [jobs, stateFilter, queueFilter, marketFilter, taskSearch]);

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
      <Typography variant="h5" gutterBottom>
        Operations
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Per-market telemetry, live queue inventory, and safe background-job controls.
      </Typography>

      <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
        Market activity
      </Typography>
      {bootstrap?.background_warning && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          {bootstrap.background_warning}
        </Typography>
      )}
      {bootstrapStatusQuery.isError && (
        <Typography variant="body2" color="error" sx={{ mb: 1.5 }}>
          {bootstrapStatusQuery.data
            ? 'Failed to refresh publication status. Showing last known status.'
            : 'Failed to load publication status.'}
        </Typography>
      )}
      <UnpublishedDataAlert status={bootstrapStatusQuery.data} />
      {activityQuery.isLoading && !hasMarketActivity ? (
        <CircularProgress size={20} />
      ) : (
        <>
          {activityQuery.isError && (
            <Typography variant="body2" color="error" sx={{ mb: 1.5 }}>
              Failed to refresh runtime activity. Showing last known status.
            </Typography>
          )}
          {hasMarketActivity ? (
            <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap sx={{ mb: 3 }}>
              {marketActivity.map((activity) => (
                <MarketActivityCard key={activity.activity_id ?? activity.market} activity={activity} />
              ))}
            </Stack>
          ) : (
            !activityQuery.isLoading && (
              <Typography variant="body2" color={activityQuery.isError ? 'error' : 'text.secondary'}>
                {activityQuery.isError
                  ? 'Failed to load runtime activity.'
                  : 'No runtime activity available.'}
              </Typography>
            )
          )}
        </>
      )}

      <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
        Job console
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        Shows queued, waiting, reserved, running, stale, and stuck jobs across all Celery queues.
      </Typography>
      {jobActionError && (
        <Alert severity="warning" sx={{ mb: 1.5 }}>
          {jobActionError}
        </Alert>
      )}
      <Stack direction="row" spacing={2} sx={{ mb: 2 }} flexWrap="wrap" useFlexGap>
        <FormControl size="small" sx={{ minWidth: 160 }}>
          <InputLabel id="ops-state-filter-label">State</InputLabel>
          <Select
            labelId="ops-state-filter-label"
            value={stateFilter}
            label="State"
            onChange={(event) => setStateFilter(event.target.value)}
          >
            <MenuItem value="all">All states</MenuItem>
            {['queued', 'waiting', 'reserved', 'running', 'stale', 'stuck', 'failed', 'cancelled'].map((state) => (
              <MenuItem key={state} value={state}>{state}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 180 }}>
          <InputLabel id="ops-queue-filter-label">Queue</InputLabel>
          <Select
            labelId="ops-queue-filter-label"
            value={queueFilter}
            label="Queue"
            onChange={(event) => setQueueFilter(event.target.value)}
          >
            <MenuItem value="all">All queues</MenuItem>
            {queueOptions.map((queue) => (
              <MenuItem key={queue} value={queue}>{queue}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 160 }}>
          <InputLabel id="ops-market-filter-label">Market</InputLabel>
          <Select
            labelId="ops-market-filter-label"
            value={marketFilter}
            label="Market"
            onChange={(event) => setMarketFilter(event.target.value)}
          >
            <MenuItem value="all">All markets</MenuItem>
            {marketOptions.map((market) => (
              <MenuItem key={market} value={market}>{market}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <TextField
          size="small"
          label="Task search"
          value={taskSearch}
          onChange={(event) => setTaskSearch(event.target.value)}
          sx={{ minWidth: 240 }}
        />
      </Stack>
      <JobsData query={jobsQuery}>
        <JobsTable
          jobs={filteredJobs}
          onCancel={setConfirmJob}
          cancellingTaskId={cancelMutation.variables?.taskId}
          cancelInFlight={cancelMutation.isPending}
        />
      </JobsData>
      <Dialog open={Boolean(confirmAction)} onClose={() => setConfirmJob(null)}>
        <DialogTitle>{confirmAction?.label}</DialogTitle>
        <DialogContent>
          <DialogContentText>{confirmAction?.detail}</DialogContentText>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            {confirmJob?.task_name} · {confirmJob?.task_id} · {confirmJob?.state}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmJob(null)}>Back</Button>
          <Button
            variant="contained"
            color={confirmAction?.danger ? 'error' : 'primary'}
            onClick={() => {
              cancelMutation.mutate({
                taskId: confirmJob.task_id,
                expectedStrategy: confirmJob.cancel_strategy,
              });
              setConfirmJob(null);
            }}
          >
            {confirmAction?.label}
          </Button>
        </DialogActions>
      </Dialog>

      {features.tasks && <SchedulesSection activeMarkets={activeMarkets} />}

      {/* Anchor for "See its claim in Lease status" (Scheduled jobs); the
          heading renders even while the jobs data loads or fails. */}
      <Typography id="lease-status" variant="h6" sx={{ mt: 3, mb: 1 }}>
        Lease status
      </Typography>
      <JobsData query={jobsQuery}>
        <LeaseSummary
          leases={leases}
          onReleaseChain={setConfirmChain}
          releasingToken={releaseChainMutation.isPending ? releaseChainMutation.variables?.token : null}
          activeMarkets={activeMarkets}
        />
      </JobsData>
      <Dialog open={Boolean(confirmChain)} onClose={() => setConfirmChain(null)}>
        <DialogTitle>Release {confirmChain?.market} pipeline chain claim</DialogTitle>
        <DialogContent>
          <DialogContentText>
            The claim stops a second daily pipeline from starting while one is in progress. Release it only
            if that market&apos;s pipeline stages are gone from the job console (for example, a stage was
            revoked); otherwise the next Run starts a second pipeline alongside the first.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmChain(null)}>Back</Button>
          <Button
            variant="contained"
            color="error"
            onClick={() => {
              releaseChainMutation.mutate(confirmChain);
              setConfirmChain(null);
            }}
          >
            Release
          </Button>
        </DialogActions>
      </Dialog>

      <Stack direction={{ xs: 'column', lg: 'row' }} spacing={2} sx={{ mt: 3 }}>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            Queues
          </Typography>
          <JobsData query={jobsQuery}>
            <QueueSummaryTable queues={queueSplit.shown} />
            <InactiveMarkets markets={queueSplit.hiddenMarkets}>
              <QueueSummaryTable queues={queueSplit.hidden} />
            </InactiveMarkets>
          </JobsData>
        </Box>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            Workers
          </Typography>
          <JobsData query={jobsQuery}>
            <WorkerStatusTable workers={workers} />
          </JobsData>
        </Box>
      </Stack>

      <Box sx={{ my: 2 }}>
        {alertsQuery.isLoading ? (
          <CircularProgress size={20} />
        ) : alertsQuery.isError ? (
          <Typography variant="body2" color="error">
            Failed to load telemetry summaries. Check backend connectivity.
          </Typography>
        ) : (
          <>
            <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
              {summarySplit.shown.map((s) => (
                <MarketSummaryCard key={s.market} summary={s} />
              ))}
            </Stack>
            <InactiveMarkets markets={summarySplit.hiddenMarkets}>
              <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
                {summarySplit.hidden.map((s) => (
                  <MarketSummaryCard key={s.market} summary={s} />
                ))}
              </Stack>
            </InactiveMarkets>
          </>
        )}
      </Box>

      <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
        Active alerts
      </Typography>
      {ackError && (
        <Typography variant="caption" color="error" sx={{ display: 'block', mb: 1 }}>
          Acknowledge failed: {ackError}
        </Typography>
      )}
      {alertsQuery.isLoading ? (
        <CircularProgress size={20} />
      ) : alertsQuery.isError ? (
        <Typography variant="body2" color="error">
          Failed to load alerts.
        </Typography>
      ) : (
        <AlertsTable alerts={alerts} onAcknowledge={(id) => ackMutation.mutate(id)} />
      )}
      <SocialSignalHealthPanel />
      {features.themes && <ExposureResearchWorkspace />}
    </Container>
  );
}
