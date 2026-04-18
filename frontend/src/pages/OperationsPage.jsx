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
import { cancelOperationsJob, fetchOperationsJobs } from '../api/operations';
import { useRuntimeActivity } from '../hooks/useRuntimeActivity';

const POLL_MS = 30000;
const ACTIVE_JOB_STATES = new Set(['queued', 'waiting', 'reserved', 'running', 'stale', 'stuck']);
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

function formatCancelLabel(strategy) {
  switch (strategy) {
    case 'revoke':
      return 'Revoke';
    case 'scan_cancel':
      return 'Cancel scan';
    case 'force_cancel_refresh':
      return 'Force cancel';
    case 'revoke_and_remove_from_queue':
      return 'Cancel';
    default:
      return 'Unavailable';
  }
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
  const progressValue = Number.isFinite(activity.percent) ? activity.percent : null;

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

function LeaseSummary({ leases }) {
  const externalHolder = leases?.external_fetch_global;
  const marketHolders = leases?.market_workload || {};

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
            {Object.entries(marketHolders).map(([market, holder]) => (
              <Box key={market}>
                <Typography variant="body2">
                  <strong>{market}:</strong> {holder?.task_name || 'idle'}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {holder?.task_id ? `Task ID: ${holder.task_id}` : 'No active workload lease'}
                </Typography>
              </Box>
            ))}
          </Stack>
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
        </TableRow>
      </TableHead>
      <TableBody>
        {queues.map((queue) => (
          <TableRow key={queue.queue}>
            <TableCell>{queue.queue}</TableCell>
            <TableCell>{queue.depth}</TableCell>
            <TableCell>{formatLagSeconds(queue.oldest_age_seconds)}</TableCell>
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
          <TableCell>Age</TableCell>
          <TableCell>Blocker</TableCell>
          <TableCell>Heartbeat</TableCell>
          <TableCell>Action</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {jobs.map((job) => {
          const canCancel = job.cancel_strategy && job.cancel_strategy !== 'unsupported';
          const isCancelling = cancelInFlight && cancellingTaskId === job.task_id;
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
              <TableCell>{formatLagSeconds(job.age_seconds)}</TableCell>
              <TableCell>{job.wait_reason || '—'}</TableCell>
              <TableCell>{formatLagSeconds(job.heartbeat_lag_seconds)}</TableCell>
              <TableCell>
                {canCancel ? (
                  <Button
                    size="small"
                    variant="outlined"
                    color={job.cancel_strategy === 'force_cancel_refresh' ? 'error' : 'primary'}
                    onClick={() => onCancel(job.task_id)}
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

  const alertsQuery = useQuery({
    queryKey: ['telemetry', 'alerts'],
    queryFn: () => fetchAlerts({ evaluate: true }),
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
    mutationFn: cancelOperationsJob,
    onSuccess: (payload) => {
      setJobActionError(null);
      queryClient.invalidateQueries({ queryKey: ['operations', 'jobs'] });
      queryClient.invalidateQueries({ queryKey: ['runtimeActivity'] });
      if (payload.status !== 'accepted') {
        setJobActionError(payload.message);
      }
    },
    onError: (err) => setJobActionError(err?.response?.data?.detail || err.message),
  });

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
                <MarketActivityCard key={activity.market} activity={activity} />
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
      {jobsQuery.isLoading ? (
        <CircularProgress size={20} />
      ) : jobsQuery.isError ? (
        <Typography variant="body2" color="error">
          Failed to load Operations job inventory.
        </Typography>
      ) : (
        <JobsTable
          jobs={filteredJobs}
          onCancel={(taskId) => cancelMutation.mutate(taskId)}
          cancellingTaskId={cancelMutation.variables}
          cancelInFlight={cancelMutation.isPending}
        />
      )}

      <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
        Lease status
      </Typography>
      {jobsQuery.isLoading ? (
        <CircularProgress size={20} />
      ) : (
        <LeaseSummary leases={leases} />
      )}

      <Stack direction={{ xs: 'column', lg: 'row' }} spacing={2} sx={{ mt: 3 }}>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            Queues
          </Typography>
          {jobsQuery.isLoading ? <CircularProgress size={20} /> : <QueueSummaryTable queues={queues} />}
        </Box>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            Workers
          </Typography>
          {jobsQuery.isLoading ? <CircularProgress size={20} /> : <WorkerStatusTable workers={workers} />}
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
          <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
            {summaries.map((s) => (
              <MarketSummaryCard key={s.market} summary={s} />
            ))}
          </Stack>
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
    </Container>
  );
}
