/**
 * Operations / telemetry dashboard (bead asia.10.2).
 *
 * Polls the backend telemetry API every 30s for per-market gauges and
 * active alerts. The header entry is a discreet icon button in Layout.jsx —
 * this page isn't a top-level tab.
 */
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  LinearProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
  Button,
} from '@mui/material';
import { fetchAlerts, acknowledgeAlert } from '../api/telemetry';
import { useRuntimeActivity } from '../hooks/useRuntimeActivity';

const POLL_MS = 30000;

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
};

function formatLagSeconds(seconds) {
  if (seconds == null) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
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

function formatTimestamp(value) {
  if (!value) return '—';
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
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

export default function OperationsPage() {
  const queryClient = useQueryClient();
  const [ackError, setAckError] = useState(null);
  const activityQuery = useRuntimeActivity();

  // Single poll: /alerts returns both summaries and alerts so the dashboard
  // doesn't fan out to two endpoints (the eval already builds the summaries).
  const alertsQuery = useQuery({
    queryKey: ['telemetry', 'alerts'],
    queryFn: () => fetchAlerts({ evaluate: true }),
    refetchInterval: POLL_MS,
  });

  const ackMutation = useMutation({
    mutationFn: (alertId) => acknowledgeAlert(alertId, 'operator'),
    onSuccess: () => {
      setAckError(null);
      queryClient.invalidateQueries({ queryKey: ['telemetry', 'alerts'] });
    },
    onError: (err) => setAckError(err?.response?.data?.detail || err.message),
  });

  const summaries = alertsQuery.data?.summaries || [];
  const alerts = alertsQuery.data?.alerts || [];
  const marketActivity = activityQuery.data?.markets || [];
  const bootstrap = activityQuery.data?.bootstrap;

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
      <Typography variant="h5" gutterBottom>
        Operations
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Per-market telemetry and active alerts. Polls every {POLL_MS / 1000}s.
      </Typography>

      <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
        Market activity
      </Typography>
      {bootstrap?.background_warning && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          {bootstrap.background_warning}
        </Typography>
      )}
      {activityQuery.isLoading ? (
        <CircularProgress size={20} />
      ) : activityQuery.isError ? (
        <Typography variant="body2" color="error">
          Failed to load runtime activity.
        </Typography>
      ) : (
        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap sx={{ mb: 3 }}>
          {marketActivity.map((activity) => (
            <MarketActivityCard key={activity.market} activity={activity} />
          ))}
        </Stack>
      )}

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
