import { Link as RouterLink } from 'react-router-dom';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  Link,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

function formatPercentValue(value) {
  if (value == null) return '-';
  return `${Number(value).toFixed(1)}%`;
}

function formatRate(value) {
  if (value == null) return '-';
  return `${(Number(value) * 100).toFixed(0)}%`;
}

function formatDate(value) {
  if (!value) return '-';
  if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [year, month, day] = value.split('-').map(Number);
    return new Date(year, month - 1, day).toLocaleDateString();
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString();
}

function MetricCard({ title, horizon }) {
  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
          {title}
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.75 }}>
          <MetricRow label="Sample" value={horizon?.sample_size ?? 0} />
          <MetricRow label="Positive" value={formatRate(horizon?.positive_rate)} />
          <MetricRow label="Avg Return" value={formatPercentValue(horizon?.avg_return_pct)} />
          <MetricRow label="Median" value={formatPercentValue(horizon?.median_return_pct)} />
          <MetricRow label="Avg MFE" value={formatPercentValue(horizon?.avg_mfe_pct)} />
          <MetricRow label="Avg MAE" value={formatPercentValue(horizon?.avg_mae_pct)} />
          <MetricRow label="Skipped" value={horizon?.skipped_missing_history ?? 0} />
        </Box>
      </CardContent>
    </Card>
  );
}

function MetricRow({ label, value }) {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1 }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" sx={{ fontSize: '0.8rem', fontWeight: 600 }}>
        {value}
      </Typography>
    </Box>
  );
}

export function ValidationSummaryCards({ horizons }) {
  const oneSession = horizons?.find((item) => item.horizon_sessions === 1);
  const fiveSessions = horizons?.find((item) => item.horizon_sessions === 5);

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 1.5 }}>
      <MetricCard title="1 Session" horizon={oneSession} />
      <MetricCard title="5 Sessions" horizon={fiveSessions} />
    </Box>
  );
}

export function ValidationRecentEventsTable({ events, showSourceKind = true }) {
  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            {showSourceKind && <TableCell>Source</TableCell>}
            <TableCell>Event</TableCell>
            <TableCell>Entry</TableCell>
            <TableCell align="right">1S</TableCell>
            <TableCell align="right">5S</TableCell>
            <TableCell align="right">MFE</TableCell>
            <TableCell align="right">MAE</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {(events || []).map((event) => (
            <TableRow key={`${event.source_ref}:${event.event_at}:${event.attributes?.symbol || 'symbol'}`}>
              <TableCell>
                {event.attributes?.symbol ? (
                  <Link component={RouterLink} to={`/stocks/${encodeURIComponent(event.attributes.symbol)}`} underline="hover">
                    {event.attributes.symbol}
                  </Link>
                ) : '-'}
              </TableCell>
              {showSourceKind && (
                <TableCell>
                  <Chip size="small" variant="outlined" label={event.source_kind.replace('_', ' ')} />
                </TableCell>
              )}
              <TableCell>{formatDate(event.event_at)}</TableCell>
              <TableCell>{formatDate(event.entry_at)}</TableCell>
              <TableCell align="right">{formatPercentValue(event.return_1s_pct)}</TableCell>
              <TableCell align="right">{formatPercentValue(event.return_5s_pct)}</TableCell>
              <TableCell align="right">{formatPercentValue(event.mfe_5s_pct)}</TableCell>
              <TableCell align="right">{formatPercentValue(event.mae_5s_pct)}</TableCell>
            </TableRow>
          ))}
          {(!events || events.length === 0) && (
            <TableRow>
              <TableCell colSpan={showSourceKind ? 8 : 7} align="center">
                <Typography variant="body2" color="text.secondary">
                  No validation events are available for this selection.
                </Typography>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export function ValidationFailureClustersTable({ clusters }) {
  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Cluster</TableCell>
            <TableCell align="right">Sample</TableCell>
            <TableCell align="right">Avg 5S</TableCell>
            <TableCell align="right">Median 5S</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {(clusters || []).map((cluster) => (
            <TableRow key={cluster.cluster_key}>
              <TableCell>{cluster.label}</TableCell>
              <TableCell align="right">{cluster.sample_size}</TableCell>
              <TableCell align="right">{formatPercentValue(cluster.avg_return_pct)}</TableCell>
              <TableCell align="right">{formatPercentValue(cluster.median_return_pct)}</TableCell>
            </TableRow>
          ))}
          {(!clusters || clusters.length === 0) && (
            <TableRow>
              <TableCell colSpan={4} align="center">
                <Typography variant="body2" color="text.secondary">
                  No losing clusters are available for this selection.
                </Typography>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export function ValidationDegradedAlert({ degradedReasons }) {
  if (!degradedReasons?.length) {
    return null;
  }

  return (
    <Alert severity="info">
      Validation is partially degraded: {degradedReasons.join(', ')}.
    </Alert>
  );
}

export function ValidationSection({ title, degradedReasons, horizons, recentEvents, failureClusters, showSourceKind = true }) {
  return (
    <Stack spacing={1.5}>
      {title && (
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
      )}
      <ValidationDegradedAlert degradedReasons={degradedReasons} />
      <ValidationSummaryCards horizons={horizons} />
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
          Recent Events
        </Typography>
        <ValidationRecentEventsTable events={recentEvents} showSourceKind={showSourceKind} />
      </Box>
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
          Failure Clusters
        </Typography>
        <ValidationFailureClustersTable clusters={failureClusters} />
      </Box>
    </Stack>
  );
}
