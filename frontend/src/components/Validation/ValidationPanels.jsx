import { Fragment, useState } from 'react';
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
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';

import { ACTION_STATE_META } from '../../features/opportunityState/actionState';
import { formatPatternName } from '../../utils/formatUtils';

const GROUP_HORIZONS = [1, 5, 20, 60];
const SMALL_SAMPLE_SIZE = 20;
// Dimensions whose values have a display name; the others show their value as it is.
const GROUP_VALUE_LABELS = {
  action_state: (value) => ACTION_STATE_META[value]?.label ?? value,
  setup_pattern: formatPatternName,
};

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

function MetricCard({ title, horizon, benchmarkLabel, stopLossPct }) {
  const excessSample = horizon?.excess_sample_size ?? 0;
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
          <MetricRow label={`Avg vs ${benchmarkLabel} (no stop, n=${excessSample})`} value={formatPercentValue(horizon?.avg_excess_return_pct)} />
          <MetricRow label={`Median vs ${benchmarkLabel} (no stop, n=${excessSample})`} value={formatPercentValue(horizon?.median_excess_return_pct)} />
          <MetricRow label={`Avg with ${stopLossPct}% stop`} value={formatPercentValue(horizon?.avg_stopped_return_pct)} />
          <MetricRow label={`Median with ${stopLossPct}% stop`} value={formatPercentValue(horizon?.median_stopped_return_pct)} />
          <MetricRow label="Avg MFE (no stop)" value={formatPercentValue(horizon?.avg_mfe_pct)} />
          <MetricRow label="Avg MAE (no stop)" value={formatPercentValue(horizon?.avg_mae_pct)} />
          <MetricRow label="Stopped out" value={formatRate(horizon?.stopped_out_rate)} />
          <MetricRow label="Skipped" value={horizon?.skipped_missing_history ?? 0} />
          <MetricRow label="Pending" value={horizon?.pending_windows ?? 0} />
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

export function ValidationSummaryCards({ horizons, benchmarkSymbol, stopLossPct }) {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)', xl: 'repeat(4, 1fr)' }, gap: 1.5 }}>
      {(horizons || []).map((horizon) => (
        <MetricCard
          key={horizon.horizon_sessions}
          title={`${horizon.horizon_sessions} ${horizon.horizon_sessions === 1 ? 'Session' : 'Sessions'}`}
          horizon={horizon}
          benchmarkLabel={benchmarkSymbol || 'Benchmark'}
          stopLossPct={stopLossPct}
        />
      ))}
    </Box>
  );
}

export function ValidationRecentEventsTable({ events, showSourceKind = true, benchmarkSymbol }) {
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
            <TableCell align="right">20S</TableCell>
            <TableCell align="right">60S</TableCell>
            <TableCell align="right">5S vs {benchmarkSymbol || 'Benchmark'}</TableCell>
            <TableCell align="right">MFE</TableCell>
            <TableCell align="right">MAE</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {(events || []).map((event, index) => {
            const horizonSx = (horizon) => (
              event.counted_horizons?.includes(horizon) === false ? { color: 'text.secondary' } : undefined
            );
            const fiveSessionSx = horizonSx(5);
            return (
            <TableRow
              key={`${event.source_ref}:${event.event_at}:${event.attributes?.symbol || 'symbol'}:${index}`}
            >
              <TableCell>
                {event.attributes?.symbol ? (
                  <Link component={RouterLink} to={`/stocks/${encodeURIComponent(event.attributes.symbol)}`} underline="hover">
                    {event.attributes.symbol}
                  </Link>
                ) : '-'}
                {event.counted === false && (
                  <Chip
                    size="small"
                    label="repeat"
                    title="Inside an earlier event's 5-session window; left out of the 5-session summary"
                    sx={{ ml: 0.75 }}
                  />
                )}
              </TableCell>
              {showSourceKind && (
                <TableCell>
                  <Chip size="small" variant="outlined" label={event.source_kind.replace('_', ' ')} />
                </TableCell>
              )}
              <TableCell>{formatDate(event.event_at)}</TableCell>
              <TableCell>{formatDate(event.entry_at)}</TableCell>
              <TableCell align="right">{formatPercentValue(event.return_1s_pct)}</TableCell>
              <TableCell align="right" sx={fiveSessionSx}>{formatPercentValue(event.return_5s_pct)}</TableCell>
              <TableCell align="right" sx={horizonSx(20)}>{formatPercentValue(event.return_20s_pct)}</TableCell>
              <TableCell align="right" sx={horizonSx(60)}>{formatPercentValue(event.return_60s_pct)}</TableCell>
              <TableCell align="right" sx={fiveSessionSx}>{formatPercentValue(event.excess_return_5s_pct)}</TableCell>
              <TableCell align="right" sx={fiveSessionSx}>{formatPercentValue(event.mfe_5s_pct)}</TableCell>
              <TableCell align="right" sx={fiveSessionSx}>{formatPercentValue(event.mae_5s_pct)}</TableCell>
            </TableRow>
            );
          })}
          {(!events || events.length === 0) && (
            <TableRow>
              <TableCell colSpan={showSourceKind ? 11 : 10} align="center">
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

export function ValidationGroupsTable({ groups, benchmarkSymbol, stopLossPct }) {
  const [horizonSessions, setHorizonSessions] = useState(20);
  return (
    <Stack spacing={1}>
      <ToggleButtonGroup
        exclusive
        size="small"
        aria-label="Group horizon"
        value={horizonSessions}
        onChange={(_, value) => value && setHorizonSessions(value)}
      >
        {GROUP_HORIZONS.map((value) => (
          <ToggleButton key={value} value={value}>{value}S</ToggleButton>
        ))}
      </ToggleButtonGroup>
      <TableContainer>
        <Table size="small" aria-label="By group">
          <TableHead>
            <TableRow>
              <TableCell>Group</TableCell>
              <TableCell align="right">Sample</TableCell>
              <TableCell align="right">Positive</TableCell>
              <TableCell align="right">Avg Return</TableCell>
              <TableCell align="right">{`Avg vs ${benchmarkSymbol || 'Benchmark'} (no stop)`}</TableCell>
              <TableCell align="right">{`Avg with ${stopLossPct}% stop`}</TableCell>
              <TableCell align="right">Stopped out</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(groups || []).map((group, index) => {
              const horizon = group.horizons.find((item) => item.horizon_sessions === horizonSessions);
              const sampleSize = horizon?.sample_size ?? 0;
              const excessSampleSize = horizon?.excess_sample_size ?? 0;
              const isSmallSample = sampleSize < SMALL_SAMPLE_SIZE;
              return (
                <Fragment key={`${group.dimension}:${group.value}`}>
                  {group.dimension !== groups[index - 1]?.dimension && (
                    <TableRow>
                      <TableCell colSpan={7} sx={{ fontWeight: 700 }}>{group.label}</TableCell>
                    </TableRow>
                  )}
                  <TableRow
                    title={isSmallSample ? `Small sample: fewer than ${SMALL_SAMPLE_SIZE} events` : undefined}
                    sx={isSmallSample ? { '& td': { color: 'text.secondary' } } : undefined}
                  >
                    <TableCell>{GROUP_VALUE_LABELS[group.dimension]?.(group.value) ?? group.value}</TableCell>
                    <TableCell align="right">{sampleSize}</TableCell>
                    <TableCell align="right">{formatRate(horizon?.positive_rate)}</TableCell>
                    <TableCell align="right">{formatPercentValue(horizon?.avg_return_pct)}</TableCell>
                    <TableCell align="right">
                      {formatPercentValue(horizon?.avg_excess_return_pct)}
                      {excessSampleSize < sampleSize && ` (n=${excessSampleSize})`}
                    </TableCell>
                    <TableCell align="right">{formatPercentValue(horizon?.avg_stopped_return_pct)}</TableCell>
                    <TableCell align="right">{formatRate(horizon?.stopped_out_rate)}</TableCell>
                  </TableRow>
                </Fragment>
              );
            })}
            {(!groups || groups.length === 0) && (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No grouped results for this selection.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Stack>
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

export function ValidationSection({ title, degradedReasons, horizons, recentEvents, groups, showSourceKind = true, benchmarkSymbol, stopLossPct }) {
  return (
    <Stack spacing={1.5}>
      {title && (
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
      )}
      <ValidationDegradedAlert degradedReasons={degradedReasons} />
      <ValidationSummaryCards horizons={horizons} benchmarkSymbol={benchmarkSymbol} stopLossPct={stopLossPct} />
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
          Recent Events
        </Typography>
        <ValidationRecentEventsTable events={recentEvents} showSourceKind={showSourceKind} benchmarkSymbol={benchmarkSymbol} />
      </Box>
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
          By Group
        </Typography>
        <ValidationGroupsTable groups={groups} benchmarkSymbol={benchmarkSymbol} stopLossPct={stopLossPct} />
      </Box>
    </Stack>
  );
}
