/**
 * Market coverage: one row per active market saying whether it can run, and
 * in plain words what is missing when it cannot (see coverageProblems.js).
 */
import {
  Alert,
  Box,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import { marketCoverage } from './coverageProblems';

const WORKER_LABELS = [
  ['market_jobs', 'Pipeline'],
  ['data_fetch', 'Prices'],
  ['user_scans', 'Scans'],
];

function ageLabel(iso) {
  const minutes = Math.floor((Date.now() - Date.parse(iso)) / 60000);
  if (Number.isNaN(minutes)) return '—';
  if (minutes < 60) return `${Math.max(minutes, 0)}m ago`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
  return `${Math.floor(minutes / 1440)}d ago`;
}

function Workers({ workers }) {
  return (
    <Stack direction="row" spacing={1.5}>
      {WORKER_LABELS.map(([family, label]) => {
        const count = workers[family];
        return (
          <Typography key={family} variant="body2" color={count === 0 ? 'error' : 'text.primary'}>
            {label}: {count ?? '—'}
          </Typography>
        );
      })}
    </Stack>
  );
}

// Problems first (warning), then what is unknown or merely notable; OK only
// when there is nothing at all to say.
function Status({ row, tracked }) {
  const unchecked = uncheckedOf(row, tracked);
  if (!row.problems.length && !unchecked.length && !row.notes.length) {
    return <Typography variant="body2" color="success.main">OK</Typography>;
  }
  return (
    <Box>
      {row.problems.map((problem) => (
        <Typography key={problem} variant="body2" color="warning.main">{problem}</Typography>
      ))}
      {[...unchecked, ...row.notes].map((line) => (
        <Typography key={line} variant="body2" color="text.secondary">{line}</Typography>
      ))}
    </Box>
  );
}

// What could not be checked: never read as OK, and keeps the summary from
// calling the market wired up. tracked false: pipeline runs are not shown here.
function uncheckedOf(row, tracked) {
  if (row.outsideDeployment) return [];
  const unchecked = [];
  if (Object.values(row.workers).some((count) => count === null)) unchecked.push('Worker counts unknown');
  if (tracked && row.enabled && !row.pipelineKnown) unchecked.push('Pipeline runs unknown');
  if (tracked && row.pipelineNeverRan) unchecked.push('No daily pipeline run recorded yet');
  if (tracked && row.pipelineScheduleUnconfirmed) unchecked.push("Last run was manual, so Beat's schedule is not confirmed");
  return unchecked;
}

// markets: active markets in catalog order. queues: the jobs payload's queue
// summaries. tasks: scheduled tasks; undefined while loading, null when the
// tasks feature is off (pipeline runs are then left out). deploymentMarkets:
// the deployment's ENABLED_MARKETS, undefined when unknown.
export default function MarketCoverage({ markets, enabledMarkets = [], deploymentMarkets, queues = [], tasks }) {
  const tracked = tasks !== null;
  const enabled = new Set(enabledMarkets);
  const rows = [...markets]
    .sort((a, b) => Number(enabled.has(b)) - Number(enabled.has(a)))
    .map((market) => marketCoverage({
      market,
      enabled: enabled.has(market),
      deploymentMarkets,
      queues,
      pipelineTask: tasks?.find((task) => task.market === market && task.name.startsWith('daily-market-pipeline-')),
    }));
  const needAttention = rows.filter((row) => row.problems.length).map((row) => row.market);
  const allChecked = rows.length > 0 && rows.every((row) => uncheckedOf(row, tracked).length === 0);

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" sx={{ mb: 0.5 }}>Market coverage</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        Whether each market can run. The app&apos;s enabled markets are chosen at setup; workers and Beat follow
        ENABLED_MARKETS in the deployment environment.
      </Typography>
      {needAttention.length > 0 ? (
        <Alert severity="warning" sx={{ mb: 1.5 }}>
          {needAttention.length === 1 ? '1 market needs' : `${needAttention.length} markets need`} attention:{' '}
          {needAttention.join(', ')}.
        </Alert>
      ) : allChecked ? (
        <Typography variant="body2" color="success.main" sx={{ mb: 1 }}>Every market shown is wired up.</Typography>
      ) : rows.length > 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Some checks are not available yet.</Typography>
      )}
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Market</TableCell>
            <TableCell>In app</TableCell>
            <TableCell>Workers</TableCell>
            <TableCell>Daily pipeline</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.market}>
              <TableCell component="th" scope="row">{row.market}</TableCell>
              <TableCell>
                <Chip
                  size="small"
                  label={row.enabled ? 'Enabled' : 'Not enabled'}
                  color={row.enabled ? 'success' : 'default'}
                  variant={row.enabled ? 'filled' : 'outlined'}
                />
              </TableCell>
              <TableCell><Workers workers={row.workers} /></TableCell>
              <TableCell>
                <Typography variant="body2">
                  {!row.pipelineKnown ? '—' : row.pipelineRun?.started_at ? ageLabel(row.pipelineRun.started_at) : 'Never'}
                </Typography>
              </TableCell>
              <TableCell><Status row={row} tracked={tracked} /></TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}
