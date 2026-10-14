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

function Status({ row }) {
  if (row.problems.length) {
    return (
      <Box>
        {row.problems.map((problem) => (
          <Typography key={problem} variant="body2" color="warning.main">{problem}</Typography>
        ))}
      </Box>
    );
  }
  const unchecked = uncheckedOf(row);
  return (
    <Typography variant="body2" color={unchecked.length ? 'text.secondary' : 'success.main'}>
      {unchecked.length ? unchecked.join('; ') : 'OK'}
    </Typography>
  );
}

// What could not be checked for a row with no problems: never read as OK.
function uncheckedOf(row) {
  const unchecked = [];
  if (Object.values(row.workers).some((count) => count === null)) unchecked.push('Worker counts unknown');
  if (row.enabled && !row.pipelineKnown) unchecked.push('Pipeline runs unknown');
  return unchecked;
}

// markets: active markets in catalog order. queues: the jobs payload's queue
// summaries. tasks: scheduled tasks, or undefined while not loaded.
export default function MarketCoverage({ markets, enabledMarkets = [], queues = [], tasks }) {
  const enabled = new Set(enabledMarkets);
  const rows = [...markets]
    .sort((a, b) => Number(enabled.has(b)) - Number(enabled.has(a)))
    .map((market) => marketCoverage({
      market,
      enabled: enabled.has(market),
      queues,
      pipelineTask: tasks?.find((task) => task.market === market && task.name.startsWith('daily-market-pipeline-')),
    }));
  const needAttention = rows.filter((row) => row.problems.length).map((row) => row.market);
  const allChecked = rows.every((row) => uncheckedOf(row).length === 0);

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
      ) : (
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
              <TableCell><Status row={row} /></TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}
