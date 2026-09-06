import { Table, TableBody, TableCell, TableContainer, TableRow, Tooltip, Typography } from '@mui/material';
import { formatOptionsMetric } from './optionsFormatting';

const humanize = (value) => String(value || '').replaceAll('_', ' ');

export function UnavailableMetric({ metric }) {
  const reason = metric?.reason_codes?.map(humanize).join(', ') || 'metric unavailable';
  return (
    <Tooltip title={reason}>
      <Typography component="span" color="text.secondary" aria-label={`Unavailable: ${reason}`}>—</Typography>
    </Tooltip>
  );
}

export default function OptionsMetricTable({ metrics }) {
  const percentageMetrics = new Set([
    'atm_iv',
    'skew_25_delta',
    'realized_volatility',
    'vrp',
    'iv_percentile',
    'iv_rank',
    'atm_iv_change_5',
    'skew_25_delta_change_5',
    'realized_volatility_change_5',
    'vrp_change_5',
  ]);
  return (
    <TableContainer>
      <Table size="small" aria-label="Options metrics">
        <TableBody>
          {Object.entries(metrics).map(([name, metric]) => (
            <TableRow key={name}>
              <TableCell component="th" scope="row">{metric.label}</TableCell>
              <TableCell align="right">
                {metric.available
                  ? formatOptionsMetric(metric, {
                      percent: percentageMetrics.has(name),
                    })
                  : <UnavailableMetric metric={metric} />}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
