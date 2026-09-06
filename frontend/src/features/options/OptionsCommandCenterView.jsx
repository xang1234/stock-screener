import { useMemo, useState } from 'react';
import {
  Box,
  Button,
  Paper,
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
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';

import OptionsQualityBadge from './OptionsQualityBadge';
import OptionsSourceBadges from './OptionsSourceBadges';
import OptionsStatusBanner from './OptionsStatusBanner';
import { UnavailableMetric } from './OptionsMetricTable';
import { formatOptionsMetric } from './optionsFormatting';

const LENSES = {
  gamma: [
    { name: 'net_gex' },
    { name: 'gamma_flip' },
    { name: 'call_wall' },
    { name: 'put_wall' },
  ],
  volatility: [
    { name: 'atm_iv', percent: true },
    { name: 'realized_volatility', percent: true },
    { name: 'vrp', percent: true },
  ],
  skew: [
    { name: 'skew_25_delta', percent: true },
    { name: 'near_spot_volume_concentration', percent: true },
  ],
  activity: [
    { name: 'activity_intensity' },
    { name: 'call_put_volume_ratio' },
    { name: 'volume_oi_ratio' },
    { name: 'near_spot_volume_concentration', percent: true },
    { name: 'near_spot_open_interest_concentration', percent: true },
  ],
};

const sortRows = (items, metricName, direction) => [...items].sort((left, right) => {
  const leftMetric = left.metrics[metricName];
  const rightMetric = right.metrics[metricName];
  if (!leftMetric.available && !rightMetric.available) return left.symbol.localeCompare(right.symbol);
  if (!leftMetric.available) return 1;
  if (!rightMetric.available) return -1;
  const difference = leftMetric.value - rightMetric.value;
  return difference === 0 ? left.symbol.localeCompare(right.symbol) : difference * direction;
});

export default function OptionsCommandCenterView({ data, onOpenSymbol }) {
  const [lens, setLens] = useState('gamma');
  const [sort, setSort] = useState({ metric: LENSES.gamma[0].name, direction: -1 });
  const columns = LENSES[lens];
  const rows = useMemo(
    () => sortRows(data.items, sort.metric, sort.direction),
    [data.items, sort],
  );
  const rankedSymbols = useMemo(
    () => rows.filter((item) => item.metrics[sort.metric].available).map((item) => item.symbol),
    [rows, sort.metric],
  );

  const changeLens = (_, nextLens) => {
    if (!nextLens) return;
    setLens(nextLens);
    setSort({ metric: LENSES[nextLens][0].name, direction: -1 });
  };

  const changeSort = (metric) => {
    setSort((current) => ({
      metric,
      direction: current.metric === metric ? current.direction * -1 : -1,
    }));
  };

  const openFromKeyboard = (event, symbol) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onOpenSymbol(symbol);
    }
  };

  return (
    <Box>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h4" component="h1">Options Command Center</Typography>
        <Typography color="text.secondary">
          A focused read of the current liquid Candidates and Leaders—not an option-chain scanner.
        </Typography>
      </Box>
      <OptionsStatusBanner data={data} />
      <Paper variant="outlined">
        <Box sx={{ p: 1.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
          <Typography variant="subtitle2">{data.items.length} current symbols</Typography>
          <ToggleButtonGroup exclusive size="small" value={lens} onChange={changeLens} aria-label="Metric focus">
            <ToggleButton value="gamma">Gamma</ToggleButton>
            <ToggleButton value="volatility">Volatility</ToggleButton>
            <ToggleButton value="skew">Skew</ToggleButton>
            <ToggleButton value="activity">Activity</ToggleButton>
          </ToggleButtonGroup>
        </Box>
        <TableContainer>
          <Table size="small" aria-label="Options Command Center current symbols">
            <TableHead>
              <TableRow>
                <TableCell>#</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Equity source</TableCell>
                <TableCell>Quality</TableCell>
                {columns.map(({ name }) => {
                  const label = data.items.find((item) => item.metrics[name])?.metrics[name]?.label || name;
                  const selected = sort.metric === name;
                  const direction = selected && sort.direction === 1 ? 'ascending' : 'descending';
                  const muiDirection = direction === 'ascending' ? 'asc' : 'desc';
                  return (
                    <TableCell key={name} align="right" sortDirection={selected ? muiDirection : false}>
                      <Button
                        size="small"
                        color="inherit"
                        aria-pressed={selected}
                        aria-label={`Sort by ${label}${selected ? `, ${direction}` : ''}`}
                        onClick={() => changeSort(name)}
                        endIcon={selected
                          ? (sort.direction === 1 ? <ArrowUpwardIcon /> : <ArrowDownwardIcon />)
                          : null}
                      >
                        {label}
                      </Button>
                    </TableCell>
                  );
                })}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((item) => {
                const rank = rankedSymbols.indexOf(item.symbol) + 1;
                return (
                  <TableRow
                    hover
                    key={item.symbol}
                    tabIndex={0}
                    onClick={() => onOpenSymbol(item.symbol)}
                    onKeyDown={(event) => openFromKeyboard(event, item.symbol)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>{rank > 0 && <span aria-label={`Metric rank ${rank}`}>{rank}</span>}</TableCell>
                    <TableCell>
                      <Typography fontWeight={700}>{item.symbol}</Typography>
                      <Typography variant="caption" color="text.secondary">Max Pain: {item.metrics.max_pain.available ? formatOptionsMetric(item.metrics.max_pain) : '—'}</Typography>
                    </TableCell>
                    <TableCell>
                      <OptionsSourceBadges
                        badges={item.source_badges}
                        candidateRank={item.candidate_rank}
                        leaderRank={item.leader_rank}
                      />
                    </TableCell>
                    <TableCell><OptionsQualityBadge state={item.state} reasonCodes={item.reason_codes} /></TableCell>
                    {columns.map(({ name, percent }) => {
                      const metric = item.metrics[name];
                      return (
                        <TableCell key={name} align="right">
                          {metric.available
                            ? formatOptionsMetric(metric, { percent })
                            : <UnavailableMetric metric={metric} />}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  );
}
