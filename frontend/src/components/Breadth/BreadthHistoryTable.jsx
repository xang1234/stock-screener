import {
  Box,
  ButtonBase,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { format, parseISO } from 'date-fns';
import { useMemo } from 'react';

import BreadthMetricTooltip from './BreadthMetricTooltip';
import {
  breadthMetricDefinitions,
  primaryBreadthMetrics,
  secondaryBreadthMetrics,
  tableContextMetrics,
} from './breadthMetricDefinitions';
import {
  BREADTH_VISUAL_COLORS,
  buildDirectionalToneThresholds,
  isDirectionalMetric,
  metricTone,
} from './breadthVisualEncoding';

const formatValue = (row, metric) => {
  const value = row?.[metric];
  if (value == null) return '—';
  if (metric === 'ratio_5day' || metric === 'ratio_10day') {
    return Number(value).toFixed(2);
  }
  const eligibleField = breadthMetricDefinitions[metric]?.eligibleField;
  const eligibleValue = eligibleField ? row?.[eligibleField] : null;
  if (eligibleValue != null && Number(eligibleValue) <= 0) return '—';
  if (breadthMetricDefinitions[metric]?.countField) return `${Number(value).toFixed(2)}%`;
  return value;
};

const metricCellSx = (metric, tone) => ({
  fontFamily: 'monospace',
  fontWeight: isDirectionalMetric(metric) ? 700 : 500,
  color: '#fff',
  backgroundColor: BREADTH_VISUAL_COLORS[tone],
  borderColor: 'rgba(255, 255, 255, 0.06)',
  borderLeft: groupStartMetrics.has(metric)
    ? `3px solid ${metricGroupColors[metric]}`
    : undefined,
  px: 0.5,
  whiteSpace: 'nowrap',
  transition: 'background-color 120ms ease',
});

const metricHeaderLines = {
  stocks_up_4pct: ['Stocks Up', '4%+ Today'],
  stocks_down_4pct: ['Stocks Down', '4%+ Today'],
  ratio_5day: ['5 Day', 'Ratio'],
  ratio_10day: ['10 Day', 'Ratio'],
  stocks_up_25pct_quarter: ['Up 25%+', 'Quarter'],
  stocks_down_25pct_quarter: ['Down 25%+', 'Quarter'],
  stocks_up_25pct_month: ['Up 25%+', 'Month'],
  stocks_down_25pct_month: ['Down 25%+', 'Month'],
  stocks_up_50pct_month: ['Up 50%+', 'Month'],
  stocks_down_50pct_month: ['Down 50%+', 'Month'],
  stocks_up_13pct_34days: ['Up 13%+', '34 Days'],
  stocks_down_13pct_34days: ['Down 13%+', '34 Days'],
  atr_10x_extension_count: ['10x ATR', 'Extension'],
  t2108_pct: ['T2108', '> 40D SMA'],
  above_sma200_pct: ['% Above', '200D SMA'],
  down_20pct_from_high_pct: ['20%+ Off', '52W High'],
  broad_universe_count: ['Broad Universe'],
};

const GROUPS = [
  {
    key: 'primary',
    label: 'Primary Breadth Indicators',
    description: 'Daily movers & ratios',
    metrics: primaryBreadthMetrics,
    color: '#9a6700',
  },
  {
    key: 'secondary',
    label: 'Secondary Breadth Indicators',
    description: 'Trend windows',
    metrics: secondaryBreadthMetrics,
    color: '#176b43',
  },
  {
    key: 'context',
    label: 'Context',
    description: 'Market context',
    metrics: tableContextMetrics,
    color: '#315f9b',
  },
];

const groupStartMetrics = new Set(GROUPS.map((group) => group.metrics[0]));
const metricGroupColors = Object.fromEntries(
  GROUPS.map((group) => [group.metrics[0], group.color]),
);
const EMPTY_CONTRIBUTOR_DATES = new Set();

function MetricHeader({ metric }) {
  const lines = metricHeaderLines[metric] ?? [breadthMetricDefinitions[metric].label];
  const isGroupStart = groupStartMetrics.has(metric);
  return (
    <TableCell
      align="center"
      data-testid={`breadth-header-${metric}`}
      aria-label={breadthMetricDefinitions[metric].label}
      sx={{
        position: 'relative',
        height: 54,
        px: 0.25,
        py: 0.5,
        borderLeft: isGroupStart ? `3px solid ${metricGroupColors[metric]}` : undefined,
        fontSize: 10,
        lineHeight: 1.15,
        whiteSpace: 'normal',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Box component="span">
          {lines.map((line) => (
            <Box component="span" key={line} sx={{ display: 'block' }}>
              {line}
            </Box>
          ))}
        </Box>
        <BreadthMetricTooltip metric={metric} compact />
      </Box>
    </TableCell>
  );
}

function BreadthHistoryTable({
  rows = [],
  maxRows = 90,
  contributorDates = EMPTY_CONTRIBUTOR_DATES,
  onContributorCellClick,
}) {
  const metrics = [
    ...primaryBreadthMetrics,
    ...secondaryBreadthMetrics,
    ...tableContextMetrics,
  ];
  const visibleRows = useMemo(() => rows.slice(0, maxRows), [maxRows, rows]);
  const toneThresholds = useMemo(
    () => buildDirectionalToneThresholds(visibleRows),
    [visibleRows],
  );
  return (
    <TableContainer
      data-testid="breadth-history-scroll"
      sx={{ overflowX: 'auto', maxHeight: 'calc(100vh - 360px)' }}
    >
      <Table
        stickyHeader
        size="small"
        data-testid="breadth-history-table"
        sx={{ width: '100%', minWidth: 980, tableLayout: 'fixed' }}
      >
        <colgroup>
          <Box component="col" sx={{ width: 76 }} />
          {metrics.map((metric) => (
            <Box
              component="col"
              key={metric}
              sx={{
                width: tableContextMetrics.includes(metric) ? 72 : 62,
              }}
            />
          ))}
        </colgroup>
        <TableHead>
          <TableRow>
            <TableCell
              rowSpan={2}
              sx={{
                position: 'sticky',
                left: 0,
                zIndex: 5,
                px: 0.75,
                fontWeight: 700,
                borderRight: `3px solid ${GROUPS[0].color}`,
              }}
            >
              Date
            </TableCell>
            {GROUPS.map((group) => (
              <TableCell
                key={group.key}
                align="center"
                colSpan={group.metrics.length}
                data-testid={`breadth-group-${group.key}`}
                sx={{
                  bgcolor: group.color,
                  color: '#fff',
                  borderLeft: `3px solid ${group.color}`,
                  borderColor: 'rgba(255, 255, 255, 0.2)',
                  py: 0.6,
                }}
              >
                <Box sx={{ fontSize: 11, fontWeight: 800, lineHeight: 1.2 }}>
                  {group.label}
                </Box>
                <Box sx={{ fontSize: 9, fontWeight: 600, opacity: 0.78, lineHeight: 1.2 }}>
                  {group.description}
                </Box>
              </TableCell>
            ))}
          </TableRow>
          <TableRow>
            {metrics.map((metric) => <MetricHeader key={metric} metric={metric} />)}
          </TableRow>
        </TableHead>
        <TableBody>
          {visibleRows.map((row) => (
            <TableRow key={row.date} hover>
              <TableCell
                sx={{
                  position: 'sticky',
                  left: 0,
                  zIndex: 2,
                  bgcolor: BREADTH_VISUAL_COLORS.neutral,
                  color: '#fff',
                  fontFamily: 'monospace',
                  fontWeight: 600,
                  px: 0.75,
                  borderRight: `3px solid ${GROUPS[0].color}`,
                  whiteSpace: 'nowrap',
                }}
              >
                {format(parseISO(row.date), 'MM/dd/yy')}
              </TableCell>
              {metrics.map((metric) => {
                const tone = metricTone(row, metric, toneThresholds);
                const definition = breadthMetricDefinitions[metric];
                const value = Number(row?.[metric]);
                const interactive = Boolean(
                  definition?.contributor
                  && contributorDates.has(row.date)
                  && Number.isFinite(value)
                  && value > 0
                  && onContributorCellClick,
                );
                return (
                  <TableCell
                    key={metric}
                    align="right"
                    data-testid={`breadth-cell-${metric}`}
                    data-tone={tone}
                    sx={{
                      ...metricCellSx(metric, tone),
                      ...(interactive ? { p: 0 } : {}),
                    }}
                  >
                    {interactive ? (
                      <ButtonBase
                        aria-label={`View ${value} contributing stocks for ${definition.label}`}
                        onClick={(event) => onContributorCellClick(metric, row, event.currentTarget)}
                        sx={{
                          width: '100%',
                          minHeight: 33,
                          justifyContent: 'flex-end',
                          px: 0.5,
                          color: 'inherit',
                          font: 'inherit',
                          fontWeight: 'inherit',
                          '&:focus-visible': {
                            outline: '2px solid #fff',
                            outlineOffset: '-3px',
                          },
                        }}
                      >
                        {formatValue(row, metric)}
                      </ButtonBase>
                    ) : formatValue(row, metric)}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default BreadthHistoryTable;
