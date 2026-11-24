import { Box, Paper, Typography } from '@mui/material';

import BreadthMetricTooltip from './BreadthMetricTooltip';
import { breadthMetricDefinitions } from './breadthMetricDefinitions';
import { BREADTH_VISUAL_COLORS } from './breadthVisualEncoding';


const contextMetrics = [
  't2108_pct',
  'above_sma20_pct',
  'above_sma50_pct',
  'above_sma200_pct',
  'down_20pct_from_high_pct',
  'atr_10x_extension_count',
  'broad_universe_count',
];

const eligibleValue = (row, metric) => {
  const eligibleField = breadthMetricDefinitions[metric].eligibleField;
  return eligibleField ? row?.[eligibleField] : null;
};

const formatContextValue = (row, metric) => {
  if (!row) return '—';
  if (metric === 'broad_universe_count') {
    return row[metric] ?? '—';
  }
  const eligible = eligibleValue(row, metric);
  if (!eligible) return '—';
  const { countField } = breadthMetricDefinitions[metric];
  if (countField) {
    const count = row[countField] ?? 0;
    const percentage = row[metric] ?? (count / eligible) * 100;
    return `${Number(percentage).toFixed(2)}% (${count} / ${eligible})`;
  }
  return `${row[metric] ?? 0} / ${eligible}`;
};

const percent = (count, denominator) => (
  denominator > 0 ? Math.max(0, Math.min(100, (count / denominator) * 100)) : 0
);

function HealthBar({
  testId,
  left,
  right,
  neutral,
  isAvailable,
  eligibleLabel,
  colors,
}) {
  const leftPercent = percent(left.count, left.denominator);
  const rightPercent = percent(right.count, right.denominator);
  const neutralPercent = neutral ? percent(neutral.count, neutral.denominator) : 0;
  const ariaLabel = isAvailable
    ? [
      `${left.label} ${leftPercent.toFixed(1)}%`,
      `${right.label} ${rightPercent.toFixed(1)}%`,
      neutral ? `${neutral.label} ${neutralPercent.toFixed(1)}%` : null,
    ].filter(Boolean).join(', ')
    : `${left.label} and ${right.label} unavailable`;

  return (
    <Paper
      variant="outlined"
      data-testid={testId}
      aria-label={ariaLabel}
      sx={{ p: 1.25, minWidth: 0 }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
        {[left, right].map((item, index) => {
          const itemPercent = index === 0 ? leftPercent : rightPercent;
          return (
            <Box key={item.metric} sx={{ minWidth: 0, textAlign: index === 0 ? 'left' : 'right' }}>
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: index === 0 ? 'flex-start' : 'flex-end',
                }}
              >
                <Typography sx={{ color: colors[index], fontSize: 11, fontWeight: 800 }}>
                  {item.label}
                </Typography>
                <BreadthMetricTooltip metric={item.metric} compact />
              </Box>
              <Typography
                sx={{
                  color: isAvailable ? colors[index] : 'text.secondary',
                  fontFamily: 'monospace',
                  fontSize: 12,
                  fontWeight: 700,
                }}
              >
                {isAvailable ? `${item.count} (${itemPercent.toFixed(1)}%)` : '—'}
              </Typography>
            </Box>
          );
        })}
      </Box>
      <Box
        role="img"
        aria-label={ariaLabel}
        sx={{
          display: 'flex',
          height: 9,
          mt: 0.75,
          overflow: 'hidden',
          bgcolor: BREADTH_VISUAL_COLORS.neutral,
          borderRadius: 999,
        }}
      >
        <Box
          data-testid={`${testId}-${left.segment}`}
          sx={{ width: `${leftPercent}%`, bgcolor: colors[0] }}
        />
        {neutral && (
          <Box
            data-testid={`${testId}-${neutral.segment}`}
            sx={{ width: `${neutralPercent}%`, bgcolor: BREADTH_VISUAL_COLORS.unchanged }}
          />
        )}
        <Box
          data-testid={`${testId}-${right.segment}`}
          sx={{ width: `${rightPercent}%`, bgcolor: colors[1] }}
        />
      </Box>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ display: 'block', mt: 0.5, fontSize: 9.5 }}
      >
        {eligibleLabel}
      </Typography>
    </Paper>
  );
}

function BreadthContextStrip({ row }) {
  const advanceDeclineEligible = Number(row?.advance_decline_eligible_count) || 0;
  const advancing = Number(row?.advancing_count) || 0;
  const declining = Number(row?.declining_count) || 0;
  const unchanged = Number(row?.unchanged_count) || 0;
  const newHighs = Number(row?.new_high_52week_count) || 0;
  const newLows = Number(row?.new_low_52week_count) || 0;
  const highLowEvents = newHighs + newLows;
  const highLowEligible = Number(row?.high_low_52week_eligible_count) || 0;

  return (
    <Box
      aria-label="Breadth context"
      sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
      }}
    >
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' },
          gap: 1,
        }}
      >
        <HealthBar
          testId="breadth-health-advance-decline"
          left={{
            metric: 'advancing_count',
            label: 'Advancing',
            count: advancing,
            denominator: advanceDeclineEligible,
            segment: 'advancing',
          }}
          right={{
            metric: 'declining_count',
            label: 'Declining',
            count: declining,
            denominator: advanceDeclineEligible,
            segment: 'declining',
          }}
          neutral={{
            label: 'Unchanged',
            count: unchanged,
            denominator: advanceDeclineEligible,
            segment: 'unchanged',
          }}
          isAvailable={advanceDeclineEligible > 0}
          eligibleLabel={`${advanceDeclineEligible || '—'} advance/decline eligible stocks`}
          colors={[
            BREADTH_VISUAL_COLORS['up-strong'],
            BREADTH_VISUAL_COLORS['down-strong'],
          ]}
        />
        <HealthBar
          testId="breadth-health-high-low"
          left={{
            metric: 'new_high_52week_count',
            label: 'New High',
            count: newHighs,
            denominator: highLowEvents,
            segment: 'high',
          }}
          right={{
            metric: 'new_low_52week_count',
            label: 'New Low',
            count: newLows,
            denominator: highLowEvents,
            segment: 'low',
          }}
          isAvailable={highLowEligible > 0}
          eligibleLabel={`${highLowEligible || '—'} high/low eligible stocks`}
          colors={[BREADTH_VISUAL_COLORS.newHigh, BREADTH_VISUAL_COLORS.newLow]}
        />
      </Box>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: 'repeat(1, minmax(0, 1fr))',
            sm: 'repeat(3, minmax(0, 1fr))',
            lg: `repeat(${contextMetrics.length}, minmax(0, 1fr))`,
          },
          gap: 1,
        }}
      >
        {contextMetrics.map((metric) => (
          <Paper
            key={metric}
            variant="outlined"
            data-testid={`breadth-context-${metric}`}
            sx={{ p: 1, minWidth: 0 }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontSize: 10, fontWeight: 700 }}
              >
                {breadthMetricDefinitions[metric].label}
              </Typography>
              <BreadthMetricTooltip metric={metric} compact />
            </Box>
            <Typography
              sx={{ mt: 0.25, fontFamily: 'monospace', fontSize: 13, fontWeight: 700 }}
            >
              {formatContextValue(row, metric)}
            </Typography>
          </Paper>
        ))}
      </Box>
    </Box>
  );
}

export default BreadthContextStrip;
