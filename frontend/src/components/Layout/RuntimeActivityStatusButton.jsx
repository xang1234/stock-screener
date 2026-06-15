import { useEffect, useState } from 'react';
import { Box, Button, Chip, Typography } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { useRuntimeActivity } from '../../hooks/useRuntimeActivity';

// Header status is best-effort chrome; let route-critical data requests start first.
const HEADER_ACTIVITY_DELAY_MS = 1500;

function buildSummary(activity, { isPending = false, isError = false } = {}) {
  if (isError && isPending) {
    return {
      badge: 'Warn',
      badgeColor: 'warning',
      title: 'Activity unavailable',
      detail: 'View operations',
    };
  }

  if (isPending) {
    return {
      badge: '...',
      badgeColor: 'default',
      title: 'Checking activity',
      detail: 'View operations',
    };
  }

  const bootstrap = activity?.bootstrap ?? {};
  const summary = activity?.summary ?? {};
  const markets = activity?.markets ?? [];
  const activeMarket = markets.find((market) => (
    market.status === 'running' || market.status === 'queued'
  ));
  const failedMarket = markets.find((market) => market.status === 'failed');

  if (bootstrap.state === 'running') {
    const determinateBootstrap = (
      bootstrap.progress_mode === 'determinate'
      && Number.isFinite(bootstrap.percent)
    );
    const percent = determinateBootstrap ? Math.round(bootstrap.percent) : null;
    return {
      badge: determinateBootstrap ? `${percent}%` : 'Sync',
      badgeColor: 'info',
      title: `Bootstrapping ${bootstrap.primary_market || 'market'}`,
      detail: determinateBootstrap && bootstrap.current_stage
        ? `${bootstrap.current_stage} · ${percent}%`
        : bootstrap.current_stage
          || bootstrap.message
          || 'Preparing market data',
    };
  }

  if (summary.status === 'warning' && failedMarket) {
    return {
      badge: 'Warn',
      badgeColor: 'warning',
      title: 'Refresh warning',
      detail: `${failedMarket.market} · ${failedMarket.stage_label || failedMarket.message || 'Task failed'}`,
    };
  }

  if ((summary.active_market_count ?? 0) > 0) {
    const activeLabel = summary.active_market_count === 1 ? '1 market active' : `${summary.active_market_count} markets active`;
    return {
      badge: String(summary.active_market_count),
      badgeColor: 'secondary',
      title: activeLabel,
      detail: activeMarket
        ? `${activeMarket.market} · ${activeMarket.stage_label || activeMarket.message || activeMarket.status}`
        : 'Data refresh in progress',
    };
  }

  return {
    badge: 'OK',
    badgeColor: 'success',
    title: 'Markets ready',
    detail: 'View operations',
  };
}

export default function RuntimeActivityStatusButton() {
  const [activityEnabled, setActivityEnabled] = useState(false);
  useEffect(() => {
    const timerId = window.setTimeout(() => {
      setActivityEnabled(true);
    }, HEADER_ACTIVITY_DELAY_MS);

    return () => window.clearTimeout(timerId);
  }, []);

  const activityQuery = useRuntimeActivity({ enabled: activityEnabled });
  const summary = buildSummary(activityQuery.data, {
    isError: activityQuery.isError,
    isPending: !activityQuery.dataUpdatedAt,
  });

  return (
    <Button
      color="inherit"
      component={RouterLink}
      to="/operations"
      sx={{
        minWidth: 0,
        px: 1,
        py: 0.25,
        ml: 2,
        borderRadius: 1.5,
        textTransform: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        maxWidth: 280,
        border: '1px solid rgba(255,255,255,0.16)',
        backgroundColor: 'rgba(255,255,255,0.08)',
        '&:hover': {
          backgroundColor: 'rgba(255,255,255,0.14)',
        },
      }}
    >
      <Chip
        label={summary.badge}
        size="small"
        color={summary.badgeColor}
        sx={{
          height: 18,
          fontSize: '10px',
          fontWeight: 700,
          '& .MuiChip-label': {
            px: 0.75,
          },
        }}
      />
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', overflow: 'hidden' }}>
        <Typography
          variant="caption"
          sx={{
            color: 'inherit',
            fontWeight: 700,
            lineHeight: 1.2,
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
            overflow: 'hidden',
            maxWidth: 220,
          }}
        >
          {summary.title}
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: 'rgba(255,255,255,0.78)',
            lineHeight: 1.2,
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
            overflow: 'hidden',
            maxWidth: 220,
          }}
        >
          {summary.detail}
        </Typography>
      </Box>
    </Button>
  );
}
