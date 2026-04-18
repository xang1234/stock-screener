import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  Divider,
  FormControl,
  FormControlLabel,
  FormGroup,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';
import { useRuntimeActivity } from '../../hooks/useRuntimeActivity';

const STATUS_COLOR = {
  running: 'info',
  queued: 'warning',
  completed: 'success',
  failed: 'error',
  idle: 'default',
};
const STAGE_LOCAL_PROGRESS_LABELS = new Set(['Price Refresh', 'Fundamentals Refresh']);

function formatCount(value) {
  return new Intl.NumberFormat('en-US').format(value);
}

function resolveStageLocalProgress(activity) {
  if (!activity || activity.progress_mode !== 'determinate' || activity.percent === null || activity.percent === undefined) {
    return null;
  }
  const stageLabel = activity.stage_label || '';
  if (!STAGE_LOCAL_PROGRESS_LABELS.has(stageLabel)) {
    return null;
  }
  return {
    percent: Math.max(0, Math.min(100, Number(activity.percent))),
    detail: (
      activity.current !== null
      && activity.current !== undefined
      && activity.total !== null
      && activity.total !== undefined
    )
      ? `${formatCount(activity.current)} / ${formatCount(activity.total)} stocks`
      : null,
  };
}

function normalizeEnabled(primaryMarket, enabledMarkets) {
  const next = enabledMarkets.includes(primaryMarket)
    ? enabledMarkets
    : [primaryMarket, ...enabledMarkets];
  return Array.from(new Set(next));
}

export default function BootstrapSetupScreen({
  primaryMarket,
  enabledMarkets,
  supportedMarkets,
  bootstrapState,
  isStartingBootstrap,
  bootstrapError,
  onStartBootstrap,
}) {
  const [selectedPrimary, setSelectedPrimary] = useState(primaryMarket || 'US');
  const [selectedMarkets, setSelectedMarkets] = useState(() => (
    normalizeEnabled(primaryMarket || 'US', enabledMarkets?.length ? enabledMarkets : ['US'])
  ));

  useEffect(() => {
    const nextPrimary = primaryMarket || 'US';
    setSelectedPrimary(nextPrimary);
    setSelectedMarkets(
      normalizeEnabled(nextPrimary, enabledMarkets?.length ? enabledMarkets : [nextPrimary])
    );
  }, [enabledMarkets, primaryMarket]);

  const normalizedSelection = useMemo(
    () => normalizeEnabled(selectedPrimary, selectedMarkets),
    [selectedMarkets, selectedPrimary]
  );
  const running = bootstrapState === 'running';
  const activityQuery = useRuntimeActivity({ enabled: running || isStartingBootstrap });
  const bootstrap = activityQuery.data?.bootstrap ?? null;
  const marketActivity = useMemo(() => {
    const markets = activityQuery.data?.markets ?? [];
    const byMarket = new Map(markets.map((item) => [item.market, item]));
    return normalizedSelection.map((market) => (
      byMarket.get(market) ?? {
        market,
        lifecycle: running ? 'bootstrap' : 'idle',
        stage_label: running ? 'Queued' : 'Idle',
        status: running && market === (bootstrap?.primary_market || primaryMarket) ? 'running' : 'queued',
        message: running ? 'Waiting for bootstrap task' : 'Idle',
      }
    ));
  }, [activityQuery.data?.markets, bootstrap?.primary_market, normalizedSelection, primaryMarket, running]);
  const primaryActivity = useMemo(
    () => marketActivity.find((market) => market.market === (primaryMarket || selectedPrimary)) ?? marketActivity[0],
    [marketActivity, primaryMarket, selectedPrimary]
  );
  const stageLocalProgress = resolveStageLocalProgress(primaryActivity);
  const bootstrapProgressMode = (
    stageLocalProgress ? 'determinate' : (
      bootstrap?.progress_mode
      || primaryActivity?.progress_mode
      || 'indeterminate'
    )
  );
  const bootstrapPercent = (
    bootstrapProgressMode === 'determinate'
      ? (stageLocalProgress?.percent
        ?? Math.max(0, Math.min(100, Number(bootstrap?.percent ?? primaryActivity?.percent ?? 0))))
      : null
  );

  const toggleMarket = (market) => {
    if (market === selectedPrimary) {
      return;
    }
    setSelectedMarkets((previous) => (
      previous.includes(market)
        ? previous.filter((value) => value !== market)
        : [...previous, market]
    ));
  };

  const handlePrimaryChange = (event) => {
    const nextPrimary = event.target.value;
    setSelectedPrimary(nextPrimary);
    setSelectedMarkets((previous) => normalizeEnabled(nextPrimary, previous));
  };

  const handleStart = async () => {
    try {
      await onStartBootstrap({
        primaryMarket: selectedPrimary,
        enabledMarkets: normalizedSelection,
      });
    } catch {
      // Mutation state already drives the visible error UI.
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 2,
        py: 4,
      }}
    >
      <Card sx={{ width: '100%', maxWidth: 720 }}>
        <CardContent sx={{ p: 4 }}>
          <Stack spacing={3}>
            <Box>
              <Chip size="small" color="primary" label="Local Setup" sx={{ mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                First-run market bootstrap
              </Typography>
              <Typography color="text.secondary">
                Pick the primary market to hydrate first. Additional enabled markets will queue after
                the primary market is usable.
              </Typography>
            </Box>

            {bootstrapError && (
              <Alert severity="error">
                {typeof bootstrapError === 'string' ? bootstrapError : 'Failed to start bootstrap.'}
              </Alert>
            )}

            {running && (
              <Stack spacing={2}>
                <Alert severity="info">
                  Initial sync is running for {primaryMarket}. The workspace will open as soon as that
                  market has core data.
                </Alert>
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    border: 1,
                    borderColor: 'divider',
                    backgroundColor: 'action.hover',
                  }}
                >
                  <Stack spacing={1.5}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="subtitle2">
                        {bootstrap?.current_stage || primaryActivity?.stage_label || 'Preparing bootstrap'}
                      </Typography>
                      {bootstrapProgressMode === 'determinate' && bootstrapPercent !== null && (
                        <Typography variant="body2" color="text.secondary">
                          {Math.round(bootstrapPercent)}%
                        </Typography>
                      )}
                    </Stack>
                    <LinearProgress
                      variant={bootstrapProgressMode === 'determinate' ? 'determinate' : 'indeterminate'}
                      value={bootstrapProgressMode === 'determinate' ? bootstrapPercent : undefined}
                      aria-label="Bootstrap progress"
                    />
                    {stageLocalProgress?.detail && (
                      <Typography variant="caption" color="text.secondary">
                        {stageLocalProgress.detail}
                      </Typography>
                    )}
                    <Typography variant="body2" color="text.secondary">
                      {bootstrap?.message || primaryActivity?.message || 'Preparing primary market data.'}
                    </Typography>
                  </Stack>
                </Box>
                <Alert severity="warning">
                  {bootstrap?.background_warning
                    || 'Data loading will continue after bootstrap. Secondary markets and follow-up refresh jobs keep running in the background.'}
                </Alert>
                <Box>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Enabled market queue
                  </Typography>
                  <Stack spacing={1}>
                    {marketActivity.map((market) => (
                      <Box
                        key={market.market}
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: 2,
                          px: 1.5,
                          py: 1,
                          borderRadius: 1.5,
                          border: 1,
                          borderColor: 'divider',
                        }}
                      >
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>
                            {market.market}
                            {market.market === (bootstrap?.primary_market || primaryMarket) ? ' (primary)' : ''}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {market.stage_label || 'Queued'}{market.message ? ` · ${market.message}` : ''}
                          </Typography>
                        </Box>
                        <Chip
                          size="small"
                          color={STATUS_COLOR[market.status] || 'default'}
                          label={market.status || 'idle'}
                        />
                      </Box>
                    ))}
                  </Stack>
                </Box>
              </Stack>
            )}

            <Divider />

            <FormControl fullWidth>
              <InputLabel id="bootstrap-primary-market-label">Primary market</InputLabel>
              <Select
                labelId="bootstrap-primary-market-label"
                value={selectedPrimary}
                label="Primary market"
                onChange={handlePrimaryChange}
                disabled={running || isStartingBootstrap}
              >
                {supportedMarkets.map((market) => (
                  <MenuItem key={market} value={market}>
                    {market}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Enabled markets
              </Typography>
              <FormGroup>
                {supportedMarkets.map((market) => (
                  <FormControlLabel
                    key={market}
                    control={(
                      <Checkbox
                        checked={normalizedSelection.includes(market)}
                        disabled={running || isStartingBootstrap || market === selectedPrimary}
                        onChange={() => toggleMarket(market)}
                      />
                    )}
                    label={market === selectedPrimary ? `${market} (primary)` : market}
                  />
                ))}
              </FormGroup>
            </Box>

            <Divider />

            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Bootstrap order
              </Typography>
              <Typography color="text.secondary">
                1. Universe refresh
              </Typography>
              <Typography color="text.secondary">
                2. Benchmark and price refresh
              </Typography>
              <Typography color="text.secondary">
                3. Fundamentals refresh
              </Typography>
              <Typography color="text.secondary">
                4. Breadth and group rankings
              </Typography>
              <Typography color="text.secondary">
                5. Initial autoscan snapshot
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                onClick={handleStart}
                disabled={running || isStartingBootstrap}
              >
                {isStartingBootstrap ? 'Starting...' : 'Start bootstrap'}
              </Button>
            </Box>
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
}
