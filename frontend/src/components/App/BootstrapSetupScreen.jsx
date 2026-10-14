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

const FALLBACK_BOOTSTRAP_STAGES = [
  { key: 'universe', label: 'Universe Refresh' },
  { key: 'prices', label: 'Price Refresh' },
  { key: 'fundamentals', label: 'Fundamentals Refresh' },
  { key: 'groups', label: 'Group Rankings' },
  { key: 'scan', label: 'Scan' },
  { key: 'breadth', label: 'Breadth Calculation' },
];

function formatCount(value) {
  return new Intl.NumberFormat('en-US').format(value);
}

function normalizeProgressPercent(percent) {
  if (percent === null || percent === undefined) {
    return null;
  }
  const value = Number(percent);
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.max(0, Math.min(100, value));
}

function normalizeEnabled(primaryMarket, enabledMarkets) {
  const next = enabledMarkets.includes(primaryMarket)
    ? enabledMarkets
    : [primaryMarket, ...enabledMarkets];
  return Array.from(new Set(next));
}

// The saved choice can name markets this deployment does not run (stored
// before the deployment changed); start from the ones it does.
function initialSelection(primaryMarket, enabledMarkets, offeredCodes) {
  const primary = offeredCodes.includes(primaryMarket) ? primaryMarket : (offeredCodes[0] ?? primaryMarket);
  const enabled = (enabledMarkets ?? []).filter((market) => offeredCodes.includes(market));
  return { primary, enabled: normalizeEnabled(primary, enabled) };
}

function normalizeStages(stages) {
  if (!Array.isArray(stages) || stages.length === 0) {
    return FALLBACK_BOOTSTRAP_STAGES;
  }
  const normalized = stages
    .map((stage) => ({
      key: stage?.key,
      label: stage?.label || stage?.key,
    }))
    .filter((stage) => stage.key && stage.label);
  return normalized.length > 0 ? normalized : FALLBACK_BOOTSTRAP_STAGES;
}

export default function BootstrapSetupScreen({
  primaryMarket,
  enabledMarkets,
  deploymentMarkets,
  marketCatalog,
  bootstrapState,
  isStartingBootstrap,
  bootstrapError,
  onStartBootstrap,
}) {
  const marketOptions = useMemo(() => {
    const deployable = deploymentMarkets ?? [];
    const catalogMarkets = marketCatalog?.markets;
    if (Array.isArray(catalogMarkets) && catalogMarkets.length > 0) {
      return catalogMarkets
        .filter((market) => deployable.includes(market.code))
        .map((market) => ({
          code: market.code,
          label: market.label || market.code,
        }));
    }
    return deployable.map((market) => ({
      code: market,
      label: market,
    }));
  }, [deploymentMarkets, marketCatalog?.markets]);
  const offeredCodes = useMemo(() => marketOptions.map((market) => market.code), [marketOptions]);

  const [selectedPrimary, setSelectedPrimary] = useState(
    () => initialSelection(primaryMarket || 'US', enabledMarkets, offeredCodes).primary
  );
  const [selectedMarkets, setSelectedMarkets] = useState(
    () => initialSelection(primaryMarket || 'US', enabledMarkets, offeredCodes).enabled
  );

  useEffect(() => {
    const next = initialSelection(primaryMarket || 'US', enabledMarkets, offeredCodes);
    setSelectedPrimary(next.primary);
    setSelectedMarkets(next.enabled);
  }, [enabledMarkets, offeredCodes, primaryMarket]);

  const normalizedSelection = useMemo(
    () => normalizeEnabled(selectedPrimary, selectedMarkets),
    [selectedMarkets, selectedPrimary]
  );
  const running = bootstrapState === 'running';
  const activityQuery = useRuntimeActivity({ enabled: running || isStartingBootstrap });
  const bootstrap = activityQuery.data?.bootstrap ?? null;
  const marketActivity = useMemo(
    () => activityQuery.data?.markets ?? [],
    [activityQuery.data?.markets]
  );
  const bootstrapResolvedPercent = normalizeProgressPercent(bootstrap?.percent);
  const requestedBootstrapProgressMode = (
    bootstrap?.progress_mode
    || 'indeterminate'
  );
  const bootstrapProgressMode = (
    requestedBootstrapProgressMode === 'determinate' && bootstrapResolvedPercent === null
      ? 'indeterminate'
      : requestedBootstrapProgressMode
  );
  const bootstrapPercent = (
    bootstrapProgressMode === 'determinate'
      ? bootstrapResolvedPercent
      : null
  );
  const bootstrapProgressDetail = (
    bootstrap?.current !== null
    && bootstrap?.current !== undefined
    && bootstrap?.total !== null
    && bootstrap?.total !== undefined
  )
    ? `${formatCount(bootstrap.current)} / ${formatCount(bootstrap.total)} stocks`
    : null;
  const bootstrapMessage = bootstrap?.message || 'Preparing market data.';
  const bootstrapStages = useMemo(
    () => normalizeStages(bootstrap?.stages),
    [bootstrap?.stages]
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
                Pick the primary market for startup defaults. The workspace opens after the
                primary market finishes its first scan-backed snapshot.
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
                  Initial sync is running. The workspace will open after the primary market has
                  current data and a published scan. Additional markets continue loading in the background.
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
                        {bootstrap?.current_stage || 'Preparing bootstrap'}
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
                    {bootstrapProgressDetail && (
                      <Typography variant="caption" color="text.secondary">
                        {bootstrapProgressDetail}
                      </Typography>
                    )}
                    <Typography variant="body2" color="text.secondary">
                      {bootstrapMessage}
                    </Typography>
                  </Stack>
                </Box>
                {bootstrap?.background_warning && (
                  <Alert severity="warning">
                    {bootstrap.background_warning}
                  </Alert>
                )}
                {marketActivity.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Enabled market queue
                    </Typography>
                    <Stack spacing={1}>
                      {marketActivity.map((market) => (
                        <Box
                          key={market.activity_id ?? market.market}
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
                )}
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
                {marketOptions.map((market) => (
                  <MenuItem key={market.code} value={market.code}>
                    {market.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box>
              <Typography variant="subtitle2">
                Enabled markets
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                This deployment&apos;s ENABLED_MARKETS decides which markets are listed.
              </Typography>
              <FormGroup>
                {marketOptions.map((market) => (
                  <FormControlLabel
                    key={market.code}
                    control={(
                      <Checkbox
                        checked={normalizedSelection.includes(market.code)}
                        disabled={running || isStartingBootstrap || market.code === selectedPrimary}
                        onChange={() => toggleMarket(market.code)}
                      />
                    )}
                    label={market.code === selectedPrimary ? `${market.label} (primary)` : market.label}
                  />
                ))}
              </FormGroup>
            </Box>

            <Divider />

            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Bootstrap order
              </Typography>
              {bootstrapStages.map((stage, index) => (
                <Typography key={stage.key} color="text.secondary">
                  {index + 1}. {stage.label}
                </Typography>
              ))}
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
