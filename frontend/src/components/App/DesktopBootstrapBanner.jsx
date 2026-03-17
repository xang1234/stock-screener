import { Alert, Box, Button, LinearProgress, Stack, Typography } from '@mui/material';
import { useRuntime } from '../../contexts/RuntimeContext';

function DesktopBootstrapBanner() {
  const {
    bootstrap,
    bootstrapFailed,
    bootstrapIncomplete,
    bootstrapRunning,
    bootstrapWarnings,
    desktopMode,
    startBootstrap,
    isStartingBootstrap,
  } = useRuntime();

  if (!desktopMode) {
    return null;
  }

  if (!bootstrapIncomplete && bootstrapWarnings.length === 0) {
    return null;
  }

  const severity = bootstrapFailed ? 'error' : bootstrapWarnings.length > 0 && !bootstrapIncomplete ? 'warning' : 'info';
  const title = bootstrapFailed
    ? 'Desktop setup failed'
    : bootstrapRunning
      ? 'Preparing local market data'
      : bootstrapIncomplete
        ? 'Desktop setup required'
        : 'Desktop setup completed with warnings';

  const message = bootstrap?.message
    || (bootstrapRunning
      ? 'Scans, breadth, and group rankings will populate as the local setup completes.'
      : 'Run the local bootstrap to seed the starter universe and cache baseline data.');

  return (
    <Alert
      severity={severity}
      action={bootstrapFailed || (!bootstrapRunning && bootstrapIncomplete) ? (
        <Button
          color="inherit"
          size="small"
          onClick={() => startBootstrap(bootstrapFailed)}
          disabled={isStartingBootstrap}
        >
          {bootstrapFailed ? 'Retry setup' : 'Start setup'}
        </Button>
      ) : null}
      sx={{ mb: 1.5 }}
    >
      <Stack spacing={1}>
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
            {title}
          </Typography>
          <Typography variant="body2">{message}</Typography>
        </Box>
        {bootstrapRunning && (
          <Box>
            <LinearProgress
              variant="determinate"
              value={bootstrap?.percent ?? 0}
              sx={{ mb: 0.5 }}
            />
            <Typography variant="caption" color="text.secondary">
              {(bootstrap?.percent ?? 0).toFixed(0)}% complete
            </Typography>
          </Box>
        )}
        {bootstrapWarnings.length > 0 && (
          <Typography variant="caption" color="text.secondary">
            {bootstrapWarnings[0]}
          </Typography>
        )}
      </Stack>
    </Alert>
  );
}

export default DesktopBootstrapBanner;
