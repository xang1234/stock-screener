import { Alert, Box, Button, Card, CardContent, Chip, Divider, Stack, Typography } from '@mui/material';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import CloudDoneRoundedIcon from '@mui/icons-material/CloudDoneRounded';
import UpdateRoundedIcon from '@mui/icons-material/UpdateRounded';
import { useRuntime } from '../../contexts/RuntimeContext';

function formatFreshnessLabel(timestamp) {
  if (!timestamp) {
    return 'Not ready';
  }
  const value = new Date(timestamp);
  if (Number.isNaN(value.getTime())) {
    return timestamp;
  }
  return value.toLocaleString();
}

function DataChip({ label, ready, value }) {
  return (
    <Chip
      size="small"
      color={ready ? 'success' : 'default'}
      variant={ready ? 'filled' : 'outlined'}
      label={`${label}: ${value}`}
    />
  );
}

function DesktopBootstrapBanner() {
  const {
    dataStatus,
    desktopMode,
    isRefreshingNow,
    refreshNow,
    setupRequired,
    update,
    updateFailed,
    updateRunning,
  } = useRuntime();

  if (!desktopMode || setupRequired) {
    return null;
  }

  return (
    <Card sx={{ mb: 1.5 }}>
      <CardContent>
        <Stack spacing={1.5}>
          <Box display="flex" justifyContent="space-between" gap={2} alignItems="flex-start">
            <Box>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                <CloudDoneRoundedIcon fontSize="small" color="primary" />
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  Local desktop data
                </Typography>
                {dataStatus.starter_baseline_active && (
                  <Chip size="small" color="warning" label="Starter baseline active" />
                )}
              </Stack>
              <Typography variant="body2" color="text.secondary">
                {update.message || 'Starter data is local. Regular updates keep prices, breadth, groups, and fundamentals fresh.'}
              </Typography>
            </Box>
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshRoundedIcon />}
              onClick={() => refreshNow('manual')}
              disabled={isRefreshingNow || updateRunning}
            >
              Refresh now
            </Button>
          </Box>

          {(updateRunning || updateFailed) && (
            <Alert severity={updateFailed ? 'error' : 'info'} icon={<UpdateRoundedIcon fontSize="inherit" />}>
              {update.error || update.message || (updateRunning ? 'Automatic update in progress.' : 'Automatic update failed.')}
            </Alert>
          )}

          <Divider />

          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            <DataChip
              label="Prices"
              ready={dataStatus.prices.ready}
              value={formatFreshnessLabel(dataStatus.prices.last_success_at)}
            />
            <DataChip
              label="Breadth"
              ready={dataStatus.breadth.ready}
              value={formatFreshnessLabel(dataStatus.breadth.last_success_at)}
            />
            <DataChip
              label="Groups"
              ready={dataStatus.groups.ready}
              value={formatFreshnessLabel(dataStatus.groups.last_success_at)}
            />
            <DataChip
              label="Fundamentals"
              ready={dataStatus.fundamentals.ready}
              value={formatFreshnessLabel(dataStatus.fundamentals.last_success_at)}
            />
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}

export default DesktopBootstrapBanner;
