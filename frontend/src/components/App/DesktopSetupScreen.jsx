import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Divider,
  Grid,
  LinearProgress,
  Stack,
  Typography,
} from '@mui/material';
import PlayArrowRoundedIcon from '@mui/icons-material/PlayArrowRounded';
import DownloadRoundedIcon from '@mui/icons-material/DownloadRounded';
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';
import { useRuntime } from '../../contexts/RuntimeContext';

function DesktopSetupScreen() {
  const {
    isStartingSetup,
    setup,
    setupFailed,
    setupOptions,
    setupRunning,
    startSetup,
  } = useRuntime();

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background:
          'radial-gradient(circle at top left, rgba(25, 118, 210, 0.18), transparent 38%), linear-gradient(180deg, rgba(17,24,39,0.98) 0%, rgba(15,23,42,1) 100%)',
        py: 6,
      }}
    >
      <Container maxWidth="lg">
        <Stack spacing={3}>
          <Box>
            <Chip
              size="small"
              icon={<AutoAwesomeRoundedIcon fontSize="small" />}
              label="macOS desktop setup"
              sx={{ mb: 2, bgcolor: 'rgba(255,255,255,0.08)', color: 'common.white' }}
            />
            <Typography variant="h3" sx={{ fontWeight: 700, color: 'common.white', mb: 1 }}>
              Start with local data, then keep it fresh automatically.
            </Typography>
            <Typography variant="body1" sx={{ color: 'rgba(255,255,255,0.72)', maxWidth: 760 }}>
              This Mac install uses local SQLite data and background updates. Choose whether to open immediately with bundled starter data or wait for the first live core refresh to finish.
            </Typography>
          </Box>

          {setupFailed && (
            <Alert severity="error">
              {setup.error || setup.message || 'Desktop setup failed. Retry with either startup option.'}
            </Alert>
          )}

          {setupRunning && (
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardContent>
                <Stack spacing={1.5}>
                  <Box display="flex" justifyContent="space-between" gap={2}>
                    <Box>
                      <Typography variant="h6">Setting up local market data</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {setup.message || 'Preparing your local database and starter baseline.'}
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {(setup.percent ?? 0).toFixed(0)}%
                    </Typography>
                  </Box>
                  <LinearProgress variant="determinate" value={setup.percent ?? 0} />
                  {setup.steps?.length > 0 && (
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                      {setup.steps.map((step) => (
                        <Chip
                          key={step.name}
                          size="small"
                          label={`${step.label}: ${step.status}`}
                          color={step.status === 'completed' ? 'success' : step.status === 'failed' ? 'error' : 'default'}
                          variant={step.status === 'running' ? 'filled' : 'outlined'}
                        />
                      ))}
                    </Stack>
                  )}
                </Stack>
              </CardContent>
            </Card>
          )}

          <Grid container spacing={2}>
            {setupOptions.map((option) => {
              const isQuickStart = option.id === 'quick_start';
              return (
                <Grid item xs={12} md={6} key={option.id}>
                  <Card
                    sx={{
                      height: '100%',
                      border: option.recommended ? '1px solid rgba(25, 118, 210, 0.35)' : '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center" gap={1}>
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {option.label}
                        </Typography>
                        {option.recommended && <Chip size="small" color="primary" label="Recommended" />}
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {option.description}
                      </Typography>
                      <Divider />
                      <Stack spacing={1} sx={{ flex: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          {isQuickStart
                            ? 'Starter data is installed immediately. Breadth, group rankings, and prices continue refreshing in the background.'
                            : 'Starter data installs first, then the app waits for the first core live refresh of prices, breadth, group rankings, and core fundamentals.'}
                        </Typography>
                      </Stack>
                      <Button
                        variant={isQuickStart ? 'contained' : 'outlined'}
                        startIcon={isQuickStart ? <PlayArrowRoundedIcon /> : <DownloadRoundedIcon />}
                        onClick={() => startSetup(option.id, setupFailed)}
                        disabled={isStartingSetup || setupRunning}
                      >
                        {setupFailed ? `Retry ${option.label}` : option.label}
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </Stack>
      </Container>
    </Box>
  );
}

export default DesktopSetupScreen;
