import { Box, LinearProgress, Tooltip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import ErrorIcon from '@mui/icons-material/Error';

/**
 * Compact inline scan progress indicator
 */
function ScanProgress({ status, progress, totalStocks, completedStocks, passedStocks, etaSeconds }) {
  const formatETA = (seconds) => {
    if (!seconds || seconds <= 0) return '';
    if (seconds < 60) return `~${Math.round(seconds)}s`;
    if (seconds < 3600) return `~${Math.round(seconds / 60)}m`;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.round((seconds % 3600) / 60);
    return `~${hours}h${minutes}m`;
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'completed': return <CheckCircleIcon sx={{ fontSize: 14, color: 'success.main' }} />;
      case 'queued': return <HourglassEmptyIcon sx={{ fontSize: 14, color: 'warning.main' }} />;
      case 'failed': return <ErrorIcon sx={{ fontSize: 14, color: 'error.main' }} />;
      default: return null;
    }
  };

  // Don't show anything if no status
  if (!status) return null;

  // Completed or failed - show simple icon
  if (status === 'completed' || status === 'failed') {
    return (
      <Tooltip title={status === 'completed' ? `Completed: ${passedStocks} passed` : 'Scan failed'} arrow>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {getStatusIcon()}
        </Box>
      </Tooltip>
    );
  }

  // Queued - show waiting indicator
  if (status === 'queued') {
    return (
      <Tooltip title="Scan queued..." arrow>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {getStatusIcon()}
        </Box>
      </Tooltip>
    );
  }

  // Running - show progress bar with stats
  const progressPercent = progress || 0;
  const eta = formatETA(etaSeconds);

  return (
    <Tooltip
      title={
        <Box sx={{ fontSize: '11px' }}>
          <Box>{completedStocks}/{totalStocks} stocks ({progressPercent.toFixed(0)}%)</Box>
          {passedStocks != null && <Box>Passed: {passedStocks}</Box>}
          {eta && <Box>ETA: {eta}</Box>}
        </Box>
      }
      arrow
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 80 }}>
        <LinearProgress
          variant="determinate"
          value={progressPercent}
          sx={{
            flex: 1,
            height: 6,
            borderRadius: 3,
            minWidth: 50,
            backgroundColor: 'grey.200',
          }}
        />
        <Box sx={{ fontSize: '10px', color: 'text.secondary', fontFamily: 'monospace', minWidth: 28 }}>
          {progressPercent.toFixed(0)}%
        </Box>
      </Box>
    </Tooltip>
  );
}

export default ScanProgress;
