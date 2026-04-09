import {
  Box,
  Paper,
  Typography,
  IconButton,
  LinearProgress,
  CircularProgress,
  Alert,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RemoveIcon from '@mui/icons-material/Remove';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import { usePipeline } from '../contexts/usePipeline';

const PipelineProgressCard = () => {
  const { pipelineStatus, isCardVisible, isMinimized, closePipelineCard, toggleMinimize } = usePipeline();

  const steps = ['Ingestion', 'Retry', 'Extraction', 'Metrics', 'Alerts'];

  const getActiveStep = () => {
    if (!pipelineStatus) return 0;
    const stepMap = { ingestion: 0, reprocessing: 1, extraction: 2, metrics: 3, alerts: 4, completed: 5 };
    return stepMap[pipelineStatus.current_step] ?? 0;
  };

  const isComplete = pipelineStatus?.status === 'completed';
  const isFailed = pipelineStatus?.status === 'failed';

  if (!isCardVisible) return null;

  // Minimized view - small pill
  if (isMinimized) {
    return (
      <Paper
        elevation={8}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          width: 180,
          zIndex: 1300,
          borderRadius: 3,
          overflow: 'hidden',
          bgcolor: isComplete ? 'success.main' : isFailed ? 'error.main' : 'primary.main',
          color: 'white',
          cursor: 'pointer',
          transition: 'transform 0.2s, box-shadow 0.2s',
          '&:hover': {
            transform: 'scale(1.02)',
            boxShadow: 12,
          },
        }}
        onClick={toggleMinimize}
      >
        <Box
          sx={{
            px: 1.5,
            py: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Box display="flex" alignItems="center" gap={1}>
            {!isComplete && !isFailed && <CircularProgress size={14} sx={{ color: 'white' }} />}
            <Typography variant="caption" fontWeight="bold">
              {isComplete ? 'Complete' : isFailed ? 'Failed' : `Running ${pipelineStatus?.percent?.toFixed(0) || 0}%`}
            </Typography>
          </Box>
          <IconButton size="small" sx={{ color: 'white', p: 0.25 }}>
            <OpenInFullIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </Box>
      </Paper>
    );
  }

  // Full view
  return (
    <Paper
      elevation={8}
      sx={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        width: 380,
        zIndex: 1300,
        borderRadius: 2,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 2,
          py: 1.5,
          bgcolor: isComplete ? 'success.main' : isFailed ? 'error.main' : 'primary.main',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box display="flex" alignItems="center">
          {!isComplete && !isFailed && <CircularProgress size={16} sx={{ color: 'white', mr: 1 }} />}
          <Typography variant="subtitle2" fontWeight="bold">
            {isComplete ? 'Pipeline Complete' : isFailed ? 'Pipeline Failed' : 'Running Pipeline...'}
          </Typography>
        </Box>
        <Box display="flex" alignItems="center" gap={0.5}>
          <IconButton size="small" onClick={toggleMinimize} sx={{ color: 'white', p: 0.5 }}>
            <RemoveIcon fontSize="small" />
          </IconButton>
          {(isComplete || isFailed) && (
            <IconButton size="small" onClick={closePipelineCard} sx={{ color: 'white', p: 0.5 }}>
              <CloseIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
      </Box>

      {/* Content */}
      <Box sx={{ p: 2 }}>
        {/* Mini stepper */}
        <Box display="flex" justifyContent="space-between" mb={1.5}>
          {steps.map((label, index) => (
            <Box
              key={label}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                flex: 1,
              }}
            >
              <Box
                sx={{
                  width: 20,
                  height: 20,
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  bgcolor: index < getActiveStep() ? 'success.main' : index === getActiveStep() && !isComplete && !isFailed ? 'primary.main' : 'grey.300',
                  color: index <= getActiveStep() ? 'white' : 'text.secondary',
                }}
              >
                {index < getActiveStep() ? '✓' : index + 1}
              </Box>
              <Typography variant="caption" sx={{ fontSize: '9px', mt: 0.5, color: 'text.secondary' }}>
                {label}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Progress bar */}
        <LinearProgress
          variant="determinate"
          value={pipelineStatus?.percent || 0}
          color={isComplete ? 'success' : isFailed ? 'error' : 'primary'}
          sx={{ height: 6, borderRadius: 3, mb: 1 }}
        />
        <Typography variant="caption" color="text.secondary" display="block" textAlign="center">
          {pipelineStatus?.percent?.toFixed(0) || 0}% - {pipelineStatus?.message || 'Initializing...'}
        </Typography>

        {/* Error message */}
        {isFailed && (
          <Alert severity="error" sx={{ mt: 1.5, py: 0.5 }}>
            <Typography variant="caption">{pipelineStatus?.error_message || 'Pipeline failed'}</Typography>
          </Alert>
        )}

        {/* Step results summary */}
        {(pipelineStatus?.ingestion_result || pipelineStatus?.reprocessing_result || pipelineStatus?.extraction_result || pipelineStatus?.metrics_result) && (
          <Box sx={{ mt: 1.5, pt: 1.5, borderTop: 1, borderColor: 'divider' }}>
            {pipelineStatus?.ingestion_result && (
              <Typography variant="caption" color="text.secondary" display="block">
                Ingested: {pipelineStatus.ingestion_result.new_items || 0} items from {pipelineStatus.ingestion_result.total_sources || 0} sources
              </Typography>
            )}
            {pipelineStatus?.reprocessing_result && (
              <Typography variant="caption" color="text.secondary" display="block">
                Retried: {pipelineStatus.reprocessing_result.reprocessed_count || 0} failed,
                {' '}{pipelineStatus.reprocessing_result.processed || 0} recovered
              </Typography>
            )}
            {pipelineStatus?.extraction_result && (
              <Typography variant="caption" color="text.secondary" display="block">
                Extracted: {pipelineStatus.extraction_result.processed || 0} processed, {pipelineStatus.extraction_result.new_themes?.length || 0} new themes
              </Typography>
            )}
            {pipelineStatus?.metrics_result && (
              <Typography variant="caption" color="text.secondary" display="block">
                Metrics: {pipelineStatus.metrics_result.themes_updated || 0} themes updated
              </Typography>
            )}
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default PipelineProgressCard;
