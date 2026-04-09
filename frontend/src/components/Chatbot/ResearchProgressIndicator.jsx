import { Box, Typography, LinearProgress, Chip, Stack } from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import SearchIcon from '@mui/icons-material/Search';
import CompressIcon from '@mui/icons-material/Compress';
import EditNoteIcon from '@mui/icons-material/EditNote';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

/**
 * Shows progress indicator for deep research mode.
 * Displays current phase, unit progress, and sources found.
 */
function ResearchProgressIndicator({
  phase,
  totalUnits,
  completedUnits,
  sourcesFound,
  subQuestions,
}) {
  // Phase configuration
  const phases = {
    planning: {
      label: 'Planning Research',
      icon: <ScienceIcon fontSize="small" />,
      color: 'info',
      progress: 10,
    },
    researching: {
      label: 'Gathering Data',
      icon: <SearchIcon fontSize="small" />,
      color: 'primary',
      progress: totalUnits > 0 ? 20 + (completedUnits / totalUnits) * 50 : 30,
    },
    compressing: {
      label: 'Analyzing Findings',
      icon: <CompressIcon fontSize="small" />,
      color: 'secondary',
      progress: 75,
    },
    writing: {
      label: 'Writing Report',
      icon: <EditNoteIcon fontSize="small" />,
      color: 'success',
      progress: 90,
    },
    done: {
      label: 'Complete',
      icon: <CheckCircleIcon fontSize="small" />,
      color: 'success',
      progress: 100,
    },
  };

  const currentPhase = phases[phase] || phases.planning;

  return (
    <Box
      sx={{
        p: 2,
        backgroundColor: 'background.paper',
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
        mb: 2,
      }}
    >
      {/* Phase Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
        <Chip
          icon={currentPhase.icon}
          label={currentPhase.label}
          color={currentPhase.color}
          size="small"
          sx={{ fontWeight: 500 }}
        />
        {phase === 'researching' && totalUnits > 0 && (
          <Typography variant="caption" color="text.secondary">
            Unit {completedUnits + 1} of {totalUnits}
          </Typography>
        )}
      </Box>

      {/* Progress Bar */}
      <LinearProgress
        variant="determinate"
        value={currentPhase.progress}
        color={currentPhase.color}
        sx={{
          height: 6,
          borderRadius: 1,
          mb: 1.5,
          backgroundColor: 'action.hover',
        }}
      />

      {/* Sub-questions (during planning/researching) */}
      {subQuestions && subQuestions.length > 0 && phase !== 'writing' && phase !== 'done' && (
        <Box sx={{ mb: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
            Research Questions:
          </Typography>
          <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
            {subQuestions.slice(0, 4).map((q, i) => (
              <Chip
                key={i}
                label={q.length > 30 ? q.substring(0, 30) + '...' : q}
                size="small"
                variant={completedUnits > i ? 'filled' : 'outlined'}
                color={completedUnits > i ? 'success' : 'default'}
                sx={{
                  fontSize: '0.7rem',
                  height: 22,
                  mb: 0.5,
                }}
              />
            ))}
            {subQuestions.length > 4 && (
              <Chip
                label={`+${subQuestions.length - 4} more`}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem', height: 22, mb: 0.5 }}
              />
            )}
          </Stack>
        </Box>
      )}

      {/* Stats */}
      {(sourcesFound > 0 || phase === 'researching') && (
        <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
          {sourcesFound > 0 && (
            <Typography variant="caption" color="text.secondary">
              Sources: <strong>{sourcesFound}</strong>
            </Typography>
          )}
          {phase === 'researching' && completedUnits > 0 && (
            <Typography variant="caption" color="text.secondary">
              Units complete: <strong>{completedUnits}/{totalUnits}</strong>
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
}

export default ResearchProgressIndicator;
