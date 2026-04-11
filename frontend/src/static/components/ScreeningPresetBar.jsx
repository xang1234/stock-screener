import { memo } from 'react';
import { Box, Chip, Tooltip, Typography } from '@mui/material';

function ScreeningPresetBar({ presets = [], activePresetId, onSelectPreset, disabled = false }) {
  if (!presets.length) {
    return null;
  }

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 0.75, fontSize: '0.75rem' }}>
        Screens
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
        {presets.map((preset) => {
          const isActive = preset.id === activePresetId;
          return (
            <Tooltip key={preset.id} title={preset.description || ''} arrow enterDelay={400}>
              <Chip
                label={preset.name}
                size="small"
                color={isActive ? 'primary' : 'default'}
                variant={isActive ? 'filled' : 'outlined'}
                onClick={() => onSelectPreset(isActive ? null : preset.id)}
                disabled={disabled}
                sx={{
                  height: 26,
                  fontSize: '0.75rem',
                  fontWeight: isActive ? 600 : 400,
                  '& .MuiChip-label': { px: 1.25 },
                }}
              />
            </Tooltip>
          );
        })}
      </Box>
    </Box>
  );
}

export default memo(ScreeningPresetBar);
