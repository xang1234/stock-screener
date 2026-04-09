import { Box, ToggleButtonGroup, ToggleButton, Typography } from '@mui/material';

const ALL_VALUE = '__all__';

/**
 * Compact tri-state toggle for boolean filters
 * States: null (All), true (Yes), false (No)
 */
function CompactCheckbox({ label, value, onChange }) {
  const handleChange = (event, newValue) => {
    if (newValue === null && value !== null) {
      onChange(null);
      return;
    }
    if (newValue === ALL_VALUE) {
      onChange(null);
      return;
    }
    onChange(newValue);
  };

  return (
    <Box sx={{ minWidth: 60 }}>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ display: 'block', mb: 0.5, fontSize: '0.7rem' }}
      >
        {label}
      </Typography>
      <ToggleButtonGroup
        value={value ?? ALL_VALUE}
        exclusive
        onChange={handleChange}
        size="small"
        sx={{
          height: 28,
          '& .MuiToggleButton-root': {
            padding: '2px 6px',
            fontSize: '0.65rem',
            textTransform: 'none',
            minWidth: 28,
          },
        }}
      >
        <ToggleButton value={ALL_VALUE} aria-label="all">
          All
        </ToggleButton>
        <ToggleButton value={true} aria-label="yes" sx={{ color: 'success.main' }}>
          Yes
        </ToggleButton>
        <ToggleButton value={false} aria-label="no" sx={{ color: 'error.main' }}>
          No
        </ToggleButton>
      </ToggleButtonGroup>
    </Box>
  );
}

export default CompactCheckbox;
