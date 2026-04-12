import { Box, Chip, Divider, Tooltip } from '@mui/material';

const chipSx = (isActive) => ({
  fontSize: '11px',
  fontWeight: isActive ? 600 : 400,
  cursor: 'pointer',
  '& .MuiChip-label': { px: 1 },
});

function ScreenSelector({ screens, activeScreenId, onSelectScreen, matchCounts }) {
  if (!screens?.length) return null;

  const tier1 = screens.filter((s) => s.tier === 1);
  const tier2 = screens.filter((s) => s.tier !== 1);

  const renderChip = (screen) => {
    const isActive = activeScreenId === screen.id;
    const count = matchCounts?.[screen.id];
    const label = count != null ? `${screen.short_name} (${count})` : screen.short_name;

    return (
      <Tooltip key={screen.id} title={`${screen.name} — ${screen.description}`} arrow>
        <Chip
          label={label}
          size="small"
          variant={isActive ? 'filled' : 'outlined'}
          color={isActive ? 'primary' : 'default'}
          onClick={() => onSelectScreen(isActive ? null : screen.id)}
          sx={chipSx(isActive)}
        />
      </Tooltip>
    );
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
        overflowX: 'auto',
        pb: 0.5,
        mb: 1.5,
        '&::-webkit-scrollbar': { height: 4 },
        '&::-webkit-scrollbar-thumb': { borderRadius: 2, bgcolor: 'divider' },
      }}
    >
      <Chip
        label="All Stocks"
        size="small"
        variant={activeScreenId == null ? 'filled' : 'outlined'}
        color={activeScreenId == null ? 'primary' : 'default'}
        onClick={() => onSelectScreen(null)}
        sx={chipSx(activeScreenId == null)}
      />
      {tier1.map(renderChip)}
      {tier2.length > 0 && (
        <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
      )}
      {tier2.map(renderChip)}
    </Box>
  );
}

export default ScreenSelector;
