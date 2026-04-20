import { useState } from 'react';
import {
  Box,
  Chip,
  Popover,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';

const WRAP_CHIP_SX = {
  height: 'auto',
  '& .MuiChip-label': {
    whiteSpace: 'normal',
    py: 0.25,
    lineHeight: 1.3,
  },
};

function WrapVariant({ themes, emptyText }) {
  if (!themes?.length) {
    return (
      <Typography variant="caption" color="text.secondary">
        {emptyText}
      </Typography>
    );
  }
  return (
    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
      {themes.map((theme, i) => (
        <Chip
          key={`${theme}-${i}`}
          label={theme}
          size="small"
          variant="outlined"
          color="primary"
          sx={WRAP_CHIP_SX}
        />
      ))}
    </Stack>
  );
}

function CompactVariant({ themes, maxInlineChips }) {
  const [anchorEl, setAnchorEl] = useState(null);

  if (!themes?.length) {
    return <Typography component="span" variant="body2" color="text.secondary">-</Typography>;
  }

  const inline = themes.slice(0, maxInlineChips);
  const overflow = themes.length - inline.length;
  const tooltipText = themes.join(' · ');

  const handleOpen = (event) => {
    event.stopPropagation();
    setAnchorEl(event.currentTarget);
  };
  const handleClose = (event) => {
    event?.stopPropagation?.();
    setAnchorEl(null);
  };

  // Intentionally no onClick on this Box: clicks on visible chips should
  // still bubble to the TableRow handler (open-chart on row click), matching
  // every other text cell. We stop propagation only on (a) the +N chip and
  // (b) the Popover's Paper, so that popover interactions don't also open
  // the chart via synthetic-event bubbling through the React tree.
  return (
    <Tooltip title={tooltipText} placement="top" arrow>
      <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5, maxWidth: '100%' }}>
        {inline.map((theme, i) => (
          <Chip
            key={`${theme}-${i}`}
            label={theme}
            size="small"
            variant="outlined"
            color="primary"
            sx={{
              height: 20,
              fontSize: '0.7rem',
              maxWidth: 110,
              '& .MuiChip-label': { px: 0.75 },
            }}
          />
        ))}
        {overflow > 0 && (
          <>
            <Chip
              label={`+${overflow}`}
              size="small"
              variant="outlined"
              onClick={handleOpen}
              sx={{
                height: 20,
                fontSize: '0.7rem',
                cursor: 'pointer',
                '& .MuiChip-label': { px: 0.75 },
              }}
            />
            <Popover
              open={Boolean(anchorEl)}
              anchorEl={anchorEl}
              onClose={handleClose}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
              transformOrigin={{ vertical: 'top', horizontal: 'left' }}
              slotProps={{
                paper: {
                  sx: { p: 1.25, maxWidth: 320 },
                  onClick: (e) => e.stopPropagation(),
                },
              }}
            >
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: 'block', mb: 0.5 }}
              >
                Market Themes ({themes.length})
              </Typography>
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {themes.map((theme, i) => (
                  <Chip
                    key={`${theme}-${i}`}
                    label={theme}
                    size="small"
                    variant="outlined"
                    color="primary"
                    sx={WRAP_CHIP_SX}
                  />
                ))}
              </Stack>
            </Popover>
          </>
        )}
      </Box>
    </Tooltip>
  );
}

export default function MarketThemesList({
  themes,
  variant = 'wrap',
  maxInlineChips = 1,
  emptyText = 'No market taxonomy themes are available for this symbol.',
}) {
  if (variant === 'compact') {
    return <CompactVariant themes={themes} maxInlineChips={maxInlineChips} />;
  }
  return <WrapVariant themes={themes} emptyText={emptyText} />;
}
