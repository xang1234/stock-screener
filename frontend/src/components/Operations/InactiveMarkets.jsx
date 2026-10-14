// Collapsed rows of markets that are not enabled, have no worker, and hold
// no live state (see activeMarkets.js). Still there to inspect or run.
import { useState } from 'react';
import { Box, Button, Collapse, Typography } from '@mui/material';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

export default function InactiveMarkets({ markets, children }) {
  const [open, setOpen] = useState(false);
  if (!markets.length) return null;
  return (
    <Box sx={{ mt: 1 }}>
      <Button
        size="small"
        color="inherit"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        startIcon={open ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        sx={{ color: 'text.secondary', textTransform: 'none' }}
      >
        Inactive markets ({markets.length}): {markets.join(', ')}
      </Button>
      <Collapse in={open} unmountOnExit>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Not enabled, no worker serves them, and nothing is queued or held.
        </Typography>
        {children}
      </Collapse>
    </Box>
  );
}
