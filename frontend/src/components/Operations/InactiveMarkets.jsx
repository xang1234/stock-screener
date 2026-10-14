// Collapsed rows of markets that are not enabled and show no worker or live
// state in the data loaded (see activeMarkets.js). Still there to inspect or run.
import { useId, useState } from 'react';
import { Box, Button, Collapse, Typography } from '@mui/material';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

export default function InactiveMarkets({ markets, children }) {
  const [open, setOpen] = useState(false);
  const regionId = useId();
  if (!markets.length) return null;
  return (
    <Box sx={{ mt: 1 }}>
      <Button
        size="small"
        color="inherit"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-controls={regionId}
        startIcon={open ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        sx={{ color: 'text.secondary', textTransform: 'none' }}
      >
        Inactive markets ({markets.length}): {markets.join(', ')}
      </Button>
      <Collapse in={open} unmountOnExit id={regionId}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Not enabled, and no worker or activity seen for them on this page.
        </Typography>
        {children}
      </Collapse>
    </Box>
  );
}
