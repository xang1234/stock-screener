import { useState, useMemo, useCallback } from 'react';
import {
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  List,
  ListItem,
  ListItemText,
  Typography,
  Tooltip,
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

// Status values that should *not* be surfaced as "unavailable" — e.g. when
// the server returned an entry with status "available", we ignore it.
const SURFACED_STATUSES = new Set(['unavailable', 'unsupported', 'missing', 'computed']);

/**
 * Per-row transparency chip. Renders when the scan result carries any
 * field_availability entries whose status means "not a clean supported
 * value", or when growth_metric_basis is the "unavailable" sentinel.
 *
 * Silent in the common US-quarterly case (empty/null availability dict),
 * so US rows stay visually quiet.
 */
function FieldAvailabilityChip({ fieldAvailability, growthMetricBasis }) {
  const [open, setOpen] = useState(false);

  const entries = useMemo(() => {
    const list = [];
    if (fieldAvailability && typeof fieldAvailability === 'object') {
      for (const [field, entry] of Object.entries(fieldAvailability)) {
        if (!entry || typeof entry !== 'object') continue;
        if (!SURFACED_STATUSES.has(entry.status)) continue;
        list.push({ field, ...entry });
      }
    }
    return list;
  }, [fieldAvailability]);

  const cadenceNote = growthMetricBasis === 'unavailable'
    ? 'Growth metrics are unavailable for this row (insufficient statement history).'
    : null;

  const count = entries.length;
  const handleOpen = useCallback((e) => {
    e.stopPropagation();
    setOpen(true);
  }, []);
  const handleClose = useCallback((e) => {
    e?.stopPropagation?.();
    setOpen(false);
  }, []);

  if (count === 0 && !cadenceNote) return null;

  const tooltipText = count > 0
    ? `${count} field${count === 1 ? '' : 's'} unavailable or computed — click for details`
    : 'Growth metrics unavailable — click for details';

  return (
    <>
      <Tooltip title={tooltipText} arrow>
        <Chip
          size="small"
          icon={<InfoOutlinedIcon sx={{ fontSize: 12 }} />}
          label={count > 0 ? String(count) : '!'}
          onClick={handleOpen}
          sx={{
            height: 16,
            fontSize: 10,
            ml: 0.5,
            '& .MuiChip-label': { px: 0.5 },
            '& .MuiChip-icon': { ml: 0.25, mr: -0.25 },
          }}
          color="warning"
          variant="outlined"
          data-testid="field-availability-chip"
        />
      </Tooltip>
      <Dialog open={open} onClose={handleClose} onClick={(e) => e.stopPropagation()}>
        <DialogTitle>Data Availability</DialogTitle>
        <DialogContent dividers>
          {cadenceNote && (
            <Typography variant="body2" sx={{ mb: entries.length ? 2 : 0 }}>
              {cadenceNote}
            </Typography>
          )}
          {entries.length > 0 && (
            <List dense disablePadding>
              {entries.map(({ field, status, reason_code }) => (
                <ListItem key={field} disableGutters>
                  <ListItemText
                    primary={field}
                    secondary={
                      reason_code
                        ? `${status} — ${reason_code}`
                        : status
                    }
                    primaryTypographyProps={{
                      sx: { fontFamily: 'monospace', fontSize: 13 },
                    }}
                    secondaryTypographyProps={{ sx: { fontSize: 12 } }}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions>
          <Button size="small" onClick={handleClose}>Close</Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default FieldAvailabilityChip;
