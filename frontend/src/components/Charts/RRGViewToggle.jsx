/**
 * Shared Table/RRG + Groups/Sectors toggle used by both the live and static
 * Group Rankings pages, so the control can't drift between the two trees.
 */
import { Box, ToggleButton, ToggleButtonGroup } from '@mui/material';

export default function RRGViewToggle({ view, onView, scope, onScope, sx }) {
  return (
    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center', ...sx }}>
      <ToggleButtonGroup size="small" exclusive value={view} onChange={(e, v) => v && onView(v)}>
        <ToggleButton value="table">Table</ToggleButton>
        <ToggleButton value="rrg">RRG</ToggleButton>
      </ToggleButtonGroup>
      {view === 'rrg' && (
        <ToggleButtonGroup size="small" exclusive value={scope} onChange={(e, v) => v && onScope(v)}>
          <ToggleButton value="groups">Groups</ToggleButton>
          <ToggleButton value="sectors">Sectors</ToggleButton>
        </ToggleButtonGroup>
      )}
    </Box>
  );
}
