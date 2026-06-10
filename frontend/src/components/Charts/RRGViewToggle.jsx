/**
 * Shared Table/RRG + Groups/Sectors toggle used by both the live and static
 * Group Rankings pages, so the control can't drift between the two trees.
 */
import { Box, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { RRG_SCOPE_LABELS, normalizeRrgScopes } from '../../utils/rrgScopes';

export default function RRGViewToggle({
  view,
  onView,
  scope,
  onScope,
  sx,
  rrgAvailable = true,
  availableScopes,
}) {
  const scopes = normalizeRrgScopes(availableScopes);
  const canShowRRG = rrgAvailable && scopes.length > 0;
  const viewValue = canShowRRG ? view : 'table';
  const showScopeToggle = canShowRRG && view === 'rrg' && scopes.length > 1;

  return (
    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center', ...sx }}>
      <ToggleButtonGroup size="small" exclusive value={viewValue} onChange={(e, v) => v && onView(v)}>
        <ToggleButton value="table">Table</ToggleButton>
        {canShowRRG && <ToggleButton value="rrg">RRG</ToggleButton>}
      </ToggleButtonGroup>
      {showScopeToggle && (
        <ToggleButtonGroup size="small" exclusive value={scope} onChange={(e, v) => v && onScope(v)}>
          {scopes.map((availableScope) => (
            <ToggleButton key={availableScope} value={availableScope}>
              {RRG_SCOPE_LABELS[availableScope]}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      )}
    </Box>
  );
}
