import { Chip, Tooltip } from '@mui/material';

const QUALITY_META = {
  available: { label: 'Available', color: 'success' },
  building_history: { label: 'Building history', color: 'info' },
  insufficient_quality: { label: 'Limited quality', color: 'warning' },
  unavailable: { label: 'Unavailable', color: 'default' },
  failed: { label: 'Unavailable', color: 'error' },
};

const humanize = (value) => String(value || '').replaceAll('_', ' ');

export default function OptionsQualityBadge({ state, reasonCodes = [] }) {
  const meta = QUALITY_META[state] || QUALITY_META.unavailable;
  const explanation = reasonCodes.length > 0
    ? reasonCodes.map(humanize).join(', ')
    : meta.label;

  return (
    <Tooltip title={explanation}>
      <Chip
        size="small"
        variant="outlined"
        color={meta.color}
        label={meta.label}
        aria-label={`${meta.label}: ${explanation}`}
      />
    </Tooltip>
  );
}
