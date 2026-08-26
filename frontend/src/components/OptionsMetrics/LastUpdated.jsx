/**
 * LastUpdated
 *
 * Shows when a piece of data was actually fetched/computed, e.g.
 * "Updated 12 min ago (5:32 PM)". Reused across every options data section
 * so staleness is never ambiguous -- a page load can mix a 7-day-old Redis
 * cache hit, a nightly batch row from hours ago, and a just-computed live
 * result, and those can look identical without a timestamp.
 */
import { Typography } from '@mui/material';
import { formatDistanceToNow, format } from 'date-fns';

/**
 * Backend timestamps are serialized from naive UTC datetimes
 * (datetime.utcnow().isoformat()) with no 'Z'/offset suffix. Without one,
 * JS Date parsing treats the string as local time instead of UTC, which
 * would show the wrong hour (off by the browser's UTC offset). Every
 * timestamp field across these endpoints (fetched_at, timestamp, as_of,
 * computed_at) follows this same naive-UTC convention.
 */
function parseBackendUtcTimestamp(isoString) {
  if (!isoString) return null;
  const hasTimezone = /[Zz]|[+-]\d{2}:?\d{2}$/.test(isoString);
  const date = new Date(hasTimezone ? isoString : `${isoString}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

/**
 * @param {Object} props
 * @param {string} props.timestamp - ISO timestamp string (naive UTC, backend convention)
 * @param {string} [props.prefix='Updated']
 * @param {object} [props.sx]
 */
export default function LastUpdated({ timestamp, prefix = 'Updated', sx }) {
  const date = parseBackendUtcTimestamp(timestamp);
  if (!date) return null;

  return (
    <Typography variant="caption" color="text.secondary" sx={sx}>
      {prefix} {formatDistanceToNow(date, { addSuffix: true })} ({format(date, 'MMM d, h:mm a')})
    </Typography>
  );
}
