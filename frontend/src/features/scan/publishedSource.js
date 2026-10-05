// Last-published scans (#492): when current data or compute is busy, the user
// can ask the backend for a completed scan from the last published snapshot.

// Create-scan conflicts that block a `current` scan but not a last-published
// read, which never computes and ignores price freshness.
const LAST_PUBLISHED_FALLBACK_CODES = new Set([
  'market_refresh_active',
  'scan_already_active',
  'market_data_stale',
]);

export function canOfferLastPublished(refreshConflict, createScanError) {
  const code = createScanError?.detail?.code;
  // The backend already said no publication qualifies; retrying is pointless.
  if (code === 'snapshot_unavailable') {
    return false;
  }
  return Boolean(refreshConflict) || LAST_PUBLISHED_FALLBACK_CODES.has(code);
}

/**
 * Describe where a completed scan's rows came from, or null for computed scans.
 * Dates stay ISO (YYYY-MM-DD): they are market sessions, not local instants.
 * ``is_current`` and ``expected_session`` were recorded when the scan was
 * created, so the text says so rather than claiming they still hold.
 */
export function describePublishedSource(source) {
  if (!source?.as_of_date) {
    return null;
  }
  const asOf = source.as_of_date;
  const lastPublished = source.data_mode === 'last_published';
  const prefix = lastPublished ? 'Last published data' : 'Served from the published snapshot';
  if (source.is_current === false) {
    const session = source.expected_session
      ? ` When this scan was created, the latest completed session was ${source.expected_session}.`
      : ' It was older than the latest completed session when this scan was created.';
    return {
      severity: 'warning',
      text: `${prefix} as of ${asOf}.${session}`,
    };
  }
  const current = source.is_current === true
    ? ', the latest completed session when this scan was created'
    : '';
  return {
    severity: 'info',
    text: `${prefix} as of ${asOf}${current}.`,
  };
}
