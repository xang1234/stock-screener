const SECTION_DATE_LABELS = [
  ['Breadth', 'breadth_latest_date'],
  ['Groups', 'groups_latest_date'],
  ['Exposure', 'exposure_latest_date'],
  ['Key markets', 'key_markets_latest_date'],
];

export function formatSnapshotFreshnessLabel(freshness = {}) {
  const snapshotDate = freshness?.snapshot_as_of_date || null;
  const scanDate = freshness?.scan_as_of_date || null;
  const status = freshness?.date_coherence_status || null;
  const parts = snapshotDate
    ? [
        `As of ${snapshotDate}`,
        freshness?.market_timezone,
        status && status !== 'coherent' ? status : null,
      ]
    : [
        'Snapshot date unavailable',
        scanDate ? `Scan ${scanDate}` : null,
        status && status !== 'unknown' ? status : null,
      ];

  const sectionDates = SECTION_DATE_LABELS
    .map(([label, key]) => [label, freshness?.[key]])
    .filter(([, value]) => value && value !== snapshotDate);

  return [
    ...parts,
    ...sectionDates.map(([label, value]) => `${label} ${value}`),
  ].filter(Boolean).join(' · ');
}
