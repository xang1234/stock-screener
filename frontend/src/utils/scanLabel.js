const NYSE_TIMEZONE = 'America/New_York';
const ISO_WITHOUT_TZ = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$/;

const formatUniverseLabel = (scan) => {
  const ud = scan.universe_def;
  if (!ud || !ud.type) {
    return 'Unknown';
  }
  switch (ud.type) {
    case 'all':
      return 'All';
    case 'market':
      return ud.market || 'Market';
    case 'exchange':
      return ud.exchange || 'Exchange';
    case 'index':
      return ud.index === 'SP500' ? 'S&P500' : (ud.index || 'Index');
    case 'custom':
      return `Custom (${ud.symbols?.length ?? '?'})`;
    case 'test':
      return `Test (${ud.symbols?.length ?? '?'})`;
    default:
      return ud.type;
  }
};

const formatScanTimestamp = (startedAt, triggerSource) => {
  const normalizedStartedAt =
    typeof startedAt === 'string' && ISO_WITHOUT_TZ.test(startedAt)
      ? `${startedAt}Z`
      : startedAt;
  const date = new Date(normalizedStartedAt);
  if (Number.isNaN(date.getTime())) {
    return '-';
  }

  const isAuto = triggerSource === 'auto';
  return new Intl.DateTimeFormat('en-US', {
    timeZone: NYSE_TIMEZONE,
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    ...(isAuto ? {} : { hour: 'numeric', minute: '2-digit' }),
  }).format(date);
};

export const formatScanDropdownLabel = (scan) => {
  const sourceLabel = scan.trigger_source === 'auto' ? 'auto' : 'man';
  const summary = `${formatUniverseLabel(scan)} (${scan.passed_stocks}/${scan.total_stocks})`;
  return `${sourceLabel} ${formatScanTimestamp(scan.started_at, scan.trigger_source)} | ${summary}`;
};
