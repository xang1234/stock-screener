const NYSE_TIMEZONE = 'America/New_York';

const formatUniverseLabel = (scan) => {
  if (scan.universe_type) {
    switch (scan.universe_type) {
      case 'all':
        return 'All';
      case 'exchange':
        return scan.universe_exchange || 'Exchange';
      case 'index':
        return scan.universe_index === 'SP500' ? 'S&P500' : (scan.universe_index || 'Index');
      case 'custom':
        return `Custom (${scan.universe_symbols_count || '?'})`;
      case 'test':
        return `Test (${scan.universe_symbols_count || '?'})`;
      default:
        return scan.universe_type;
    }
  }

  const legacyUniverse = (scan.universe || '').toLowerCase();
  if (legacyUniverse === 'custom') return 'Test';
  if (legacyUniverse === 'sp500') return 'S&P500';
  if (legacyUniverse === 'all' || legacyUniverse === 'all stocks') return 'All';
  return scan.universe ? scan.universe.toUpperCase() : 'Unknown';
};

const formatScanTimestamp = (startedAt, triggerSource) => {
  const date = new Date(startedAt);
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
