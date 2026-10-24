// App capabilities include auth state and must always be confirmed live.
// High-churn or bulky query families are cheap to refetch and expensive to
// serialize on every poll or prefetch.
const NON_PERSISTED_QUERY_ROOTS = new Set([
  'appCapabilities',
  'priceHistory',
  'runtimeActivity',
  'allFilteredSymbols',
  'calculationStatus',
  'setupDetails',
  // Scanned as-is, so a restored copy must never stand in for the live list.
  'userWatchlistSymbols',
]);

// Families where a reload only needs the most recently loaded entry. Every
// page / sort / filter variant of the scan results table is a separate
// sizeable query; persisting all of them bloats each (synchronous) cache
// write and pushes localStorage toward its quota.
const LATEST_ONLY_QUERY_ROOTS = new Set([
  'scanResultsQuery',
]);

export const PERSISTED_QUERY_CACHE_BUSTER = 'v2';

const isLatestInFamily = (query, queryCache) => {
  if (!queryCache) return true;
  const updatedAt = query.state.dataUpdatedAt ?? 0;
  return !queryCache.findAll({ queryKey: [query.queryKey[0]] }).some((other) => (
    other !== query
    && other.state.status === 'success'
    && (other.state.dataUpdatedAt ?? 0) > updatedAt
  ));
};

export const shouldDehydratePersistedQuery = (query, queryCache) => {
  if (query.state.status !== 'success') return false;
  const root = query.queryKey[0];
  if (NON_PERSISTED_QUERY_ROOTS.has(root)) return false;
  if (LATEST_ONLY_QUERY_ROOTS.has(root)) return isLatestInFamily(query, queryCache);
  return true;
};
