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
]);

export const PERSISTED_QUERY_CACHE_BUSTER = 'v2';

export const shouldDehydratePersistedQuery = (query) => (
  query.state.status === 'success'
  && !NON_PERSISTED_QUERY_ROOTS.has(query.queryKey[0])
);
