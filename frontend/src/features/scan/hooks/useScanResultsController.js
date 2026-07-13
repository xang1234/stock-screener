import { useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';

import { getScanResults, queryScanResults } from '../../../api/scans';
import { buildFilterParams, getStableFilterKey } from '../../../utils/filterUtils';
import { buildScanQueryRequest } from '../filterExpression';
import { useScanFilterQueryState } from './useScanFilterQueryState';

export function useScanResultsController({
  currentScanId,
  scanStatus,
  groupedFilteringEnabled,
  debouncedFilters,
  initialExpression,
}) {
  const queryState = useScanFilterQueryState(initialExpression);
  const {
    requested,
    requestedKey,
    appliedSnapshot,
    requestQuickFilters,
    markRequestSucceeded,
  } = queryState;
  const {
    expression,
    page,
    perPage,
    sortBy,
    sortOrder,
  } = requested;

  useEffect(() => {
    requestQuickFilters(debouncedFilters);
  }, [debouncedFilters, requestQuickFilters]);

  const stableFilterKey = useMemo(
    () => getStableFilterKey(debouncedFilters),
    [debouncedFilters],
  );
  const legacyFilterParams = useMemo(
    () => buildFilterParams(debouncedFilters, { page, perPage, sortBy, sortOrder }),
    [debouncedFilters, page, perPage, sortBy, sortOrder],
  );
  const groupedQueryRequest = useMemo(
    () => buildScanQueryRequest(expression, {
      page,
      perPage,
      sortBy,
      sortOrder,
      includeSparklines: true,
      detailLevel: 'table',
    }),
    [expression, page, perPage, sortBy, sortOrder],
  );

  const resultQuery = useQuery({
    queryKey: groupedFilteringEnabled
      ? ['scanResultsQuery', currentScanId, requestedKey]
      : ['scanResults', currentScanId, page, perPage, sortBy, sortOrder, stableFilterKey],
    queryFn: ({ signal }) => (
      groupedFilteringEnabled
        ? queryScanResults(currentScanId, groupedQueryRequest, { signal })
        : getScanResults(currentScanId, legacyFilterParams)
    ),
    enabled: Boolean(currentScanId) && (scanStatus === 'completed' || scanStatus === 'cancelled'),
    staleTime: 10 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    placeholderData: (previousData, previousQuery) => (
      previousQuery?.queryKey?.[1] === currentScanId ? previousData : undefined
    ),
  });

  const currentSuccessSnapshot = useMemo(
    () => (
      groupedFilteringEnabled
      && resultQuery.isSuccess
      && !resultQuery.isPlaceholderData
        ? {
            request: requested,
            requestKey: requestedKey,
            scanId: currentScanId,
            data: resultQuery.data,
          }
        : null
    ),
    [
      currentScanId,
      groupedFilteringEnabled,
      requested,
      requestedKey,
      resultQuery.data,
      resultQuery.isPlaceholderData,
      resultQuery.isSuccess,
    ],
  );

  useEffect(() => {
    if (!currentSuccessSnapshot) return;
    markRequestSucceeded({
      requestKey: currentSuccessSnapshot.requestKey,
      scanId: currentSuccessSnapshot.scanId,
      data: currentSuccessSnapshot.data,
    });
  }, [currentSuccessSnapshot, markRequestSucceeded]);

  const displayedSnapshot = currentSuccessSnapshot
    ?? (appliedSnapshot?.scanId === currentScanId ? appliedSnapshot : null);
  const displayedResultsData = groupedFilteringEnabled
    ? displayedSnapshot?.data
    : resultQuery.data;
  const displayedQuery = groupedFilteringEnabled
    ? (displayedSnapshot?.request ?? requested)
    : requested;

  return {
    ...queryState,
    requestedExpression: expression,
    page,
    perPage,
    sortBy,
    sortOrder,
    displayedQuery,
    displayedResultsData,
    stableFilterKey,
    resultsLoading: resultQuery.isLoading,
    resultsFetching: resultQuery.isFetching,
    resultsError: resultQuery.isError ? resultQuery.error : null,
    refetchResults: resultQuery.refetch,
  };
}
