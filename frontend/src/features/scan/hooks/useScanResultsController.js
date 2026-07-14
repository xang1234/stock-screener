import { useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';

import { queryScanResults } from '../../../api/scans';
import { buildScanQueryRequest } from '../filterExpressionModel';
import { useScanFilterQueryState } from './useScanFilterQueryState';

export function useScanResultsController({
  currentScanId,
  scanStatus,
  initialFilters,
  initialExpression,
}) {
  const queryState = useScanFilterQueryState({
    defaultFilters: initialFilters,
    expression: initialExpression,
  });
  const {
    requested,
    requestedKey,
    appliedSnapshot,
    filterKey,
    committedFilterKey,
    commitQuickFilters,
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
    if (filterKey === committedFilterKey) return undefined;
    const timer = setTimeout(() => commitQuickFilters(filterKey), 300);
    return () => clearTimeout(timer);
  }, [commitQuickFilters, committedFilterKey, filterKey]);

  const queryRequest = useMemo(
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
    queryKey: ['scanResultsQuery', currentScanId, requestedKey],
    queryFn: ({ signal }) => queryScanResults(
      currentScanId,
      queryRequest,
      { signal },
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
      resultQuery.isSuccess
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
  const displayedResultsData = displayedSnapshot?.data;
  const displayedQuery = displayedSnapshot?.request ?? requested;

  return {
    ...queryState,
    requestedExpression: expression,
    page,
    perPage,
    sortBy,
    sortOrder,
    displayedQuery,
    displayedResultsData,
    stableFilterKey: committedFilterKey,
    resultsLoading: resultQuery.isLoading,
    resultsFetching: resultQuery.isFetching,
    resultsError: resultQuery.isError ? resultQuery.error : null,
    refetchResults: resultQuery.refetch,
  };
}
