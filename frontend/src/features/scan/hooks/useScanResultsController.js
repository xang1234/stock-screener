import { useEffect, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import { queryScanResults } from '../../../api/scans';
import { buildScanQueryRequest } from '../filterExpressionModel';
import { useScanFilterQueryState } from './useScanFilterQueryState';

function latestCachedEnvelope(queryClient, scanId) {
  if (!scanId) return undefined;
  return queryClient.getQueryCache().findAll({
    queryKey: ['scanResultsQuery', scanId],
  }).reduce((latest, query) => {
    if (query.state.data == null) return latest;
    if (latest == null || query.state.dataUpdatedAt > latest.updatedAt) {
      return { data: query.state.data, updatedAt: query.state.dataUpdatedAt };
    }
    return latest;
  }, null)?.data;
}

export function useScanResultsController({
  currentScanId,
  scanStatus,
  initialFilters,
  initialExpression,
}) {
  const queryClient = useQueryClient();
  const queryState = useScanFilterQueryState({
    defaultFilters: initialFilters,
    expression: initialExpression,
  });
  const {
    requested,
    requestedKey,
    draftExpressionKey,
    committedExpressionKey,
    commitQuickFilters,
  } = queryState;
  const {
    expression,
    page,
    perPage,
    sortBy,
    sortOrder,
  } = requested;

  useEffect(() => {
    if (draftExpressionKey === committedExpressionKey) return undefined;
    const timer = setTimeout(() => commitQuickFilters(draftExpressionKey), 300);
    return () => clearTimeout(timer);
  }, [commitQuickFilters, committedExpressionKey, draftExpressionKey]);

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
    queryFn: async ({ signal }) => ({
      data: await queryScanResults(currentScanId, queryRequest, { signal }),
      request: requested,
      requestKey: requestedKey,
      scanId: currentScanId,
    }),
    enabled: Boolean(currentScanId) && (scanStatus === 'completed' || scanStatus === 'cancelled'),
    staleTime: 10 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    placeholderData: (previousData, previousQuery) => (
      previousQuery?.queryKey?.[1] === currentScanId ? previousData : undefined
    ),
  });

  const displayedEnvelope = resultQuery.data
    ?? latestCachedEnvelope(queryClient, currentScanId);
  const displayedResultsData = displayedEnvelope?.data;
  const displayedQuery = displayedEnvelope?.request ?? requested;

  return {
    ...queryState,
    page,
    perPage,
    sortBy,
    sortOrder,
    displayedQuery,
    displayedResultsData,
    stableFilterKey: committedExpressionKey,
    resultsLoading: resultQuery.isPending && !displayedEnvelope,
    resultsFetching: resultQuery.isFetching,
    resultsError: resultQuery.error ?? null,
    refetchResults: resultQuery.refetch,
  };
}
