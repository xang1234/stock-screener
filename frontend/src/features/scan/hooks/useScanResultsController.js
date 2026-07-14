import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import { queryScanResults } from '../../../api/scans';
import { buildScanQueryRequest } from '../filterExpressionModel';
import { useScanFilterQueryState } from './useScanFilterQueryState';

export function useScanResultsController({
  currentScanId,
  scanStatus,
  initialFilters,
  initialExpression,
}) {
  const queryClient = useQueryClient();
  const [lastDisplayedRequest, setLastDisplayedRequest] = useState(null);
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

  const resultQueryKey = useMemo(
    () => ['scanResultsQuery', currentScanId, requestedKey],
    [currentScanId, requestedKey],
  );
  const resultQuery = useQuery({
    queryKey: resultQueryKey,
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

  useEffect(() => {
    const envelope = resultQuery.data;
    if (
      resultQuery.isPlaceholderData
      || envelope?.scanId !== currentScanId
      || envelope?.requestKey !== requestedKey
    ) return;
    setLastDisplayedRequest((previous) => (
      previous?.scanId === currentScanId && previous?.requestKey === requestedKey
        ? previous
        : { scanId: currentScanId, requestKey: requestedKey }
    ));
  }, [currentScanId, requestedKey, resultQuery.data, resultQuery.isPlaceholderData]);

  const cachedDisplayedEnvelope = lastDisplayedRequest?.scanId === currentScanId
    ? queryClient.getQueryData([
      'scanResultsQuery',
      lastDisplayedRequest.scanId,
      lastDisplayedRequest.requestKey,
    ])
    : undefined;
  const displayedEnvelope = resultQuery.data
    ?? cachedDisplayedEnvelope;
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
