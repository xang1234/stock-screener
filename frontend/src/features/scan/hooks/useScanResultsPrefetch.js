import { useCallback, useEffect, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import {
  fetchPriceHistory,
  prefetchPriceHistoryBatch,
  priceHistoryKeys,
  PRICE_HISTORY_STALE_TIME,
} from '../../../api/priceHistory';
import { DEFAULT_FILTER_KEY } from '../constants';

const ROW_HOVER_PREFETCH_DELAY_MS = 150;

// Warms the chart's price history for the scan rows on screen. Returns the
// row-hover handler for the results table.
export function useScanResultsPrefetch({
  results,
  viewScanId,
  bootstrappedScanId,
  displayedQuery,
  stableFilterKey,
}) {
  const queryClient = useQueryClient();

  // Prefetch only once the pointer rests on a row; sweeping across the table
  // would otherwise fire one history request per row passed over.
  const hoverPrefetchTimerRef = useRef(null);
  const handleRowHover = useCallback(
    (symbol) => {
      clearTimeout(hoverPrefetchTimerRef.current);
      hoverPrefetchTimerRef.current = setTimeout(() => {
        queryClient.prefetchQuery({
          queryKey: priceHistoryKeys.symbol(symbol, '6mo'),
          queryFn: () => fetchPriceHistory(symbol, '6mo'),
          staleTime: PRICE_HISTORY_STALE_TIME,
        });
      }, ROW_HOVER_PREFETCH_DELAY_MS);
    },
    [queryClient]
  );
  useEffect(() => () => clearTimeout(hoverPrefetchTimerRef.current), []);

  useEffect(() => {
    if (!results || results.length === 0) {
      return;
    }
    if (
      bootstrappedScanId === viewScanId &&
      displayedQuery.page === 1 &&
      displayedQuery.perPage === 50 &&
      displayedQuery.sortBy === 'composite_score' &&
      displayedQuery.sortOrder === 'desc' &&
      stableFilterKey === DEFAULT_FILTER_KEY
    ) {
      return;
    }

    const visibleSymbols = results
      .slice(0, 20)
      .map((result) => result.symbol)
      .filter(Boolean);
    if (visibleSymbols.length === 0) {
      return;
    }

    let cancelled = false;
    const run = () => {
      if (cancelled) return;
      prefetchPriceHistoryBatch(queryClient, visibleSymbols, '6mo');
    };

    if ('requestIdleCallback' in window) {
      const handle = window.requestIdleCallback(run, { timeout: 1000 });
      return () => {
        cancelled = true;
        if (window.cancelIdleCallback) window.cancelIdleCallback(handle);
      };
    }
    const timer = setTimeout(run, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [
    bootstrappedScanId,
    viewScanId,
    queryClient,
    results,
    displayedQuery.page,
    displayedQuery.perPage,
    displayedQuery.sortBy,
    displayedQuery.sortOrder,
    stableFilterKey,
  ]);

  return handleRowHover;
}
