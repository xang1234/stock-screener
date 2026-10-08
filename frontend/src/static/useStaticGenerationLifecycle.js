import { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import {
  STATIC_GENERATION_EXPIRED_EVENT,
  getStaticGeneration,
  queryKeyGeneration,
  useStaticManifest,
} from './dataClient';

/**
 * Tab-level data generation housekeeping (#504).
 *
 * - Bounds the query cache: a query from another generation is removed as
 *   soon as nothing renders it, so only the current generation (plus one still
 *   on screen as placeholder) is held despite ``gcTime: Infinity``.
 * - A file missing under the tab's generation means a later publish replaced
 *   it: check the manifest now. ``expired`` stays true only while the tab is
 *   still on that generation afterwards, so the UI can offer a reload.
 */
export function useStaticGenerationLifecycle() {
  const queryClient = useQueryClient();
  const manifestQuery = useStaticManifest();
  const { generation, dataRoot } = getStaticGeneration(manifestQuery.data);
  const [expiredRoot, setExpiredRoot] = useState(null);

  useEffect(() => {
    const cache = queryClient.getQueryCache();
    const evict = (query) => {
      const queryGeneration = queryKeyGeneration(query.queryKey);
      if (queryGeneration !== null && queryGeneration !== generation && query.getObserversCount() === 0) {
        cache.remove(query);
      }
    };
    cache.getAll().forEach(evict);
    return cache.subscribe((event) => {
      if (event.type === 'observerRemoved') {
        evict(event.query);
      }
    });
  }, [generation, queryClient]);

  useEffect(() => {
    const onExpired = (event) => {
      setExpiredRoot(event.detail?.dataRoot ?? null);
      queryClient.invalidateQueries({ queryKey: ['staticManifest'] });
    };
    window.addEventListener(STATIC_GENERATION_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(STATIC_GENERATION_EXPIRED_EVENT, onExpired);
  }, [queryClient]);

  return {
    expired: Boolean(expiredRoot) && expiredRoot === dataRoot && !manifestQuery.isFetching,
  };
}
