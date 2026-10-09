import { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import {
  STATIC_GENERATION_EXPIRED_EVENT,
  getStaticGeneration,
  queryKeyGeneration,
  useStaticManifest,
} from './dataClient';

/**
 * Tab-level data generation housekeeping (#504). Mount once (the layout).
 *
 * - Polls the root manifest (focus and every few minutes).
 * - Bounds the query cache: a query from another generation is removed as
 *   soon as nothing renders it, so only the current generation (plus one still
 *   on screen as placeholder) is held despite ``gcTime: Infinity``.
 * - A file missing under the tab's generation means a later publish replaced
 *   it: re-check the manifest. ``expired`` is true once that check finished and
 *   left the tab on the same generation, so the UI can offer a reload.
 */
export function useStaticGenerationLifecycle() {
  const queryClient = useQueryClient();
  const manifestQuery = useStaticManifest({ poll: true });
  const { generation, dataRoot } = getStaticGeneration(manifestQuery.data);
  const [checkedExpiredRoot, setCheckedExpiredRoot] = useState(null);

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
    let active = true;
    const onExpired = (event) => {
      const expiredRoot = event.detail?.dataRoot ?? null;
      queryClient.refetchQueries({ queryKey: ['staticManifest'] }).finally(() => {
        if (active) setCheckedExpiredRoot(expiredRoot);
      });
    };
    window.addEventListener(STATIC_GENERATION_EXPIRED_EVENT, onExpired);
    return () => {
      active = false;
      window.removeEventListener(STATIC_GENERATION_EXPIRED_EVENT, onExpired);
    };
  }, [queryClient]);

  return {
    expired: Boolean(checkedExpiredRoot) && checkedExpiredRoot === dataRoot,
  };
}
