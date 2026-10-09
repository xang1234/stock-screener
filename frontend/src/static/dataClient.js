import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getStaticDataUrl } from '../config/runtimeMode';
import { STATIC_DEFAULT_MARKET } from './StaticMarketContext';

// A file missing under a generation's data root: a later publish replaced that
// generation (only the current one is deployed, #504), so the tab must move on.
export class StaticGenerationExpiredError extends Error {
  constructor(relativePath, dataRoot) {
    super(`Static data was replaced by a newer publish: ${dataRoot}${relativePath}`);
    this.name = 'StaticGenerationExpiredError';
    this.dataRoot = dataRoot;
  }
}

// Announced on window so fetches outside React Query (scan chunk hydration)
// reach the tab's generation lifecycle too.
export const STATIC_GENERATION_EXPIRED_EVENT = 'static-data:generation-expired';

export const fetchStaticJson = async (relativePath, dataRoot = '') => {
  const response = await fetch(getStaticDataUrl(relativePath, dataRoot), {
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    if (response.status === 404 && dataRoot) {
      window.dispatchEvent(
        new CustomEvent(STATIC_GENERATION_EXPIRED_EVENT, { detail: { dataRoot } }),
      );
      throw new StaticGenerationExpiredError(relativePath, dataRoot);
    }
    throw new Error(`Failed to load static data: ${relativePath} (${response.status})`);
  }

  return response.json();
};

// Pages serves files with max-age=600; ``no-cache`` revalidates (usually a 304)
// so a tab learns about a new publish instead of holding the old pointer.
export const fetchStaticManifest = async () => {
  const response = await fetch(getStaticDataUrl('manifest.json'), {
    headers: { Accept: 'application/json' },
    cache: 'no-cache',
  });
  if (!response.ok) {
    throw new Error(`Failed to load static data: manifest.json (${response.status})`);
  }
  return response.json();
};

export const STATIC_MANIFEST_REFRESH_MS = 5 * 60 * 1000;

// A flat-layout publish (rollback switch) still gets its own generation from
// generated_at, so data keys switch together when the manifest changes.
// ``generation`` is null until the manifest has loaded: no data path is known yet.
export const getStaticGeneration = (manifest) => ({
  generation: !manifest
    ? null
    : manifest.generation || (manifest.generated_at ? `flat-${manifest.generated_at}` : 'flat'),
  dataRoot: manifest?.data_root || '',
});

const GENERATION_PREFIX = 'gen:';

export const withGeneration = (key, generation) => [...key, `${GENERATION_PREFIX}${generation}`];

export const queryKeyGeneration = (queryKey) => {
  const last = Array.isArray(queryKey) ? queryKey[queryKey.length - 1] : null;
  return typeof last === 'string' && last.startsWith(GENERATION_PREFIX)
    ? last.slice(GENERATION_PREFIX.length)
    : null;
};

const withoutGeneration = (queryKey) => (
  queryKeyGeneration(queryKey) === null ? queryKey : queryKey.slice(0, -1)
);

// placeholderData: keep showing the previous data while a new generation of
// the same query loads, never data from a different query (another market).
export const keepGenerationData = (queryKey) => (previousData, previousQuery) => {
  if (!previousQuery) return undefined;
  const same = JSON.stringify(withoutGeneration(previousQuery.queryKey))
    === JSON.stringify(withoutGeneration(queryKey));
  return same ? previousData : undefined;
};

export const fetchStaticBreadthContributorIndex = (indexPath, dataRoot = '') => (
  fetchStaticJson(indexPath, dataRoot)
);

export const fetchStaticBreadthContributors = (indexPath, date, dataRoot = '') => {
  if (!/\/index\.json$/.test(indexPath || '') || !/^\d{4}-\d{2}-\d{2}$/.test(date || '')) {
    throw new Error('Invalid static breadth contributor path');
  }
  return fetchStaticJson(indexPath.replace(/index\.json$/, `${date}.json`), dataRoot);
};

// Every observer reads the manifest; only the tab's lifecycle observer polls
// it (``{ poll: true }``), so mounted views do not each start an interval.
export const useStaticManifest = ({ poll = false } = {}) => useQuery({
  queryKey: ['staticManifest'],
  queryFn: fetchStaticManifest,
  staleTime: STATIC_MANIFEST_REFRESH_MS,
  refetchOnWindowFocus: poll,
  ...(poll ? { refetchInterval: STATIC_MANIFEST_REFRESH_MS } : {}),
  gcTime: Infinity,
});

export const useStaticGeneration = () => {
  const manifest = useStaticManifest().data;
  const loaded = Boolean(manifest);
  const generation = manifest?.generation;
  const dataRoot = manifest?.data_root;
  const generatedAt = manifest?.generated_at;
  return useMemo(
    () => getStaticGeneration(
      loaded ? { generation, data_root: dataRoot, generated_at: generatedAt } : null,
    ),
    [loaded, generation, dataRoot, generatedAt],
  );
};

// Options for a static data query: the generation joins the key, and the
// previous generation's data stays visible while the new one loads. gcTime is
// left to the app default unless the caller keeps data for good.
export const staticQueryOptions = ({ key, generation, ...options }) => {
  const queryKey = withGeneration(key, generation);
  return {
    staleTime: Infinity,
    ...options,
    // Nothing loads before the manifest names the generation.
    enabled: generation !== null && (options.enabled ?? true),
    queryKey,
    placeholderData: keepGenerationData(queryKey),
  };
};

export const useStaticGroupsRRG = (marketEntry) => {
  const path = marketEntry?.assets?.groups_rrg?.path;
  const { generation, dataRoot } = useStaticGeneration();
  return useQuery(staticQueryOptions({
    key: ['staticGroupsRRG', path],
    generation,
    queryFn: () => fetchStaticJson(path, dataRoot),
    enabled: Boolean(path),
    gcTime: Infinity,
  }));
};

export const getStaticSupportedMarkets = (manifest) => {
  if (Array.isArray(manifest?.supported_markets) && manifest.supported_markets.length > 0) {
    return manifest.supported_markets;
  }
  if (manifest?.default_market) {
    return [manifest.default_market];
  }
  return [STATIC_DEFAULT_MARKET];
};

export const getStaticUnavailableMarkets = (manifest) => (
  Array.isArray(manifest?.unavailable_markets) ? manifest.unavailable_markets : []
);

export const resolveStaticMarketEntry = (manifest, selectedMarket) => {
  const defaultMarket = String(manifest?.default_market || STATIC_DEFAULT_MARKET).toUpperCase();
  const supportedMarkets = getStaticSupportedMarkets(manifest).map((market) => String(market).toUpperCase());
  const normalizedMarket = String(selectedMarket || defaultMarket).toUpperCase();
  const resolvedMarket = supportedMarkets.includes(normalizedMarket) ? normalizedMarket : defaultMarket;
  const marketEntry = manifest?.markets?.[resolvedMarket];

  if (marketEntry) {
    return {
      market: resolvedMarket,
      display_name: marketEntry.display_name || resolvedMarket,
      as_of_date: marketEntry.as_of_date || manifest?.as_of_date || null,
      features: marketEntry.features || {},
      pages: marketEntry.pages || {},
      assets: marketEntry.assets || {},
      freshness: marketEntry.freshness || {},
    };
  }

  return {
    market: resolvedMarket,
    display_name: resolvedMarket,
    as_of_date: manifest?.as_of_date || null,
    features: manifest?.features || {},
    pages: manifest?.pages || {},
    assets: manifest?.assets || {},
    freshness: manifest?.freshness || {},
  };
};

export const useStaticGroupMatrix = (marketEntry, enabled) => {
  const path = marketEntry?.assets?.groups_matrix?.path;
  const { generation, dataRoot } = useStaticGeneration();
  return useQuery(staticQueryOptions({
    key: ['staticGroupMatrix', marketEntry?.market, path],
    generation,
    queryFn: () => fetchStaticJson(path, dataRoot),
    enabled: enabled && Boolean(path),
  }));
};
