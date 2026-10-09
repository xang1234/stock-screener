import { useQuery } from '@tanstack/react-query';
import { fetchStaticJson, staticQueryOptions, useStaticGeneration } from './dataClient';

export const staticChartKeys = {
  index: (path) => ['staticChartsIndex', path],
  payload: (symbol, path) => ['staticChartsPayload', symbol, path],
};

// The index records the generation it was loaded from. While a new generation
// loads, the previous index is shown as placeholder; its entries must not be
// fetched from the new data root (see ``isCurrentChartIndex``).
export const useStaticChartIndex = (path, enabled = true) => {
  const { generation, dataRoot } = useStaticGeneration();
  return useQuery(staticQueryOptions({
    key: staticChartKeys.index(path),
    generation,
    queryFn: async () => ({ ...(await fetchStaticJson(path, dataRoot)), generation }),
    enabled: Boolean(path) && enabled,
    gcTime: Infinity,
  }));
};

// An index not loaded through ``useStaticChartIndex`` carries no generation
// and is taken as current.
export const isCurrentChartIndex = (chartIndex, generation) => (
  Boolean(chartIndex)
  && (chartIndex.generation === undefined || chartIndex.generation === generation)
);

export const fetchStaticChartPayload = (path, dataRoot = '') => fetchStaticJson(path, dataRoot);

// Query options for one symbol's chart payload in the current generation.
export const staticChartPayloadQuery = (symbol, path, { generation, dataRoot }) => staticQueryOptions({
  key: staticChartKeys.payload(symbol, path),
  generation,
  queryFn: () => fetchStaticChartPayload(path, dataRoot),
  enabled: Boolean(path),
  gcTime: Infinity,
});
