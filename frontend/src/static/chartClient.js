import { useQuery } from '@tanstack/react-query';
import { fetchStaticJson, staticQueryOptions, useStaticGeneration } from './dataClient';

export const staticChartKeys = {
  index: (path) => ['staticChartsIndex', path],
  payload: (symbol, path) => ['staticChartsPayload', symbol, path],
};

export const useStaticChartIndex = (path, enabled = true) => {
  const { generation, dataRoot } = useStaticGeneration();
  return useQuery(staticQueryOptions({
    key: staticChartKeys.index(path),
    generation,
    queryFn: () => fetchStaticJson(path, dataRoot),
    enabled: Boolean(path) && enabled,
    gcTime: Infinity,
  }));
};

export const fetchStaticChartPayload = (path, dataRoot = '') => fetchStaticJson(path, dataRoot);

// Query options for one symbol's chart payload in the current generation.
export const staticChartPayloadQuery = (symbol, path, { generation, dataRoot }) => staticQueryOptions({
  key: staticChartKeys.payload(symbol, path),
  generation,
  queryFn: () => fetchStaticChartPayload(path, dataRoot),
  enabled: Boolean(path),
  gcTime: Infinity,
});
