import { useQuery } from '@tanstack/react-query';
import { fetchStaticJson } from './dataClient';

export const staticChartKeys = {
  index: (path) => ['staticChartsIndex', path],
  payload: (symbol, path) => ['staticChartsPayload', symbol, path],
};

export const useStaticChartIndex = (path, enabled = true) => useQuery({
  queryKey: staticChartKeys.index(path),
  queryFn: () => fetchStaticJson(path),
  enabled: Boolean(path) && enabled,
  staleTime: Infinity,
  gcTime: Infinity,
});

export const fetchStaticChartPayload = (path) => fetchStaticJson(path);
