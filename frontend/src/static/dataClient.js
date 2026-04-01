import { useQuery } from '@tanstack/react-query';
import { getStaticDataUrl } from '../config/runtimeMode';

export const fetchStaticJson = async (relativePath) => {
  const response = await fetch(getStaticDataUrl(relativePath), {
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to load static data: ${relativePath} (${response.status})`);
  }

  return response.json();
};

export const useStaticManifest = () => useQuery({
  queryKey: ['staticManifest'],
  queryFn: () => fetchStaticJson('manifest.json'),
  staleTime: Infinity,
  gcTime: Infinity,
});
