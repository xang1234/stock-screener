import { fetchStaticJson } from './dataClient';
import {
  normalizeOptionsCommandCenter,
  normalizeOptionsManifest,
  normalizeOptionsSymbolDetail,
  optionsCommandCenterQueryKey,
  optionsManifestRunContext,
  optionsSymbolQueryKey,
} from '../features/options/optionsContract';

const advertisedSymbol = (manifest, symbol) => {
  const normalized = String(symbol || '').trim().toUpperCase();
  const entry = manifest.symbols[normalized];
  if (!entry) throw new Error(`Options symbol ${normalized || '(empty)'} is not advertised`);
  return { symbol: normalized, entry };
};

export const getStaticOptionsManifest = async (marketEntry) => {
  const path = marketEntry?.pages?.options?.path;
  if (!path) throw new Error('Options Command Center is not advertised for this market');
  const manifest = await fetchStaticJson(path);
  return normalizeOptionsManifest(manifest);
};

export const getStaticOptionsCommandCenter = async (rawManifest) => {
  const manifest = normalizeOptionsManifest(rawManifest);
  const payload = await fetchStaticJson(manifest.command_center_path);
  return normalizeOptionsCommandCenter(payload, optionsManifestRunContext(manifest));
};

export const getStaticOptionsSymbolDetail = async (rawManifest, rawSymbol) => {
  const manifest = normalizeOptionsManifest(rawManifest);
  const { symbol, entry } = advertisedSymbol(manifest, rawSymbol);
  const payload = await fetchStaticJson(entry.path);
  return normalizeOptionsSymbolDetail(payload, {
    ...optionsManifestRunContext(manifest),
    expectedSymbol: symbol,
  });
};

export const staticOptionsCommandCenterQueryOptions = (rawManifest) => {
  const manifest = normalizeOptionsManifest(rawManifest);
  return {
    queryKey: optionsCommandCenterQueryKey({
      mode: 'static',
      runId: manifest.published_run_id,
      path: manifest.command_center_path,
    }),
    queryFn: () => getStaticOptionsCommandCenter(manifest),
    staleTime: Infinity,
    gcTime: Infinity,
  };
};

export const staticOptionsSymbolQueryOptions = (rawManifest, rawSymbol) => {
  const manifest = normalizeOptionsManifest(rawManifest);
  const { symbol, entry } = advertisedSymbol(manifest, rawSymbol);
  return {
    queryKey: optionsSymbolQueryKey({
      mode: 'static',
      runId: manifest.published_run_id,
      symbol,
      path: entry.path,
    }),
    queryFn: () => getStaticOptionsSymbolDetail(manifest, symbol),
    staleTime: Infinity,
    gcTime: Infinity,
  };
};
