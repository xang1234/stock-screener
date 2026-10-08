import { fetchStaticJson, getStaticGeneration, keepGenerationData, withGeneration } from './dataClient';
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

const FLAT = getStaticGeneration({});

const generationOptions = (key, generation) => {
  const queryKey = withGeneration(key, generation);
  return { queryKey, placeholderData: keepGenerationData(queryKey) };
};

export const getStaticOptionsManifest = async (marketEntry, dataRoot = '') => {
  const path = marketEntry?.pages?.options?.path;
  if (!path) throw new Error('Options Command Center is not advertised for this market');
  const manifest = await fetchStaticJson(path, dataRoot);
  return normalizeOptionsManifest(manifest);
};

export const getStaticOptionsCommandCenter = async (rawManifest, dataRoot = '') => {
  const manifest = normalizeOptionsManifest(rawManifest);
  const payload = await fetchStaticJson(manifest.command_center_path, dataRoot);
  return normalizeOptionsCommandCenter(payload, optionsManifestRunContext(manifest));
};

export const getStaticOptionsSymbolDetail = async (rawManifest, rawSymbol, dataRoot = '') => {
  const manifest = normalizeOptionsManifest(rawManifest);
  const { symbol, entry } = advertisedSymbol(manifest, rawSymbol);
  const payload = await fetchStaticJson(entry.path, dataRoot);
  return normalizeOptionsSymbolDetail(payload, {
    ...optionsManifestRunContext(manifest),
    expectedSymbol: symbol,
  });
};

export const staticOptionsCommandCenterQueryOptions = (rawManifest, { generation, dataRoot } = FLAT) => {
  const manifest = normalizeOptionsManifest(rawManifest);
  return {
    ...generationOptions(optionsCommandCenterQueryKey({
      mode: 'static',
      runId: manifest.published_run_id,
      path: manifest.command_center_path,
    }), generation),
    queryFn: () => getStaticOptionsCommandCenter(manifest, dataRoot),
    staleTime: Infinity,
    gcTime: Infinity,
  };
};

export const staticOptionsSymbolQueryOptions = (rawManifest, rawSymbol, { generation, dataRoot } = FLAT) => {
  const manifest = normalizeOptionsManifest(rawManifest);
  const { symbol, entry } = advertisedSymbol(manifest, rawSymbol);
  return {
    ...generationOptions(optionsSymbolQueryKey({
      mode: 'static',
      runId: manifest.published_run_id,
      symbol,
      path: entry.path,
    }), generation),
    queryFn: () => getStaticOptionsSymbolDetail(manifest, symbol, dataRoot),
    staleTime: Infinity,
    gcTime: Infinity,
  };
};
