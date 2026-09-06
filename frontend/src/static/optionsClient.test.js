import { beforeEach, describe, expect, it, vi } from 'vitest';

import { fetchStaticJson } from './dataClient';
import {
  getStaticOptionsCommandCenter,
  getStaticOptionsManifest,
  getStaticOptionsSymbolDetail,
  staticOptionsCommandCenterQueryOptions,
  staticOptionsSymbolQueryOptions,
} from './optionsClient';
import {
  commandCenterFixture,
  optionsManifestFixture,
  symbolDetailFixture,
} from '../features/options/__fixtures__/optionsResponses';

vi.mock('./dataClient', () => ({ fetchStaticJson: vi.fn() }));

describe('static options client', () => {
  beforeEach(() => vi.clearAllMocks());

  it('loads only manifest-advertised paths and returns live-equivalent models', async () => {
    fetchStaticJson
      .mockResolvedValueOnce(optionsManifestFixture)
      .mockResolvedValueOnce(commandCenterFixture)
      .mockResolvedValueOnce(symbolDetailFixture);
    const marketEntry = { pages: { options: { path: 'options/manifest.json' } } };

    const manifest = await getStaticOptionsManifest(marketEntry);
    const command = await getStaticOptionsCommandCenter(manifest);
    const detail = await getStaticOptionsSymbolDetail(manifest, 'aapl');

    expect(command).toEqual(commandCenterFixture);
    expect(detail).toEqual(symbolDetailFixture);
    expect(fetchStaticJson.mock.calls).toEqual([
      ['options/manifest.json'],
      ['options/command-center.json'],
      ['options/symbols/QUFQTA.json'],
    ]);
  });

  it('uses immutable query caching and cannot derive an unadvertised symbol path', () => {
    const command = staticOptionsCommandCenterQueryOptions(optionsManifestFixture);
    const detail = staticOptionsSymbolQueryOptions(optionsManifestFixture, 'AAPL');
    expect(command.staleTime).toBe(Infinity);
    expect(command.gcTime).toBe(Infinity);
    expect(detail.queryKey).toContain('AAPL');
    expect(() => staticOptionsSymbolQueryOptions(optionsManifestFixture, 'NVDA'))
      .toThrow(/advertised/i);
  });

  it('has no static refresh mutation', async () => {
    const module = await import('./optionsClient');
    expect(module.refreshOptionsAnalytics).toBeUndefined();
  });
});
