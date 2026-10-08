import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  StaticGenerationExpiredError,
  fetchStaticJson,
  fetchStaticManifest,
  getStaticGeneration,
  keepGenerationData,
  withGeneration,
} from './dataClient';

const respond = (status, body = {}) => vi.fn(async () => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => body,
}));

describe('static data generations (#504)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads data files under the generation data root', async () => {
    const fetchMock = respond(200, { rows: [] });
    vi.stubGlobal('fetch', fetchMock);

    await fetchStaticJson('markets/us/home.json', 'g/20261008T054712Z-abcd1234/');

    expect(fetchMock.mock.calls[0][0]).toMatch(/static-data\/g\/20261008T054712Z-abcd1234\/markets\/us\/home\.json$/);
  });

  it('keeps the flat layout for a manifest without a data root', async () => {
    const fetchMock = respond(200, {});
    vi.stubGlobal('fetch', fetchMock);

    await fetchStaticJson('markets/us/home.json');

    expect(fetchMock.mock.calls[0][0]).toMatch(/static-data\/markets\/us\/home\.json$/);
  });

  it('reports a missing file under a generation as an expired generation', async () => {
    vi.stubGlobal('fetch', respond(404));

    await expect(fetchStaticJson('markets/us/home.json', 'g/old/')).rejects.toBeInstanceOf(
      StaticGenerationExpiredError,
    );
  });

  it('reports a missing flat file as an ordinary error', async () => {
    vi.stubGlobal('fetch', respond(404));

    const error = await fetchStaticJson('markets/us/home.json').catch((caught) => caught);

    expect(error).toBeInstanceOf(Error);
    expect(error).not.toBeInstanceOf(StaticGenerationExpiredError);
  });

  it('revalidates the root manifest instead of trusting the browser cache', async () => {
    const fetchMock = respond(200, { generation: 'g1' });
    vi.stubGlobal('fetch', fetchMock);

    await fetchStaticManifest();

    expect(fetchMock.mock.calls[0][0]).toMatch(/static-data\/manifest\.json$/);
    expect(fetchMock.mock.calls[0][1].cache).toBe('no-cache');
  });

  it('derives the generation from the manifest', () => {
    expect(getStaticGeneration({ generation: 'g1', data_root: 'g/g1/' })).toEqual({
      generation: 'g1',
      dataRoot: 'g/g1/',
    });
    expect(getStaticGeneration({})).toEqual({ generation: 'flat', dataRoot: '' });
    expect(getStaticGeneration(undefined)).toEqual({ generation: 'flat', dataRoot: '' });
  });

  it('keeps previous data only across a generation change of the same query', () => {
    const key = withGeneration(['staticHome', 'markets/us/home.json'], 'g2');
    const keep = keepGenerationData(key);

    const sameQuery = { queryKey: withGeneration(['staticHome', 'markets/us/home.json'], 'g1') };
    const otherMarket = { queryKey: withGeneration(['staticHome', 'markets/hk/home.json'], 'g1') };

    expect(keep('old-us', sameQuery)).toBe('old-us');
    expect(keep('old-hk', otherMarket)).toBeUndefined();
    expect(keep(undefined, undefined)).toBeUndefined();
  });
});
