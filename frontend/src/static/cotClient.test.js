import { beforeEach, describe, expect, it, vi } from 'vitest';

import { fetchStaticJson } from './dataClient';
import {
  getStaticCotHistory,
  getStaticCotIndex,
} from './cotClient';
import {
  makeCotHistory,
  staticCotIndexFixture,
} from '../features/cot/__fixtures__/cotResponses';

vi.mock('./dataClient', async (importOriginal) => ({ ...(await importOriginal()), fetchStaticJson: vi.fn() }));

describe('static COT client', () => {
  beforeEach(() => vi.clearAllMocks());

  it('fetches only root-manifest-advertised safe paths and slices locally', async () => {
    const fiveYear = makeCotHistory({ weekCount: 260, range: '5y' });
    fetchStaticJson
      .mockResolvedValueOnce(staticCotIndexFixture)
      .mockResolvedValueOnce(fiveYear);
    const rootManifest = { assets: { cot: { path: 'cot/index.json' } } };

    const index = await getStaticCotIndex(rootManifest);
    const history = await getStaticCotHistory(index, 'sp-500', '1y');

    expect(history.weeks).toHaveLength(52);
    expect(fetchStaticJson.mock.calls).toEqual([
      ['cot/index.json', ''],
      ['cot/sp-500.json', ''],
    ]);
  });

  it('rejects unadvertised and unlisted reads', async () => {
    await expect(getStaticCotIndex({ assets: {} })).rejects.toThrow(/advertised/i);
    await expect(getStaticCotHistory(staticCotIndexFixture, 'not-listed', '1y'))
      .rejects.toThrow(/advertised/i);
  });

  it('contains no live API route', async () => {
    const module = await import('./cotClient');
    expect(Object.values(module).some((value) => String(value).includes('/api'))).toBe(false);
  });
});
