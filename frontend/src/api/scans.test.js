import apiClient from './client';
import {
  getAllFilteredSymbols,
  getScanResults,
  getSetupDetails,
  getSingleResult,
} from './scans';

vi.mock('./client', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

const deferred = () => {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
};

describe('scan api helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('defaults /results to table detail level', async () => {
    apiClient.get.mockResolvedValueOnce({ data: { results: [] } });

    await getScanResults('scan-1', { page: 2, per_page: 50 });

    expect(apiClient.get).toHaveBeenCalledWith('/v1/scans/scan-1/results', {
      params: {
        detail_level: 'table',
        page: 2,
        per_page: 50,
      },
    });
  });

  it('defaults /result/{symbol} to core detail level', async () => {
    apiClient.get.mockResolvedValueOnce({ data: { symbol: 'AAPL' } });

    await getSingleResult('scan-2', 'AAPL');

    expect(apiClient.get).toHaveBeenCalledWith('/v1/scans/scan-2/result/AAPL', {
      params: {
        detail_level: 'core',
      },
    });
  });

  it('fetches setup details from dedicated endpoint', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: { symbol: 'MSFT', se_explain: { summary: 'ok' } },
    });

    await getSetupDetails('scan-3', 'MSFT');

    expect(apiClient.get).toHaveBeenCalledWith('/v1/scans/scan-3/setup/MSFT');
  });

  it('uses /symbols fast path when available', async () => {
    apiClient.get.mockResolvedValueOnce({ data: { symbols: ['AAPL', 'MSFT'] } });

    const symbols = await getAllFilteredSymbols('scan-4', { sort_by: 'symbol' });

    expect(symbols).toEqual(['AAPL', 'MSFT']);
    expect(apiClient.get).toHaveBeenCalledTimes(1);
    expect(apiClient.get).toHaveBeenCalledWith('/v1/scans/scan-4/symbols', {
      params: { sort_by: 'symbol' },
    });
  });

  it('fallback /results batching does not eagerly start all pages', async () => {
    const page2 = deferred();
    const page3 = deferred();
    const page4 = deferred();

    apiClient.get.mockImplementation((url, config = {}) => {
      if (url === '/v1/scans/scan-5/symbols') {
        return Promise.reject(new Error('symbols endpoint unavailable'));
      }
      if (url !== '/v1/scans/scan-5/results') {
        return Promise.reject(new Error(`unexpected url ${url}`));
      }

      const page = config.params?.page;
      if (page === 1) {
        return Promise.resolve({
          data: { total: 500, results: [{ symbol: 'S1' }] },
        });
      }
      if (page === 2) {
        return page2.promise;
      }
      if (page === 3) {
        return page3.promise;
      }
      if (page === 4) {
        return page4.promise;
      }
      if (page === 5) {
        return Promise.resolve({
          data: { total: 500, results: [{ symbol: 'S5' }] },
        });
      }
      return Promise.reject(new Error(`unexpected page ${page}`));
    });

    const pending = getAllFilteredSymbols('scan-5');
    await vi.waitFor(() => {
      const earlyPages = apiClient.get.mock.calls
        .filter(([url]) => url === '/v1/scans/scan-5/results')
        .map(([, cfg]) => cfg?.params?.page);
      expect(earlyPages).toEqual([1, 2, 3, 4]);
      expect(earlyPages).not.toContain(5);
    });

    page2.resolve({ data: { total: 500, results: [{ symbol: 'S2' }] } });
    page3.resolve({ data: { total: 500, results: [{ symbol: 'S3' }] } });
    page4.resolve({ data: { total: 500, results: [{ symbol: 'S4' }] } });

    const symbols = await pending;
    expect(symbols).toEqual(['S1', 'S2', 'S3', 'S4', 'S5']);

    const allPages = apiClient.get.mock.calls
      .filter(([url]) => url === '/v1/scans/scan-5/results')
      .map(([, cfg]) => cfg?.params?.page);
    expect(allPages).toEqual([1, 2, 3, 4, 5]);
  });
});
