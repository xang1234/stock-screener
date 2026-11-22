import apiClient from './client';
import {
  getAllFilteredSymbols,
  getScanBootstrap,
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

describe('scan api helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('requests the scan bootstrap by scan id, else by market, else unscoped', async () => {
    apiClient.get.mockResolvedValue({ data: {} });

    await getScanBootstrap('scan-1', 'HK');
    await getScanBootstrap(null, 'HK');
    await getScanBootstrap();

    expect(apiClient.get.mock.calls).toEqual([
      ['/v1/scans/bootstrap', { params: { scan_id: 'scan-1' } }],
      ['/v1/scans/bootstrap', { params: { market: 'HK' } }],
      ['/v1/scans/bootstrap', { params: undefined }],
    ]);
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

  it('propagates a /symbols failure instead of crawling /results', async () => {
    apiClient.get.mockRejectedValueOnce(new Error('symbols endpoint failed'));

    await expect(getAllFilteredSymbols('scan-5')).rejects.toThrow('symbols endpoint failed');
    expect(apiClient.get).toHaveBeenCalledTimes(1);
  });
});
