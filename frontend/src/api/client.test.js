import { beforeEach, describe, expect, it, vi } from 'vitest';

const mockRequestUse = vi.fn();
const mockResponseUse = vi.fn();
const mockCreate = vi.fn();

vi.mock('axios', () => ({
  default: {
    create: mockCreate,
  },
}));

const loadRequestInterceptor = async () => {
  vi.resetModules();
  mockRequestUse.mockReset();
  mockResponseUse.mockReset();

  const mockClient = {
    interceptors: {
      request: { use: mockRequestUse },
      response: { use: mockResponseUse },
    },
  };
  mockCreate.mockReturnValue(mockClient);

  await import('./client');
  return mockRequestUse.mock.calls[0][0];
};

const loadClientModule = async () => {
  vi.resetModules();
  mockRequestUse.mockReset();
  mockResponseUse.mockReset();

  const mockClient = {
    interceptors: {
      request: { use: mockRequestUse },
      response: { use: mockResponseUse },
    },
  };
  mockCreate.mockReturnValue(mockClient);

  return import('./client');
};

describe('api client timeout policy', () => {
  beforeEach(() => {
    mockCreate.mockReset();
  });

  it('applies 5-minute timeout for theme URLs when timeout is undefined', async () => {
    const interceptor = await loadRequestInterceptor();
    const config = { url: '/v1/themes/rankings', method: 'get' };

    const result = interceptor(config);

    expect(result.timeout).toBe(300000);
  });

  it('raises small explicit timeout to 5 minutes for theme URLs', async () => {
    const interceptor = await loadRequestInterceptor();
    const config = { url: '/api/v1/themes/pipeline/run', method: 'post', timeout: 10000 };

    const result = interceptor(config);

    expect(result.timeout).toBe(300000);
  });

  it('preserves larger explicit timeout for theme URLs', async () => {
    const interceptor = await loadRequestInterceptor();
    const config = { url: '/v1/themes/pipeline/status', method: 'get', timeout: 600000 };

    const result = interceptor(config);

    expect(result.timeout).toBe(600000);
  });

  it('leaves non-theme URLs unchanged', async () => {
    const interceptor = await loadRequestInterceptor();
    const config = { url: '/v1/scans/active', method: 'get', timeout: 45000 };

    const result = interceptor(config);

    expect(result.timeout).toBe(45000);
  });
});

describe('api client auth failure handling', () => {
  it('notifies the unauthorized handler for non-auth 401 responses', async () => {
    const { setUnauthorizedResponseHandler } = await loadClientModule();
    const onUnauthorized = vi.fn();
    setUnauthorizedResponseHandler(onUnauthorized);
    const responseErrorInterceptor = mockResponseUse.mock.calls[0][1];
    const error = {
      response: { status: 401, data: {} },
      config: { url: '/v1/scans/active', headers: {} },
    };

    await expect(responseErrorInterceptor(error)).rejects.toBe(error);

    expect(onUnauthorized).toHaveBeenCalledWith(error);
  });

  it('does not notify the unauthorized handler for auth endpoint failures', async () => {
    const { setUnauthorizedResponseHandler } = await loadClientModule();
    const onUnauthorized = vi.fn();
    setUnauthorizedResponseHandler(onUnauthorized);
    const responseErrorInterceptor = mockResponseUse.mock.calls[0][1];
    const error = {
      response: { status: 401, data: {} },
      config: { url: '/v1/auth/login', headers: {} },
    };

    await expect(responseErrorInterceptor(error)).rejects.toBe(error);

    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it('does not notify the unauthorized handler for admin-key config failures', async () => {
    const { setUnauthorizedResponseHandler } = await loadClientModule();
    const onUnauthorized = vi.fn();
    setUnauthorizedResponseHandler(onUnauthorized);
    const responseErrorInterceptor = mockResponseUse.mock.calls[0][1];
    const error = {
      response: { status: 401, data: {} },
      config: { url: '/v1/config/theme-policies', headers: { 'X-Admin-Key': 'bad-key' } },
    };

    await expect(responseErrorInterceptor(error)).rejects.toBe(error);

    expect(onUnauthorized).not.toHaveBeenCalled();
  });
});
