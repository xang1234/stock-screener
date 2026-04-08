import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { RuntimeProvider, useRuntime } from './RuntimeContext';

const {
  getAppCapabilities,
  loginServer,
  logoutServer,
  setUnauthorizedResponseHandler,
} = vi.hoisted(() => ({
  getAppCapabilities: vi.fn(),
  loginServer: vi.fn(),
  logoutServer: vi.fn(),
  setUnauthorizedResponseHandler: vi.fn(),
}));

vi.mock('../api/client', () => ({
  setUnauthorizedResponseHandler,
}));

vi.mock('../api/appRuntime', () => ({
  getAppCapabilities: (...args) => getAppCapabilities(...args),
}));

vi.mock('../api/auth', () => ({
  loginServer: (...args) => loginServer(...args),
  logoutServer: (...args) => logoutServer(...args),
}));

function RuntimeProbe() {
  const { auth, runtimeReady } = useRuntime();

  return (
    <>
      <div data-testid="runtime-ready">{String(runtimeReady)}</div>
      <div data-testid="auth-required">{String(auth.required)}</div>
    </>
  );
}

function renderRuntime() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retryDelay: 1,
        gcTime: 0,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <RuntimeProvider>
        <RuntimeProbe />
      </RuntimeProvider>
    </QueryClientProvider>
  );
}

describe('RuntimeProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getAppCapabilities.mockReset();
    loginServer.mockReset();
    logoutServer.mockReset();
    setUnauthorizedResponseHandler.mockReset();
  });

  it('marks runtime ready after app capabilities errors out', async () => {
    getAppCapabilities.mockRejectedValue(new Error('backend unavailable'));

    renderRuntime();

    expect(screen.getByTestId('runtime-ready')).toHaveTextContent('false');

    await waitFor(() => {
      expect(screen.getByTestId('runtime-ready')).toHaveTextContent('true');
    });

    expect(screen.getByTestId('auth-required')).toHaveTextContent('false');
    expect(getAppCapabilities).toHaveBeenCalledTimes(2);
  });
});
