import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { RuntimeProvider, useRuntime } from './RuntimeContext';

const {
  getAppCapabilities,
  getDesktopSetupStatus,
  getDesktopUpdateStatus,
  runDesktopUpdateNow,
  startDesktopSetup,
  loginServer,
  logoutServer,
  setUnauthorizedResponseHandler,
} = vi.hoisted(() => ({
  getAppCapabilities: vi.fn(),
  getDesktopSetupStatus: vi.fn(),
  getDesktopUpdateStatus: vi.fn(),
  runDesktopUpdateNow: vi.fn(),
  startDesktopSetup: vi.fn(),
  loginServer: vi.fn(),
  logoutServer: vi.fn(),
  setUnauthorizedResponseHandler: vi.fn(),
}));

vi.mock('../api/client', () => ({
  setUnauthorizedResponseHandler,
}));

vi.mock('../api/appRuntime', () => ({
  getAppCapabilities: (...args) => getAppCapabilities(...args),
  getDesktopSetupStatus: (...args) => getDesktopSetupStatus(...args),
  getDesktopUpdateStatus: (...args) => getDesktopUpdateStatus(...args),
  runDesktopUpdateNow: (...args) => runDesktopUpdateNow(...args),
  startDesktopSetup: (...args) => startDesktopSetup(...args),
}));

vi.mock('../api/auth', () => ({
  loginServer: (...args) => loginServer(...args),
  logoutServer: (...args) => logoutServer(...args),
}));

function RuntimeProbe() {
  const { auth, desktopMode, runtimeReady } = useRuntime();

  return (
    <>
      <div data-testid="runtime-ready">{String(runtimeReady)}</div>
      <div data-testid="desktop-mode">{String(desktopMode)}</div>
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
    getDesktopSetupStatus.mockReset();
    getDesktopUpdateStatus.mockReset();
    runDesktopUpdateNow.mockReset();
    startDesktopSetup.mockReset();
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

    expect(screen.getByTestId('desktop-mode')).toHaveTextContent('false');
    expect(screen.getByTestId('auth-required')).toHaveTextContent('false');
    expect(getAppCapabilities).toHaveBeenCalledTimes(2);
  });
});
