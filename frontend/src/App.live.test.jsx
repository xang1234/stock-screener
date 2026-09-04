import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import App from './App';

const runtimeMock = vi.hoisted(() => ({ state: null }));

vi.mock('./config/runtimeMode', () => ({ STATIC_SITE_MODE: false }));

vi.mock('./contexts/RuntimeContext', () => ({
  RuntimeProvider: ({ children }) => <>{children}</>,
  useRuntime: () => runtimeMock.state,
}));

vi.mock('./components/App/ServerLoginScreen', () => ({
  default: ({ auth }) => (
    <div>
      Server Login {auth?.configured ? 'configured' : 'missing'}
    </div>
  ),
}));

vi.mock('./components/App/BootstrapSetupScreen', () => ({
  default: ({ primaryMarket }) => <div>Bootstrap Setup {primaryMarket}</div>,
}));

vi.mock('./components/Layout/Layout', () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock('./contexts/StrategyProfileContext', () => ({
  StrategyProfileProvider: ({ children }) => <>{children}</>,
}));
vi.mock('./pages/MarketScanPage', () => ({ default: () => <div>Live Market Scan Page</div> }));
vi.mock('./pages/OptionsPage', () => ({ default: () => <div>Live Options Page</div> }));
vi.mock('./pages/OptionsSymbolPage', () => ({ default: () => <div>Live Options Symbol Page</div> }));

const baseRuntimeState = () => ({
  auth: { required: false, authenticated: true, configured: true },
  bootstrapRequired: false,
  bootstrapState: null,
  enabledMarkets: ['US'],
  features: { themes: true, chatbot: true, options_analytics: true },
  isLoggingIn: false,
  isStartingBootstrap: false,
  login: vi.fn(),
  marketCatalog: {
    markets: [{ code: 'US', capabilities: { options_analytics: true } }],
  },
  primaryMarket: 'US',
  loginError: null,
  runtimeReady: true,
  startBootstrap: vi.fn(),
  supportedMarkets: ['US'],
  bootstrapError: null,
});

describe('App live mode shell', () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.history.pushState({}, '', '/');
    runtimeMock.state = baseRuntimeState();
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    window.localStorage.clear();
    window.history.pushState({}, '', '/');
  });

  it('lazy-loads the server login screen only when auth blocks the app', async () => {
    runtimeMock.state = {
      ...baseRuntimeState(),
      auth: { required: true, authenticated: false, configured: true },
    };

    render(<App />);

    expect(await screen.findByText('Server Login configured')).toBeInTheDocument();
  });

  it('lazy-loads the bootstrap setup screen only when bootstrap is required', async () => {
    runtimeMock.state = {
      ...baseRuntimeState(),
      bootstrapRequired: true,
    };

    render(<App />);

    expect(await screen.findByText('Bootstrap Setup US')).toBeInTheDocument();
  });

  it('lazy-loads Options only when the feature and selected US capability are enabled', async () => {
    window.history.pushState({}, '', '/options');
    render(<App />);
    expect(await screen.findByText('Live Options Page')).toBeInTheDocument();

    cleanup();
    runtimeMock.state = {
      ...baseRuntimeState(),
      features: { ...baseRuntimeState().features, options_analytics: false },
    };
    render(<App />);
    expect(await screen.findByText('Live Market Scan Page')).toBeInTheDocument();
    expect(screen.queryByText('Live Options Page')).not.toBeInTheDocument();
  });
});
