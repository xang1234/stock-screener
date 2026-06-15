import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  StrategyProfileProvider,
  useStrategyProfile,
  useStrategyProfileData,
} from './StrategyProfileContext';

const { getStrategyProfile, getStrategyProfiles } = vi.hoisted(() => ({
  getStrategyProfile: vi.fn(),
  getStrategyProfiles: vi.fn(),
}));

vi.mock('../api/strategyProfiles', () => ({
  getStrategyProfile: (...args) => getStrategyProfile(...args),
  getStrategyProfiles: (...args) => getStrategyProfiles(...args),
}));

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: Infinity,
      },
    },
  });
}

function StrategyProfileSelectionProbe() {
  const { activeProfile } = useStrategyProfile();

  return (
    <div data-testid="active-profile">{activeProfile}</div>
  );
}

function StrategyProfileDataProbe() {
  const {
    activeProfileDetail,
    effectiveProfile,
    hasProfileLoadError,
    isLoadingProfiles,
    profiles,
    requestProfile,
  } = useStrategyProfileData();

  return (
    <>
      <div data-testid="profile-count">{profiles.length}</div>
      <div data-testid="effective-profile">{effectiveProfile || ''}</div>
      <div data-testid="request-profile">{requestProfile || ''}</div>
      <div data-testid="profile-load-error">{String(hasProfileLoadError)}</div>
      <div data-testid="profile-label">{activeProfileDetail?.label || ''}</div>
      <div data-testid="loading-profiles">{String(isLoadingProfiles)}</div>
    </>
  );
}

function renderStrategyProfileProvider({
  queryClient = createTestQueryClient(),
  showData = false,
} = {}) {
  const rendered = render(
    <QueryClientProvider client={queryClient}>
      <StrategyProfileProvider>
        <StrategyProfileSelectionProbe />
        {showData && <StrategyProfileDataProbe />}
      </StrategyProfileProvider>
    </QueryClientProvider>
  );

  return {
    queryClient,
    ...rendered,
  };
}

describe('StrategyProfileProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    getStrategyProfiles.mockResolvedValue({
      profiles: [{ profile: 'default', label: 'Default' }],
    });
    getStrategyProfile.mockResolvedValue({
      profile: 'default',
      label: 'Default',
    });
  });

  it('defers profile API requests until profile data is consumed', async () => {
    const rendered = renderStrategyProfileProvider();

    expect(screen.getByTestId('active-profile')).toHaveTextContent('default');
    expect(getStrategyProfiles).not.toHaveBeenCalled();
    expect(getStrategyProfile).not.toHaveBeenCalled();

    rendered.rerender(
      <QueryClientProvider client={rendered.queryClient}>
        <StrategyProfileProvider>
          <StrategyProfileSelectionProbe />
          <StrategyProfileDataProbe />
        </StrategyProfileProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('profile-label')).toHaveTextContent('Default');
    });
    expect(screen.getByTestId('effective-profile')).toHaveTextContent('default');
    expect(screen.getByTestId('request-profile')).toHaveTextContent('default');
    expect(getStrategyProfiles).toHaveBeenCalledTimes(1);
    expect(getStrategyProfile).toHaveBeenCalledWith('default');
  });

  it('normalizes an invalid stored profile to the default effective profile', async () => {
    window.localStorage.setItem('stockscanner.activeStrategyProfile', 'stale-profile');

    renderStrategyProfileProvider({ showData: true });

    expect(screen.getByTestId('active-profile')).toHaveTextContent('stale-profile');
    await waitFor(() => {
      expect(screen.getByTestId('effective-profile')).toHaveTextContent('default');
    });
    await waitFor(() => {
      expect(screen.getByTestId('active-profile')).toHaveTextContent('default');
    });
    expect(getStrategyProfile).toHaveBeenCalledWith('stale-profile');
  });

  it('reports profile metadata outage only when list and detail requests fail', async () => {
    getStrategyProfiles.mockRejectedValue(new Error('profiles unavailable'));
    getStrategyProfile.mockRejectedValue(new Error('profile unavailable'));

    renderStrategyProfileProvider({ showData: true });

    await waitFor(() => {
      expect(screen.getByTestId('profile-load-error')).toHaveTextContent('true');
    });
    expect(screen.getByTestId('effective-profile')).toHaveTextContent('');
    expect(screen.getByTestId('request-profile')).toHaveTextContent('default');
  });
});
