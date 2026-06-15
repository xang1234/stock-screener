import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { StrategyProfileProvider, useStrategyProfile } from './StrategyProfileContext';

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

function StrategyProfileProbe() {
  const {
    activeProfile,
    activeProfileDetail,
    isLoadingProfiles,
    profiles,
    requestProfileData,
  } = useStrategyProfile();

  return (
    <>
      <div data-testid="active-profile">{activeProfile}</div>
      <div data-testid="profile-count">{profiles.length}</div>
      <div data-testid="profile-label">{activeProfileDetail?.label || ''}</div>
      <div data-testid="loading-profiles">{String(isLoadingProfiles)}</div>
      <button type="button" onClick={requestProfileData}>Load profiles</button>
    </>
  );
}

function renderStrategyProfileProvider(queryClient = createTestQueryClient()) {
  const rendered = render(
    <QueryClientProvider client={queryClient}>
      <StrategyProfileProvider>
        <StrategyProfileProbe />
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

  it('defers profile API requests until profile data is requested', async () => {
    renderStrategyProfileProvider();

    expect(screen.getByTestId('active-profile')).toHaveTextContent('default');
    expect(screen.getByTestId('profile-count')).toHaveTextContent('0');
    expect(getStrategyProfiles).not.toHaveBeenCalled();
    expect(getStrategyProfile).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Load profiles' }));

    await waitFor(() => {
      expect(screen.getByTestId('profile-label')).toHaveTextContent('Default');
    });
    expect(getStrategyProfiles).toHaveBeenCalledTimes(1);
    expect(getStrategyProfile).toHaveBeenCalledWith('default');
  });
});
