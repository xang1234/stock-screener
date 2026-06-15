/* eslint-disable react-refresh/only-export-components */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getStrategyProfile, getStrategyProfiles } from '../api/strategyProfiles';

const STORAGE_KEY = 'stockscanner.activeStrategyProfile';
const DEFAULT_PROFILE = 'default';

const StrategyProfileContext = createContext(null);

function getStoredProfile() {
  if (typeof window === 'undefined') {
    return DEFAULT_PROFILE;
  }
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored || DEFAULT_PROFILE;
}

export function StrategyProfileProvider({ children }) {
  const [activeProfile, setActiveProfileState] = useState(getStoredProfile);
  const setActiveProfile = useCallback((nextProfile) => {
    setActiveProfileState(nextProfile || DEFAULT_PROFILE);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, activeProfile);
  }, [activeProfile]);

  const value = useMemo(() => ({
    activeProfile,
    setActiveProfile,
  }), [
    activeProfile,
    setActiveProfile,
  ]);

  return (
    <StrategyProfileContext.Provider value={value}>
      {children}
    </StrategyProfileContext.Provider>
  );
}

export function useStrategyProfile() {
  const context = useContext(StrategyProfileContext);
  if (!context) {
    throw new Error('useStrategyProfile must be used within a StrategyProfileProvider');
  }
  return context;
}

export function useStrategyProfileData() {
  const { activeProfile, setActiveProfile } = useStrategyProfile();

  const profilesQuery = useQuery({
    queryKey: ['strategyProfiles'],
    queryFn: getStrategyProfiles,
    staleTime: 5 * 60 * 1000,
  });

  const profileDetailQuery = useQuery({
    queryKey: ['strategyProfile', activeProfile],
    queryFn: () => getStrategyProfile(activeProfile),
    enabled: Boolean(activeProfile),
    staleTime: 5 * 60 * 1000,
  });

  const profiles = useMemo(() => profilesQuery.data?.profiles || [], [profilesQuery.data]);
  const activeProfileIsListed = useMemo(
    () => profiles.some((profile) => profile.profile === activeProfile),
    [activeProfile, profiles]
  );

  useEffect(() => {
    if (!profiles.length) {
      return;
    }
    if (!activeProfileIsListed) {
      setActiveProfile(DEFAULT_PROFILE);
    }
  }, [activeProfileIsListed, profiles.length, setActiveProfile]);

  const activeProfileDetail = useMemo(() => {
    if (profileDetailQuery.data) {
      return profileDetailQuery.data;
    }
    return profiles.find((profile) => profile.profile === activeProfile) || null;
  }, [activeProfile, profileDetailQuery.data, profiles]);
  const effectiveProfile = activeProfileDetail?.profile
    || (activeProfileIsListed ? activeProfile : null);
  const hasProfileLoadError = profilesQuery.isError && profileDetailQuery.isError;
  const requestProfile = effectiveProfile ?? (hasProfileLoadError ? activeProfile : null);

  return useMemo(() => ({
    activeProfile,
    effectiveProfile,
    requestProfile,
    setActiveProfile,
    profiles,
    activeProfileDetail,
    hasProfileLoadError,
    isLoadingProfiles: profilesQuery.isLoading || profileDetailQuery.isLoading,
  }), [
    activeProfile,
    activeProfileDetail,
    effectiveProfile,
    hasProfileLoadError,
    profileDetailQuery.isLoading,
    profiles,
    profilesQuery.isLoading,
    requestProfile,
    setActiveProfile,
  ]);
}
