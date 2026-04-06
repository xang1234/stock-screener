/* eslint-disable react-refresh/only-export-components */

import { createContext, useContext, useEffect, useMemo, useState } from 'react';
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

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, activeProfile);
  }, [activeProfile]);

  useEffect(() => {
    if (!profiles.length) {
      return;
    }
    if (!profiles.some((profile) => profile.profile === activeProfile)) {
      setActiveProfileState(DEFAULT_PROFILE);
    }
  }, [activeProfile, profiles]);

  const activeProfileDetail = useMemo(() => {
    if (profileDetailQuery.data) {
      return profileDetailQuery.data;
    }
    return profiles.find((profile) => profile.profile === activeProfile) || null;
  }, [activeProfile, profileDetailQuery.data, profiles]);

  const value = useMemo(() => ({
    activeProfile,
    setActiveProfile: (nextProfile) => setActiveProfileState(nextProfile || DEFAULT_PROFILE),
    profiles,
    activeProfileDetail,
    isLoadingProfiles: profilesQuery.isLoading || profileDetailQuery.isLoading,
  }), [activeProfile, activeProfileDetail, profileDetailQuery.isLoading, profiles, profilesQuery.isLoading]);

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
