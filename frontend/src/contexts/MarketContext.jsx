import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

import { useRuntime } from './RuntimeContext';
import { marketOptionsForCapability, normalizeMarketCode } from '../utils/marketCapabilities';

export const MARKET_STORAGE_KEY = 'server-mode:selected-market';

// Default used when no provider is mounted (tests, isolated renders):
// selectedMarket null lets consumers fall back to the runtime primary market.
const NO_PROVIDER_CONTEXT = {
  selectedMarket: null,
  setSelectedMarket: () => {},
  selectableMarkets: [],
  marketLabel: (code) => normalizeMarketCode(code),
};

const MarketContext = createContext(NO_PROVIDER_CONTEXT);

const clampMarket = (value, selectableMarkets, fallback) => {
  const normalized = normalizeMarketCode(value);
  if (normalized && selectableMarkets.includes(normalized)) {
    return normalized;
  }
  return fallback;
};

export function MarketProvider({ children }) {
  const { enabledMarkets, supportedMarkets, primaryMarket, marketCatalog } = useRuntime();
  const [searchParams, setSearchParams] = useSearchParams();

  const selectableMarkets = useMemo(() => {
    const enabled = (enabledMarkets || []).map(normalizeMarketCode).filter(Boolean);
    if (enabled.length > 0) return Array.from(new Set(enabled));
    const supported = (supportedMarkets || []).map(normalizeMarketCode).filter(Boolean);
    return supported.length > 0 ? Array.from(new Set(supported)) : ['US'];
  }, [enabledMarkets, supportedMarkets]);

  const defaultMarket = clampMarket(primaryMarket, selectableMarkets, selectableMarkets[0] || 'US');

  // null = no explicit user choice yet — follow the runtime primary market.
  // An explicit choice (URL param, stored value, or selector click) sticks.
  const [explicitMarket, setExplicitMarket] = useState(() => {
    const fromQuery = normalizeMarketCode(searchParams.get('market'));
    if (fromQuery) return fromQuery;
    if (typeof window !== 'undefined') {
      return normalizeMarketCode(window.localStorage.getItem(MARKET_STORAGE_KEY)) || null;
    }
    return null;
  });

  // The provider stays mounted across navigations, so the initializer above
  // only sees the first URL. Keep a later ?market= change (back/forward, an
  // in-app link) in sync. An absent param keeps the current selection: tab
  // navigation drops query params and must not reset the market.
  useEffect(() => {
    const fromQuery = normalizeMarketCode(searchParams.get('market'));
    if (fromQuery && fromQuery !== explicitMarket) {
      setExplicitMarket(fromQuery);
    }
  }, [searchParams, explicitMarket]);

  const selectedMarket = explicitMarket
    ? clampMarket(explicitMarket, selectableMarkets, defaultMarket)
    : defaultMarket;

  const setSelectedMarket = useCallback((market) => {
    const normalized = normalizeMarketCode(market);
    if (!normalized) return;
    setExplicitMarket(normalized);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(MARKET_STORAGE_KEY, normalized);
    }
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set('market', normalized);
    setSearchParams(nextParams, { replace: true });
  }, [searchParams, setSearchParams]);

  const marketLabel = useCallback((code) => {
    const normalized = normalizeMarketCode(code);
    const entries = Array.isArray(marketCatalog?.markets) ? marketCatalog.markets : [];
    const entry = entries.find((item) => normalizeMarketCode(item?.code) === normalized);
    return entry?.label || entry?.display_name || normalized;
  }, [marketCatalog]);

  const value = useMemo(() => ({
    selectedMarket,
    setSelectedMarket,
    selectableMarkets,
    marketLabel,
  }), [selectedMarket, setSelectedMarket, selectableMarkets, marketLabel]);

  return (
    <MarketContext.Provider value={value}>
      {children}
    </MarketContext.Provider>
  );
}

export function useMarket() {
  return useContext(MarketContext);
}

/**
 * Global market selection clamped to the markets supporting a capability.
 *
 * Pages whose data only exists for some markets (group rankings, breadth)
 * follow the header selector when possible and fall back to the first
 * capable market otherwise.
 */
export function useMarketForCapability(capability, fallbackCodes) {
  const { enabledMarkets, supportedMarkets, primaryMarket, marketCatalog } = useRuntime();
  const { selectedMarket } = useMarket();

  const availableMarkets = useMemo(() => marketOptionsForCapability({
    marketCatalog,
    capability,
    fallbackCodes,
    enabledMarkets,
    supportedMarkets,
  }), [capability, enabledMarkets, fallbackCodes, marketCatalog, supportedMarkets]);

  const market = useMemo(() => {
    const preferred = normalizeMarketCode(selectedMarket || primaryMarket || 'US');
    return availableMarkets.includes(preferred)
      ? preferred
      : (availableMarkets[0] || 'US');
  }, [availableMarkets, primaryMarket, selectedMarket]);

  return { market, availableMarkets };
}
