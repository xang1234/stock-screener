import { createContext, useCallback, useContext, useMemo, useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';

export const STATIC_MARKET_STORAGE_KEY = 'static-site:selected-market';
export const STATIC_DEFAULT_MARKET = 'US';

const StaticMarketContext = createContext({
  selectedMarket: STATIC_DEFAULT_MARKET,
  setSelectedMarket: () => {},
});

// An unavailable market (listed by the manifest but with no data in this
// publish) stays selected so the site can say so (#504), instead of silently
// showing another market.
const normalizeMarket = (value, supportedMarkets, defaultMarket, unavailableMarkets = []) => {
  const normalized = String(value || defaultMarket || STATIC_DEFAULT_MARKET).trim().toUpperCase();
  if (unavailableMarkets.includes(normalized)) {
    return normalized;
  }
  if (Array.isArray(supportedMarkets) && supportedMarkets.length > 0) {
    return supportedMarkets.includes(normalized) ? normalized : (supportedMarkets[0] || defaultMarket || STATIC_DEFAULT_MARKET);
  }
  return normalized || defaultMarket || STATIC_DEFAULT_MARKET;
};

const EMPTY_MARKETS = [];

export function StaticMarketProvider({
  children,
  supportedMarkets = EMPTY_MARKETS,
  unavailableMarkets = EMPTY_MARKETS,
  defaultMarket = STATIC_DEFAULT_MARKET,
}) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedMarket, setSelectedMarketState] = useState(() => {
    const fromQuery = searchParams.get('market');
    const fromStorage = typeof window !== 'undefined'
      ? window.localStorage.getItem(STATIC_MARKET_STORAGE_KEY)
      : null;
    return normalizeMarket(fromQuery || fromStorage, supportedMarkets, defaultMarket, unavailableMarkets);
  });

  useEffect(() => {
    const fromQuery = searchParams.get('market');
    const fromStorage = typeof window !== 'undefined'
      ? window.localStorage.getItem(STATIC_MARKET_STORAGE_KEY)
      : null;
    const nextMarket = normalizeMarket(fromQuery || fromStorage, supportedMarkets, defaultMarket, unavailableMarkets);
    if (typeof window !== 'undefined' && fromQuery) {
      window.localStorage.setItem(STATIC_MARKET_STORAGE_KEY, nextMarket);
    }
    if (nextMarket !== selectedMarket) {
      setSelectedMarketState(nextMarket);
    }
  }, [defaultMarket, searchParams, selectedMarket, supportedMarkets, unavailableMarkets]);

  const setSelectedMarket = useCallback((market) => {
    const normalized = normalizeMarket(market, supportedMarkets, defaultMarket, unavailableMarkets);
    setSelectedMarketState(normalized);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STATIC_MARKET_STORAGE_KEY, normalized);
    }

    const nextParams = new URLSearchParams(searchParams);
    if (normalized === (defaultMarket || STATIC_DEFAULT_MARKET)) {
      nextParams.delete('market');
    } else {
      nextParams.set('market', normalized);
    }
    setSearchParams(nextParams, { replace: true });
  }, [defaultMarket, searchParams, setSearchParams, supportedMarkets, unavailableMarkets]);

  const value = useMemo(() => ({
    selectedMarket,
    setSelectedMarket,
  }), [selectedMarket, setSelectedMarket]);

  return (
    <StaticMarketContext.Provider value={value}>
      {children}
    </StaticMarketContext.Provider>
  );
}

export const useStaticMarket = () => useContext(StaticMarketContext);
