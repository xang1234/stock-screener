import { createContext, useContext, useMemo, useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';

export const STATIC_MARKET_STORAGE_KEY = 'static-site:selected-market';
export const STATIC_DEFAULT_MARKET = 'US';

const StaticMarketContext = createContext({
  selectedMarket: STATIC_DEFAULT_MARKET,
  setSelectedMarket: () => {},
});

const normalizeMarket = (value, supportedMarkets, defaultMarket) => {
  const normalized = String(value || defaultMarket || STATIC_DEFAULT_MARKET).trim().toUpperCase();
  if (Array.isArray(supportedMarkets) && supportedMarkets.length > 0) {
    return supportedMarkets.includes(normalized) ? normalized : (supportedMarkets[0] || defaultMarket || STATIC_DEFAULT_MARKET);
  }
  return normalized || defaultMarket || STATIC_DEFAULT_MARKET;
};

export function StaticMarketProvider({
  children,
  supportedMarkets = [],
  defaultMarket = STATIC_DEFAULT_MARKET,
}) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedMarket, setSelectedMarketState] = useState(() => {
    const fromQuery = searchParams.get('market');
    const fromStorage = typeof window !== 'undefined'
      ? window.localStorage.getItem(STATIC_MARKET_STORAGE_KEY)
      : null;
    return normalizeMarket(fromQuery || fromStorage, supportedMarkets, defaultMarket);
  });

  useEffect(() => {
    const fromQuery = searchParams.get('market');
    const fromStorage = typeof window !== 'undefined'
      ? window.localStorage.getItem(STATIC_MARKET_STORAGE_KEY)
      : null;
    const nextMarket = normalizeMarket(fromQuery || fromStorage, supportedMarkets, defaultMarket);
    if (nextMarket !== selectedMarket) {
      setSelectedMarketState(nextMarket);
    }
  }, [defaultMarket, searchParams, selectedMarket, supportedMarkets]);

  const setSelectedMarket = (market) => {
    const normalized = normalizeMarket(market, supportedMarkets, defaultMarket);
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
  };

  const value = useMemo(() => ({
    selectedMarket,
    setSelectedMarket,
  }), [selectedMarket]);

  return (
    <StaticMarketContext.Provider value={value}>
      {children}
    </StaticMarketContext.Provider>
  );
}

export const useStaticMarket = () => useContext(StaticMarketContext);
