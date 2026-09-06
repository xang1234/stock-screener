import { normalizeMarketCode } from '../../utils/marketCapabilities';

export const isLiveOptionsAvailable = ({ features, marketCatalog, selectedMarket, primaryMarket }) => {
  const market = normalizeMarketCode(selectedMarket || primaryMarket);
  if (!features?.options_analytics || market !== 'US') return false;
  const entries = Array.isArray(marketCatalog?.markets) ? marketCatalog.markets : [];
  const us = entries.find((entry) => normalizeMarketCode(entry?.code) === 'US');
  return us?.capabilities?.options_analytics === true;
};

export const isStaticOptionsAvailable = (marketEntry) => (
  normalizeMarketCode(marketEntry?.market) === 'US'
  && typeof marketEntry?.pages?.options?.path === 'string'
  && marketEntry.pages.options.path.length > 0
);
