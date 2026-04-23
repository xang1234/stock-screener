import { formatLargeNumber, getCurrencyPrefix } from './formatUtils';

/**
 * Resolve the best market-cap value to display from a primary row plus a
 * fallback fundamentals payload.
 *
 * By default this prefers the native/local market-cap value so per-market
 * tables stay aligned with their local-currency filters. Cross-market popup
 * contexts can opt into USD-first resolution via preferUsd.
 */
export const resolveMarketCapDisplay = (
  primary = null,
  fallback = null,
  { preferUsd = false } = {},
) => {
  const marketCapUsd = primary?.market_cap_usd ?? fallback?.market_cap_usd ?? null;
  const marketCapLocal = primary?.market_cap ?? fallback?.market_cap ?? null;
  const currency = primary?.currency ?? fallback?.currency ?? null;

  if (preferUsd && marketCapUsd != null) {
    return {
      source: 'market_cap_usd',
      label: 'Mkt Cap (USD)',
      value: marketCapUsd,
      formattedValue: formatLargeNumber(marketCapUsd, '$'),
    };
  }

  if (marketCapLocal != null) {
    return {
      source: 'market_cap',
      label: 'Mkt Cap (local)',
      value: marketCapLocal,
      formattedValue: formatLargeNumber(marketCapLocal, getCurrencyPrefix(currency)),
    };
  }

  if (marketCapUsd != null) {
    return {
      source: 'market_cap_usd',
      label: 'Mkt Cap (USD)',
      value: marketCapUsd,
      formattedValue: formatLargeNumber(marketCapUsd, '$'),
    };
  }

  return {
    source: null,
    label: 'Mkt Cap',
    value: null,
    formattedValue: '-',
  };
};
