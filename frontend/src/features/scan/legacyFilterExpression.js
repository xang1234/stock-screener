import { createEmptyExpression } from './filterExpressionModel';
import {
  quickFiltersToConditions,
  resolveIpoCutoff,
} from './quickFilterExpression';
import { STATIC_ONLY_LEGACY_FILTER_KEYS } from './scanFilterFields';

export { resolveIpoCutoff };

const STATIC_ONLY_FILTER_KEYS = new Set(STATIC_ONLY_LEGACY_FILTER_KEYS);

export function legacyFiltersToExpression(filters = {}, now = new Date()) {
  return createEmptyExpression(quickFiltersToConditions(filters, now));
}

export function legacyLiveFiltersToExpression(filters = {}, now = new Date()) {
  const liveFilters = Object.fromEntries(
    Object.entries(filters).filter(([key]) => !STATIC_ONLY_FILTER_KEYS.has(key)),
  );
  return legacyFiltersToExpression(liveFilters, now);
}
