import { createEmptyExpression } from './filterExpressionModel';
import {
  quickFiltersToConditions,
  resolveIpoCutoff,
} from './quickFilterExpression';

export { resolveIpoCutoff };

export function legacyFiltersToExpression(filters = {}, now = new Date()) {
  return createEmptyExpression(quickFiltersToConditions(filters, now));
}
