import { getStableFilterKey } from '../../utils/filterUtils';
import {
  expressionToLegacyFilters,
  legacyFiltersToExpression,
} from './legacyFilterExpression';

const clone = (value) => structuredClone(value);

export function createFilterState({ defaultFilters, expression = null }) {
  const defaults = clone(defaultFilters);
  const canonicalExpression = expression ?? legacyFiltersToExpression(defaults);
  const filters = expressionToLegacyFilters(canonicalExpression, defaults);
  const filterKey = getStableFilterKey(filters);
  return {
    defaultFilters: defaults,
    filters,
    filterKey,
    committedFilterKey: filterKey,
    expression: canonicalExpression,
  };
}

function applyExpression(state, expression) {
  const filters = expressionToLegacyFilters(expression, state.defaultFilters);
  const filterKey = getStableFilterKey(filters);
  return {
    ...state,
    filters,
    filterKey,
    committedFilterKey: filterKey,
    expression,
  };
}

function commitQuickFilters(state, filterKey) {
  if (filterKey !== state.filterKey || filterKey === state.committedFilterKey) return state;
  return {
    ...state,
    committedFilterKey: filterKey,
    expression: legacyFiltersToExpression(state.filters, state.expression),
  };
}

export function filterStateReducer(state, action) {
  if (action.type === 'edit-quick-filters') {
    const filterKey = getStableFilterKey(action.filters);
    if (filterKey === state.filterKey) return state;
    return { ...state, filters: action.filters, filterKey };
  }
  if (action.type === 'commit-quick-filters') {
    return commitQuickFilters(state, action.filterKey);
  }
  if (action.type === 'apply-quick-filters') {
    const filters = action.filters;
    const filterKey = getStableFilterKey(filters);
    return {
      ...state,
      filters,
      filterKey,
      committedFilterKey: filterKey,
      expression: legacyFiltersToExpression(filters, state.expression),
    };
  }
  if (action.type === 'apply-expression') {
    return applyExpression(state, action.expression);
  }
  if (action.type === 'reset-filters') {
    return createFilterState({
      defaultFilters: action.defaultFilters ?? state.defaultFilters,
    });
  }
  return state;
}
