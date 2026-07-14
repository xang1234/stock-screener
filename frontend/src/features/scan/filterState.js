import {
  expressionToLegacyFilters,
  legacyFiltersToExpression,
  patchExpressionQuickFilters,
} from './legacyFilterExpression';
import { stableExpressionKey } from './filterExpressionModel';

const clone = (value) => structuredClone(value);

export function createFilterState({ defaultFilters, expression = null }) {
  const defaults = clone(defaultFilters);
  const canonicalExpression = expression ?? legacyFiltersToExpression(defaults);
  return {
    defaultFilters: defaults,
    draftExpression: canonicalExpression,
    committedExpression: canonicalExpression,
  };
}

function applyExpression(state, expression) {
  return {
    ...state,
    draftExpression: expression,
    committedExpression: expression,
  };
}

function commitQuickFilters(state, expressionKey) {
  if (
    expressionKey !== stableExpressionKey(state.draftExpression)
    || expressionKey === stableExpressionKey(state.committedExpression)
  ) return state;
  return {
    ...state,
    committedExpression: state.draftExpression,
  };
}

export function filterStateReducer(state, action) {
  if (action.type === 'edit-quick-filters') {
    const draftExpression = patchExpressionQuickFilters(
      state.draftExpression,
      action.filters,
    );
    if (
      stableExpressionKey(draftExpression)
      === stableExpressionKey(state.draftExpression)
    ) return state;
    return { ...state, draftExpression };
  }
  if (action.type === 'commit-quick-filters') {
    return commitQuickFilters(state, action.expressionKey);
  }
  if (action.type === 'apply-quick-filters') {
    const expression = patchExpressionQuickFilters(
      state.draftExpression,
      action.filters,
    );
    return {
      ...state,
      draftExpression: expression,
      committedExpression: expression,
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

export function selectQuickFilters(state) {
  return expressionToLegacyFilters(state.draftExpression, state.defaultFilters);
}
