import {
  legacyFiltersToExpression,
} from './legacyFilterExpression';
import {
  expressionToQuickFilters,
  patchExpressionQuickFilter,
} from './quickFilterExpression';
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

function updateQuickFilter(state, action, commit) {
  const filters = {
    ...selectQuickFilters(state),
    [action.key]: action.value,
  };
  const draftExpression = patchExpressionQuickFilter(
    state.draftExpression,
    action.key,
    action.value,
    filters,
  );
  if (
    stableExpressionKey(draftExpression)
    === stableExpressionKey(state.draftExpression)
  ) return state;
  return {
    ...state,
    draftExpression,
    committedExpression: commit ? draftExpression : state.committedExpression,
  };
}

export function filterStateReducer(state, action) {
  if (action.type === 'edit-quick-filter') {
    return updateQuickFilter(state, action, false);
  }
  if (action.type === 'apply-quick-filter') {
    return updateQuickFilter(state, action, true);
  }
  if (action.type === 'commit-quick-filters') {
    return commitQuickFilters(state, action.expressionKey);
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
  return expressionToQuickFilters(state.draftExpression, state.defaultFilters);
}
