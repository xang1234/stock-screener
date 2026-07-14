import { useCallback, useMemo, useReducer } from 'react';

import {
  canonicalizeExpression,
  stableExpressionKey,
} from '../filterExpressionModel';
import {
  createFilterState,
  filterStateReducer,
  selectQuickFilters,
} from '../filterState';

const DEFAULT_QUERY = Object.freeze({
  page: 1,
  perPage: 50,
  sortBy: 'composite_score',
  sortOrder: 'desc',
});

export function createScanFilterQuery(expression, overrides = {}) {
  return {
    expression,
    ...DEFAULT_QUERY,
    ...overrides,
  };
}

export function stableScanFilterQueryKey(query) {
  return JSON.stringify({
    expression: canonicalizeExpression(query.expression),
    page: query.page,
    perPage: query.perPage,
    sortBy: query.sortBy,
    sortOrder: query.sortOrder,
  });
}

export function createScanFilterQueryState({ defaultFilters, expression = null }) {
  const filterState = createFilterState({ defaultFilters, expression });
  const requested = createScanFilterQuery(filterState.committedExpression);
  return {
    filterState,
    requested,
    requestedKey: stableScanFilterQueryKey(requested),
  };
}

function updateRequestedQuery(state, updates) {
  const requested = { ...state.requested, ...updates };
  const requestedKey = stableScanFilterQueryKey(requested);
  return requestedKey === state.requestedKey
    ? state
    : { ...state, requested, requestedKey };
}

function updateFilterAndRequestedQuery(state, action) {
  const filterState = filterStateReducer(state.filterState, action);
  if (filterState === state.filterState) return state;
  const nextState = { ...state, filterState };
  if (
    filterState.committedExpression
    === state.filterState.committedExpression
  ) return nextState;
  return updateRequestedQuery(nextState, {
    expression: filterState.committedExpression,
    page: 1,
  });
}

export function scanFilterQueryReducer(state, action) {
  if (action.type === 'request-expression') {
    return updateFilterAndRequestedQuery(state, {
      type: 'apply-expression',
      expression: action.expression,
    });
  }

  if (action.type === 'edit-quick-filters' || action.type === 'commit-quick-filters') {
    return updateFilterAndRequestedQuery(state, action);
  }

  if (action.type === 'reset-filters') {
    return updateFilterAndRequestedQuery(state, action);
  }

  if (action.type === 'request-page') {
    return updateRequestedQuery(state, { page: action.page });
  }

  if (action.type === 'request-per-page') {
    return updateRequestedQuery(state, { page: 1, perPage: action.perPage });
  }

  if (action.type === 'request-sort') {
    return updateRequestedQuery(state, {
      page: 1,
      sortBy: action.sortBy,
      sortOrder: action.sortOrder,
    });
  }

  if (action.type === 'request-query') {
    const expression = action.query.expression;
    const withFilters = expression
      ? updateFilterAndRequestedQuery(state, { type: 'apply-expression', expression })
      : state;
    return updateRequestedQuery(withFilters, {
      ...action.query,
      expression: withFilters.filterState.committedExpression,
      page: action.query.page ?? 1,
    });
  }

  return state;
}

export function useScanFilterQueryState(initialState) {
  const [state, dispatch] = useReducer(
    scanFilterQueryReducer,
    initialState,
    createScanFilterQueryState,
  );

  const requestExpression = useCallback((expression) => {
    dispatch({ type: 'request-expression', expression });
  }, []);
  const editQuickFilters = useCallback((filters) => {
    dispatch({ type: 'edit-quick-filters', filters });
  }, []);
  const commitQuickFilters = useCallback((expressionKey) => {
    dispatch({ type: 'commit-quick-filters', expressionKey });
  }, []);
  const resetFilters = useCallback((defaultFilters) => {
    dispatch({ type: 'reset-filters', defaultFilters });
  }, []);
  const requestPage = useCallback((page) => {
    dispatch({ type: 'request-page', page });
  }, []);
  const requestPerPage = useCallback((perPage) => {
    dispatch({ type: 'request-per-page', perPage });
  }, []);
  const requestSort = useCallback((sortBy, sortOrder) => {
    dispatch({ type: 'request-sort', sortBy, sortOrder });
  }, []);
  const requestQuery = useCallback((query) => {
    dispatch({ type: 'request-query', query });
  }, []);
  const filters = useMemo(
    () => selectQuickFilters(state.filterState),
    [state.filterState],
  );
  const draftExpression = state.filterState.draftExpression;
  const draftExpressionKey = stableExpressionKey(draftExpression);
  const committedExpressionKey = stableExpressionKey(
    state.filterState.committedExpression,
  );
  return {
    ...state,
    filters,
    draftExpression,
    draftExpressionKey,
    committedExpressionKey,
    requestExpression,
    editQuickFilters,
    commitQuickFilters,
    resetFilters,
    requestPage,
    requestPerPage,
    requestSort,
    requestQuery,
  };
}
