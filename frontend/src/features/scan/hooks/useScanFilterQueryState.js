import { useCallback, useReducer } from 'react';

import {
  canonicalizeExpression,
  legacyFiltersToExpression,
} from '../filterExpression';

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

export function createScanFilterQueryState(expression) {
  const requested = createScanFilterQuery(expression);
  return {
    requested,
    requestedKey: stableScanFilterQueryKey(requested),
    appliedSnapshot: null,
  };
}

function updateRequestedQuery(state, updates) {
  const requested = { ...state.requested, ...updates };
  const requestedKey = stableScanFilterQueryKey(requested);
  return requestedKey === state.requestedKey
    ? state
    : { ...state, requested, requestedKey };
}

export function scanFilterQueryReducer(state, action) {
  if (action.type === 'request-expression') {
    return updateRequestedQuery(state, { expression: action.expression, page: 1 });
  }

  if (action.type === 'request-quick-filters') {
    return updateRequestedQuery(state, {
      expression: legacyFiltersToExpression(action.filters, state.requested.expression),
      page: 1,
    });
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
    return updateRequestedQuery(state, { ...action.query, page: action.query.page ?? 1 });
  }

  if (action.type === 'request-succeeded') {
    if (action.requestKey !== state.requestedKey) return state;
    const snapshot = {
      request: state.requested,
      requestKey: state.requestedKey,
      scanId: action.scanId,
      data: action.data,
    };
    if (
      snapshot.requestKey === state.appliedSnapshot?.requestKey
      && snapshot.scanId === state.appliedSnapshot.scanId
      && snapshot.data === state.appliedSnapshot.data
    ) return state;
    return { ...state, appliedSnapshot: snapshot };
  }

  return state;
}

export function useScanFilterQueryState(initialExpression) {
  const [state, dispatch] = useReducer(
    scanFilterQueryReducer,
    initialExpression,
    createScanFilterQueryState,
  );

  const requestExpression = useCallback((expression) => {
    dispatch({ type: 'request-expression', expression });
  }, []);
  const requestQuickFilters = useCallback((filters) => {
    dispatch({ type: 'request-quick-filters', filters });
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
  const markRequestSucceeded = useCallback((success) => {
    dispatch({ type: 'request-succeeded', ...success });
  }, []);
  return {
    ...state,
    requestExpression,
    requestQuickFilters,
    requestPage,
    requestPerPage,
    requestSort,
    requestQuery,
    markRequestSucceeded,
  };
}
