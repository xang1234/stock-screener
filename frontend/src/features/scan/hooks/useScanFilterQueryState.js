import { useCallback, useReducer } from 'react';

import { legacyFiltersToExpression, stableExpressionKey } from '../filterExpression';

export function createScanFilterQueryState(expression) {
  const key = stableExpressionKey(expression);
  return {
    requestedExpression: expression,
    requestedKey: key,
    appliedExpression: expression,
    appliedKey: key,
    appliedResultsData: null,
    appliedScanId: null,
  };
}

export function scanFilterQueryReducer(state, action) {
  if (action.type === 'request-expression') {
    const requestedKey = stableExpressionKey(action.expression);
    if (requestedKey === state.requestedKey) return state;
    return { ...state, requestedExpression: action.expression, requestedKey };
  }

  if (action.type === 'request-quick-filters') {
    const expression = legacyFiltersToExpression(action.filters, state.requestedExpression);
    const requestedKey = stableExpressionKey(expression);
    if (requestedKey === state.requestedKey) return state;
    return { ...state, requestedExpression: expression, requestedKey };
  }

  if (action.type === 'request-succeeded') {
    if (action.key !== state.requestedKey) return state;
    if (
      action.key === state.appliedKey
      && action.scanId === state.appliedScanId
      && action.data === state.appliedResultsData
    ) return state;
    return {
      ...state,
      appliedExpression: state.requestedExpression,
      appliedKey: state.requestedKey,
      appliedResultsData: action.data,
      appliedScanId: action.scanId,
    };
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
  const markRequestSucceeded = useCallback((key, data, scanId) => {
    dispatch({ type: 'request-succeeded', key, data, scanId });
  }, []);
  return {
    ...state,
    requestExpression,
    requestQuickFilters,
    markRequestSucceeded,
  };
}
