import {
  evaluateExpression,
  legacyFiltersToExpression,
} from '../features/scan/filterExpression';

const RATING_SORT_ORDER = {
  'Strong Buy': 5,
  Buy: 4,
  Watch: 3,
  Pass: 2,
  Error: 1,
  'Insufficient Data': 0,
};

const compareValues = (left, right) => {
  if (left == null && right == null) return 0;
  if (left == null) return 1;
  if (right == null) return -1;
  if (typeof left === 'string' || typeof right === 'string') {
    return String(left).localeCompare(String(right));
  }
  return Number(left) - Number(right);
};

const getSortValue = (row, sortBy) => {
  if (sortBy === 'rating') {
    return RATING_SORT_ORDER[row.rating] ?? 0;
  }
  return row?.[sortBy];
};

export const filterStaticScanRows = (rows, filters) => {
  const expression = legacyFiltersToExpression(filters);
  return rows.filter((row) => evaluateExpression(row, expression));
};

export const sortStaticScanRows = (
  rows,
  sortBy,
  sortOrder = 'desc',
) => {
  const direction = sortOrder === 'asc' ? 1 : -1;
  return [...rows].sort((left, right) => {
    const leftValue = getSortValue(left, sortBy);
    const rightValue = getSortValue(right, sortBy);
    if (sortBy === 'composite_score' && sortOrder === 'desc') {
      if (leftValue == null && rightValue != null) {
        return 1;
      }
      if (leftValue != null && rightValue == null) {
        return -1;
      }
    }
    const comparison = compareValues(leftValue, rightValue);
    if (comparison !== 0) {
      return comparison * direction;
    }
    return compareValues(left.symbol, right.symbol);
  });
};

export const paginateStaticScanRows = (rows, page, perPage) => {
  const offset = (page - 1) * perPage;
  return rows.slice(offset, offset + perPage);
};
