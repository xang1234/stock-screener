export function createEmptyExpression(requiredConditions = []) {
  return {
    expression_version: 1,
    required: {
      id: 'required',
      name: 'Always require',
      match: 'all',
      enabled: true,
      conditions: requiredConditions,
    },
    group_join: 'any',
    groups: [],
  };
}

export function canonicalizeExpression(expression) {
  const defaults = createEmptyExpression();
  const source = expression && typeof expression === 'object' && !Array.isArray(expression)
    ? expression
    : defaults;
  const normalized = structuredClone(source);
  const normalizeCondition = (condition) => {
    const safeCondition = condition && typeof condition === 'object' && !Array.isArray(condition)
      ? condition
      : {};
    return safeCondition.kind === 'categorical'
      ? {
        ...safeCondition,
        values: [...new Set(Array.isArray(safeCondition.values) ? safeCondition.values : [])].sort(),
      }
      : { ...safeCondition };
  };
  const normalizeGroup = (group, fallback = {}) => {
    const safeGroup = group && typeof group === 'object' && !Array.isArray(group)
      ? group
      : fallback;
    return {
      id: safeGroup.id,
      name: String(safeGroup.name || '').trim(),
      match: safeGroup.match === 'any' ? 'any' : 'all',
      enabled: safeGroup.enabled !== false,
      conditions: (Array.isArray(safeGroup.conditions) ? safeGroup.conditions : [])
        .map(normalizeCondition),
    };
  };
  return {
    expression_version: 1,
    required: normalizeGroup(normalized.required, defaults.required),
    group_join: normalized.group_join === 'all' ? 'all' : 'any',
    groups: (Array.isArray(normalized.groups) ? normalized.groups : []).map(
      (group) => normalizeGroup(group),
    ),
  };
}

export function stableExpressionKey(expression) {
  return JSON.stringify(canonicalizeExpression(expression));
}

export function buildScanQueryRequest(expression, {
  page = null,
  perPage = null,
  sortBy = 'composite_score',
  sortOrder = 'desc',
  includeSparklines = true,
  detailLevel = 'table',
} = {}) {
  return {
    ...canonicalizeExpression(expression),
    sort: { field: sortBy, order: sortOrder },
    page: page == null ? null : { number: page, size: perPage ?? 50 },
    options: { detail_level: detailLevel, include_sparklines: includeSparklines },
    passes_only: false,
  };
}
