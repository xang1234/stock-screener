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
  const normalized = structuredClone(expression ?? createEmptyExpression());
  const normalizeGroup = (group) => ({
    id: group.id,
    name: String(group.name || '').trim(),
    match: group.match === 'any' ? 'any' : 'all',
    enabled: group.enabled !== false,
    conditions: (group.conditions || []).map((condition) => (
      condition.kind === 'categorical'
        ? { ...condition, values: [...new Set(condition.values || [])].sort() }
        : { ...condition }
    )),
  });
  return {
    expression_version: 1,
    required: normalizeGroup(normalized.required),
    group_join: normalized.group_join === 'all' ? 'all' : 'any',
    groups: (normalized.groups || []).map(normalizeGroup),
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
