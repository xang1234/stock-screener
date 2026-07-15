import { EXPRESSION_LIMITS, fieldMeta } from './scanFilterFields';

const rangeValue = (value, isDate) => {
  if (value == null) return null;
  if (isDate) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return Number.NaN;
    const parsed = new Date(`${value}T00:00:00Z`);
    return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value
      ? value : Number.NaN;
  }
  return Number(value);
};

const CONDITION_HANDLERS = {
  range: {
    create: (field) => ({ kind: 'range', field, min: null, max: null }),
    label: (condition, label) => {
      if (condition.min != null && condition.max != null) return `${label} ${condition.min}–${condition.max}`;
      if (condition.min != null) return `${label} ≥ ${condition.min}`;
      return `${label} ≤ ${condition.max}`;
    },
    validate: (condition, label, meta) => {
      if (condition.min == null && condition.max == null) {
        return [`${label} needs a minimum or maximum.`];
      }
      const isDate = meta.value_type === 'date';
      const minimum = rangeValue(condition.min, isDate);
      const maximum = rangeValue(condition.max, isDate);
      if ((minimum != null && (isDate ? Number.isNaN(minimum) : !Number.isFinite(minimum)))
        || (maximum != null && (isDate ? Number.isNaN(maximum) : !Number.isFinite(maximum)))) {
        return [`${label} needs ${isDate ? 'valid ISO dates' : 'finite numeric values'}.`];
      }
      if (minimum != null && maximum != null && minimum > maximum) {
        return [`${label} minimum cannot exceed maximum.`];
      }
      return [];
    },
  },
  categorical: {
    create: (field) => ({ kind: 'categorical', field, values: [], mode: 'include' }),
    label: (condition, label) => (
      `${label} ${condition.mode === 'exclude' ? 'excludes' : 'is'} ${condition.values.join(', ')}`
    ),
    validate: (condition, label) => {
      if (!condition.values?.length) return [`${label} needs at least one value.`];
      if (condition.values.length > EXPRESSION_LIMITS.maxCategoricalValues) {
        return [
          `${label} allows at most ${EXPRESSION_LIMITS.maxCategoricalValues} values.`,
        ];
      }
      return [];
    },
  },
  boolean: {
    create: (field) => ({ kind: 'boolean', field, value: true }),
    label: (condition, label) => `${label} is ${condition.value ? 'Yes' : 'No'}`,
    validate: (condition, label) => (
      typeof condition.value === 'boolean' ? [] : [`${label} needs a Yes or No value.`]
    ),
  },
  text: {
    create: (field) => ({ kind: 'text', field, pattern: '' }),
    label: (condition, label) => `${label} contains “${condition.pattern}”`,
    validate: (condition, label) => (
      condition.pattern?.trim() ? [] : [`${label} needs search text.`]
    ),
  },
};

export function conditionLabel(condition) {
  const meta = fieldMeta(condition.field, condition.kind);
  return CONDITION_HANDLERS[condition.kind]?.label(condition, meta.label) ?? 'Unknown rule';
}

export function expressionSummary(expression) {
  const requiredCount = expression?.required?.conditions?.length ?? 0;
  const enabledGroups = (expression?.groups || []).filter((group) => group.enabled !== false);
  if (!enabledGroups.length) {
    return requiredCount ? `${requiredCount} always-required ${requiredCount === 1 ? 'rule' : 'rules'}` : 'No filters applied';
  }
  const join = expression.group_join === 'all' ? 'all' : 'any';
  return `${requiredCount} required · match ${join} of ${enabledGroups.length} named ${enabledGroups.length === 1 ? 'setup' : 'setups'}`;
}

export function newCondition(field = 'composite_score') {
  const meta = fieldMeta(field);
  return CONDITION_HANDLERS[meta.type]?.create(meta.field)
    ?? CONDITION_HANDLERS.range.create(meta.field);
}

function validateCondition(condition) {
  const meta = fieldMeta(condition.field, condition.kind);
  return CONDITION_HANDLERS[condition.kind]?.validate(condition, meta.label, meta)
    ?? ['Unsupported rule type.'];
}

export function validateExpression(expression) {
  const errors = [];
  const groups = expression?.groups || [];
  if (groups.length > EXPRESSION_LIMITS.maxGroups) {
    errors.push(`Use at most ${EXPRESSION_LIMITS.maxGroups} setup groups.`);
  }
  const ids = new Set();
  const requiredConditions = expression?.required?.conditions ?? [];
  let total = requiredConditions.length;
  requiredConditions.forEach((condition) => errors.push(...validateCondition(condition)));
  groups.forEach((group, index) => {
    const groupConditionCount = group.conditions?.length ?? 0;
    total += groupConditionCount;
    if (!group.name?.trim()) errors.push(`Setup ${index + 1} needs a name.`);
    if (ids.has(group.id)) errors.push('Setup group IDs must be unique.');
    ids.add(group.id);
    if (group.enabled !== false && !group.conditions?.length) {
      errors.push(`${group.name || `Setup ${index + 1}`} needs at least one rule or must be disabled.`);
    }
    if (groupConditionCount > EXPRESSION_LIMITS.maxGroupConditions) {
      errors.push(
        `${group.name || `Setup ${index + 1}`} can contain at most ${EXPRESSION_LIMITS.maxGroupConditions} rules.`,
      );
    }
    (group.conditions || []).forEach((condition) => errors.push(...validateCondition(condition)));
  });
  if (total > EXPRESSION_LIMITS.maxConditions) {
    errors.push(`Use at most ${EXPRESSION_LIMITS.maxConditions} rules in one filter.`);
  }
  return [...new Set(errors)];
}
