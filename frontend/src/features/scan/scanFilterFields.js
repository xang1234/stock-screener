import scanFilterContract from './scanFilterFields.json';

export const EXPRESSION_LIMITS = Object.freeze({
  maxGroups: scanFilterContract.expression_limits.max_groups,
  maxGroupConditions: scanFilterContract.expression_limits.max_group_conditions,
  maxConditions: scanFilterContract.expression_limits.max_conditions,
  maxGroupIdLength: scanFilterContract.expression_limits.max_group_id_length,
  maxGroupNameLength: scanFilterContract.expression_limits.max_group_name_length,
  maxTextPatternLength: scanFilterContract.expression_limits.max_text_pattern_length,
  maxCategoricalValues: scanFilterContract.expression_limits.max_categorical_values,
});

const LEGACY_FILTER_FIELDS = Object.freeze(
  scanFilterContract.fields
    .filter((item) => item.legacy_key && ['range', 'boolean'].includes(item.kind))
    .map((item) => [item.legacy_key, item.field, item.kind, item.api_filter !== false]),
);

export const STATIC_ONLY_LEGACY_FILTER_KEYS = Object.freeze(
  LEGACY_FILTER_FIELDS
    .filter(([, , , apiFilter]) => !apiFilter)
    .map(([key]) => key),
);

export const FILTER_FIELD_CATALOG = Object.freeze(
  scanFilterContract.fields
    .filter((item) => item.kind)
    .map((item) => ({
      field: item.field,
      label: item.builder?.label ?? item.field.replaceAll('_', ' '),
      type: item.kind,
      value_type: item.value_type ?? (item.kind === 'range' ? 'number' : item.kind),
      category: item.builder?.category ?? 'Other',
      sortable: item.sortable === true,
      option_source: item.builder?.option_source ?? null,
      options: item.builder?.options ?? [],
      builder_visible: Boolean(item.builder?.label) && item.api_filter !== false,
    })),
);

export const BUILDER_FIELD_CATALOG = Object.freeze(
  FILTER_FIELD_CATALOG.filter((item) => item.builder_visible),
);

const mapLegacyFields = (kind) => Object.fromEntries(
  LEGACY_FILTER_FIELDS
    .filter(([, , fieldKind]) => fieldKind === kind)
    .map(([key, field]) => [key, field]),
);

export const RANGE_FILTER_TO_FIELD = Object.freeze(mapLegacyFields('range'));
export const BOOLEAN_FILTER_TO_FIELD = Object.freeze(mapLegacyFields('boolean'));
export const FIELD_TO_RANGE_FILTER = Object.freeze(Object.fromEntries(
  LEGACY_FILTER_FIELDS
    .filter(([, , kind]) => kind === 'range')
    .map(([key, field]) => [field, key]),
));
export const FIELD_TO_BOOLEAN_FILTER = Object.freeze(Object.fromEntries(
  LEGACY_FILTER_FIELDS
    .filter(([, , kind]) => kind === 'boolean')
    .map(([key, field]) => [field, key]),
));

export function fieldMeta(field, fallbackType = 'range') {
  return FILTER_FIELD_CATALOG.find((item) => item.field === field) ?? {
    field,
    label: String(field || 'Unknown field').replaceAll('_', ' '),
    type: fallbackType,
    value_type: fallbackType === 'range' ? 'number' : fallbackType,
    category: 'Other',
    sortable: false,
    option_source: null,
    options: [],
    builder_visible: false,
  };
}

export function fieldValueOptions(field, optionValues = {}) {
  const meta = fieldMeta(field);
  if (meta.options.length) return meta.options;
  return meta.option_source ? (optionValues[meta.option_source] ?? []) : [];
}
