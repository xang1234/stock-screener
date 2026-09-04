export const OPTIONS_ANALYTICS_SCHEMA_VERSION = 'options-analytics-v1';
export const STATIC_OPTIONS_SCHEMA_VERSION = 'static-options-v1';

const METRIC_NAMES = [
  'max_pain',
  'net_gex',
  'gamma_flip',
  'call_wall',
  'put_wall',
  'atm_iv',
  'skew_25_delta',
  'realized_volatility',
  'vrp',
  'activity_intensity',
  'volume_oi_ratio',
  'near_spot_volume_concentration',
];

const fail = (message) => {
  throw new Error(`Invalid options analytics contract: ${message}`);
};

const isRecord = (value) => (
  value !== null && typeof value === 'object' && !Array.isArray(value)
);

const requireFiniteNumbers = (value, location = 'payload') => {
  if (typeof value === 'number' && !Number.isFinite(value)) {
    fail(`non-finite number at ${location}`);
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => requireFiniteNumbers(item, `${location}[${index}]`));
  } else if (isRecord(value)) {
    Object.entries(value).forEach(([key, item]) => (
      requireFiniteNumbers(item, `${location}.${key}`)
    ));
  }
};

const requireRecord = (value, location) => {
  if (!isRecord(value)) fail(`${location} must be an object`);
};

const requireArray = (value, location) => {
  if (!Array.isArray(value)) fail(`${location} must be an array`);
};

const requireInteger = (value, location, { nullable = false } = {}) => {
  if (nullable && value === null) return;
  if (!Number.isInteger(value)) fail(`${location} must be an integer`);
};

const requireString = (value, location) => {
  if (typeof value !== 'string' || value.length === 0) fail(`${location} must be a string`);
};

const requireNumber = (value, location, { min, max, nullable = false } = {}) => {
  if (nullable && value === null) return;
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${location} must be a number`);
  if (min !== undefined && value < min) fail(`${location} must be at least ${min}`);
  if (max !== undefined && value > max) fail(`${location} must be at most ${max}`);
};

const validateQualityEvidence = (evidence, location) => {
  requireRecord(evidence, location);
  requireNumber(evidence.source_spot_price, `${location}.source_spot_price`, {
    min: 0,
    nullable: true,
  });
  if (evidence.source_spot_price === 0) fail(`${location}.source_spot_price must be positive`);
  requireNumber(evidence.provider_spot_price, `${location}.provider_spot_price`, { nullable: true });
  requireNumber(evidence.spot_disagreement_ratio, `${location}.spot_disagreement_ratio`, {
    min: 0,
    nullable: true,
  });
  if (evidence.latest_contract_trade_at !== null) {
    requireString(evidence.latest_contract_trade_at, `${location}.latest_contract_trade_at`);
  }
  requireInteger(evidence.days_to_expiration, `${location}.days_to_expiration`, { nullable: true });
  if (evidence.days_to_expiration !== null && evidence.days_to_expiration < 0) {
    fail(`${location}.days_to_expiration must be non-negative`);
  }
  [
    'normalized_call_count',
    'normalized_put_count',
    'distinct_strike_count',
  ].forEach((field) => {
    requireInteger(evidence[field], `${location}.${field}`);
    if (evidence[field] < 0) fail(`${location}.${field} must be non-negative`);
  });
  [
    'open_interest_coverage',
    'iv_coverage',
    'volume_coverage',
    'two_sided_quote_coverage',
  ].forEach((field) => requireNumber(evidence[field], `${location}.${field}`, { min: 0, max: 1 }));
};

const requireSafeOptionsPath = (path, location) => {
  requireString(path, location);
  const segments = path.split('/');
  if (
    path.startsWith('/')
    || path.includes('\\')
    || path.includes('://')
    || segments[0] !== 'options'
    || segments.some((segment) => segment === '' || segment === '.' || segment === '..')
  ) {
    fail(`unsafe ${location}`);
  }
};

const validateMetric = (metric, location) => {
  requireRecord(metric, location);
  if (typeof metric.available !== 'boolean') fail(`${location}.available must be boolean`);
  if (metric.value !== null && typeof metric.value !== 'number') {
    fail(`${location}.value must be a number or null`);
  }
  if (metric.available !== (metric.value !== null)) {
    fail(`${location} availability does not match value`);
  }
  if (metric.label !== null) requireString(metric.label, `${location}.label`);
  requireArray(metric.reason_codes, `${location}.reason_codes`);
  requireRecord(metric.evidence, `${location}.evidence`);
};

const validateItem = (item, location) => {
  requireRecord(item, location);
  requireString(item.symbol, `${location}.symbol`);
  if (item.symbol !== item.symbol.toUpperCase()) fail(`${location}.symbol must be uppercase`);
  requireArray(item.source_badges, `${location}.source_badges`);
  requireInteger(item.candidate_rank, `${location}.candidate_rank`, { nullable: true });
  requireInteger(item.leader_rank, `${location}.leader_rank`, { nullable: true });
  validateQualityEvidence(item.quality_evidence, `${location}.quality_evidence`);
  requireRecord(item.metrics, `${location}.metrics`);
  METRIC_NAMES.forEach((name) => validateMetric(item.metrics[name], `${location}.metrics.${name}`));
  requireArray(item.warnings, `${location}.warnings`);
  requireArray(item.reason_codes, `${location}.reason_codes`);
};

const validateRun = (payload, context = {}) => {
  requireRecord(payload, 'payload');
  requireFiniteNumbers(payload);
  if (payload.schema_version !== OPTIONS_ANALYTICS_SCHEMA_VERSION) {
    fail('unsupported data schema version');
  }
  if (payload.calculation_version !== OPTIONS_ANALYTICS_SCHEMA_VERSION) {
    fail('unsupported calculation version');
  }
  requireInteger(payload.run_id, 'run_id');
  requireInteger(payload.source_feature_run_id, 'source_feature_run_id', { nullable: true });
  if (payload.market !== 'US') fail('market must be US');
  requireString(payload.provider, 'provider');
  if (typeof payload.coverage !== 'number' || payload.coverage < 0 || payload.coverage > 1) {
    fail('coverage must be between zero and one');
  }
  if (typeof payload.stale !== 'boolean') fail('stale must be boolean');
  requireArray(payload.reason_codes, 'reason_codes');
  requireRecord(payload.assumptions, 'assumptions');

  const expected = {
    run_id: context.expectedRunId,
    schema_version: context.expectedSchemaVersion,
    calculation_version: context.expectedCalculationVersion,
    source_feature_run_id: context.expectedSourceRunId,
    provider: context.expectedProvider,
    market: context.expectedMarket,
    latest_observation_at: context.expectedLatestObservationAt,
    coverage: context.expectedCoverage,
    stale: context.expectedStale,
  };
  Object.entries(expected).forEach(([field, value]) => {
    if (value !== undefined && payload[field] !== value) fail(`run identity mismatch: ${field}`);
  });
};

export const normalizeOptionsManifest = (manifest) => {
  requireRecord(manifest, 'manifest');
  requireFiniteNumbers(manifest, 'manifest');
  if (manifest.schema_version !== STATIC_OPTIONS_SCHEMA_VERSION) fail('unsupported static schema version');
  if (manifest.data_schema_version !== OPTIONS_ANALYTICS_SCHEMA_VERSION) fail('unsupported data schema version');
  if (manifest.calculation_version !== OPTIONS_ANALYTICS_SCHEMA_VERSION) fail('unsupported calculation version');
  requireInteger(manifest.published_run_id, 'published_run_id');
  requireInteger(manifest.source_feature_run_id, 'source_feature_run_id', { nullable: true });
  if (manifest.market !== 'US') fail('market must be US');
  if (typeof manifest.coverage !== 'number' || manifest.coverage < 0 || manifest.coverage > 1) {
    fail('coverage must be between zero and one');
  }
  if (typeof manifest.stale !== 'boolean' || typeof manifest.stale_relative_to_equity !== 'boolean') {
    fail('stale markers must be boolean');
  }
  requireArray(manifest.reason_codes, 'reason_codes');
  requireSafeOptionsPath(manifest.command_center_path, 'command_center_path');
  requireRecord(manifest.symbols, 'symbols');
  if (Object.keys(manifest.symbols).length > 80) fail('symbol map exceeds 80 current symbols');
  const paths = new Set();
  Object.entries(manifest.symbols).forEach(([symbol, entry]) => {
    if (!symbol || symbol !== symbol.toUpperCase()) fail('symbol map keys must be uppercase');
    requireRecord(entry, `symbols.${symbol}`);
    requireString(entry.key, `symbols.${symbol}.key`);
    requireSafeOptionsPath(entry.path, `symbols.${symbol}.path`);
    if (paths.has(entry.path)) fail('symbol detail paths must be unique');
    paths.add(entry.path);
  });
  return manifest;
};

export const normalizeOptionsCommandCenter = (payload, context = {}) => {
  validateRun(payload, context);
  requireArray(payload.items, 'items');
  if (payload.items.length > 80) fail('command center exceeds 80 current symbols');
  requireInteger(payload.current_count, 'current_count');
  if (payload.current_count !== payload.items.length) fail('current count does not match items');
  const symbols = new Set();
  payload.items.forEach((item, index) => {
    validateItem(item, `items[${index}]`);
    if (symbols.has(item.symbol)) fail(`duplicate symbol: ${item.symbol}`);
    symbols.add(item.symbol);
  });
  return payload;
};

export const normalizeOptionsSymbolDetail = (payload, context = {}) => {
  validateRun(payload, context);
  validateItem(payload.item, 'item');
  if (context.expectedSymbol && payload.item.symbol !== context.expectedSymbol) {
    fail('symbol detail identity mismatch');
  }
  requireArray(payload.strike_points, 'strike_points');
  requireArray(payload.history, 'history');
  const strikes = new Set();
  payload.strike_points.forEach((point, index) => {
    requireRecord(point, `strike_points[${index}]`);
    if (typeof point.strike !== 'number') fail(`strike_points[${index}].strike must be a number`);
    if (strikes.has(point.strike)) fail(`duplicate strike: ${point.strike}`);
    strikes.add(point.strike);
  });
  return payload;
};

export const optionsManifestRunContext = (manifest) => ({
  expectedRunId: manifest.published_run_id,
  expectedSchemaVersion: manifest.data_schema_version,
  expectedCalculationVersion: manifest.calculation_version,
  expectedSourceRunId: manifest.source_feature_run_id,
  expectedProvider: manifest.provider,
  expectedMarket: manifest.market,
  expectedLatestObservationAt: manifest.latest_observation_at,
  expectedCoverage: manifest.coverage,
  expectedStale: manifest.stale,
});

export const optionsCommandCenterQueryKey = ({ mode, runId, path = null }) => [
  'options-analytics',
  'command-center',
  mode,
  runId,
  path,
];

export const optionsSymbolQueryKey = ({ mode, runId, symbol, path = null }) => [
  'options-analytics',
  'symbol',
  mode,
  runId,
  String(symbol || '').trim().toUpperCase(),
  path,
];
