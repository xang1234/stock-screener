import optionsSchema from './optionsSchema.json';
import {
  commandCenter as validateCommandCenter,
  manifest as validateManifest,
  symbolDetail as validateSymbolDetail,
} from 'virtual:options-validators';

const { manifest } = optionsSchema.models;

export const OPTIONS_ANALYTICS_SCHEMA_VERSION = manifest.properties.data_schema_version.const;
export const STATIC_OPTIONS_SCHEMA_VERSION = manifest.properties.schema_version.const;

const fail = (message) => {
  throw new Error(`Invalid options analytics contract: ${message}`);
};

const validationErrorsText = (errors) => (errors || [])
  .map((error) => `${error.instancePath || 'data'} ${error.message}`)
  .join('; ');

const rejectNonFiniteNumbers = (value) => {
  if (typeof value === 'number' && !Number.isFinite(value)) fail('non-finite number');
  if (Array.isArray(value)) value.forEach(rejectNonFiniteNumbers);
  else if (value !== null && typeof value === 'object') {
    Object.values(value).forEach(rejectNonFiniteNumbers);
  }
};

const validateShape = (validator, payload, label) => {
  rejectNonFiniteNumbers(payload);
  if (!validator(payload)) {
    fail(`${label}: ${validationErrorsText(validator.errors)}`);
  }
};

const requireSafeOptionsPath = (path, location) => {
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

const validateRunIdentity = (payload, context) => {
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

const validateItemSemantics = (item, location) => {
  if (item.symbol !== item.symbol.toUpperCase()) fail(`${location}.symbol must be uppercase`);
  Object.entries(item.metrics).forEach(([name, metric]) => {
    if (metric.available !== (metric.value !== null)) {
      fail(`${location}.metrics.${name} availability does not match value`);
    }
  });
  Object.entries(item.historical_metrics).forEach(([name, metric]) => {
    if (metric.available !== (metric.value !== null)) {
      fail(`${location}.historical_metrics.${name} availability does not match value`);
    }
  });
};

export const normalizeOptionsManifest = (payload) => {
  validateShape(validateManifest, payload, 'manifest');
  requireSafeOptionsPath(payload.command_center_path, 'command_center_path');
  const paths = new Set();
  Object.entries(payload.symbols).forEach(([symbol, entry]) => {
    if (symbol !== symbol.toUpperCase()) fail('symbol map keys must be uppercase');
    requireSafeOptionsPath(entry.path, `symbols.${symbol}.path`);
    if (paths.has(entry.path)) fail('symbol detail paths must be unique');
    paths.add(entry.path);
  });
  if (
    payload.stale_relative_to_equity
    && (!payload.stale || !payload.reason_codes.includes('stale_relative_to_equity'))
  ) {
    fail('stale options metadata is inconsistent');
  }
  return payload;
};

export const normalizeOptionsCommandCenter = (payload, context = {}) => {
  validateShape(validateCommandCenter, payload, 'command center');
  validateRunIdentity(payload, context);
  if (payload.current_count !== payload.items.length) fail('current count does not match items');
  const symbols = new Set();
  payload.items.forEach((item, index) => {
    validateItemSemantics(item, `items[${index}]`);
    if (symbols.has(item.symbol)) fail(`duplicate symbol: ${item.symbol}`);
    symbols.add(item.symbol);
  });
  return payload;
};

export const normalizeOptionsSymbolDetail = (payload, context = {}) => {
  validateShape(validateSymbolDetail, payload, 'symbol detail');
  validateRunIdentity(payload, context);
  validateItemSemantics(payload.item, 'item');
  if (context.expectedSymbol && payload.item.symbol !== context.expectedSymbol) {
    fail('symbol detail identity mismatch');
  }
  const strikes = new Set();
  payload.strike_points.forEach((point) => {
    if (strikes.has(point.strike)) fail(`duplicate strike: ${point.strike}`);
    strikes.add(point.strike);
  });
  return payload;
};

export const optionsManifestRunContext = (manifestPayload) => ({
  expectedRunId: manifestPayload.published_run_id,
  expectedSchemaVersion: manifestPayload.data_schema_version,
  expectedCalculationVersion: manifestPayload.calculation_version,
  expectedSourceRunId: manifestPayload.source_feature_run_id,
  expectedProvider: manifestPayload.provider,
  expectedMarket: manifestPayload.market,
  expectedLatestObservationAt: manifestPayload.latest_observation_at,
  expectedCoverage: manifestPayload.coverage,
  expectedStale: manifestPayload.stale,
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
