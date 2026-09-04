import { describe, expect, it } from 'vitest';

import {
  normalizeOptionsCommandCenter,
  normalizeOptionsManifest,
  normalizeOptionsSymbolDetail,
  optionsCommandCenterQueryKey,
  optionsSymbolQueryKey,
} from './optionsContract';
import {
  commandCenterFixture,
  optionsManifestFixture,
  symbolDetailFixture,
} from './__fixtures__/optionsResponses';

describe('options contract', () => {
  it('preserves truthful metric states, model labels, ranks, zero, and null', () => {
    const result = normalizeOptionsCommandCenter(commandCenterFixture);
    const aapl = result.items[0];

    expect(aapl.candidate_rank).toBe(1);
    expect(aapl.leader_rank).toBe(2);
    expect(aapl.metrics.max_pain.value).toBe(0);
    expect(aapl.metrics.net_gex.label).toBe('Estimated Net GEX');
    expect(aapl.metrics.skew_25_delta.value).toBeNull();
    expect(aapl.metrics.skew_25_delta.reason_codes).toEqual(['building_history']);
    expect(aapl.quality_evidence.provider_spot_price).toBe(201);
  });

  it('keys command and symbol reads by mode, run, path, and symbol identity', () => {
    expect(optionsCommandCenterQueryKey({ mode: 'static', runId: 7, path: 'options/command-center.json' }))
      .toEqual(['options-analytics', 'command-center', 'static', 7, 'options/command-center.json']);
    expect(optionsSymbolQueryKey({ mode: 'static', runId: 7, symbol: 'aapl', path: 'options/symbols/QUFQTA.json' }))
      .not.toEqual(optionsSymbolQueryKey({ mode: 'static', runId: 7, symbol: 'MSFT', path: 'options/symbols/TVNGVA.json' }));
  });

  it('rejects malformed schema, mixed runs, non-finite values, and unsafe paths', () => {
    expect(() => normalizeOptionsManifest({ ...optionsManifestFixture, schema_version: 'future-v9' }))
      .toThrow(/schema/i);
    expect(() => normalizeOptionsCommandCenter({ ...commandCenterFixture, run_id: 99 }, { expectedRunId: 7 }))
      .toThrow(/run/i);
    expect(() => normalizeOptionsSymbolDetail({ ...symbolDetailFixture, coverage: Infinity }))
      .toThrow(/finite/i);
    expect(() => normalizeOptionsManifest({
      ...optionsManifestFixture,
      command_center_path: '../secret.json',
    })).toThrow(/path/i);
    expect(() => normalizeOptionsCommandCenter({
      ...commandCenterFixture,
      unexpected_root_field: true,
    })).toThrow(/contract/i);
  });

  it('rejects incomplete or mistyped quality evidence', () => {
    const missingDte = structuredClone(commandCenterFixture);
    delete missingDte.items[0].quality_evidence.days_to_expiration;
    expect(() => normalizeOptionsCommandCenter(missingDte)).toThrow(/days_to_expiration/i);

    const invalidCoverage = structuredClone(commandCenterFixture);
    invalidCoverage.items[0].quality_evidence.iv_coverage = 1.1;
    expect(() => normalizeOptionsCommandCenter(invalidCoverage)).toThrow(/iv_coverage/i);
  });
});
