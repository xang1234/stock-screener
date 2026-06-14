import { describe, expect, it } from 'vitest';

import { shouldDehydratePersistedQuery } from './appQueryPersistence';

const successfulQuery = (queryKey) => ({
  queryKey,
  state: { status: 'success' },
});

const failedQuery = (queryKey) => ({
  queryKey,
  state: { status: 'error' },
});

describe('app query persistence', () => {
  it('does not persist volatile runtime or auth state', () => {
    expect(shouldDehydratePersistedQuery(successfulQuery(['appCapabilities']))).toBe(false);
    expect(shouldDehydratePersistedQuery(successfulQuery(['runtimeActivity']))).toBe(false);
  });

  it('persists ordinary successful data queries only', () => {
    expect(shouldDehydratePersistedQuery(successfulQuery(['scanHistory', 'US']))).toBe(true);
    expect(shouldDehydratePersistedQuery(failedQuery(['scanHistory', 'US']))).toBe(false);
  });
});
