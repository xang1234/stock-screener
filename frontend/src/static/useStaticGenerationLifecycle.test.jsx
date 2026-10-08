import { act, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  STATIC_GENERATION_EXPIRED_EVENT,
  fetchStaticJson,
  queryKeyGeneration,
  staticQueryOptions,
  useStaticGeneration,
  useStaticManifest,
} from './dataClient';
import { useStaticGenerationLifecycle } from './useStaticGenerationLifecycle';

const manifestFor = (generation) => ({ generation, data_root: `g/${generation}/` });

// A fetch stub: the manifest names the current generation; data files resolve
// only when the test releases them, so replacement data can arrive late.
function stubSite() {
  const site = { generation: 'g1', pending: [] };
  vi.stubGlobal('fetch', vi.fn((url) => {
    if (url.endsWith('static-data/manifest.json')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => manifestFor(site.generation) });
    }
    const generation = url.match(/static-data\/g\/([^/]+)\//)?.[1];
    return new Promise((resolve) => {
      site.pending.push(() => resolve({ ok: true, status: 200, json: async () => ({ generation }) }));
    });
  }));
  site.release = () => {
    const pending = site.pending.splice(0);
    pending.forEach((resolve) => resolve());
  };
  return site;
}

function HomeView() {
  // Pages take every data path from the manifest, so nothing loads before it.
  const manifest = useStaticManifest().data;
  const { generation, dataRoot } = useStaticGeneration();
  const { expired } = useStaticGenerationLifecycle();
  const query = useQuery(staticQueryOptions({
    key: ['staticHome', 'markets/us/home.json'],
    generation,
    queryFn: () => fetchStaticJson('markets/us/home.json', dataRoot),
    enabled: Boolean(manifest),
  }));
  return (
    <>
      <div data-testid="shown">{query.data?.generation ?? 'loading'}</div>
      <div data-testid="expired">{String(expired)}</div>
    </>
  );
}

const renderView = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <HomeView />
    </QueryClientProvider>,
  );
  return queryClient;
};

const homeGenerations = (queryClient) => queryClient.getQueryCache().getAll()
  .filter((query) => query.queryKey[0] === 'staticHome')
  .map((query) => queryKeyGeneration(query.queryKey));

describe('static data generation lifecycle (#504)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('keeps old data visible until each new generation loads, then drops the old one', async () => {
    const site = stubSite();
    const queryClient = renderView();
    await waitFor(() => expect(site.pending).toHaveLength(1));
    await act(async () => site.release());
    await waitFor(() => expect(screen.getByTestId('shown').textContent).toBe('g1'));

    for (const next of ['g2', 'g3', 'g4']) {
      site.generation = next;
      await act(async () => {
        await queryClient.refetchQueries({ queryKey: ['staticManifest'] });
      });
      // The replacement is still in flight: the previous data stays on screen.
      await waitFor(() => expect(site.pending).toHaveLength(1));
      expect(screen.getByTestId('shown').textContent).not.toBe('loading');
      await act(async () => site.release());
      await waitFor(() => expect(screen.getByTestId('shown').textContent).toBe(next));
      // Only the generation on screen stays cached.
      await waitFor(() => expect(homeGenerations(queryClient)).toEqual([next]));
    }
  });

  it('offers a reload only while the tab is still on the replaced generation', async () => {
    const site = stubSite();
    const queryClient = renderView();
    await waitFor(() => expect(site.pending).toHaveLength(1));
    await act(async () => site.release());
    await waitFor(() => expect(screen.getByTestId('shown').textContent).toBe('g1'));

    // A file under g1 is gone and the manifest still says g1 (e.g. a lagging edge).
    await act(async () => {
      window.dispatchEvent(new CustomEvent(STATIC_GENERATION_EXPIRED_EVENT, { detail: { dataRoot: 'g/g1/' } }));
    });
    await waitFor(() => expect(screen.getByTestId('expired').textContent).toBe('true'));

    // A routine manifest check that finds the same generation keeps the offer up.
    let refetching;
    act(() => {
      refetching = queryClient.refetchQueries({ queryKey: ['staticManifest'] });
    });
    expect(screen.getByTestId('expired').textContent).toBe('true');
    await act(async () => refetching);
    expect(screen.getByTestId('expired').textContent).toBe('true');

    // The next manifest check moves the tab forward and the reload offer goes away.
    site.generation = 'g2';
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ['staticManifest'] });
    });
    await waitFor(() => expect(screen.getByTestId('expired').textContent).toBe('false'));
  });

  it('announces a file missing under a generation', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404, json: async () => ({}) })));
    const listener = vi.fn();
    window.addEventListener(STATIC_GENERATION_EXPIRED_EVENT, listener);

    await expect(fetchStaticJson('markets/us/home.json', 'g/old/')).rejects.toThrow();

    window.removeEventListener(STATIC_GENERATION_EXPIRED_EVENT, listener);
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].detail.dataRoot).toBe('g/old/');
  });
});
