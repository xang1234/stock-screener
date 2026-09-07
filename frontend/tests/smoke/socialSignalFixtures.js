const now = '2026-09-07T12:00:00Z';

const baseExplanation = {
  state_reasons: ['setup_ready', 'theme_confirmed'],
  readiness: 'ready', setup_score: 88, rs_rating_3m: 92, group_rank: 4,
  theme: 'AI infrastructure', theme_id: 17, market_exposure: 68,
  acceleration: 2.4,
  social_components: { authors: { value: 90, available_weight: 25, total_weight: 25 } },
  confirmation_components: { setup: { value: 88, available_weight: 30, total_weight: 30 } },
};

const row = (symbol, social, confirmation, queueScore, market = 'US') => ({
  run_id: 'fixture-run', candidate_key: `${market}:${symbol}`,
  canonical_symbol: symbol, market, state: 'actionable', social_score: social,
  confirmation_score: confirmation, queue_score: queueScore, mention_count: 3,
  observed_list_count: 2, enabled_list_count: 3, latest_mention: now,
  formula_version: 'social-signal-v1', coverage: [], explanation: baseExplanation,
});

export const blendedRows = [row('AMD', 75, 95, 85), row('NVDA', 96, 30, 60)];
export const pureRows = [blendedRows[1], blendedRows[0]];

const source = (sourceId, name, listId, lifecycle = 'enabled', version = 1) => ({
  source_id: String(sourceId), name, list_id: listId,
  canonical_url: `https://x.com/i/lists/${listId}`, lifecycle,
  provenance: sourceId < 3 ? 'system_seed' : 'admin', version,
  test_outcome: sourceId < 3 ? null : { provider: 'official', status: 'passed', sample_count: 5, tested_at: now },
  test_progress: null, collected_at: now,
  audit: [{ action: 'created', occurred_at: now, actor: 'fixture-admin' }],
});

export async function installSocialSignalFixtures(page) {
  const state = {
    refreshes: 0,
    watchlistAdds: [],
    sources: [
      source(1, 'Minervini Research List', '1522014550211457024'),
      source(2, 'Asia-Pacific Growth List', '1986290701492232693'),
    ],
  };
  const json = (route, payload, status = 200, headers = {}) => route.fulfill({
    status, headers, contentType: 'application/json', body: JSON.stringify(payload),
  });

  await page.route('**/api/v1/**', async (route, request) => {
    const url = new URL(request.url());
    const path = url.pathname.replace(/^\/api/, '');
    const method = request.method();

    if (path === '/v1/app-capabilities') return json(route, {
      features: { themes: true, chatbot: false, tasks: false, social_signals: true },
      auth: { required: false, configured: true, authenticated: true, mode: 'session_cookie', message: null },
      ui_snapshots: { enabled: false, scan: false, breadth: false, groups: false, themes: false },
      bootstrap_required: false, primary_market: 'US',
      enabled_markets: ['US', 'HK', 'CN', 'JP', 'TW'],
      supported_markets: ['US', 'HK', 'CN', 'JP', 'TW'],
      market_catalog: { version: 'fixture-v1', markets: [
        ['US', 'United States'], ['HK', 'Hong Kong'], ['CN', 'China A-shares'],
        ['JP', 'Japan'], ['TW', 'Taiwan'],
      ].map(([code, label]) => ({ code, label, capabilities: {} })) },
      api_base_path: '/api',
    });
    if (path === '/v1/runtime/activity') return json(route, {
      bootstrap: { state: 'ready', app_ready: true, primary_market: 'US', enabled_markets: ['US'], percent: 100 },
      summary: { active_market_count: 0, active_markets: [], status: 'idle' }, markets: [],
    });
    if (path === '/v1/strategy-profiles') return json(route, { profiles: [] });
    if (path === '/v1/pipeline/status') return json(route, { status: 'idle' });
    if (path === '/v1/operations/jobs') return json(route, { jobs: [], queues: [], workers: [], leases: {} });
    if (path === '/v1/telemetry/alerts') return json(route, { summaries: [], alerts: [] });
    if (path === '/v1/market-scan/daily-snapshot') return json(route, {
      market: url.searchParams.get('market') || 'US', market_display_name: 'United States', scan_id: 'fixture-scan',
      freshness: { as_of_date: '2026-09-07' }, key_markets: [], top_groups: [], leaders: { rows: [] },
      correction_survivors: { rows: [] }, top_candidates: { rows: [] }, market_health_exposure: null,
    });

    if (path === '/v1/social-signals/summary') return json(route, {
      available: true, stale: false, published_at: now,
      participating_source_count: 3, enabled_source_count: 3,
      top_signals: [...blendedRows, row('AVGO', 70, 70, 70), row('ANET', 69, 69, 69), row('VRT', 68, 68, 68), row('SMCI', 67, 67, 67)],
      dominant_themes: [{ theme_key: 'ai_infrastructure', accepted_company_count: 4 }],
    });
    if (path === '/v1/social-signals/queue') {
      const items = url.searchParams.get('rank_mode') === 'pure_social' ? pureRows : blendedRows;
      return json(route, { available: true, stale: false, total: items.length, items, published_at: now });
    }
    if (path === '/v1/social-signals/context') return json(route, {
      available: true, total: 1, items: [{ ...row('SPY', null, null, null), state: 'context' }],
    });
    if (path === '/v1/social-signals/unresolved') return json(route, {
      available: true, total: 1, items: [{ ...row('$MYSTERY', null, null, null), market: null, state: 'unresolved' }],
    });
    if (path.includes('/v1/social-signals/candidates/') && path.endsWith('/evidence')) return json(route, {
      available: true, item: pureRows[0], related_listings: [{ market: 'HK', canonical_symbol: 'NVDA.HK' }],
      posts: [1, 2, 3, 4].map((id) => ({ post_id: String(id), author_handle: `author${id}`,
        created_at: now, excerpt: `Fixture evidence ${id}`, url: `https://x.com/author${id}/status/${id}`,
        source_names: id === 1 ? ['Minervini Research List', 'Asia-Pacific Growth List'] : ['Japan Growth'],
        engagement: { likes: id },
      })),
    });
    if (path === '/v1/social-signals/theme-pulse') return json(route, {
      available: true, items: [{ theme_key: 'ai_infrastructure', name: 'AI Infrastructure',
        status: 'confirmed', social_strength: 91, market_strength: 73,
        measured_company_count: 3, accepted_company_count: 4, benchmark_symbol: 'SPY' }],
    });

    if (path === '/v1/social-signals/admin/runtime') return json(route, { mode: 'live', provider: 'official', version: 7 });
    if (path === '/v1/social-signals/admin/health') return json(route, {
      provider: 'official', source_count: state.sources.length,
      enabled_source_count: state.sources.filter((item) => item.lifecycle === 'enabled').length,
      participating_source_count: 2, social_fresh: true, collection_status: 'complete', processing_status: 'complete',
      reason_codes: [], unknown_company_identity_count: 0,
      budget: { limit_usd: '2', remaining_usd: '1.50', spent_usd: '.50', reserved_usd: '0', timezone: 'Asia/Singapore', pricing_status: 'configured' },
      backlog: { waiting: 0, failed: 0, outside_window: 0 }, last_collection_at: now,
    });
    if (path === '/v1/social-signals/admin/sources' && method === 'GET') {
      const includeArchived = url.searchParams.get('include_archived') === 'true';
      return json(route, state.sources.filter((item) => includeArchived || item.lifecycle !== 'archived'));
    }
    if (path === '/v1/social-signals/admin/sources' && method === 'POST') {
      const body = request.postDataJSON();
      const created = source(3, body.name, String(body.list_ref).split('/').pop(), 'pending', 1);
      created.test_outcome = null;
      state.sources.push(created);
      return json(route, created, 201);
    }
    const sourceMatch = path.match(/^\/v1\/social-signals\/admin\/sources\/(\d+)(?:\/(test|transition))?$/);
    if (sourceMatch) {
      const selected = state.sources.find((item) => item.source_id === sourceMatch[1]);
      const action = sourceMatch[2];
      if (action === 'test') {
        selected.version += 1;
        selected.test_outcome = { provider: 'official', status: 'passed', sample_count: 5, tested_at: now };
        selected.audit.push({ action: 'test_completed', occurred_at: now, actor: 'fixture-worker' });
        return json(route, { task_id: 'fixture-test' }, 202);
      }
      if (action === 'transition') {
        const { target } = request.postDataJSON();
        if (selected.source_id === '1' && target === 'disabled') return json(route, { detail: 'minimum_two_enabled' }, 409);
        selected.lifecycle = target;
        selected.version += 1;
        selected.audit.push({ action: target, occurred_at: now, actor: 'fixture-admin' });
        return json(route, selected);
      }
      const { name } = request.postDataJSON();
      selected.name = name;
      selected.version += 1;
      selected.audit.push({ action: 'renamed', occurred_at: now, actor: 'fixture-admin' });
      return json(route, selected);
    }
    if (path === '/v1/social-signals/admin/refresh') {
      state.refreshes += 1;
      return state.refreshes === 1
        ? json(route, { task_id: 'fixture-refresh' }, 202)
        : json(route, { detail: 'refresh_cooldown' }, 429, { 'retry-after': '900' });
    }
    if (path === '/v1/social-signals/admin/analysis') return json(route, []);
    if (path === '/v1/social-signals/admin/associations') return json(route, []);
    if (path === '/v1/social-signals/admin/company-identities') return json(route, { registry_version: 7, entries: [] });
    if (path === '/v1/social-signals/admin/runs') return json(route, []);
    if (path === '/v1/user-watchlists') return json(route, { watchlists: [{ id: 7, name: 'Leaders' }] });
    if (path === '/v1/user-watchlists/memberships') return json(route, { memberships: {} });
    if (path === '/v1/user-watchlists/7/items' && method === 'POST') {
      state.watchlistAdds.push(request.postDataJSON());
      return json(route, { id: 1, watchlist_id: 7, ...request.postDataJSON() }, 201);
    }

    return json(route, {});
  });
  return state;
}
