import { expect, test } from '@playwright/test';

const defaultScanDefaults = {
  universe: 'test',
  screeners: ['minervini'],
  composite_method: 'weighted_average',
  criteria: {
    include_vcp: true,
    custom_filters: {
      price_min: 5,
      price_max: 500,
      rs_rating_min: 70,
      volume_min: 100000,
      eps_growth_min: 20,
      sales_growth_min: 20,
    },
  },
};

const defaultCapabilities = (authenticated) => ({
  features: {
    themes: true,
    chatbot: true,
    tasks: false,
  },
  auth: {
    required: true,
    configured: true,
    authenticated,
    mode: 'session_cookie',
    message: null,
  },
  ui_snapshots: {
    enabled: false,
    scan: false,
    breadth: false,
    groups: false,
    themes: false,
  },
  scan_defaults: defaultScanDefaults,
  api_base_path: '/api',
});

const defaultRuntimeActivity = {
  bootstrap: {
    state: 'ready',
    app_ready: true,
    primary_market: 'US',
    enabled_markets: ['US'],
    current_stage: null,
    progress_mode: 'determinate',
    percent: 100,
    message: 'Primary market is ready.',
    background_warning: null,
  },
  summary: {
    active_market_count: 0,
    active_markets: [],
    status: 'idle',
  },
  markets: [
    {
      market: 'US',
      lifecycle: 'idle',
      stage_key: null,
      stage_label: null,
      status: 'idle',
      progress_mode: 'determinate',
      percent: null,
      current: null,
      total: null,
      message: 'Idle',
      task_name: null,
      task_id: null,
      updated_at: null,
    },
  ],
};

const jsonResponse = (route, payload, status = 200) =>
  route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(payload),
  });

const ensureLoginVisible = async (page) => {
  const loginTitle = page.getByText('Sign in to Stock Scanner');
  if (!(await loginTitle.isVisible().catch(() => false))) {
    await page.goto('/scan');
  }
  await expect(loginTitle).toBeVisible({ timeout: 20000 });
};

test('single-tenant operator smoke path (assistant -> scan -> themes review -> auth expiry -> relogin)', async ({ page }) => {
  let authenticated = false;
  let expireNextScanRequest = false;
  let latestScanId = 'scan-smoke';
  const assistantConversationId = 'assistant-smoke';
  let assistantMessages = [];

  const assistantReply = {
    id: 2,
    conversation_id: assistantConversationId,
    role: 'assistant',
    content: 'NVDA remains the strongest internal AI theme candidate [1]. External coverage is still constructive, so it is worth keeping on a watchlist.',
    agent_type: 'hermes',
    tool_calls: [
      {
        tool: 'stock_snapshot',
        args: { symbol: 'NVDA' },
      },
    ],
    source_references: [
      {
        reference_number: 1,
        type: 'internal',
        title: 'Feature run snapshot',
        url: '/stocks/NVDA',
        section: 'As of 2026-04-09',
        snippet: 'Latest published scan posture and theme context.',
      },
    ],
    created_at: '2026-04-09T00:01:00Z',
  };

  await page.route('**/*', async (route, request) => {
    const url = new URL(request.url());
    const method = request.method();
    const path = url.pathname.replace(/^\/api/, '');

    if (!path.startsWith('/v1/')) {
      return route.continue();
    }

    if ((path === '/v1/app-capabilities' || path === '/v1/app-capabilities/') && method === 'GET') {
      return jsonResponse(route, defaultCapabilities(authenticated));
    }

    if (path === '/v1/auth/login' && method === 'POST') {
      authenticated = true;
      return jsonResponse(route, { success: true });
    }

    if (path === '/v1/auth/logout' && method === 'POST') {
      authenticated = false;
      return jsonResponse(route, { success: true });
    }

    if (path === '/v1/assistant/health' && method === 'GET') {
      if (!authenticated) {
        return jsonResponse(route, { detail: 'Unauthorized' }, 401);
      }
      return jsonResponse(route, {
        status: 'ok',
        available: true,
        streaming: true,
        popup_enabled: true,
        model: 'hermes-smoke',
        detail: null,
      });
    }

    if (!authenticated) {
      return jsonResponse(route, { detail: 'Unauthorized' }, 401);
    }

    if (path === '/v1/assistant/conversations' && method === 'POST') {
      return jsonResponse(route, {
        id: 1,
        conversation_id: assistantConversationId,
        title: 'Assistant',
        created_at: '2026-04-09T00:00:00Z',
        updated_at: '2026-04-09T00:00:00Z',
        is_active: true,
        message_count: assistantMessages.length,
      });
    }

    if (path === '/v1/assistant/conversations' && method === 'GET') {
      return jsonResponse(route, {
        conversations: [
          {
            id: 1,
            conversation_id: assistantConversationId,
            title: 'Assistant',
            created_at: '2026-04-09T00:00:00Z',
            updated_at: '2026-04-09T00:00:00Z',
            is_active: true,
            message_count: assistantMessages.length,
          },
        ],
        total: 1,
      });
    }

    if (path === `/v1/assistant/conversations/${assistantConversationId}` && method === 'GET') {
      return jsonResponse(route, {
        id: 1,
        conversation_id: assistantConversationId,
        title: 'Assistant',
        created_at: '2026-04-09T00:00:00Z',
        updated_at: '2026-04-09T00:00:00Z',
        is_active: true,
        message_count: assistantMessages.length,
        messages: assistantMessages,
      });
    }

    if (path === `/v1/assistant/conversations/${assistantConversationId}/messages` && method === 'POST') {
      const body = request.postDataJSON();
      assistantMessages = [
        ...assistantMessages,
        {
          id: assistantMessages.length + 1,
          conversation_id: assistantConversationId,
          role: 'user',
          content: body.content,
          created_at: '2026-04-09T00:00:30Z',
        },
        assistantReply,
      ];

      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: [
          `data: ${JSON.stringify({ type: 'content', content: 'NVDA remains the strongest internal AI theme candidate [1]. ' })}`,
          '',
          `data: ${JSON.stringify({ type: 'tool_call', tool: 'stock_snapshot', params: { symbol: 'NVDA' } })}`,
          '',
          `data: ${JSON.stringify({ type: 'tool_result', tool: 'stock_snapshot', status: 'ok', result: { symbol: 'NVDA' } })}`,
          '',
          `data: ${JSON.stringify({ type: 'content', content: 'External coverage is still constructive, so it is worth keeping on a watchlist.' })}`,
          '',
          `data: ${JSON.stringify({ type: 'done', message: assistantReply, tool_calls: assistantReply.tool_calls, references: assistantReply.source_references })}`,
          '',
        ].join('\n'),
      });
    }

    if (path === '/v1/user-watchlists' && method === 'GET') {
      return jsonResponse(route, {
        watchlists: [
          { id: 7, name: 'Leaders' },
        ],
      });
    }

    if (path === '/v1/assistant/watchlist-add-preview' && method === 'POST') {
      return jsonResponse(route, {
        watchlist: { id: 7, name: 'Leaders' },
        requested_symbols: ['NVDA'],
        addable_symbols: ['NVDA'],
        existing_symbols: [],
        invalid_symbols: [],
        reason: null,
        summary: '1 symbol can be added to Leaders.',
      });
    }

    if (path === '/v1/user-watchlists/7/items/bulk' && method === 'POST') {
      return jsonResponse(route, [
        {
          id: 101,
          watchlist_id: 7,
          position: 1,
          symbol: 'NVDA',
          created_at: '2026-04-09T00:02:00Z',
          updated_at: '2026-04-09T00:02:00Z',
        },
      ]);
    }

    if (path === '/v1/strategy-profiles' && method === 'GET') {
      return jsonResponse(route, {
        profiles: [{ profile: 'default', label: 'Default' }],
      });
    }

    if (path === '/v1/strategy-profiles/default' && method === 'GET') {
      return jsonResponse(route, {
        profile: 'default',
        label: 'Default',
        scan_defaults: defaultScanDefaults,
      });
    }

    if (path === '/v1/cache/dashboard-stats' && method === 'GET') {
      return jsonResponse(route, {
        fundamentals: {
          last_update: '2026-04-09T00:00:00Z',
          cached_count: 1,
          fresh_count: 1,
        },
      });
    }

    if (path === '/v1/cache/health' && method === 'GET') {
      return jsonResponse(route, {
        status: 'fresh',
        spy_last_date: '2026-04-09',
        message: 'Cache up to date',
        can_refresh: true,
      });
    }

    if (path === '/v1/runtime/activity' && method === 'GET') {
      return jsonResponse(route, defaultRuntimeActivity);
    }

    if (path.startsWith('/v1/market-scan/watchlist/') && method === 'GET') {
      return jsonResponse(route, { symbols: ['SPY', 'QQQ', 'IWM'] });
    }

    if (path === '/v1/universe/stats' && method === 'GET') {
      return jsonResponse(route, {
        active: 2,
        sp500: 2,
        by_exchange: {
          NYSE: 1,
          NASDAQ: 1,
          AMEX: 0,
        },
      });
    }

    if (path === '/v1/scans' && method === 'GET') {
      return jsonResponse(route, {
        scans: [
          {
            scan_id: latestScanId,
            status: 'completed',
            created_at: '2026-04-09T00:00:00Z',
            universe: 'test',
            screeners: ['minervini'],
            composite_method: 'weighted_average',
            total_stocks: 2,
            passed_stocks: 1,
          },
        ],
      });
    }

    if (path === '/v1/scans' && method === 'POST') {
      if (expireNextScanRequest) {
        expireNextScanRequest = false;
        authenticated = false;
        return jsonResponse(route, { detail: 'Session expired' }, 401);
      }
      latestScanId = 'scan-smoke';
      return jsonResponse(route, { scan_id: latestScanId, status: 'queued' });
    }

    if (/^\/v1\/scans\/[^/]+\/status$/.test(path) && method === 'GET') {
      return jsonResponse(route, {
        scan_id: latestScanId,
        status: 'completed',
        progress: 100,
        total_stocks: 2,
        completed_stocks: 2,
        passed_stocks: 1,
        eta_seconds: 0,
      });
    }

    if (/^\/v1\/scans\/[^/]+\/results$/.test(path) && method === 'GET') {
      return jsonResponse(route, {
        scan_id: latestScanId,
        total: 1,
        results: [
          {
            symbol: 'NVDA',
            company_name: 'NVIDIA Corporation',
            composite_score: 97.5,
            minervini_score: 92.1,
            current_price: 903.42,
            volume: 150000000,
            market_cap: 2100000000000,
            rs_rating: 98,
            eps_growth_qq: 41,
            sales_growth_qq: 32,
            stage: 2,
            ibd_industry_group: 'Semiconductors',
            gics_sector: 'Technology',
            ma_alignment: true,
            vcp_detected: true,
            passes_template: true,
            rating: 'Buy',
          },
        ],
      });
    }

    if (/^\/v1\/scans\/[^/]+\/filter-options$/.test(path) && method === 'GET') {
      return jsonResponse(route, {
        ibd_industries: ['Semiconductors'],
        gics_sectors: ['Technology'],
        ratings: ['Buy'],
      });
    }

    if (path === '/v1/themes/rankings' && method === 'GET') {
      return jsonResponse(route, {
        date: '2026-04-09',
        total_themes: 1,
        rankings: [
          {
            theme_cluster_id: 1,
            theme: 'AI Infrastructure',
            rank: 1,
            momentum_score: 84,
            mention_velocity: 1.7,
            mentions_7d: 18,
            basket_rs_vs_spy: 72,
            basket_return_1w: 3.8,
            pct_above_50ma: 78,
            num_constituents: 4,
            top_tickers: ['NVDA', 'AVGO'],
            status: 'trending',
            first_seen: '2026-03-01',
          },
        ],
      });
    }

    if (path === '/v1/themes/taxonomy/l1' && method === 'GET') {
      return jsonResponse(route, {
        rankings: [
          {
            id: 100,
            display_name: 'Semiconductors',
            rank: 1,
            momentum_score: 84,
          },
        ],
      });
    }

    if (path === '/v1/themes/taxonomy/categories' && method === 'GET') {
      return jsonResponse(route, { categories: [] });
    }

    if (path === '/v1/themes/emerging' && method === 'GET') {
      return jsonResponse(route, {
        count: 1,
        themes: [{ theme: 'AI Infrastructure', velocity: 1.7, mentions_7d: 18 }],
      });
    }

    if (path === '/v1/themes/alerts' && method === 'GET') {
      return jsonResponse(route, {
        total: 1,
        unread: 1,
        alerts: [
          {
            id: 11,
            title: 'AI breakout',
            alert_type: 'breakout',
            severity: 'warning',
            is_read: false,
          },
        ],
      });
    }

    if (path === '/v1/themes/merge-suggestions' && method === 'GET') {
      return jsonResponse(route, {
        total: 1,
        suggestions: [
          {
            id: 21,
            source_theme_id: 2,
            source_theme_name: 'AI Infra',
            source_aliases: ['AI Infra'],
            target_theme_id: 1,
            target_theme_name: 'AI Infrastructure',
            target_aliases: ['AI Infrastructure'],
            similarity_score: 0.93,
            llm_confidence: 0.91,
            relationship_type: 'duplicate',
            reasoning: 'Alias overlap',
            suggested_name: 'AI Infrastructure',
            status: 'pending',
            created_at: '2026-04-09T00:00:00Z',
          },
        ],
      });
    }

    if (path === '/v1/themes/candidates/queue' && method === 'GET') {
      return jsonResponse(route, {
        total: 1,
        items: [
          {
            theme_cluster_id: 31,
            theme_name: 'Edge AI',
            theme_display_name: 'Edge AI',
            confidence_band: '0.70-0.84',
            avg_confidence_30d: 0.78,
            mentions_7d: 7,
            source_diversity_7d: 2,
            persistence_days_7d: 4,
            queue_reason: 'candidate_review_pending',
          },
        ],
        confidence_bands: [{ band: '0.70-0.84', count: 1 }],
      });
    }

    if (path === '/v1/themes/pipeline/failed-count' && method === 'GET') {
      return jsonResponse(route, { failed_count: 0 });
    }

    if (path === '/v1/themes/pipeline/observability' && method === 'GET') {
      return jsonResponse(route, { status: 'healthy' });
    }

    if (path === '/v1/filter-presets' && method === 'GET') {
      return jsonResponse(route, { presets: [] });
    }

    if (/^\/v1\/stocks\/[^/]+\/history$/.test(path) && method === 'GET') {
      return jsonResponse(route, []);
    }

    if (path === '/v1/stocks/history/batch' && method === 'POST') {
      return jsonResponse(route, { data: {}, missing: [] });
    }

    throw new Error(`Unhandled smoke mock for ${method} ${path}`);
  });

  await page.goto('/');
  await ensureLoginVisible(page);

  await page.getByLabel('Server password').fill('smoke-password');
  await page.getByRole('button', { name: 'Sign in' }).click();

  await page.goto('/chatbot');
  await expect(page.getByRole('heading', { name: 'Assistant' })).toBeVisible();
  await expect(page.getByText('Hermes online')).toBeVisible();
  await page.getByPlaceholder(/ask about scans, themes, breadth, symbols/i).fill(
    'Compare StockScreenClaude signals with current web context for NVDA.'
  );
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.getByText(/NVDA remains the strongest internal AI theme candidate/i)).toBeVisible();
  await page.getByText('Sources (1)').click();
  await expect(page.getByText('Feature run snapshot')).toBeVisible();

  await page.goto('/scan');
  await expect(page.getByLabel('Open assistant')).toBeVisible();
  await page.getByLabel('Open assistant').click();
  await expect(page.getByText(/NVDA remains the strongest internal AI theme candidate/i)).toBeVisible();
  await page.getByRole('button', { name: 'Add tickers to watchlist' }).click();
  await expect(page.getByText('1 symbol can be added to Leaders.')).toBeVisible();
  await page.getByRole('button', { name: 'Confirm add' }).click();
  await expect(page.getByText('1 symbol can be added to Leaders.')).not.toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Scan' })).toBeVisible();
  await page.getByRole('button', { name: 'Scan' }).click();
  await expect(page.getByText(/Results:\s*1 stocks/i)).toBeVisible();

  await page.goto('/themes');
  await expect(page.getByText('Theme Discovery')).toBeVisible();
  await page.getByRole('button', { name: 'Review' }).click();
  await expect(page.getByRole('heading', { name: 'Theme Review' }).first()).toBeVisible();

  expireNextScanRequest = true;
  await page.goto('/scan');
  await page.getByRole('button', { name: 'Scan' }).click();
  await ensureLoginVisible(page);

  await page.getByLabel('Server password').fill('smoke-password');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.goto('/scan');
  await expect(page.getByRole('button', { name: 'Scan' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Assistant' })).toBeVisible();
});
