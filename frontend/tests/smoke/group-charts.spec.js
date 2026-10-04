import { expect, test } from '@playwright/test';

// Group detail -> Charts with 40 constituents (#496). Uses the real chart
// library inside the real scrollable dialog; only the API is faked.

const GROUP = 'Fixture Group';
const SYMBOLS = Array.from({ length: 40 }, (_, i) => `FX${String(i).padStart(2, '0')}`);

function bars(seed) {
  const days = [];
  const day = new Date(Date.UTC(2026, 3, 1));
  while (days.length < 126) {
    if (day.getUTCDay() % 6 !== 0) {
      const close = 100 + seed + Math.sin(days.length / 7) * 5;
      days.push({
        date: day.toISOString().slice(0, 10),
        open: close - 1, high: close + 2, low: close - 2, close, volume: 1_000_000 + days.length,
      });
    }
    day.setUTCDate(day.getUTCDate() + 1);
  }
  return days;
}

const rankingRow = {
  industry_group: GROUP, date: '2026-09-30', rank: 1, avg_rs_rating: 90, avg_rs_rating_1d: 89,
  avg_rs_rating_1w: 88, avg_rs_rating_1m: 85, avg_rs_rating_3m: 80, avg_rs_rating_6m: 75,
  median_rs_rating: 90, weighted_avg_rs_rating: 90, rs_std_dev: 3, num_stocks: 40,
  num_stocks_rs_above_80: 30, pct_rs_above_80: 75, top_symbol: 'FX00', top_rs_rating: 99,
  rank_change_1w: 1, rank_change_1m: 2, rank_change_3m: null, rank_change_6m: null,
};

async function installGroupChartFixtures(page) {
  const unexpected = [];
  const batchRequests = [];
  const json = (route, body, status = 200) =>
    route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  await page.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.abort());
  await page.route('**/api/v1/**', async (route) => {
    const { pathname } = new URL(route.request().url());
    if (pathname.endsWith('/app-capabilities')) {
      return json(route, {
        features: { tasks: false, themes: false, chatbot: false },
        auth: { required: false, authenticated: true }, bootstrap_required: false,
        primary_market: 'US', enabled_markets: ['US'], supported_markets: ['US'],
        ui_snapshots: { enabled: false, groups: false }, api_base_path: '/api',
        market_catalog: {
          version: 'test',
          markets: [{ code: 'US', label: 'United States', capabilities: { group_rankings: true, rrg_scopes: [] } }],
        },
      });
    }
    if (pathname.endsWith('/runtime/activity')) {
      return json(route, {
        bootstrap: { state: 'ready', app_ready: true, primary_market: 'US', enabled_markets: ['US'] },
        summary: { status: 'idle', active_markets: [] }, markets: [],
      });
    }
    if (pathname.endsWith('/groups/rankings/current')) {
      return json(route, { date: '2026-09-30', total_groups: 1, market_scope: 'US', rankings: [rankingRow] });
    }
    if (pathname.endsWith('/groups/rankings/movers')) {
      return json(route, { period: '1w', market_scope: 'US', gainers: [rankingRow], losers: [] });
    }
    if (pathname.endsWith('/groups/rankings/detail')) {
      return json(route, {
        industry_group: GROUP, current_rank: 1, current_avg_rs: 90, num_stocks: 40,
        top_symbol: 'FX00', top_rs_rating: 99, history: [],
        stocks: SYMBOLS.map((symbol) => ({ symbol })),
      });
    }
    if (pathname.endsWith('/stocks/history/batch')) {
      const { symbols } = route.request().postDataJSON();
      batchRequests.push(symbols.length);
      return json(route, {
        data: Object.fromEntries(symbols.map((symbol, i) => [symbol, bars(i)])),
        missing: [],
      });
    }
    if (pathname.includes('/groups/rrg')) return json(route, { detail: 'not in fixture' }, 503);
    unexpected.push(pathname);
    return json(route, { detail: `unexpected ${pathname}` }, 500);
  });
  return { unexpected, batchRequests };
}

// One `.tv-lightweight-charts` root per createChart() call, whatever its canvas count.
const chartInstances = (page) => page.locator('[role="dialog"] .tv-lightweight-charts').count();

async function openGroupCharts(page) {
  await page.goto('/groups');
  await page.getByText(GROUP, { exact: true }).first().click();
  await page.getByRole('tab', { name: 'Charts (40)' }).click();
  await expect(page.getByTestId('group-chart-cell')).toHaveCount(40);
  await expect.poll(() => chartInstances(page)).toBeGreaterThan(0);
}

// Cells overlapping the dialog content's visible box (the scroll root).
const visibleCells = (page) =>
  page.locator('.MuiDialogContent-root').evaluate((root) => {
    const box = root.getBoundingClientRect();
    return [...root.querySelectorAll('[data-testid="group-chart-cell"]')].filter((cell) => {
      const rect = cell.getBoundingClientRect();
      return rect.bottom > box.top && rect.top < box.bottom;
    }).length;
  });

const scrollDialog = (page, to) =>
  page.locator('.MuiDialogContent-root').evaluate((el, target) => {
    el.scrollTop = target === 'bottom' ? el.scrollHeight : 0;
  }, to);

for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
  test(`group charts build near the viewport only (${viewport.width}px)`, async ({ page }, testInfo) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const { unexpected, batchRequests } = await installGroupChartFixtures(page);
    await page.setViewportSize(viewport);

    await openGroupCharts(page);
    const initial = await chartInstances(page);
    // Bounded well below 40, and more than the visible cells: the prewarm row
    // below exists only if the observer is rooted at the dialog's scroller
    // (rooted at the viewport, the dialog clips it and nothing is prewarmed).
    expect(initial).toBeLessThanOrEqual(viewport.width > 900 ? 10 : 6);
    expect(initial).toBeGreaterThan(await visibleCells(page));

    await scrollDialog(page, 'bottom');
    await expect(
      page.getByTestId('group-chart-cell').last().locator('.tv-lightweight-charts'),
    ).toHaveCount(1);
    const afterScroll = await chartInstances(page);
    expect(afterScroll).toBeGreaterThan(initial);

    await scrollDialog(page, 'top');
    // Revealed charts stay mounted; never more than the 40-symbol cap.
    expect(await chartInstances(page)).toBe(afterScroll);
    expect(afterScroll).toBeLessThanOrEqual(40);

    await page.screenshot({ path: testInfo.outputPath(`group-charts-${viewport.width}.png`) });
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toBeHidden();
    expect(await page.locator('.tv-lightweight-charts').count()).toBe(0);

    // Reopen within the freshness window: cached batch, fresh observations.
    await page.getByText(GROUP, { exact: true }).first().click();
    await page.getByRole('tab', { name: 'Charts (40)' }).click();
    await expect.poll(() => chartInstances(page)).toBeGreaterThan(0);
    expect(await chartInstances(page)).toBeLessThanOrEqual(initial);

    // Keyboard scrolling from the focused tab reaches and builds the last chart.
    await expect(page.getByRole('tab', { name: 'Charts (40)' })).toBeFocused();
    const lastChart = page.getByTestId('group-chart-cell').last().locator('.tv-lightweight-charts');
    // Mobile is one column of 40 rows (~14 pages); stop once the last chart exists.
    for (let i = 0; i < 30 && (await lastChart.count()) === 0; i += 1) {
      await page.keyboard.press('PageDown');
    }
    await expect(lastChart).toHaveCount(1);

    expect(batchRequests).toEqual([40]);
    expect(unexpected).toEqual([]);
    expect(errors).toEqual([]);
    console.log('group charts', JSON.stringify({ viewport, initial, afterScroll }));
  });
}
