// Capture refreshed README screenshots from the live full app.
//
// Auth: reads SERVER_AUTH_PASSWORD from the repo-root .env and POSTs the login
// so the session cookie rides along in the Playwright context (password is never
// logged). Override the target host with SITE_URL.
//
// Usage (from frontend/):
//   node scripts/capture-live-readme.mjs
//
// Output: PNG/JPEG files in .tmp/readme-shots/ for review, then promote the good
// ones into docs/screenshots/.
import { chromium } from 'playwright';
import { readFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');
const OUT = path.join(ROOT, '.tmp', 'readme-shots');
mkdirSync(OUT, { recursive: true });

const BASE = process.env.SITE_URL || 'https://openclaws-macbook-pro.mink-company.ts.net';
const readPassword = () => {
  const env = readFileSync(path.join(ROOT, '.env'), 'utf8');
  const m = env.match(/^SERVER_AUTH_PASSWORD=(.*)$/m);
  if (!m) throw new Error('SERVER_AUTH_PASSWORD not found in .env');
  return m[1].trim().replace(/^["']|["']$/g, '');
};

const VIEWPORT = { width: 1440, height: 900 };
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: VIEWPORT,
  deviceScaleFactor: 2, // retina-crisp PNGs
  colorScheme: 'dark',
});
const login = await context.request.post(`${BASE}/api/v1/auth/login`, {
  headers: { 'Content-Type': 'application/json' },
  data: { password: readPassword() },
});
console.log('login:', login.status());

const page = await context.newPage();
const dwell = (ms) => page.waitForTimeout(ms);
const png = (n) => page.screenshot({ path: path.join(OUT, `${n}.png`) });
const jpg = (n) => page.screenshot({ path: path.join(OUT, `${n}.jpg`), type: 'jpeg', quality: 92 });

async function goto(route, settle = 4500) {
  await page.goto(`${BASE}${route}`, { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
  await dwell(settle);
}

// A failed step keeps going (so one bad page doesn't cost the rest) but the
// run exits nonzero, so a missing image can't pass for a complete refresh.
let failed = 0;
async function step(name, fn) {
  try { await fn(); console.log(`✓ ${name}`); }
  catch (e) { failed += 1; console.error(`✗ ${name}:`, e.message); }
}

async function pickMarket(label) {
  await page.getByRole('combobox', { name: 'Market selector' }).click();
  await page.getByRole('option', { name: new RegExp(label) }).click();
  await dwell(4000);
}

// 1. Daily Snapshot dashboard (NEW hero)
await step('daily-snapshot', async () => {
  await goto('/', 5000);
  await png('daily-snapshot');
});

// 1b. Market Health & Exposure card — element crop from the same page
await step('health-exposure', async () => {
  const card = page.getByText('Market Health & Exposure', { exact: true })
    .locator('xpath=ancestor::div[contains(@class,"MuiPaper-root")][1]');
  if (await card.getByText(/No exposure data yet/).count()) throw new Error('no exposure data on this stack');
  await card.screenshot({ path: path.join(OUT, 'health-exposure.jpg'), type: 'jpeg', quality: 92 });
});

// 2. Scan results — the default sort is already Composite descending (clicking
// the header would flip it to ascending and lead with the weakest names).
// Theme chips are only populated for the Asian universes in the reference
// deployment: set SCAN_RESULTS_MARKET to the option label (e.g. Japan) there.
const SCAN_MARKET = process.env.SCAN_RESULTS_MARKET;
await step('scan-results', async () => {
  await goto('/scan', 5000);
  if (SCAN_MARKET) await pickMarket(SCAN_MARKET);
  try {
    await dwell(1500);
    await png('scan-results');
  } finally {
    // The market choice persists; later steps (Options) need US.
    if (SCAN_MARKET) await pickMarket('United States');
  }
});

// 3. Market selector — open the header market dropdown (5 markets)
await step('market-selector', async () => {
  await goto('/scan', 4000);
  await page.getByRole('combobox', { name: 'Market selector' }).click();
  await dwell(1100);
  await jpg('market-selector');
});

// 4. Breadth — 3M range
await step('breadth-chart', async () => {
  await goto('/breadth', 4500);
  const threeM = page.getByRole('button', { name: '3M', exact: true });
  if (await threeM.isVisible().catch(() => false)) { await threeM.click(); await dwell(2200); }
  await png('breadth-chart');
});

// 5. Group rankings
await step('group-rankings', async () => {
  await goto('/groups', 5000);
  await png('group-rankings');
});

// 6. RRG — capture both GROUPS (default) and SECTORS scope for comparison
await step('rrg', async () => {
  await goto('/groups', 4000);
  await page.getByRole('tab', { name: 'RRG', exact: true }).click();
  await dwell(4500);
  await png('rrg-groups');
  const sectors = page.getByRole('button', { name: 'Sectors', exact: true }).first();
  if (await sectors.count()) {
    await sectors.click(); await dwell(3000);
    await page.getByRole('checkbox', { name: /Labels/i }).first().check().catch(() => {});
    await dwell(1500); await png('rrg-sectors');
  }
});

// 6b. Options Command Center (US only; the nav hides it for other markets)
await step('options-command-center', async () => {
  await goto('/options', 6000);
  // The route redirects to / when options are unavailable; don't save that as Options.
  await page.getByRole('heading', { name: 'Options Command Center', level: 1 }).waitFor({ timeout: 5000 });
  await png('options-command-center');
});

// 7. Watchlist table
await step('watchlist-table', async () => {
  await goto('/', 3500);
  await page.locator('text=Watchlists').first().click();
  await dwell(4000);
  await png('watchlist-table');
});

await context.close();
await browser.close();
console.log('done ->', OUT);
if (failed) {
  console.error(`${failed} step(s) failed`);
  process.exitCode = 1;
}
