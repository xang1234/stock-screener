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

async function step(name, fn) {
  try { await fn(); console.log(`✓ ${name}`); }
  catch (e) { console.error(`✗ ${name}:`, e.message); }
}

// 1. Daily Snapshot dashboard (NEW hero)
await step('daily-snapshot', async () => {
  await goto('/', 5000);
  await png('daily-snapshot');
});

// 2. Scan results — sort by Composite to surface classified names
await step('scan-results', async () => {
  await goto('/scan', 5000);
  // Click the COMP column header to sort by composite score (desc).
  const comp = page.getByRole('columnheader', { name: /COMP/i }).first();
  if (await comp.count()) { await comp.click().catch(() => {}); await dwell(900); }
  await dwell(1500);
  await png('scan-results');
});

// 3. Market selector — open the header market dropdown (5 markets)
await step('market-selector', async () => {
  await goto('/scan', 4000);
  const marketBtn = page.getByRole('combobox').filter({ hasText: 'US' }).first();
  await marketBtn.click();
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
  await page.getByRole('button', { name: 'RRG', exact: true }).click();
  await dwell(4500);
  await png('rrg-groups');
  const sectors = page.getByRole('button', { name: 'Sectors', exact: true }).first();
  if (await sectors.count()) { await sectors.click(); await dwell(4000); await png('rrg-sectors'); }
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
