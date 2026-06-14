// Record the hero product-tour video of the live full app, then (via the
// companion shell script) convert WebM -> GIF with ffmpeg + gifski.
//
// Storyboard: Daily Snapshot -> Scan (drill into an individual stock detail +
// keyboard walk) -> Breadth -> Groups (ranked table + RRG rotation view).
//
// Auth: reads SERVER_AUTH_PASSWORD from repo-root .env, POSTs login so the
// session cookie rides in the Playwright context. Override host with SITE_URL.
import { chromium } from 'playwright';
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');
const OUT_DIR = path.join(ROOT, '.tmp', 'scan-capture');
const BASE = process.env.SITE_URL || 'https://openclaws-macbook-pro.mink-company.ts.net';
const VIEWPORT = { width: 1440, height: 900 };

mkdirSync(OUT_DIR, { recursive: true });
for (const f of readdirSync(OUT_DIR)) if (f.endsWith('.webm')) rmSync(path.join(OUT_DIR, f));

const readPassword = () => {
  const m = readFileSync(path.join(ROOT, '.env'), 'utf8').match(/^SERVER_AUTH_PASSWORD=(.*)$/m);
  if (!m) throw new Error('SERVER_AUTH_PASSWORD not found in .env');
  return m[1].trim().replace(/^["']|["']$/g, '');
};

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: VIEWPORT,
  deviceScaleFactor: 1, // 1x keeps the video/GIF light
  colorScheme: 'dark',
  recordVideo: { dir: OUT_DIR, size: VIEWPORT },
});
await context.request.post(`${BASE}/api/v1/auth/login`, {
  headers: { 'Content-Type': 'application/json' },
  data: { password: readPassword() },
});
// The dashboard does a slow cold data load (~15s of spinner). Warm the server
// cache on a throwaway page first (its own video file is discarded), so the
// recorded tour renders content immediately instead of spinners.
const warm = await context.newPage();
for (const r of ['/', '/scan', '/breadth', '/groups']) {
  await warm.goto(`${BASE}${r}`, { waitUntil: 'domcontentloaded' }).catch(() => {});
  await warm.waitForTimeout(r === '/' ? 12000 : 5000);
}
await warm.close();

const page = await context.newPage();
const dwell = (ms) => page.waitForTimeout(ms);

// This SPA holds long-lived connections, so 'networkidle' never fires — wait on
// each page's signature content instead. Navigate via the top-nav (SPA
// transition, no full reload); fall back to a hard goto if the link is missing.
async function navTo(name, route, sig) {
  const link = page.getByRole('link', { name: new RegExp(`^${name}$`, 'i') }).first();
  const btn = page.getByRole('button', { name: new RegExp(`^${name}$`, 'i') }).first();
  if (await link.count().catch(() => 0)) await link.click().catch(() => {});
  else if (await btn.count().catch(() => 0)) await btn.click().catch(() => {});
  else await page.goto(`${BASE}${route}`, { waitUntil: 'domcontentloaded' });
  if (sig) await page.waitForSelector(sig, { timeout: 20000 }).catch(() => {});
}

try {
  // 1. Daily Snapshot — the at-a-glance home dashboard.
  await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('text=TOP SCAN CANDIDATES', { timeout: 25000 }).catch(() => {});
  await dwell(2600);
  await page.mouse.wheel(0, 260); await dwell(900);
  await page.mouse.wheel(0, -260); await dwell(600);

  // 2. Scan — results table, then drill into an individual stock's detail.
  await navTo('Scan', '/scan', 'table tbody tr');
  await dwell(2000);
  // Open the first result's inline detail (scores + TradingView-style chart).
  const firstRowBtn = page.locator('table tbody tr').first().locator('button').first();
  await firstRowBtn.hover().catch(() => {});
  await dwell(300);
  await firstRowBtn.click().catch(() => {});
  await page.waitForSelector('text=Space: Next Stock', { timeout: 10000 }).catch(() => {});
  await dwell(2400);
  // Walk to the next couple of stocks with the keyboard shortcut.
  for (let i = 0; i < 2; i++) { await page.keyboard.press('Space'); await dwell(1500); }
  await page.keyboard.press('Escape');
  await dwell(800);

  // 3. Breadth — advance/decline with SPY overlay, 3M range.
  await navTo('Breadth', '/breadth', 'text=Market Breadth');
  await dwell(1600);
  const threeM = page.getByRole('button', { name: '3M', exact: true });
  if (await threeM.isVisible().catch(() => false)) { await threeM.click(); await dwell(2200); }
  else { await dwell(1500); }

  // 4. Groups — ranked industry groups, then the RRG rotation view.
  await navTo('Groups', '/groups', 'table');
  await dwell(2000);
  await page.getByRole('button', { name: 'RRG', exact: true }).click().catch(() => {});
  await dwell(2400);
  await page.getByRole('button', { name: 'Sectors', exact: true }).first().click().catch(() => {});
  await dwell(1400);
  const labels = page.getByRole('checkbox', { name: /Labels/i }).first();
  if (await labels.count().catch(() => 0)) { await labels.check().catch(() => labels.click()); }
  await dwell(2600);
} finally {
  const videoHandle = page.video();
  await context.close();
  await browser.close();
  if (videoHandle) {
    const produced = await videoHandle.path();
    const dst = path.join(OUT_DIR, 'video.webm');
    if (produced && produced !== dst) { if (existsSync(dst)) rmSync(dst); renameSync(produced, dst); }
    console.log('✓ saved', dst, `(${(statSync(dst).size / 1024 / 1024).toFixed(1)} MB)`);
  } else {
    const webms = readdirSync(OUT_DIR).filter((f) => f.endsWith('.webm'));
    if (webms.length) { renameSync(path.join(OUT_DIR, webms[0]), path.join(OUT_DIR, 'video.webm')); }
    else { console.error('✗ no webm produced'); process.exit(1); }
  }
}
