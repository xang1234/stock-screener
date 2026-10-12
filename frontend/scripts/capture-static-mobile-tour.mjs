// Record a phone-sized tour of the static site for the README (shown beside the
// desktop hero GIF). Encoded by `TOUR=mobile bash capture-static-site-tour.sh`.
//
// Storyboard: Daily Snapshot -> Scan (stock cards, open one) -> Breadth ->
// Groups (RRG), navigating with the bottom tab bar.
import { chromium, devices } from 'playwright';
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, '..', '..', '.tmp', 'hero-capture');
const SITE_URL = process.env.SITE_URL || 'https://xang1234.github.io/stock-screener';
const PHONE = devices['iPhone 13']; // 390x664 CSS px viewport (screen minus browser chrome)
// Video records CSS pixels; a larger size only pads the frame with grey.
const VIDEO = PHONE.viewport;

mkdirSync(OUT_DIR, { recursive: true });
for (const f of readdirSync(OUT_DIR)) if (f.endsWith('.webm')) rmSync(path.join(OUT_DIR, f));

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ ...PHONE, colorScheme: 'dark', recordVideo: { dir: OUT_DIR, size: VIDEO } });
const page = await context.newPage();
const dwell = (ms) => page.waitForTimeout(ms);
const settle = () => page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
// The bottom tab bar is the phone's navigation; a missing tab should fail the run.
const tab = async (name) => { await page.getByRole('link', { name, exact: true }).last().click(); await settle(); };
const scroll = async (dy) => { await page.mouse.wheel(0, dy); await dwell(900); };

try {
  // 1. Daily Snapshot.
  await page.goto(`${SITE_URL}/#/`, { waitUntil: 'domcontentloaded' });
  await settle();
  await dwell(2500);
  await scroll(520);
  await scroll(-520);

  // 2. Scan — the card list, then open a stock's chart.
  await tab('Scan');
  await page.locator('[data-testid="card-sort-note"], button[aria-label^="Open "]').first().waitFor({ timeout: 15000 });
  await dwell(1200);
  await scroll(560);
  await dwell(800);
  await page.locator('button[aria-label^="Open "][aria-label$=" chart"]').first().click();
  await dwell(3200);
  await page.keyboard.press('Escape');
  await dwell(900);

  // 3. Breadth — health bars, then the movers chart.
  await tab('Breadth');
  await dwell(1800);
  await scroll(620);
  await dwell(1200);

  // 4. Groups — ranked table, then the RRG rotation view.
  await tab('Groups');
  await dwell(1600);
  await page.getByRole('tab', { name: 'RRG', exact: true }).click();
  await dwell(1600);
  await page.getByRole('button', { name: 'Sectors', exact: true }).first().click();
  const labels = page.getByRole('checkbox', { name: /Labels/i }).first();
  if (await labels.count().catch(() => 0)) await labels.check().catch(() => {});
  await dwell(800);
  // On a phone the plot sits below the RRG controls.
  await scroll(380);
  await dwell(2600);
} finally {
  const video = page.video();
  await context.close();
  await browser.close();
  const dst = path.join(OUT_DIR, 'video.webm');
  const produced = video ? await video.path() : null;
  if (produced && produced !== dst) { if (existsSync(dst)) rmSync(dst); renameSync(produced, dst); }
  if (existsSync(dst)) console.log('✓ saved', dst, `(${(statSync(dst).size / 1024 / 1024).toFixed(1)} MB)`);
}
