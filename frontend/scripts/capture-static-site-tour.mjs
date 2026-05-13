import { chromium } from 'playwright';
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const OUT_DIR = path.join(REPO_ROOT, '.tmp', 'hero-capture');
const SITE_URL = process.env.SITE_URL || 'https://xang1234.github.io/stock-screener';
const VIEWPORT = { width: 1440, height: 900 };

if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });
for (const f of readdirSync(OUT_DIR)) {
  if (f.endsWith('.webm')) rmSync(path.join(OUT_DIR, f));
}

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: VIEWPORT,
  deviceScaleFactor: 2,
  colorScheme: 'dark',
  recordVideo: { dir: OUT_DIR, size: VIEWPORT },
});
const page = await context.newPage();

const dwell = (ms) => page.waitForTimeout(ms);

async function navTo(label, hashRoute) {
  console.log(`→ ${label}`);
  const link = page.locator(`a[href="#${hashRoute}"]`).first();
  await link.hover();
  await dwell(350);
  await link.click();
  await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
}

async function switchMarket(label) {
  console.log(`→ market: ${label}`);
  const selector = page.getByRole('combobox', { name: 'Static market selector' });
  await selector.click();
  await dwell(700);
  await page.getByRole('option', { name: label }).click();
  await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
}

try {
  // 1. Daily (US)
  console.log('→ Daily US (#/)');
  await page.goto(`${SITE_URL}/#/`, { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
  await dwell(3000);

  // 2. Scan (US) — preset filters + space-bar stock walk
  await navTo('Scan', '/scan');
  await page.getByRole('button', { name: 'All Stocks' }).waitFor({ timeout: 15000 });
  await dwell(1500);

  for (const preset of [/^Minervini/, /^CANSLIM/, /^VCP Setups/, /^All Stocks/]) {
    const btn = page.getByRole('button', { name: preset }).first();
    await btn.hover();
    await dwell(250);
    await btn.click();
    await dwell(1100);
  }

  const firstRowExpand = page.locator('table tbody button').first();
  await firstRowExpand.click();
  await page.waitForSelector('text=Space: Next Stock', { timeout: 10000 });
  await dwell(1500);
  for (let i = 0; i < 3; i++) {
    await page.keyboard.press('Space');
    await dwell(1100);
  }
  await page.keyboard.press('Escape');
  await dwell(500);

  // 3. Breadth (US)
  await navTo('Breadth', '/breadth');
  await dwell(1300);
  const threeM = page.getByRole('button', { name: '3M', exact: true });
  if (await threeM.isVisible().catch(() => false)) await threeM.click();
  await dwell(2000);

  // 4. Groups (US) — open group → Overview → Charts tab
  await navTo('Groups', '/groups');
  await page.getByRole('heading', { name: 'Current Rankings' }).waitFor({ timeout: 15000 });
  await dwell(1200);
  const firstGroupRow = page.locator('table').last().locator('tbody tr').first();
  await firstGroupRow.click();
  await page.getByRole('dialog').waitFor({ timeout: 10000 });
  await dwell(2200);
  await page.getByRole('tab', { name: 'Charts' }).click();
  await dwell(2500);
  await page.keyboard.press('Escape');
  await dwell(500);

  // 5. Back to Daily, then switch markets
  await navTo('Daily', '/');
  await dwell(1200);
  await switchMarket('🇯🇵 Japan');
  await dwell(2500);
  await switchMarket('🇨🇦 Canada');
  await dwell(2500);
} finally {
  const videoHandle = page.video();
  await context.close();
  await browser.close();

  if (videoHandle) {
    const produced = await videoHandle.path();
    const dst = path.join(OUT_DIR, 'video.webm');
    if (produced && produced !== dst) {
      if (existsSync(dst)) rmSync(dst);
      renameSync(produced, dst);
    }
    console.log('✓ saved', dst, `(${(statSync(dst).size / 1024 / 1024).toFixed(1)} MB)`);
  } else {
    const webms = readdirSync(OUT_DIR).filter((f) => f.endsWith('.webm'));
    if (webms.length) {
      webms.sort((a, b) => statSync(path.join(OUT_DIR, b)).mtimeMs - statSync(path.join(OUT_DIR, a)).mtimeMs);
      renameSync(path.join(OUT_DIR, webms[0]), path.join(OUT_DIR, 'video.webm'));
    } else {
      console.error('✗ no webm produced');
      process.exit(1);
    }
  }
}
