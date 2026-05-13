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
  await dwell(400);
  await link.click();
  await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
}

try {
  console.log('→ Daily (#/)');
  await page.goto(`${SITE_URL}/#/`, { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
  await dwell(3500);

  await navTo('Scan', '/scan');
  await dwell(1600);
  await page.mouse.wheel(0, 400);
  await dwell(900);
  await page.mouse.wheel(0, -400);
  await dwell(900);

  await navTo('Breadth', '/breadth');
  await dwell(1600);
  const threeM = page.getByRole('button', { name: '3M', exact: true });
  if (await threeM.isVisible().catch(() => false)) {
    await threeM.click();
  }
  await dwell(2200);

  await navTo('Groups', '/groups');
  await dwell(1500);
  await page.mouse.wheel(0, 700);
  await dwell(1800);
  await page.mouse.wheel(0, 400);
  await dwell(700);
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
