// Capture the current frontend against an authenticated local backend.
// Optional snapshot JSON ({matrix: ...}) supplies published Matrix data when
// previewing frontend changes before rebuilding the backend. No data is invented.
// Usage: node scripts/capture-group-readme.mjs /path/to/published-snapshot.json
import { chromium, expect } from '@playwright/test';
import { readFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = new URL('../../', import.meta.url);
const output = new URL('.tmp/group-readme/', root);
const frontend = process.env.GROUPS_SCREENSHOT_URL || 'http://127.0.0.1:4175';
const backend = 'http://localhost';
const snapshot = process.argv[2] ? JSON.parse(await readFile(process.argv[2], 'utf8')) : null;
const env = await readFile(new URL('.env', root), 'utf8');
const password = env.match(/^SERVER_AUTH_PASSWORD=(.*)$/m)?.[1].trim().replace(/^["']|["']$/g, '');
if (!password) throw new Error('Local SERVER_AUTH_PASSWORD is required for screenshot capture');
await mkdir(output, {recursive:true});
console.log('Opening screenshot browser');
const browser = await chromium.launch({headless:true, channel:'chrome'});
try {
  const context = await browser.newContext({viewport:{width:1600,height:1100}, deviceScaleFactor:1.5, colorScheme:'dark'});
  console.log('Authenticating with local backend');
  const login = await context.request.post(`${backend}/api/v1/auth/login`, {data:{password}});
  if (!login.ok()) throw new Error(`Local login failed (${login.status()})`);
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error=>errors.push(error.message));
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/groups/matrix') && snapshot?.matrix) {
      await route.fulfill({json:snapshot.matrix});
    } else {
      if (route.request().method() !== 'GET') throw new Error('Screenshot capture only allows read-only API requests');
      await route.fulfill({response:await context.request.get(`${backend}${url.pathname}${url.search}`)});
    }
  });
  const shot = async name => {
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.mouse.move(0,0);
    await page.screenshot({path:fileURLToPath(new URL(`${name}.png`,output)), animations:'disabled'});
  };
  console.log('Loading Groups page');
  await page.goto(`${frontend}/groups`);
  await expect(page.getByRole('columnheader', {name:'RS',exact:true})).toBeVisible({timeout:60000});
  // Let the header status chip settle from "Checking activity".
  await page.getByText('Markets ready').waitFor({timeout:15000}).catch(() => {});
  await shot('group-rs-table');
  console.log('Captured Group RS table');
  await page.getByRole('tab', {name:'Matrix',exact:true}).click();
  await expect(page.getByRole('heading', {name:'US Stock Matrix'})).toBeVisible({timeout:60000});
  await page.getByRole('combobox', {name:/^Sector/}).click();
  await page.getByRole('option', {name:'Technology',exact:true}).click();
  await page.getByRole('button', {name:'Unknown cap',exact:true}).click();
  await page.getByRole('combobox', {name:/^Color /}).click();
  await page.getByRole('option', {name:'1-Week Change',exact:true}).click();
  await expect(page.locator('[data-matrix-stock]:visible').first()).toBeVisible();
  // Scroll the actual grid to the populated software rows, keeping column headers visible.
  const grid = page.getByRole('region', {name:'Stock matrix grid, industries by market cap'});
  await grid.evaluate(element => { element.scrollTop = 38 + 14 * 190; });
  await expect(grid.getByRole('button', {name:'Computer Sftwr-Enterprse',exact:true})).toBeVisible();
  await shot('group-matrix-grid');
  console.log('Captured Grid');
  await page.getByRole('button', {name:'Clusters',exact:true}).click();
  await page.getByRole('combobox', {name:/^IBD industry/}).click();
  await page.getByRole('option', {name:'Computer Sftwr-Enterprse',exact:true}).click();
  await page.getByRole('combobox', {name:/^Color /}).click();
  await page.getByRole('option', {name:'1-Month Change',exact:true}).click();
  await expect(page.locator('.group-matrix-bubble-pack').first()).toBeVisible();
  await page.setViewportSize({width:1600,height:900});
  await shot('group-matrix-clusters');
  if (errors.length) throw new Error(errors.join('\n'));
  console.log(`Screenshots saved to ${fileURLToPath(output)}`);
} finally {
  await browser.close();
}
