import { expect, test } from '@playwright/test';

import { installSocialSignalFixtures } from './socialSignalFixtures';

test.describe('Social Signals', () => {
  test('signed-in queue supports discovery, evidence, actions, ranking and market controls', async ({ page }) => {
    const state = await installSocialSignalFixtures(page);
    await page.goto('/');

    await expect(page.getByText('Social Signals').first()).toBeVisible();
    await expect(page.getByText('SMCI')).toHaveCount(0); // daily card is capped at five
    await page.getByRole('button', { name: 'Open Social Signals' }).click();
    await expect(page.getByRole('heading', { name: 'Social Signal Queue' })).toBeVisible();
    await expect(page.getByRole('table', { name: 'Ranked Social Signals' }).locator('tbody tr').first()).toContainText('AMD');

    await page.getByRole('button', { name: 'Pure Social' }).click();
    await expect(page.getByRole('table', { name: 'Ranked Social Signals' }).locator('tbody tr').first()).toContainText('NVDA');
    await page.getByRole('button', { name: 'All Signals' }).click();
    await expect(page.getByRole('table', { name: 'Context' })).toContainText('SPY');
    await expect(page.getByRole('table', { name: 'Needs resolution' })).toContainText('$MYSTERY');

    await page.getByRole('table', { name: 'Ranked Social Signals' }).locator('tbody tr').first().click();
    const evidence = page.getByRole('dialog', { name: 'Social evidence' });
    await expect(evidence).toContainText('Why it surfaced');
    await expect(evidence.getByText(/Fixture evidence/)).toHaveCount(3);
    await expect(evidence.getByRole('link', { name: 'Open on X' }).first()).toHaveAttribute('href', /x\.com/);
    await expect(evidence.getByRole('button', { name: 'Open chart & setup' })).toBeVisible();

    await evidence.getByLabel('Add to watchlist').click();
    await page.getByRole('menuitem', { name: /Leaders/ }).click();
    await expect.poll(() => state.watchlistAdds.length).toBe(1);
    await page.keyboard.press('Escape');
    await evidence.getByRole('button', { name: 'Close' }).click();
    await expect(evidence).toBeHidden();

    await page.getByLabel('Market selector').click();
    await page.getByRole('option', { name: /Japan/ }).click();
    await expect(page).toHaveURL(/market=JP/);
    await page.getByRole('button', { name: 'Send visible to Scan' }).first().click();
    await expect(page).toHaveURL(/\/scan\?.*market=JP.*symbols=/);
  });

  test('Theme pulse and admin source lifecycle stay fixture-only', async ({ page }) => {
    await installSocialSignalFixtures(page);
    await page.goto('/themes');
    await expect(page.getByRole('heading', { name: 'Social Pulse' })).toBeVisible();
    await expect(page.getByText('AI Infrastructure')).toBeVisible();
    await expect(page.getByText(/Social strength 91/)).toBeVisible();

    await page.goto('/operations');
    await page.getByLabel('Social admin key').fill('fixture-admin');
    await page.getByRole('button', { name: 'Load Social administration' }).click();
    await expect(page.getByText('Social Sources')).toBeVisible();

    await page.getByLabel('List name').fill('Japan Momentum');
    await page.getByLabel('List ID or URL').fill('3000000000000000000');
    await page.getByRole('button', { name: 'Add pending list' }).click();
    await expect(page.getByLabel('Rename Japan Momentum')).toBeVisible();
    await page.getByRole('button', { name: 'Test Japan Momentum' }).click();
    await expect(page.getByText(/official passed · 5\/5/)).toBeVisible();
    await page.getByRole('button', { name: 'Enable Japan Momentum' }).click();
    await expect(page.getByText('enabled').last()).toBeVisible();

    await page.getByLabel('Rename Japan Momentum').locator('input').fill('Japan Growth');
    await page.getByRole('button', { name: 'Rename' }).last().click();
    await expect(page.getByLabel('Rename Japan Growth')).toBeVisible();
    await page.getByRole('button', { name: 'Disable' }).last().click();
    await expect(page.getByText('disabled').last()).toBeVisible();
    page.once('dialog', (dialog) => dialog.accept());
    await page.getByRole('button', { name: 'Archive' }).last().click();
    await expect(page.getByLabel('Rename Japan Growth')).toHaveCount(0);

    await page.getByRole('button', { name: 'Disable' }).first().click();
    await expect(page.getByText('minimum_two_enabled')).toBeVisible();
    await expect(page.getByText(/created ·/).first()).toBeVisible();

    await page.getByRole('button', { name: 'Refresh Social Signals' }).click();
    await expect(page.getByText('Social refresh queued.')).toBeVisible();
    await page.getByRole('button', { name: 'Refresh Social Signals' }).click();
    await expect(page.getByText(/cooling down.*900 seconds/i)).toBeVisible();
  });
});
