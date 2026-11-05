import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  allPendingScoreboardFixture,
  liveClosedAtFiveScoreboardFixture,
  nullBenchmarkScoreboardFixture,
  pendingArtifactScoreboardFixture,
  preFixOnlyScoreboardFixture,
  scoreboardFixture,
} from '../../test/fixtures/scoreboardFixtures';
import { renderWithProviders } from '../../test/renderWithProviders';
import { getDesignTokens } from '../../theme';
import { StaticMarketProvider } from '../StaticMarketContext';
import StaticLayout from '../StaticLayout';
import StaticScoreboardPage from './StaticScoreboardPage';

const ARTIFACT_TITLE = 'Recovered from retained build artifacts, not recorded live';
const PRE_FIX_TITLE = 'Recovered from retained build artifacts, not recorded live, from before the Stage calculation fix (#671)';
const ALL_PENDING = 'No window has closed yet. Nothing here is a result.';
const DARK_TEXT = getDesignTokens('dark').palette.text;

const manifest = {
  default_market: 'US',
  generation: 'g1',
  data_root: 'g/g1/',
  supported_markets: ['US', 'HK', 'JP', 'AU', 'TW', 'CN', 'DE'],
  unavailable_markets: [],
  markets: {
    US: { display_name: 'United States', assets: { scoreboard: { path: 'markets/us/scoreboard.json' } } },
    HK: { display_name: 'Hong Kong', assets: { scoreboard: { path: 'markets/hk/scoreboard.json' } } },
    JP: { display_name: 'Japan', assets: {} },
    AU: { display_name: 'Australia', assets: { scoreboard: { path: 'markets/au/scoreboard.json' } } },
    TW: { display_name: 'Taiwan', assets: { scoreboard: { path: 'markets/tw/scoreboard.json' } } },
    CN: { display_name: 'China', assets: { scoreboard: { path: 'markets/cn/scoreboard.json' } } },
    DE: { display_name: 'Germany', assets: { scoreboard: { path: 'markets/de/scoreboard.json' } } },
  },
};

const STATIC_FILES = {
  '/static-data/manifest.json': manifest,
  '/static-data/g/g1/markets/us/scoreboard.json': scoreboardFixture,
  '/static-data/g/g1/markets/hk/scoreboard.json': allPendingScoreboardFixture,
  '/static-data/g/g1/markets/au/scoreboard.json': nullBenchmarkScoreboardFixture,
  '/static-data/g/g1/markets/tw/scoreboard.json': preFixOnlyScoreboardFixture,
  '/static-data/g/g1/markets/cn/scoreboard.json': pendingArtifactScoreboardFixture,
  '/static-data/g/g1/markets/de/scoreboard.json': liveClosedAtFiveScoreboardFixture,
};

const respondWith = (body) => (body
  ? { ok: true, status: 200, json: async () => body }
  : { ok: false, status: 500, json: async () => ({}) });

const renderRoute = (entry) => renderWithProviders(
  <MemoryRouter initialEntries={[entry]}>
    <StaticMarketProvider supportedMarkets={manifest.supported_markets} defaultMarket="US">
      <StaticLayout><StaticScoreboardPage /></StaticLayout>
    </StaticMarketProvider>
  </MemoryRouter>,
);

const rowFor = (section, name) => within(section).getByText(name).closest('tr');
const sectionFor = (title) => screen.getByText(title).closest('section');
const scoreboardFetches = () => fetch.mock.calls.map(([url]) => url).filter((url) => url.includes('scoreboard'));
const sectionToggle = (title) => screen.getByRole('button', { name: title });
const openSection = (title) => {
  const toggle = sectionToggle(title);
  if (toggle.getAttribute('aria-expanded') !== 'true') fireEvent.click(toggle);
};
const pressedHorizon = () => within(screen.getByRole('group', { name: 'Horizon' }))
  .getByRole('button', { pressed: true }).textContent;
const switchMarket = async (name) => {
  fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Static market selector' }));
  fireEvent.click(await screen.findByRole('option', { name: new RegExp(name) }));
};

describe('StaticScoreboardPage', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(async (url) => respondWith(STATIC_FILES[url])));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('shows a pending live row with counts, dashes and no percent sign', async () => {
    renderRoute('/scoreboard');
    const live = (await screen.findByText('Recorded live')).closest('section');

    const row = rowFor(live, 'Minervini');
    expect(row).toHaveTextContent('0 closed · 0 dates · 357 pending · oldest 2 of 5 sessions');
    expect(row).not.toHaveTextContent('%');
    expect(within(row).getAllByText('-')).toHaveLength(4);
    expect(row).not.toHaveTextContent('Small sample');
    expect(within(live).getByText(ALL_PENDING)).toBeInTheDocument();
  });

  it('says a zero-match cohort has no picks rather than a pending result', async () => {
    renderRoute('/scoreboard');
    const live = (await screen.findByText('Recorded live')).closest('section');

    const row = rowFor(live, 'High Tight Flag');
    expect(row).toHaveTextContent('No picks');
    expect(row).not.toHaveTextContent('pending');
    expect(row).not.toHaveTextContent('%');
  });

  it('keeps the rows of a collapsed section hidden until it is opened', async () => {
    renderRoute('/scoreboard');
    const toggle = await screen.findByRole('button', { name: PRE_FIX_TITLE });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(rowFor(sectionFor(PRE_FIX_TITLE), 'VCP')).not.toBeVisible();

    openSection(PRE_FIX_TITLE);

    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(rowFor(sectionFor(PRE_FIX_TITLE), 'VCP')).toBeVisible();
  });

  it('changes the shown number when the horizon changes', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');
    openSection(PRE_FIX_TITLE);
    const section = sectionFor(PRE_FIX_TITLE);

    expect(rowFor(section, 'VCP')).toHaveTextContent('540 closed · 13 dates, effective 9.0 · 0 pending · 80 overlapping');
    expect(rowFor(section, 'VCP')).toHaveTextContent('+0.4%');

    fireEvent.click(screen.getByRole('button', { name: '20 sessions' }));

    expect(rowFor(section, 'VCP')).toHaveTextContent('26 closed · 6 dates, effective 4.8 · 300 pending · 294 overlapping');
    expect(rowFor(section, 'VCP')).toHaveTextContent('-2.3%');
    expect(rowFor(section, 'VCP')).not.toHaveTextContent('+0.4%');
  });

  it('renders the sample cell and the median, positive, beat and excess numbers at 5 sessions', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');
    openSection(ARTIFACT_TITLE);

    const row = rowFor(sectionFor(ARTIFACT_TITLE), 'Minervini');
    expect(within(row).getAllByRole('cell')[1].textContent)
      .toBe('1,298 closed · 12 dates, effective 8.4 · 40 pending · 3 no price · 100 overlapping · excess on 1,250');
    expect(within(row).getAllByRole('cell').slice(2).map((cell) => cell.textContent))
      .toEqual(['+0.9%', '57%', '48%', '+0.1%']);
    expect(row).not.toHaveTextContent('Small sample');
  });

  it('has Median, Positive, Beat benchmark and Median excess columns and no average', async () => {
    renderRoute('/scoreboard');
    const live = (await screen.findByText('Recorded live')).closest('section');

    expect(within(live).getAllByRole('columnheader').map((cell) => cell.textContent))
      .toEqual(['Screen', 'Sample', 'Median', 'Positive', 'Beat benchmark', 'Median excess']);
    expect(document.body).not.toHaveTextContent(/average/i);
  });

  it('keeps the screen name beside the numbers when a narrow table scrolls sideways', async () => {
    renderRoute('/scoreboard');
    const live = (await screen.findByText('Recorded live')).closest('section');

    const header = within(live).getByRole('columnheader', { name: 'Screen' });
    const name = within(rowFor(live, 'Minervini')).getAllByRole('cell')[0];
    const sample = within(rowFor(live, 'Minervini')).getAllByRole('cell')[1];
    expect(header).toHaveStyle({ position: 'sticky', left: '0px' });
    expect(name).toHaveStyle({ position: 'sticky', left: '0px' });
    expect(sample).not.toHaveStyle({ position: 'sticky' });
  });

  it('omits the excess count when the benchmark covers every closed window', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');
    openSection(PRE_FIX_TITLE);

    expect(rowFor(sectionFor(PRE_FIX_TITLE), 'VCP')).toHaveTextContent('540 closed');
    expect(rowFor(sectionFor(PRE_FIX_TITLE), 'VCP')).not.toHaveTextContent('excess on');
  });

  it('opens the artifact section at 5 sessions when that is the only horizon it has closed', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');

    expect(sectionToggle(ARTIFACT_TITLE)).toHaveAttribute('aria-expanded', 'true');
    expect(sectionToggle(PRE_FIX_TITLE)).toHaveAttribute('aria-expanded', 'false');
    expect(pressedHorizon()).toBe('5 sessions');
    const row = rowFor(sectionFor(ARTIFACT_TITLE), 'Minervini');
    expect(row).toBeVisible();
    expect(within(row).getAllByRole('cell').slice(2).map((cell) => cell.textContent))
      .toEqual(['+0.9%', '57%', '48%', '+0.1%']);
    expect(within(sectionFor(ARTIFACT_TITLE)).queryByText(/Nothing here is a result/)).not.toBeInTheDocument();
  });

  it('opens the pre-fix section of a market with only pre-fix days, at 20 sessions when one closed', async () => {
    renderRoute('/scoreboard?market=TW');
    const toggle = await screen.findByRole('button', { name: PRE_FIX_TITLE });

    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(pressedHorizon()).toBe('20 sessions');
    const row = rowFor(sectionFor(PRE_FIX_TITLE), 'VCP');
    expect(row).toBeVisible();
    expect(row).toHaveTextContent('26 closed · 6 dates, effective 4.8 · 300 pending · 294 overlapping');
  });

  it('passes over an artifact section with nothing closed and opens the pre-fix section at 5 sessions', async () => {
    renderRoute('/scoreboard?market=CN');
    const toggle = await screen.findByRole('button', { name: PRE_FIX_TITLE });

    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(sectionToggle(ARTIFACT_TITLE)).toHaveAttribute('aria-expanded', 'false');
    expect(pressedHorizon()).toBe('5 sessions');
    expect(rowFor(sectionFor(PRE_FIX_TITLE), 'VCP')).toHaveTextContent('90 closed · 4 dates, effective 3.6 · 0 pending');
  });

  it('opens nothing and shows 5 sessions when live has closed only at 5', async () => {
    renderRoute('/scoreboard?market=DE');
    const live = (await screen.findByText('Recorded live')).closest('section');

    expect(sectionToggle(ARTIFACT_TITLE)).toHaveAttribute('aria-expanded', 'false');
    expect(sectionToggle(PRE_FIX_TITLE)).toHaveAttribute('aria-expanded', 'false');
    expect(pressedHorizon()).toBe('5 sessions');
    expect(rowFor(live, 'Minervini')).toHaveTextContent('42 closed · 6 dates, effective 5.2 · 18 pending');
    expect(within(live).queryByText(/Nothing here is a result/)).not.toBeInTheDocument();
  });

  it('keeps 20 sessions and the note when every window is pending', async () => {
    renderRoute('/scoreboard?market=HK');
    const live = (await screen.findByText('Recorded live')).closest('section');

    expect(pressedHorizon()).toBe('20 sessions');
    expect(within(live).getByRole('table', { name: '20-session windows' })).toBeInTheDocument();
    expect(within(live).getByText(ALL_PENDING)).toBeInTheDocument();
  });

  it('keeps a picked horizon across a market change', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');
    fireEvent.click(screen.getByRole('button', { name: '60 sessions' }));
    expect(pressedHorizon()).toBe('60 sessions');

    await switchMarket('Taiwan');
    await screen.findByRole('button', { name: PRE_FIX_TITLE });

    expect(pressedHorizon()).toBe('60 sessions');
    expect(rowFor(sectionFor(PRE_FIX_TITLE), 'VCP')).toHaveTextContent('0 closed · 0 dates · 326 pending · oldest 22 of 60 sessions');
  });

  it('works out the default again when a market changes, including back to one already loaded', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');
    expect(pressedHorizon()).toBe('5 sessions');

    await switchMarket('Taiwan');
    const preFix = await screen.findByRole('button', { name: PRE_FIX_TITLE });
    expect(preFix).toHaveAttribute('aria-expanded', 'true');
    expect(pressedHorizon()).toBe('20 sessions');

    await switchMarket('United States');
    await screen.findByRole('button', { name: ARTIFACT_TITLE });
    expect(pressedHorizon()).toBe('5 sessions');
    expect(sectionToggle(ARTIFACT_TITLE)).toHaveAttribute('aria-expanded', 'true');
    expect(sectionToggle(PRE_FIX_TITLE)).toHaveAttribute('aria-expanded', 'false');
  });

  it('says Few dates, not Small sample, when many closed windows sit on few dates', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');
    openSection(PRE_FIX_TITLE);
    const section = sectionFor(PRE_FIX_TITLE);

    expect(rowFor(section, 'VCP')).not.toHaveTextContent('Few dates');
    fireEvent.click(screen.getByRole('button', { name: '20 sessions' }));
    expect(rowFor(section, 'VCP')).toHaveTextContent('Few dates');
    expect(rowFor(section, 'VCP')).not.toHaveTextContent('Small sample');
  });

  it('names the horizon in the table caption and marks the pressed picker button', async () => {
    renderRoute('/scoreboard');
    const live = (await screen.findByText('Recorded live')).closest('section');
    const picker = screen.getByRole('group', { name: 'Horizon' });

    expect(within(live).getByRole('table', { name: '5-session windows' })).toBeInTheDocument();
    expect(within(picker).getByRole('button', { name: '5 sessions' })).toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(within(picker).getByRole('button', { name: '60 sessions' }));

    expect(within(live).getByRole('table', { name: '60-session windows' })).toBeInTheDocument();
    expect(within(picker).getByRole('button', { name: '60 sessions' })).toHaveAttribute('aria-pressed', 'true');
    expect(within(picker).getByRole('button', { name: '5 sessions' })).toHaveAttribute('aria-pressed', 'false');
  });

  it('puts the section heading around the accordion button, not inside it', async () => {
    renderRoute('/scoreboard');
    const toggle = await screen.findByRole('button', { name: ARTIFACT_TITLE });

    expect(toggle.closest('h2')).toHaveTextContent(ARTIFACT_TITLE);
    expect(toggle.querySelector('h2')).toBeNull();
    expect(screen.getByRole('heading', { level: 2, name: ARTIFACT_TITLE })).toBe(toggle.closest('h2'));
  });

  it.each([
    [ARTIFACT_TITLE, 'scoreboard-US-artifact-toggle', 'scoreboard-US-artifact-panel'],
    [PRE_FIX_TITLE, 'scoreboard-US-artifact_pre_stage_fix-toggle', 'scoreboard-US-artifact_pre_stage_fix-panel'],
  ])('connects the toggle and panel of "%s" in both directions', async (title, toggleId, panelId) => {
    renderRoute('/scoreboard');
    const toggle = await screen.findByRole('button', { name: title });
    openSection(title);
    const panel = screen.getByRole('region', { name: title });

    expect(toggle).toHaveAttribute('id', toggleId);
    expect(toggle).toHaveAttribute('aria-controls', panelId);
    expect(panel).toHaveAttribute('id', panelId);
    expect(panel).toHaveAttribute('aria-labelledby', toggleId);
    expect(toggle.closest('h2')).not.toHaveAttribute('id');
    expect(toggle.closest('h2')).not.toHaveAttribute('aria-controls');
  });

  it('takes accordion ids from the selected market', async () => {
    renderRoute('/scoreboard');
    await screen.findByRole('button', { name: ARTIFACT_TITLE });

    await switchMarket('China');

    await waitFor(() => expect(sectionToggle(ARTIFACT_TITLE)).toHaveAttribute('id', 'scoreboard-CN-artifact-toggle'));
    expect(sectionToggle(ARTIFACT_TITLE)).toHaveAttribute('aria-controls', 'scoreboard-CN-artifact-panel');
  });

  it('greys thin rows with closed windows and labels them Small sample', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');
    openSection(ARTIFACT_TITLE);
    const section = sectionFor(ARTIFACT_TITLE);

    const thin = rowFor(section, 'CANSLIM');
    expect(thin).toHaveTextContent('Small sample');
    expect(thin).toHaveTextContent('-0.3%');
    expect(within(thin).getAllByRole('cell')[2]).toHaveStyle({ color: DARK_TEXT.secondary });
    expect(within(rowFor(section, 'Minervini')).getAllByRole('cell')[2]).toHaveStyle({ color: DARK_TEXT.primary });
  });

  it('labels a retired preset', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');

    expect(rowFor(sectionFor(ARTIFACT_TITLE), 'Old Breakout')).toHaveTextContent('Retired');
    expect(rowFor(sectionFor(ARTIFACT_TITLE), 'CANSLIM')).not.toHaveTextContent('Retired');
  });

  it('keeps the published cohort order whatever the returns are', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');
    openSection(ARTIFACT_TITLE);

    const names = within(sectionFor(ARTIFACT_TITLE)).getAllByRole('row').slice(1)
      .map((row) => within(row).getAllByRole('cell')[0].textContent);
    expect(names.map((name) => name.split('2026')[0])).toEqual(['Minervini', 'CANSLIM', 'Old Breakout']);
  });

  it('names a horizon with no closed window when another horizon has closed', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');

    expect(within(sectionFor(ARTIFACT_TITLE)).queryByText(/Nothing here is a result/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '20 sessions' }));
    expect(within(sectionFor(ARTIFACT_TITLE)).getByText('No 20-session window has closed yet. Nothing here is a result.'))
      .toBeInTheDocument();
  });

  it('states the method, ledger range and missing sessions in the header', async () => {
    renderRoute('/scoreboard');
    await screen.findByText('Recorded live');

    expect(screen.getByText('Recording since 2026-10-05 · Ledger through 2026-11-04 · 2 missing sessions')).toBeInTheDocument();
    expect(screen.getByText('Entry is the first open after publication. Exit is the close of the Nth session.')).toBeInTheDocument();
    expect(screen.getByText('Returns are price only, in local currency, with no costs. Excess is against SPY.')).toBeInTheDocument();
    expect(screen.getByText('Funds that match a screen are included.')).toBeInTheDocument();
    expect(screen.getByText('Stocks that later delist can drop out as no price, which can flatter results.')).toBeInTheDocument();
    expect(screen.getByText(/A stock counts once per screen until its window ends/)).toBeInTheDocument();
    expect(screen.getByText("A pick is a stock newly entering the screen. Stocks on the list on the first recorded day, or on a day the screen's definition or the Stage calculation changed, are not counted.")).toBeInTheDocument();
    expect(screen.getByText('Effective dates count each date by how many picks it holds, so one busy day counts as about one.')).toBeInTheDocument();
    const rules = screen.getAllByRole('listitem').map((item) => item.textContent);
    expect(rules[rules.indexOf("A pick is a stock newly entering the screen. Stocks on the list on the first recorded day, or on a day the screen's definition or the Stage calculation changed, are not counted.") + 1])
      .toBe('A stock that returns after its window ends counts again.');
    expect(screen.getByText('Prices are as the provider reports them, so an unadjusted split can appear as a false move. Medians limit the effect.')).toBeInTheDocument();
  });

  it('shows Scored since only when it differs from Recording since', async () => {
    renderRoute('/scoreboard?market=AU');

    expect(await screen.findByText('Recording since 2025-06-02 · Scored since 2025-11-04 · Ledger through 2026-11-04 · 0 missing sessions'))
      .toBeInTheDocument();
  });

  it('drops the benchmark sentence and prints no null benchmark when a market has none', async () => {
    renderRoute('/scoreboard?market=AU');
    const live = (await screen.findByText('Recorded live')).closest('section');

    expect(screen.getByText('Returns are price only, in local currency, with no costs.')).toBeInTheDocument();
    const row = rowFor(live, 'CANSLIM');
    expect(within(row).getAllByRole('cell')[1].textContent)
      .toBe('64 closed · 11 dates, effective 7.3 · 0 pending · excess on 0');
    expect(within(row).getAllByRole('cell').slice(2).map((cell) => cell.textContent))
      .toEqual(['+1.1%', '55%', '-', '-']);
    expect(document.body).not.toHaveTextContent(/against|null|None|undefined/);
  });

  it('opens a deep link on its market and shows the all-pending state', async () => {
    renderRoute('/scoreboard?market=HK');

    expect(await screen.findByText(ALL_PENDING)).toBeInTheDocument();
    expect(screen.getByText('Returns are price only, in local currency, with no costs. Excess is against 2800.HK.')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Static market selector' })).toHaveTextContent('Hong Kong');
    expect(scoreboardFetches()).toEqual(['/static-data/g/g1/markets/hk/scoreboard.json']);
    expect(screen.queryByRole('button', { name: ARTIFACT_TITLE })).not.toBeInTheDocument();
  });

  it('has a nav item when the market advertises the asset', async () => {
    renderRoute('/scoreboard');
    expect(await screen.findByRole('link', { name: 'Scoreboard' })).toHaveAttribute('href', '/scoreboard');
  });

  it('has no nav item and fetches nothing when the market has no asset', async () => {
    renderRoute('/scoreboard?market=JP');

    expect(await screen.findByText('The scoreboard is not available for JP in this snapshot.')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Scoreboard' })).not.toBeInTheDocument();
    expect(scoreboardFetches()).toEqual([]);
  });

  it('says so when the advertised file cannot be read', async () => {
    fetch.mockImplementation(async (url) => respondWith(url.endsWith('/manifest.json') ? manifest : null));
    renderRoute('/scoreboard');

    expect(await screen.findByText('The scoreboard is not available for US in this snapshot.')).toBeInTheDocument();
  });
});
