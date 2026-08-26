/**
 * Static mock data for the Options Command Center prototype.
 *
 * Every number here is hand-authored to *look* like a real options-flow
 * snapshot (plausible spreads between spot/flip/walls, plausible GEX
 * magnitudes, plausible VRP/skew ranges) but none of it is fetched --
 * this file is the single swap-in point for wiring the real API later.
 * Component code should never hardcode a ticker or number; it should only
 * ever read from here.
 */

export const macroHealth = {
  spx: {
    label: '$SPX',
    gammaRegime: 'Long Gamma',
    gammaPct: 65,
    flipLevel: 5480,
    spot: 5612,
  },
  indices: [
    {
      symbol: 'SPY',
      spot: 561.42,
      flipLevel: 548.0,
      callWall: 570.0,
      putWall: 550.0,
      regime: 'long_gamma',
    },
    {
      symbol: 'QQQ',
      spot: 481.15,
      flipLevel: 472.5,
      callWall: 490.0,
      putWall: 470.0,
      regime: 'long_gamma',
    },
  ],
  vix: {
    spot: 14.82,
    frontMonth: 15.1,
    secondMonth: 16.4,
    thirdMonth: 17.65,
    termStructure: 'contango', // 'contango' | 'backwardation'
  },
};

// Column A -- Structural & Dealer Risk ---------------------------------

export const volatilityAcceleration = [
  { symbol: 'TSLA', price: 258.34, totalGex: -412_500_000, distanceToFlipPct: -3.2, regime: 'short_gamma' },
  { symbol: 'MSTR', price: 178.9, totalGex: -298_100_000, distanceToFlipPct: -5.8, regime: 'short_gamma' },
  { symbol: 'SMCI', price: 41.22, totalGex: -211_400_000, distanceToFlipPct: -2.1, regime: 'short_gamma' },
  { symbol: 'COIN', price: 214.6, totalGex: -184_700_000, distanceToFlipPct: -4.4, regime: 'short_gamma' },
  { symbol: 'NVDA', price: 132.18, totalGex: -156_300_000, distanceToFlipPct: -1.3, regime: 'short_gamma' },
  { symbol: 'PLTR', price: 39.75, totalGex: -122_900_000, distanceToFlipPct: -3.6, regime: 'short_gamma' },
  { symbol: 'AMD', price: 156.02, totalGex: -98_450_000, distanceToFlipPct: -2.7, regime: 'short_gamma' },
  { symbol: 'NFLX', price: 712.4, totalGex: -76_200_000, distanceToFlipPct: -0.9, regime: 'short_gamma' },
];

export const gammaFlipProximity = [
  { symbol: 'AAPL', spot: 227.85, flipLevel: 226.5, distancePct: 0.6 },
  { symbol: 'MSFT', spot: 421.1, flipLevel: 424.0, distancePct: -0.7 },
  { symbol: 'META', spot: 512.3, flipLevel: 519.2, distancePct: -1.3 },
  { symbol: 'GOOGL', spot: 178.44, flipLevel: 177.1, distancePct: 0.8 },
  { symbol: 'AMZN', spot: 189.62, flipLevel: 191.4, distancePct: -0.9 },
  { symbol: 'AVGO', spot: 168.9, flipLevel: 167.3, distancePct: 1.0 },
  { symbol: 'XOM', spot: 118.27, flipLevel: 119.6, distancePct: -1.1 },
  { symbol: 'JPM', spot: 214.55, flipLevel: 213.1, distancePct: 0.7 },
];

// Column B -- Volatility Mispricing --------------------------------------

export const richVrp = [
  { symbol: 'MSTR', iv: 0.98, hv: 0.61, vrpPct: 37.0 },
  { symbol: 'SMCI', iv: 1.12, hv: 0.79, vrpPct: 33.0 },
  { symbol: 'COIN', iv: 0.74, hv: 0.46, vrpPct: 28.0 },
  { symbol: 'PLTR', iv: 0.63, hv: 0.39, vrpPct: 24.0 },
  { symbol: 'TSLA', iv: 0.58, hv: 0.37, vrpPct: 21.0 },
  { symbol: 'CVNA', iv: 0.81, hv: 0.61, vrpPct: 20.0 },
  { symbol: 'RIVN', iv: 0.69, hv: 0.51, vrpPct: 18.0 },
  { symbol: 'SNOW', iv: 0.52, hv: 0.36, vrpPct: 16.0 },
];

export const extremeSkew = [
  { symbol: 'GME', callIv: 0.92, putIv: 0.58, skew: -0.34 },
  { symbol: 'AMC', callIv: 0.88, putIv: 0.61, skew: -0.27 },
  { symbol: 'SMCI', callIv: 0.79, putIv: 0.55, skew: -0.24 },
  { symbol: 'COIN', callIv: 0.71, putIv: 0.51, skew: -0.20 },
  { symbol: 'MSTR', callIv: 0.85, putIv: 0.68, skew: -0.17 },
  { symbol: 'IWM', callIv: 0.34, putIv: 0.29, skew: -0.05 },
  { symbol: 'NVDA', callIv: 0.48, putIv: 0.46, skew: -0.02 },
  { symbol: 'SPY', callIv: 0.135, putIv: 0.142, skew: 0.007 },
];

// Column C -- Smart Money Flow --------------------------------------

export const netPremiumInflows = [
  { symbol: 'NVDA', callPremium: 184_200_000, putPremium: 61_400_000, netPremium: 122_800_000 },
  { symbol: 'TSLA', callPremium: 142_600_000, putPremium: 58_900_000, netPremium: 83_700_000 },
  { symbol: 'AAPL', callPremium: 98_100_000, putPremium: 41_200_000, netPremium: 56_900_000 },
  { symbol: 'META', callPremium: 76_400_000, putPremium: 62_800_000, netPremium: 13_600_000 },
  { symbol: 'SPY', callPremium: 210_500_000, putPremium: 198_300_000, netPremium: 12_200_000 },
  { symbol: 'MSFT', callPremium: 54_300_000, putPremium: 48_100_000, netPremium: 6_200_000 },
  { symbol: 'AMZN', callPremium: 39_800_000, putPremium: 46_500_000, netPremium: -6_700_000 },
  { symbol: 'BA', callPremium: 21_100_000, putPremium: 44_900_000, netPremium: -23_800_000 },
];

export const unusualVolumeOi = [
  { symbol: 'SMCI', strike: 45, type: 'call', volume: 18_420, openInterest: 612, ratio: 30.1 },
  { symbol: 'COIN', strike: 230, type: 'call', volume: 9_845, openInterest: 388, ratio: 25.4 },
  { symbol: 'MSTR', strike: 190, type: 'put', volume: 7_210, openInterest: 340, ratio: 21.2 },
  { symbol: 'PLTR', strike: 42, type: 'call', volume: 12_960, openInterest: 705, ratio: 18.4 },
  { symbol: 'RIVN', strike: 14, type: 'put', volume: 5_540, openInterest: 355, ratio: 15.6 },
  { symbol: 'TSLA', strike: 270, type: 'call', volume: 21_330, openInterest: 1_480, ratio: 14.4 },
  { symbol: 'AMD', strike: 165, type: 'call', volume: 8_120, openInterest: 610, ratio: 13.3 },
  { symbol: 'NFLX', strike: 730, type: 'call', volume: 3_960, openInterest: 320, ratio: 12.4 },
];

// Executive Alert Ticker -------------------------------------------------
//
// Severity here is hand-picked for the mock. Once wired to real data, reuse
// the same weighted Executive Signal scoring the Options Analytics
// dashboard already computes (see getMarketSignal/marketFactors in
// OptionsAnalyticsDashboardPage.jsx: Total GEX weight 2, skew/PCR/OI-skew
// weight 1 each, max pain weight 0.5, call-wall-break weight 1 -- summed
// into a single -6.5..+6.5 score) rather than inventing a second scoring
// convention for this page. Suggested thresholds against that same score:
//   |total_score| >= 4                          -> critical
//   |total_score| >= 1.5                         -> warning
//   otherwise, or a single-factor informational reading -> info
// A hard structural event (wall breach, gamma-regime flip) should probably
// always be at least "warning" regardless of the aggregate score, since
// those are the alerts a trading floor most wants to not miss.
export const executiveAlerts = [
  { id: 1, severity: 'critical', text: 'Gamma Squeeze Alert: $SMCI breached Call Wall ($45.00) on 18.4x Vol/OI' },
  { id: 2, severity: 'warning', text: '$COIN flipped Short Gamma -- expect amplified intraday moves below $215' },
  { id: 3, severity: 'info', text: '$NVDA Net Call Premium +$122.8M today, largest single-name inflow' },
  { id: 4, severity: 'warning', text: '$MSTR VRP at 37% -- implied vol running rich vs. 20D realized' },
  { id: 5, severity: 'critical', text: 'Unusual Flow: $PLTR $42C volume 18.4x open interest into the close' },
  { id: 6, severity: 'info', text: '$SPY Total GEX +$3.1B, dealers positioned to dampen swings into OPEX' },
  { id: 7, severity: 'warning', text: '$BA Net Premium -$23.8M -- put buying dominating flow' },
];
