const STOCKBEE_LIQUIDITY =
  "20-session average raw local-currency traded value must meet the selected market's fixed threshold.";

const STOCKBEE_DAILY_VOLUME =
  'Volume must meet the market-specific daily share threshold and exceed the prior session.';

const STOCKBEE_MONTH_REFERENCE =
  '21 sessions; raw close 20 sessions earlier must meet the market-specific local reference-price threshold';

const stockBee = (definition) => ({
  formulaOrigin: 'StockBee',
  groupOrigin: 'Screenshot grouping',
  liquidityRule: STOCKBEE_LIQUIDITY,
  ...definition,
});

export const breadthMetricDefinitions = {
  stocks_up_4pct: stockBee({
    label: 'Stocks Up 4%+',
    description:
      `Adjusted close rises at least 4% today. ${STOCKBEE_DAILY_VOLUME}`,
    requiredHistory: '20 sessions for liquidity plus the prior session',
    eligibleField: 'stockbee_daily_eligible_count',
    contributor: { signalKey: 'up_4pct', direction: 'up', valueKind: 'percent', qualifierLabel: '1-day change' },
  }),
  stocks_down_4pct: stockBee({
    label: 'Stocks Down 4%+',
    description:
      `Adjusted close falls at least 4% today. ${STOCKBEE_DAILY_VOLUME}`,
    requiredHistory: '20 sessions for liquidity plus the prior session',
    eligibleField: 'stockbee_daily_eligible_count',
    contributor: { signalKey: 'down_4pct', direction: 'down', valueKind: 'percent', qualifierLabel: '1-day change' },
  }),
  ratio_5day: stockBee({
    label: '5 Day Ratio',
    description: 'Today-inclusive five-session sum of up-4% counts divided by down-4% counts.',
    requiredHistory: '5 breadth sessions including today',
    eligibleField: 'stockbee_daily_eligible_count',
  }),
  ratio_10day: stockBee({
    label: '10 Day Ratio',
    description: 'Today-inclusive ten-session sum of up-4% counts divided by down-4% counts.',
    requiredHistory: '10 breadth sessions including today',
    eligibleField: 'stockbee_daily_eligible_count',
  }),
  stocks_up_25pct_quarter: stockBee({
    label: 'Up 25%+ Quarter',
    description: 'Adjusted close is at least 25% above the trailing 65-session adjusted-close low.',
    requiredHistory: '65 sessions including today',
    eligibleField: 'stockbee_quarter_eligible_count',
    contributor: { signalKey: 'up_25pct_quarter', direction: 'up', valueKind: 'percent', qualifierLabel: 'Gain from 65-session low' },
  }),
  stocks_down_25pct_quarter: stockBee({
    label: 'Down 25%+ Quarter',
    description: 'Adjusted close is at least 25% below the trailing 65-session adjusted-close high.',
    requiredHistory: '65 sessions including today',
    eligibleField: 'stockbee_quarter_eligible_count',
    contributor: { signalKey: 'down_25pct_quarter', direction: 'down', valueKind: 'percent', qualifierLabel: 'Decline from 65-session high' },
  }),
  stocks_up_25pct_month: stockBee({
    label: 'Up 25%+ Month',
    description: 'Adjusted close is at least 25% above the adjusted close exactly 20 sessions ago.',
    requiredHistory: STOCKBEE_MONTH_REFERENCE,
    eligibleField: 'stockbee_month_eligible_count',
    contributor: { signalKey: 'up_25pct_month', direction: 'up', valueKind: 'percent', qualifierLabel: '20-session change' },
  }),
  stocks_down_25pct_month: stockBee({
    label: 'Down 25%+ Month',
    description: 'Adjusted close is at least 25% below the adjusted close exactly 20 sessions ago.',
    requiredHistory: STOCKBEE_MONTH_REFERENCE,
    eligibleField: 'stockbee_month_eligible_count',
    contributor: { signalKey: 'down_25pct_month', direction: 'down', valueKind: 'percent', qualifierLabel: '20-session change' },
  }),
  stocks_up_50pct_month: stockBee({
    label: 'Up 50%+ Month',
    description: 'Adjusted close is at least 50% above the adjusted close exactly 20 sessions ago.',
    requiredHistory: STOCKBEE_MONTH_REFERENCE,
    eligibleField: 'stockbee_month_eligible_count',
    contributor: { signalKey: 'up_50pct_month', direction: 'up', valueKind: 'percent', qualifierLabel: '20-session change' },
  }),
  stocks_down_50pct_month: stockBee({
    label: 'Down 50%+ Month',
    description: 'Adjusted close is at least 50% below the adjusted close exactly 20 sessions ago.',
    requiredHistory: STOCKBEE_MONTH_REFERENCE,
    eligibleField: 'stockbee_month_eligible_count',
    contributor: { signalKey: 'down_50pct_month', direction: 'down', valueKind: 'percent', qualifierLabel: '20-session change' },
  }),
  stocks_up_13pct_34days: stockBee({
    label: 'Up 13%+ / 34 Days',
    description: 'Adjusted close is at least 13% above the trailing 34-session adjusted-close low.',
    requiredHistory: '34 sessions including today',
    eligibleField: 'stockbee_34day_eligible_count',
    contributor: { signalKey: 'up_13pct_34days', direction: 'up', valueKind: 'percent', qualifierLabel: 'Gain from 34-session low' },
  }),
  stocks_down_13pct_34days: stockBee({
    label: 'Down 13%+ / 34 Days',
    description: 'Adjusted close is at least 13% below the trailing 34-session adjusted-close high.',
    requiredHistory: '34 sessions including today',
    eligibleField: 'stockbee_34day_eligible_count',
    contributor: { signalKey: 'down_13pct_34days', direction: 'down', valueKind: 'percent', qualifierLabel: 'Decline from 34-session high' },
  }),
  advancing_count: {
    label: 'Advancing',
    formulaOrigin: 'Standard breadth',
    groupOrigin: 'Context',
    description: 'Adjusted close is above the prior session adjusted close.',
    requiredHistory: '2 sessions',
    eligibleField: 'advance_decline_eligible_count',
  },
  declining_count: {
    label: 'Declining',
    formulaOrigin: 'Standard breadth',
    groupOrigin: 'Context',
    description: 'Adjusted close is below the prior session adjusted close.',
    requiredHistory: '2 sessions',
    eligibleField: 'advance_decline_eligible_count',
  },
  new_high_52week_count: {
    label: 'New 52-Week High',
    formulaOrigin: 'StockBee',
    groupOrigin: 'Context',
    description: 'Adjusted high is strictly above every adjusted high in the prior 251 sessions.',
    requiredHistory: '252 sessions',
    eligibleField: 'high_low_52week_eligible_count',
  },
  new_low_52week_count: {
    label: 'New 52-Week Low',
    formulaOrigin: 'StockBee',
    groupOrigin: 'Context',
    description: 'Adjusted low is strictly below every adjusted low in the prior 251 sessions.',
    requiredHistory: '252 sessions',
    eligibleField: 'high_low_52week_eligible_count',
  },
  t2108_pct: {
    label: 'T2108',
    formulaOrigin: 'StockBee',
    groupOrigin: 'Context',
    description: 'Percentage of eligible stocks with adjusted close above their 40-session simple moving average.',
    requiredHistory: '40 sessions',
    eligibleField: 't2108_eligible_count',
    countField: 't2108_count',
  },
  above_sma20_pct: {
    label: '% Above 20D',
    formulaOrigin: 'Standard breadth',
    groupOrigin: 'Context',
    description: 'Percentage of eligible stocks with adjusted close above their 20-session simple moving average.',
    requiredHistory: '20 sessions',
    eligibleField: 'sma20_eligible_count',
    countField: 'above_sma20_count',
  },
  above_sma50_pct: {
    label: '% Above 50D',
    formulaOrigin: 'Standard breadth',
    groupOrigin: 'Context',
    description: 'Percentage of eligible stocks with adjusted close above their 50-session simple moving average.',
    requiredHistory: '50 sessions',
    eligibleField: 'sma50_eligible_count',
    countField: 'above_sma50_count',
  },
  above_sma200_pct: {
    label: '% Above 200D',
    formulaOrigin: 'Standard breadth',
    groupOrigin: 'Context',
    description: 'Percentage of eligible stocks with adjusted close above their 200-session simple moving average.',
    requiredHistory: '200 sessions',
    eligibleField: 'sma200_eligible_count',
    countField: 'above_sma200_count',
  },
  down_20pct_from_high_pct: {
    label: '% 20%+ Off High',
    formulaOrigin: 'Standard breadth',
    groupOrigin: 'Context',
    description: 'Percentage of eligible stocks whose adjusted close is at least 20% below their highest adjusted high of the last 252 sessions. A recent listing measures from its listing high.',
    requiredHistory: '1 session',
    eligibleField: 'off_high_eligible_count',
    countField: 'down_20pct_from_high_count',
  },
  atr_10x_extension_count: {
    label: '10x ATR',
    formulaOrigin: 'Screenshot-derived',
    groupOrigin: 'Context',
    description: 'Stocks whose positive extension above SMA50 is at least ten times Wilder ATR14 as a percentage of price.',
    requiredHistory: '50 sessions plus Wilder ATR14',
    eligibleField: 'atr_extension_eligible_count',
    contributor: { signalKey: 'atr_10x_extension', direction: 'extension', valueKind: 'multiple', qualifierLabel: 'ATR extension' },
  },
  broad_universe_count: {
    label: 'Broad Universe',
    formulaOrigin: 'Shared calculation layer',
    groupOrigin: 'Context',
    description: 'Point-in-time common-stock universe before metric-specific data eligibility is applied.',
    requiredHistory: 'Point-in-time membership',
    eligibleField: null,
  },
};

export const primaryBreadthMetrics = [
  'stocks_up_4pct',
  'stocks_down_4pct',
  'ratio_5day',
  'ratio_10day',
];

export const secondaryBreadthMetrics = [
  'stocks_up_25pct_quarter',
  'stocks_down_25pct_quarter',
  'stocks_up_25pct_month',
  'stocks_down_25pct_month',
  'stocks_up_50pct_month',
  'stocks_down_50pct_month',
  'stocks_up_13pct_34days',
  'stocks_down_13pct_34days',
];

export const tableContextMetrics = [
  'atr_10x_extension_count',
  't2108_pct',
  'above_sma20_pct',
  'above_sma50_pct',
  'above_sma200_pct',
  'down_20pct_from_high_pct',
  'broad_universe_count',
];
