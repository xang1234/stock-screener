export const STAGE_OPTIONS = [
  { value: 1, label: 'S1 - Basing' },
  { value: 2, label: 'S2 - Advancing' },
  { value: 3, label: 'S3 - Topping' },
  { value: 4, label: 'S4 - Declining' },
];

export const VOLUME_OPTIONS = [
  { value: 10000000, label: '>$10M' },
  { value: 50000000, label: '>$50M' },
  { value: 100000000, label: '>$100M' },
  { value: 500000000, label: '>$500M' },
  { value: 1000000000, label: '>$1B' },
  { value: 5000000000, label: '>$5B' },
  { value: 10000000000, label: '>$10B' },
];

export const MARKET_CAP_OPTIONS = [
  { value: 100000000, label: '>$100M' },
  { value: 200000000, label: '>$200M' },
  { value: 500000000, label: '>$500M' },
  { value: 1000000000, label: '>$1B' },
  { value: 2000000000, label: '>$2B' },
  { value: 5000000000, label: '>$5B' },
  { value: 10000000000, label: '>$10B' },
];

export const FUNDAMENTAL_KEYS = [
  'symbolSearch', 'minMarketCap', 'minVolume', 'price',
  'epsGrowth', 'salesGrowth', 'epsRating', 'ibdIndustries', 'gicsSectors', 'ipoAfter',
];

export const TECHNICAL_KEYS = [
  'stage', 'rsRating', 'rs1m', 'rs3m', 'rs12m', 'maAlignment',
  'adrPercent', 'perfDay', 'perfWeek', 'perfMonth',
  'perf3m', 'perf6m', 'gapPercent', 'volumeSurge',
  'ema10Distance', 'ema20Distance', 'ema50Distance',
  'week52HighDistance', 'week52LowDistance',
  'beta', 'betaAdjRs',
];

export const RATING_KEYS = [
  'compositeScore', 'minerviniScore', 'canslimScore', 'ipoScore',
  'customScore', 'volBreakthroughScore',
  'seSetupScore', 'seDistanceToPivot', 'seBbSqueeze', 'seVolumeVs50d',
  'seSetupReady', 'seRsLineNewHigh',
  'vcpScore', 'vcpDetected', 'vcpReady', 'passesTemplate',
];
