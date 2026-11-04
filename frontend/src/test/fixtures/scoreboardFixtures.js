const horizon = (horizonSessions, fields = {}) => ({
  horizon_sessions: horizonSessions,
  sample_size: 0,
  signal_dates: 0,
  pending_windows: 0,
  oldest_open_sessions: null,
  thin: true,
  positive_rate: null,
  avg_return_pct: null,
  median_return_pct: null,
  avg_excess_return_pct: null,
  median_excess_return_pct: null,
  excess_sample_size: 0,
  skipped_missing_history: 0,
  suppressed_overlapping: 0,
  ...fields,
});

const closed = (sampleSize, signalDates, positiveRate, avg, median, avgExcess, medianExcess, fields = {}) => ({
  sample_size: sampleSize,
  signal_dates: signalDates,
  thin: sampleSize < 20 || signalDates < 5,
  positive_rate: positiveRate,
  avg_return_pct: avg,
  median_return_pct: median,
  avg_excess_return_pct: avgExcess,
  median_excess_return_pct: medianExcess,
  excess_sample_size: sampleSize,
  ...fields,
});

const pending = (windows, oldestOpen, fields = {}) => ({
  pending_windows: windows,
  oldest_open_sessions: oldestOpen,
  ...fields,
});

export const liveAllPendingCohort = {
  preset: 'minervini',
  name: 'Minervini',
  retired: false,
  definition: '241a0f65f493c25e',
  origin: 'live',
  first_date: '2026-11-03',
  last_date: '2026-11-04',
  events: 357,
  signal_dates: 2,
  horizons: [
    horizon(5, pending(357, 2)),
    horizon(20, pending(357, 2)),
    horizon(60, pending(357, 2)),
  ],
};

const liveZeroMatchCohort = {
  preset: 'se_high_tight_flag',
  name: 'High Tight Flag',
  retired: false,
  definition: '7c1d0b3a9e24f615',
  origin: 'live',
  first_date: '2026-11-03',
  last_date: '2026-11-04',
  events: 0,
  signal_dates: 0,
  horizons: [horizon(5), horizon(20), horizon(60)],
};

const artifactClosedCohort = {
  preset: 'minervini',
  name: 'Minervini',
  retired: false,
  definition: '241a0f65f493c25e',
  origin: 'artifact',
  first_date: '2026-10-22',
  last_date: '2026-10-29',
  events: 1441,
  signal_dates: 6,
  horizons: [
    horizon(5, closed(1298, 6, 0.57, 1.42, 0.86, 0.31, 0.12, pending(40, 4, {
      skipped_missing_history: 3,
      suppressed_overlapping: 100,
    }))),
    horizon(20, pending(1341, 9, { suppressed_overlapping: 100 })),
    horizon(60, pending(1341, 9, { suppressed_overlapping: 100 })),
  ],
};

const artifactThinCohort = {
  preset: 'canslim',
  name: 'CANSLIM',
  retired: false,
  definition: '0a9b5ce31d7f2e48',
  origin: 'artifact',
  first_date: '2026-10-27',
  last_date: '2026-10-28',
  events: 14,
  signal_dates: 2,
  horizons: [
    horizon(5, closed(12, 2, 0.5, -0.74, -0.31, -1.2, -0.95, pending(2, 4))),
    horizon(20, pending(14, 7)),
    horizon(60, pending(14, 7)),
  ],
};

const artifactRetiredCohort = {
  preset: 'old_breakout',
  name: 'Old Breakout',
  retired: true,
  definition: 'e5f0a6c2b8147d39',
  origin: 'artifact',
  first_date: '2026-10-22',
  last_date: '2026-10-23',
  events: 8,
  signal_dates: 2,
  horizons: [
    horizon(5, closed(8, 2, 0.625, 2.05, 1.5, 1.1, 0.8)),
    horizon(20, pending(8, 9)),
    horizon(60, pending(8, 9)),
  ],
};

const preStageFixCohort = {
  preset: 'vcp',
  name: 'VCP',
  retired: false,
  definition: '93d2e71f0c6ab584',
  origin: 'artifact_pre_stage_fix',
  first_date: '2026-10-05',
  last_date: '2026-10-21',
  events: 620,
  signal_dates: 13,
  horizons: [
    horizon(5, closed(540, 13, 0.52, 0.91, 0.44, -0.2, -0.36, { suppressed_overlapping: 80 })),
    horizon(20, closed(26, 6, 0.46, -1.63, -2.27, -2.9, -3.05, pending(300, 19, { suppressed_overlapping: 294 }))),
    horizon(60, pending(326, 22, { suppressed_overlapping: 294 })),
  ],
};

export const scoreboardFixture = {
  schema_version: 'static-scoreboard-v1',
  market: 'US',
  as_of_date: '2026-11-04',
  generated_at: '2026-11-05T02:15:22Z',
  benchmark: 'SPY',
  recording_since: '2026-10-05',
  ledger_through: '2026-11-04',
  missing_sessions: 2,
  cohorts: [
    liveAllPendingCohort,
    liveZeroMatchCohort,
    artifactClosedCohort,
    artifactThinCohort,
    artifactRetiredCohort,
    preStageFixCohort,
  ],
};

export const allPendingScoreboardFixture = {
  ...scoreboardFixture,
  market: 'HK',
  benchmark: '2800.HK',
  missing_sessions: 0,
  cohorts: [liveAllPendingCohort, liveZeroMatchCohort],
};
