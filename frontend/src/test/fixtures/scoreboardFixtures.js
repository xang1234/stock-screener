const horizon = (horizonSessions, fields = {}) => ({
  horizon_sessions: horizonSessions,
  sample_size: 0,
  signal_dates: 0,
  effective_dates: 0,
  pending_windows: 0,
  oldest_open_sessions: null,
  thin: true,
  positive_rate: null,
  beat_benchmark_rate: null,
  median_return_pct: null,
  median_excess_return_pct: null,
  excess_sample_size: 0,
  skipped_missing_history: 0,
  suppressed_overlapping: 0,
  ...fields,
});

const closed = (sampleSize, signalDates, effectiveDates, positiveRate, beatRate, median, medianExcess, fields = {}) => ({
  sample_size: sampleSize,
  signal_dates: signalDates,
  effective_dates: effectiveDates,
  thin: sampleSize < 20 || effectiveDates < 5,
  positive_rate: positiveRate,
  beat_benchmark_rate: beatRate,
  median_return_pct: median,
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
    horizon(5, closed(1298, 12, 8.4, 0.57, 0.48, 0.86, 0.12, pending(40, 4, {
      skipped_missing_history: 3,
      suppressed_overlapping: 100,
      excess_sample_size: 1250,
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
    horizon(5, closed(12, 2, 1.9, 0.5, 0.25, -0.31, -0.95, pending(2, 4))),
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
    horizon(5, closed(8, 2, 2, 0.625, 0.5, 0.2, 0.8)),
    horizon(20, pending(8, 9)),
    horizon(60, pending(8, 9)),
  ],
};

const artifactZeroMatchCohort = {
  ...liveZeroMatchCohort,
  origin: 'artifact',
  first_date: '2026-10-22',
  last_date: '2026-10-29',
};

const artifactNoExcessCohort = {
  preset: 'pocket_pivot',
  name: 'Pocket Pivot',
  retired: false,
  definition: '5b8e2d7f1a9c0364',
  origin: 'artifact',
  first_date: '2026-10-22',
  last_date: '2026-10-28',
  events: 34,
  signal_dates: 6,
  horizons: [
    horizon(5, closed(30, 6, 5.5, 0.4, null, -1.4, null, pending(4, 3, { excess_sample_size: 0 }))),
    horizon(20, pending(34, 9)),
    horizon(60, pending(34, 9)),
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
    horizon(5, closed(540, 13, 9, 0.52, 0.44, 0.46, -0.36, { suppressed_overlapping: 80 })),
    horizon(20, closed(26, 6, 4.8, 0.46, 0.42, -2.27, -3.05, pending(300, 19, { suppressed_overlapping: 294 }))),
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
  scored_since: '2026-10-05',
  ledger_through: '2026-11-04',
  missing_sessions: 2,
  cohorts: [
    liveAllPendingCohort,
    liveZeroMatchCohort,
    artifactZeroMatchCohort,
    artifactNoExcessCohort,
    artifactClosedCohort,
    artifactThinCohort,
    artifactRetiredCohort,
    preStageFixCohort,
  ],
};

const liveClosedAtFiveCohort = {
  ...liveAllPendingCohort,
  events: 60,
  signal_dates: 8,
  horizons: [
    horizon(5, closed(42, 6, 5.2, 0.6, 0.55, 1.2, 0.3, pending(18, 3))),
    horizon(20, pending(60, 12)),
    horizon(60, pending(60, 12)),
  ],
};

const artifactAllPendingCohort = {
  ...artifactClosedCohort,
  events: 70,
  signal_dates: 3,
  horizons: [
    horizon(5, pending(70, 3)),
    horizon(20, pending(70, 3)),
    horizon(60, pending(70, 3)),
  ],
};

const preStageFixClosedAtFiveCohort = {
  ...preStageFixCohort,
  horizons: [
    horizon(5, closed(90, 4, 3.6, 0.5, 0.45, 0.7, 0.1)),
    horizon(20, pending(90, 18)),
    horizon(60, pending(90, 18)),
  ],
};

export const preFixOnlyScoreboardFixture = {
  ...scoreboardFixture,
  market: 'TW',
  benchmark: '0050.TW',
  recording_since: '2026-10-05',
  ledger_through: '2026-10-08',
  cohorts: [preStageFixCohort],
};

export const pendingArtifactScoreboardFixture = {
  ...scoreboardFixture,
  market: 'CN',
  cohorts: [artifactAllPendingCohort, preStageFixClosedAtFiveCohort],
};

export const liveClosedAtFiveScoreboardFixture = {
  ...scoreboardFixture,
  cohorts: [liveClosedAtFiveCohort, artifactClosedCohort, preStageFixCohort],
};

export const allPendingScoreboardFixture = {
  ...scoreboardFixture,
  market: 'HK',
  benchmark: '2800.HK',
  missing_sessions: 0,
  cohorts: [liveAllPendingCohort, liveZeroMatchCohort],
};

export const nullBenchmarkScoreboardFixture = {
  ...scoreboardFixture,
  market: 'AU',
  benchmark: null,
  recording_since: '2025-06-02',
  scored_since: '2025-11-04',
  missing_sessions: 0,
  cohorts: [
    {
      ...liveAllPendingCohort,
      preset: 'canslim',
      name: 'CANSLIM',
      first_date: '2025-11-05',
      last_date: '2026-10-20',
      events: 64,
      signal_dates: 14,
      horizons: [
        horizon(5, closed(64, 11, 7.3, 0.55, null, 1.1, null, { excess_sample_size: 0 })),
        horizon(20, pending(64, 9)),
        horizon(60, pending(64, 9)),
      ],
    },
  ],
};

const liveAllDroppedCohort = {
  ...liveAllPendingCohort,
  preset: 'momentum',
  name: 'Momentum',
  definition: '4d2a9f0c7e1b5386',
  events: 9,
  signal_dates: 2,
  horizons: [
    horizon(5, { skipped_missing_history: 3, suppressed_overlapping: 6 }),
    horizon(20, { skipped_missing_history: 3, suppressed_overlapping: 6 }),
    horizon(60, { skipped_missing_history: 3, suppressed_overlapping: 6 }),
  ],
};

export const nothingPendingScoreboardFixture = {
  ...scoreboardFixture,
  market: 'KR',
  cohorts: [liveAllDroppedCohort, liveZeroMatchCohort],
};
