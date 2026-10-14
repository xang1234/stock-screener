/**
 * Is each market wired up? Three places decide what runs, and they can
 * disagree: the app's enabled markets (chosen at setup), the workers started
 * for each market's queues, and Beat's schedule (both from ENABLED_MARKETS in
 * the deployment env). This turns the page's data into plain-language
 * problems per market, and raises none it cannot back up.
 */

// Each market's queues and what stops without a worker on them.
const QUEUE_ROLES = [
  ['market_jobs', 'runs its pipeline stages'],
  ['data_fetch', 'fetches its prices'],
  ['user_scans', 'runs its manual scans'],
];
// Workers that serve one market only. The shared datafetch-global worker reads
// every market's data_fetch queue under start_celery.sh, so it proves nothing.
const MARKET_WORKER_FAMILIES = ['market_jobs', 'user_scans'];

// Beat starts each daily pipeline Mon-Fri, and a worker records even a
// holiday run as Skipped, so a gap beyond a long weekend means it stopped.
const PIPELINE_SILENT_MS = 4 * 24 * 3600 * 1000;
const DAY_MS = 24 * 3600 * 1000;

// queues: the jobs payload's queue summaries (consumers null when unknown).
// pipelineTask: the market's daily-market-pipeline task row, if known.
// deploymentMarkets: the deployment's ENABLED_MARKETS. RuntimeContext falls back
// to every supported market for an older backend, which never raises a problem.
export function marketCoverage({ market, enabled, deploymentMarkets, queues = [], pipelineTask, now = Date.now() }) {
  const consumersOf = new Map(queues.map((queue) => [queue.queue, queue.consumers]));
  const workers = {};
  const problems = [];
  const notes = [];
  QUEUE_ROLES.forEach(([family, role]) => {
    const queueName = `${family}_${market.toLowerCase()}`;
    const consumers = consumersOf.get(queueName) ?? null;
    workers[family] = consumers;
    if (enabled && consumers === 0) problems.push(`No worker ${role} (${queueName}).`);
  });

  // Never run is not an alarm: a fresh install or a newly enabled market has
  // no run until its first weekday after the warm-up time.
  let pipelineNeverRan = false;
  let pipelineScheduleUnconfirmed = false;
  if (enabled && pipelineTask) {
    const startedAt = Date.parse(pipelineTask.last_run?.started_at ?? '');
    if (pipelineTask.is_enabled === false) {
      notes.push('Scheduled pipeline runs are off (CACHE_WARMUP_ENABLED).');
    } else if (Number.isNaN(startedAt)) {
      pipelineNeverRan = true;
    } else if (now - startedAt > PIPELINE_SILENT_MS) {
      // last_run is the latest of any trigger, so no run at all since.
      const days = Math.floor((now - startedAt) / DAY_MS);
      problems.push(`No daily pipeline run has started in ${days} days; check Beat and the market's pipeline worker.`);
    } else if (pipelineTask.last_run.triggered_by === 'manual') {
      // A recent Run click says nothing about Beat; the next scheduled run will.
      pipelineScheduleUnconfirmed = true;
    }
  }

  if (!enabled && MARKET_WORKER_FAMILIES.some((family) => workers[family] > 0)) {
    problems.push('Workers serve it, but it is not enabled in the app.');
  }

  // pipelineKnown false: the task list is not loaded or has no pipeline job
  // for this market, so its last run is unknown rather than Never.
  const row = {
    market,
    enabled,
    outsideDeployment: false,
    workers,
    pipelineKnown: Boolean(pipelineTask),
    pipelineRun: pipelineTask?.last_run ?? null,
    pipelineNeverRan,
    pipelineScheduleUnconfirmed,
    problems,
    notes,
  };
  if (!enabled || !deploymentMarkets || deploymentMarkets.includes(market)) return row;
  // Nothing starts its workers or pipeline, so those findings are only
  // consequences of this one.
  return {
    ...row,
    outsideDeployment: true,
    pipelineNeverRan: false,
    pipelineScheduleUnconfirmed: false,
    problems: ["Enabled in the app, but not in this deployment's ENABLED_MARKETS, so nothing runs it."],
    notes: [],
  };
}
