/**
 * Is each market wired up? Three places decide what runs, and they can
 * disagree: the app's enabled markets (chosen at setup), the workers started
 * for each market's queues, and Beat's schedule (both from ENABLED_MARKETS in
 * the deployment env). This turns the page's data into plain-language
 * problems per market.
 */

// Each market's queues and what stops without a worker on them.
const QUEUE_ROLES = [
  ['market_jobs', 'runs its pipeline stages'],
  ['data_fetch', 'fetches its prices'],
  ['user_scans', 'runs its manual scans'],
];

// Beat runs each daily pipeline Mon-Fri and records even a holiday run as
// Skipped, so a gap beyond a long weekend means Beat is not running it.
const PIPELINE_SILENT_MS = 4 * 24 * 3600 * 1000;

// queues: the jobs payload's queue summaries (consumers null when unknown).
// pipelineTask: the market's daily-market-pipeline task row, if loaded.
export function marketCoverage({ market, enabled, queues = [], pipelineTask, now = Date.now() }) {
  const consumersOf = new Map(queues.map((queue) => [queue.queue, queue.consumers]));
  const workers = {};
  const problems = [];
  QUEUE_ROLES.forEach(([family, role]) => {
    const queueName = `${family}_${market.toLowerCase()}`;
    const consumers = consumersOf.get(queueName) ?? null;
    workers[family] = consumers;
    if (enabled && consumers === 0) problems.push(`No worker ${role} (${queueName}).`);
  });

  if (enabled && pipelineTask) {
    const startedAt = Date.parse(pipelineTask.last_run?.started_at ?? '');
    if (Number.isNaN(startedAt)) {
      problems.push('No daily pipeline run recorded; Beat may not schedule it.');
    } else if (now - startedAt > PIPELINE_SILENT_MS) {
      const days = Math.floor((now - startedAt) / (24 * 3600 * 1000));
      problems.push(`No daily pipeline run in ${days} days; Beat may not schedule it.`);
    }
  }

  if (!enabled && Object.values(workers).some((count) => count > 0)) {
    problems.push('Workers serve it, but it is not enabled in the app.');
  }

  // pipelineKnown false: the task list is not loaded or has no pipeline job
  // for this market, so its last run is unknown rather than Never.
  return {
    market,
    enabled,
    workers,
    pipelineKnown: Boolean(pipelineTask),
    pipelineRun: pipelineTask?.last_run ?? null,
    problems,
  };
}
