/**
 * Which markets the Operations page shows up front. A market is active when
 * the app enables it, a market worker (market_jobs_* / user_scans_*) reads
 * its queue (the deployment runs it, e.g. CN served but not enabled), or it
 * has live state: a
 * job, a non-empty queue, a held lease or chain claim, an open alert. Rows of
 * other markets sit behind an "Inactive markets" disclosure.
 */

// Job states that are live work; failed and cancelled jobs stay in the job
// console but no longer make their market active.
export const ACTIVE_JOB_STATES = new Set(['queued', 'waiting', 'reserved', 'running', 'stale', 'stuck']);

// Queue names end in the market code: data_fetch_jp, market_jobs_jp, user_scans_jp.
export function queueMarket(queueName, supportedMarkets = []) {
  const code = String(queueName ?? '').split('_').pop().toUpperCase();
  return supportedMarkets.includes(code) ? code : null;
}

// Queues a worker serves for one market only. The shared datafetch-global
// worker reads data_fetch_* for every market (start_celery.sh subscribes it to
// all of them), so its subscription says nothing about which markets run.
const MARKET_WORKER_QUEUE = /^(market_jobs|user_scans)_/;

export function findActiveMarkets({ enabledMarkets = [], supportedMarkets = [], jobsPayload, alerts } = {}) {
  const active = new Set(enabledMarkets);
  const addQueue = (queueName) => {
    const market = queueMarket(queueName, supportedMarkets);
    if (market) active.add(market);
  };
  const addServedQueue = (queueName) => {
    if (MARKET_WORKER_QUEUE.test(queueName)) addQueue(queueName);
  };
  jobsPayload?.workers?.forEach((worker) => worker.queues?.forEach(addServedQueue));
  // consumers also counts a worker that missed this inspect poll (busy solo
  // pool) by its heartbeat, which workers[] alone would drop.
  jobsPayload?.queues?.forEach((queue) => {
    if (queue.depth > 0) addQueue(queue.queue);
    else if (queue.consumers > 0) addServedQueue(queue.queue);
  });
  jobsPayload?.jobs?.forEach((job) => {
    if (job.market && ACTIVE_JOB_STATES.has(job.state)) active.add(job.market);
  });
  Object.entries(jobsPayload?.leases?.market_workload ?? {}).forEach(([market, holder]) => {
    if (holder) active.add(market);
  });
  Object.keys(jobsPayload?.leases?.pipeline_chain ?? {}).forEach((market) => active.add(market));
  alerts?.forEach((alert) => {
    if (alert.market && alert.state !== 'closed') active.add(alert.market);
  });
  return active;
}

// Rows with no market, an active market, or that keep() marks stay shown.
// active null (not known yet) shows everything.
export function splitByMarket(items, marketOf, active, keep = () => false) {
  const shown = [];
  const hidden = [];
  const hiddenMarkets = new Set();
  items.forEach((item) => {
    const market = marketOf(item);
    if (!active || !market || active.has(market) || keep(item)) {
      shown.push(item);
    } else {
      hidden.push(item);
      hiddenMarkets.add(market);
    }
  });
  // Sorted so every section's disclosure names them in the same order.
  return { shown, hidden, hiddenMarkets: [...hiddenMarkets].sort() };
}
