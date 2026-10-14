/**
 * Which markets the Operations page shows up front. A market is active when
 * the app enables it, a worker reads one of its queues (the deployment runs
 * it, e.g. CN scheduled by Beat but not enabled), or it has live state: a
 * job, a non-empty queue, a held lease or chain claim, an open alert. Rows of
 * other markets sit behind an "Inactive markets" disclosure.
 */

// Queue names end in the market code: data_fetch_jp, market_jobs_jp, user_scans_jp.
export function queueMarket(queueName, supportedMarkets = []) {
  const code = String(queueName ?? '').split('_').pop().toUpperCase();
  return supportedMarkets.includes(code) ? code : null;
}

export function findActiveMarkets({ enabledMarkets = [], supportedMarkets = [], jobsPayload, alerts } = {}) {
  const active = new Set(enabledMarkets);
  const addQueue = (queueName) => {
    const market = queueMarket(queueName, supportedMarkets);
    if (market) active.add(market);
  };
  jobsPayload?.workers?.forEach((worker) => worker.queues?.forEach(addQueue));
  jobsPayload?.queues?.forEach((queue) => {
    if (queue.depth > 0) addQueue(queue.queue);
  });
  jobsPayload?.jobs?.forEach((job) => {
    if (job.market) active.add(job.market);
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
