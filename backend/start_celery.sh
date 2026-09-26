#!/bin/bash
#
# Start Celery workers for stock scanner background tasks.
#
# Coordinated queue topology:
#   - One global datafetch worker subscribed to all data_fetch_* queues with
#     concurrency 1 so external-provider fetches are serialized across markets.
#   - One marketjobs-<market> worker per enabled market for breadth, group
#     rankings, and feature snapshots.
#   - One userscans-<market> worker per enabled market plus a shared safety-net
#     worker for any scan task dispatched without an explicit market.
#
# Usage: ./start_celery.sh
# Override enabled markets: ENABLED_MARKETS="US,HK" ./start_celery.sh
#

cd "$(dirname "$0")"

# macOS fork() safety: required for Python deps that initialize Objective-C after fork.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

echo "Starting Celery workers..."

# On macOS, use solo pool to avoid fork() crashes with Objective-C runtime safety checks.
# Solo pool runs tasks sequentially in the main process (no forking)
# On Linux, you can change this to 'prefork' for parallel execution
POOL="${CELERY_POOL:-solo}"

# Supported markets and queue names are owned by the backend Market Catalog.
SUPPORTED_MARKETS="$(./venv/bin/python - <<'PY'
from app.tasks.market_queues import SUPPORTED_MARKETS
print(",".join(SUPPORTED_MARKETS))
PY
)"
DATA_FETCH_QUEUES="$(./venv/bin/python - <<'PY'
from app.tasks.market_queues import all_data_fetch_queues
print(",".join(all_data_fetch_queues()))
PY
)"

# Enabled markets (comma-separated). Override via env to skip markets locally.
ENABLED_MARKETS="${ENABLED_MARKETS:-$SUPPORTED_MARKETS}"

echo "  Pool: $POOL"
echo "  Enabled markets: $ENABLED_MARKETS"

# General compute queue (unchanged from pre-9.1).
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q celery \
    -n general@%h &

# Global data-fetch worker: handles all external fetch queues under a single
# concurrency-1 worker so yfinance-bound jobs never overlap across markets.
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    --concurrency=1 \
    -Q "$DATA_FETCH_QUEUES" \
    -n datafetch-global@%h &

# Shared user-scans worker — same safety-net pattern for user-initiated scans.
./venv/bin/celery -A app.celery_app worker \
    --loglevel=info \
    --pool="$POOL" \
    -Q user_scans_shared \
    -n userscans-shared@%h &

# One worker per enabled market for market compute/write jobs and market scans.
IFS=',' read -ra MARKET_ARRAY <<< "$ENABLED_MARKETS"
for RAW_MARKET in "${MARKET_ARRAY[@]}"; do
    MARKET_UPPER="$(echo "$RAW_MARKET" | tr '[:lower:]' '[:upper:]' | xargs)"
    MARKET_LOWER="$(echo "$MARKET_UPPER" | tr '[:upper:]' '[:lower:]')"

    if [[ ",$SUPPORTED_MARKETS," != *",$MARKET_UPPER,"* ]]; then
        echo "  Skipping unknown market: $MARKET_UPPER"
        continue
    fi

    echo "  Starting marketjobs-$MARKET_LOWER and userscans-$MARKET_LOWER workers"

    ./venv/bin/celery -A app.celery_app worker \
        --loglevel=info \
        --pool="$POOL" \
        -Q "market_jobs_${MARKET_LOWER}" \
        -n "marketjobs-${MARKET_LOWER}@%h" &

    ./venv/bin/celery -A app.celery_app worker \
        --loglevel=info \
        --pool="$POOL" \
        -Q "user_scans_${MARKET_LOWER}" \
        -n "userscans-${MARKET_LOWER}@%h" &
done

# Company exposure research: opt-in dedicated worker. It never consumes the
# price-fetch queues; provider pacing is shared through RateBudgetPolicy keys.
if [[ "${EXPOSURE_WORKER_ENABLED:-false}" == "true" ]]; then
    echo "  Starting exposure-research worker"
    ./venv/bin/celery -A app.celery_app worker \
        --loglevel=info \
        --pool="$POOL" \
        --concurrency=1 \
        --prefetch-multiplier=1 \
        -Q exposure_research \
        -n exposure-research@%h &
fi

echo "Workers started. Use 'pkill -f celery' to stop."
wait
